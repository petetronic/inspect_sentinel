from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse
from starlette.routing import Route

from ._exchange import Answer, Pass, Refuse, Reply, Request
from ._handler import Handler, TimeLimitError
from ._http import MAX_BODY_BYTES, health_route, read_within


class RequestMessage(BaseModel):
    """A model request, sent before it goes to the provider. It is answered with a `PassAnswer`, a `ReplaceAnswer` or a `RefuseAnswer`."""

    version: Literal[1] = Field(
        description="The version of these messages. A message in another version is not read."
    )
    phase: Literal["request"]
    id: str = Field(description="Shared by a request and its reply.")
    provider: str = Field(
        description="The provider whose format the body is in, such as `anthropic` or `openai`."
    )
    model: str = Field(description="The model, by the name the proxy shows for it.")
    user: str | None = Field(
        default=None, description="Who the proxy signed the caller in as."
    )
    run: str | None = Field(
        default=None,
        description="The run the request belongs to, where the proxy can tell. A sentinel's state is kept by run.",
    )
    headers: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="The caller's request headers the proxy passes on, without credentials. They are the caller's claims.",
    )
    body: dict[str, Any] = Field(description="The request's JSON body.")


class ReplyMessage(BaseModel):
    """A provider's reply, held whole before the caller is sent any of it. It is answered with a `PassAnswer`, a `ReplaceAnswer` or a `RefuseAnswer`."""

    version: Literal[1] = Field(
        description="The version of these messages. A message in another version is not read."
    )
    phase: Literal["reply"]
    id: str = Field(description="The `id` of the request this is the reply to.")
    provider: str
    model: str
    user: str | None = None
    status: int = Field(description="The HTTP status of the provider's reply.")
    content_type: str = Field(
        description="The reply's content type, which says whether it is one JSON object or a stream of events."
    )
    body: str = Field(description="The reply's body, as base64.")


class PassAnswer(BaseModel):
    """Send the request or the reply on as it is."""

    action: Literal["pass"] = "pass"
    headers: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Response headers to add when this answers a reply, each named `x-...`.",
    )


class ReplaceAnswer(BaseModel):
    """Send this body on in place of the one that was sent."""

    action: Literal["replace"] = "replace"
    body: dict[str, Any] | str = Field(
        description="A JSON object for a request, and base64 for a reply."
    )
    headers: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Response headers to add when this answers a reply, each named `x-...`.",
    )


class RefuseAnswer(BaseModel):
    """Send nothing on, and answer the caller with an error."""

    action: Literal["refuse"] = "refuse"
    status: int = Field(ge=400, le=599, description="The HTTP status of the error.")
    message: str = Field(description="What the caller is told.")
    headers: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="Response headers to add to the error, each named `x-...`.",
    )


_MESSAGE: TypeAdapter[RequestMessage | ReplyMessage] = TypeAdapter(
    Annotated[RequestMessage | ReplyMessage, Field(discriminator="phase")]
)


def proxy_app(handler: Handler, *, max_body_bytes: int = MAX_BODY_BYTES) -> Starlette:
    """An ASGI app that answers proxies: the sidecar's endpoint for them.

    A hook that a proxy runs, such as `MiddlemanHook`, POSTs each model request and its reply to `/` as JSON, and is answered with `pass`, `replace` or `refuse`. The hook is a package of its own, and nothing of this package runs in the proxy. A request or reply the handler can't judge is answered with a 500, one in a version of the messages this doesn't read with a 400, one larger than `max_body_bytes` with a 413, and one the sentinel didn't finish judging within the handler's time limit with a 504. The proxy's side says whether each of those refuses the exchange or lets it through.

    The app also answers a health check at `/health`, which a proxy doesn't call.

    Args:
        handler: Judges each exchange.
        max_body_bytes: The largest message that is read. It has to cover the largest reply the proxy holds, which is sent as base64.
    """

    async def exchange(http: HttpRequest) -> JSONResponse:
        sent = await read_within(http, max_body_bytes)
        if sent is None:
            return JSONResponse(
                {"error": f"A message is at most {max_body_bytes} bytes."},
                status_code=413,
            )
        try:
            message = _MESSAGE.validate_json(sent)
        except ValidationError as ex:
            return JSONResponse({"error": str(ex)}, status_code=400)
        try:
            return await judge(message)
        except TimeLimitError as ex:
            return JSONResponse({"error": str(ex)}, status_code=504)

    async def judge(message: RequestMessage | ReplyMessage) -> JSONResponse:
        if isinstance(message, RequestMessage):
            answer = await handler.request(
                Request(
                    id=message.id,
                    provider=message.provider,
                    model=message.model,
                    user=message.user,
                    body=json.dumps(message.body).encode(),
                    headers=message.headers,
                    run=message.run,
                )
            )
            return JSONResponse(_answer(answer, reply=False))
        try:
            body = base64.b64decode(message.body, validate=True)
        except binascii.Error:
            return JSONResponse({"error": "A reply's body is base64."}, status_code=400)
        answer = await handler.reply(
            Reply(
                id=message.id,
                provider=message.provider,
                model=message.model,
                user=message.user,
                status=message.status,
                content_type=message.content_type,
                body=body,
            )
        )
        return JSONResponse(_answer(answer, reply=True))

    return Starlette(routes=[Route("/", exchange, methods=["POST"]), health_route()])


def _answer(answer: Answer, *, reply: bool) -> dict[str, Any]:
    if isinstance(answer, Refuse):
        return RefuseAnswer(
            status=answer.status, message=answer.message, headers=dict(answer.headers)
        ).model_dump()
    # an answer that names no headers is sent without the field
    unset = None if answer.headers else {"headers"}
    if isinstance(answer, Pass):
        return PassAnswer(headers=dict(answer.headers)).model_dump(exclude=unset)
    # a replaced request is a JSON object, and a replaced reply is base64
    return ReplaceAnswer(
        body=base64.b64encode(answer.body).decode("ascii")
        if reply
        else json.loads(answer.body),
        headers=dict(answer.headers),
    ).model_dump(exclude=unset)


def schema() -> dict[str, Any]:
    """The JSON Schema of the messages a proxy sends and the answers it is given.

    A message fits the definition named in `messages` for its `phase`, and is answered with a JSON object that fits one of the definitions named in `answers`.
    """
    models = (RequestMessage, ReplyMessage, PassAnswer, ReplaceAnswer, RefuseAnswer)
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="Inspect Sentinel: the endpoint for proxies",
        description="JSON messages, POSTed to `/`. `messages` names the definition a message of each phase fits, and `answers` the definitions an answer fits.",
    )
    return {
        "$schema": GenerateJsonSchema.schema_dialect,
        **definitions,
        "messages": {
            "request": {"$ref": f"#/$defs/{RequestMessage.__name__}"},
            "reply": {"$ref": f"#/$defs/{ReplyMessage.__name__}"},
        },
        "answers": {
            "pass": {"$ref": f"#/$defs/{PassAnswer.__name__}"},
            "replace": {"$ref": f"#/$defs/{ReplaceAnswer.__name__}"},
            "refuse": {"$ref": f"#/$defs/{RefuseAnswer.__name__}"},
        },
    }


SCHEMA_FILE = Path(__file__).with_name("proxy_endpoint.schema.json")
"""Where `schema()` is kept, for a proxy's hook to be written and tested against without running this package."""


if __name__ == "__main__":
    # rewrites the schema file; run after changing a model above
    SCHEMA_FILE.write_text(json.dumps(schema(), indent=2) + "\n")
