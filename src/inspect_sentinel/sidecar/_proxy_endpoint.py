from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ._exchange import Answer, Pass, Refuse, Reply, Request
from ._handler import Handler, TimeLimitError
from ._http import MAX_BODY_BYTES, health_route, read_within
from ._inspect_client import SentinelField
from ._json_rpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    SENTINEL_FAILED,
    TIME_LIMIT,
    UNREADABLE,
    RpcError,
    answered,
    failed,
    invalid_params,
    read_call,
)
from ._wire import Format, UnreadableError

logger = logging.getLogger(__name__)

PATH = "/proxy/v1"
"""The one path the endpoint serves. The `v1` is the version of its methods."""

_PROVIDER = "The provider whose format the body is in, such as `anthropic` or `openai`. Where `format` is given, the body is read by `format`, and this can be any name the proxy has for the provider."
_FORMAT = "Which API the body is written in: `anthropic-messages`, `openai-chat` or `openai-responses`. A proxy that knows says, and the body is then read as that, whatever `provider` is. Without it, the API is worked out from `provider` and the body."


class JudgeRequestParams(BaseModel):
    """Parameters of `judge_request`: a model request, before it goes to the provider."""

    id: str = Field(
        description="Chosen by the proxy, and given again with the reply to this request."
    )
    provider: str = Field(description=_PROVIDER)
    format: Format | None = Field(default=None, description=_FORMAT)
    model: str = Field(description="The model, by the name the proxy shows for it.")
    user: str | None = Field(
        default=None, description="Who the proxy signed the caller in as."
    )
    headers: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="The caller's request headers the proxy passes on, without credentials. They are the caller's claims.",
    )
    body: dict[str, Any] = Field(description="The request's JSON body.")


class JudgeReplyParams(BaseModel):
    """Parameters of `judge_reply`: a provider's reply, held whole before the caller is sent any of it."""

    id: str = Field(description="The `id` of the request this is the reply to.")
    provider: str = Field(description=_PROVIDER)
    format: Format | None = Field(default=None, description=_FORMAT)
    model: str = Field(description="The model, by the name the proxy shows for it.")
    user: str | None = Field(
        default=None, description="Who the proxy signed the caller in as."
    )
    status: int = Field(description="The HTTP status of the provider's reply.")
    content_type: str = Field(
        description="The reply's content type, which says whether it is one JSON object or a stream of events."
    )
    body: str = Field(description="The reply's body, as base64.")


class PassResult(BaseModel):
    """Send the request or the reply on as it is."""

    action: Literal["pass"] = "pass"


class ReplaceRequestResult(BaseModel):
    """Send this request body to the provider in place of the one that was sent."""

    action: Literal["replace"] = "replace"
    body: dict[str, Any] = Field(description="The request's JSON body.")


class ReplaceReplyResult(BaseModel):
    """Send the caller this reply body in place of the provider's. An Inspect client's reply is always replaced, to add the `sentinel` field."""

    action: Literal["replace"] = "replace"
    body: str = Field(description="The reply's body, as base64.")


class RefuseResult(BaseModel):
    """Send nothing on, and answer the caller with an error."""

    action: Literal["refuse"] = "refuse"
    status: int = Field(ge=400, le=599, description="The HTTP status of the error.")
    message: str = Field(description="What the caller is told.")
    sentinel: SentinelField | None = Field(
        default=None,
        description="For an Inspect client: what was decided. The proxy hook puts it into the error it sends the caller, as a top-level `sentinel` field beside the error's own, without reading it.",
    )


_NOT_JUDGED = "A proxy hook refuses the call, unless its operator has chosen to let calls that weren't judged through."

_ERRORS: dict[int, tuple[str, str]] = {
    UNREADABLE: (
        "unreadable",
        f"The request or reply couldn't be read, so it was not judged. {_NOT_JUDGED}",
    ),
    SENTINEL_FAILED: (
        "sentinel_failed",
        f"The sentinel failed, so the request or reply was not judged. {_NOT_JUDGED}",
    ),
    TIME_LIMIT: (
        "time_limit",
        f"The sentinel didn't finish judging within the sidecar's time limit, and was cancelled. {_NOT_JUDGED}",
    ),
}

_METHODS: dict[str, tuple[type[BaseModel], tuple[type[BaseModel], ...]]] = {
    "judge_request": (
        JudgeRequestParams,
        (PassResult, ReplaceRequestResult, RefuseResult),
    ),
    "judge_reply": (
        JudgeReplyParams,
        (PassResult, ReplaceReplyResult, RefuseResult),
    ),
}


def proxy_app(handler: Handler, *, max_body_bytes: int = MAX_BODY_BYTES) -> Starlette:
    """An ASGI app that answers proxies: the sidecar's endpoint for them.

    Every call is a JSON-RPC 2.0 request POSTed to `/proxy/v1`. A proxy hook calls `judge_request` with each model request and `judge_reply` with the provider's reply, and each is answered with a result whose `action` is `pass`, `replace` or `refuse`. A proxy hook is a package of its own, and nothing of this package runs in the proxy.

    A refusal is a result, because it is a decision. A request or reply that was not judged is answered with a JSON-RPC error: one that couldn't be read, one the sentinel failed on, and one the sentinel didn't finish judging within the handler's time limit each have a code of their own, which the schema lists under `errors`. A proxy hook refuses the call on any of them, unless its operator has chosen to let calls that weren't judged through, as Inspect fails a sample whose sentinel fails. A call larger than `max_body_bytes` is not read, and is answered with an HTTP 413.

    The app also answers a health check at `/health`, which a proxy doesn't call.

    Args:
        handler: Judges each request and each reply.
        max_body_bytes: The largest call that is read. It has to cover the largest reply the proxy holds, which is sent as base64.
    """

    async def call(http: HttpRequest) -> Response:
        body = await read_within(http, max_body_bytes)
        if body is None:
            return JSONResponse(
                {"error": f"A call is at most {max_body_bytes} bytes."},
                status_code=413,
            )
        id: int | str | None = None
        try:
            asked = read_call(body)
            id = asked.id
            if id is None:
                raise RpcError(
                    INVALID_REQUEST,
                    "A call needs an id, because its answer says what to do.",
                )
            result = await _judge(handler, asked.method, asked.params)
        except RpcError as ex:
            return failed(id, ex)
        return answered(id, result)

    return Starlette(routes=[Route(PATH, call, methods=["POST"]), health_route()])


async def _judge(handler: Handler, method: str, params: dict[str, Any]) -> BaseModel:
    if method not in _METHODS:
        raise RpcError(METHOD_NOT_FOUND, f"There is no method {method!r}.")
    try:
        given = _METHODS[method][0].model_validate(params)
    except ValidationError as ex:
        raise invalid_params(ex) from None
    if isinstance(given, JudgeRequestParams):
        what, id = "request", given.id
        judging = handler.request(_request(given))
    elif isinstance(given, JudgeReplyParams):
        what, id = "reply", given.id
        judging = handler.reply(_reply(given))
    else:
        raise TypeError(f"No method takes {type(given).__name__}.")
    try:
        return _result(await judging, reply=what == "reply")
    except TimeLimitError as ex:
        raise RpcError(TIME_LIMIT, str(ex)) from None
    except UnreadableError as ex:
        # why is kept here, since it may quote the request or the reply
        logger.warning("The %s %r was not judged: %s", what, id, ex)
        raise RpcError(
            UNREADABLE, f"The {what} couldn't be read, so it was not judged."
        ) from None
    except Exception:
        logger.exception("The %s %r was not judged.", what, id)
        raise RpcError(
            SENTINEL_FAILED, f"The sentinel failed, so the {what} was not judged."
        ) from None


def _request(given: JudgeRequestParams) -> Request:
    return Request(
        id=given.id,
        provider=given.provider,
        format=given.format,
        model=given.model,
        user=given.user,
        body=json.dumps(given.body).encode(),
        headers={name.lower(): value for name, value in given.headers.items()},
    )


def _reply(given: JudgeReplyParams) -> Reply:
    try:
        body = base64.b64decode(given.body, validate=True)
    except ValueError:
        # binascii.Error for what isn't base64, and a plain ValueError for
        # what isn't even ASCII
        raise RpcError(
            INVALID_PARAMS, "The parameters don't fit: body: A reply's body is base64."
        ) from None
    return Reply(
        id=given.id,
        provider=given.provider,
        format=given.format,
        model=given.model,
        user=given.user,
        status=given.status,
        content_type=given.content_type,
        body=body,
    )


def _result(answer: Answer, *, reply: bool) -> BaseModel:
    if isinstance(answer, Refuse):
        return RefuseResult(
            status=answer.status,
            message=answer.message,
            sentinel=(
                SentinelField.model_validate(answer.sentinel)
                if answer.sentinel is not None
                else None
            ),
        )
    if isinstance(answer, Pass):
        return PassResult()
    if reply:
        return ReplaceReplyResult(body=base64.b64encode(answer.body).decode("ascii"))
    return ReplaceRequestResult(body=json.loads(answer.body))


def schema() -> dict[str, Any]:
    """The JSON Schema of each method's parameters and results.

    A call to method `m` has `params` that fit the definition named in `methods[m].params`, and is answered with a `result` that fits one of the definitions named in `methods[m].result`. A call that was not judged is answered with a JSON-RPC error whose code is a key of `errors`.
    """
    models = {
        model: None
        for params, results in _METHODS.values()
        for model in (params, *results)
    }
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="Inspect Sentinel: the endpoint for proxies",
        description=f"JSON-RPC 2.0 methods, POSTed to {PATH}. `methods` names the definition that a method's `params` fit, and the definitions that its `result` fits one of. `errors` lists the codes of the JSON-RPC errors that say a call was not judged, and what a proxy hook does on each.",
    )
    return {
        "$schema": GenerateJsonSchema.schema_dialect,
        **definitions,
        "methods": {
            name: {
                "params": {"$ref": f"#/$defs/{params.__name__}"},
                "result": {
                    "oneOf": [
                        {"$ref": f"#/$defs/{result.__name__}"} for result in results
                    ]
                },
            }
            for name, (params, results) in _METHODS.items()
        },
        "errors": {
            str(code): {"name": name, "description": description}
            for code, (name, description) in _ERRORS.items()
        },
    }


SCHEMA_FILE = Path(__file__).with_name("proxy_endpoint.schema.json")
"""Where `schema()` is kept, for a proxy hook to be written and tested against without running this package."""


if __name__ == "__main__":
    # rewrites the schema file; run after changing a model above
    SCHEMA_FILE.write_text(json.dumps(schema(), indent=2) + "\n")
