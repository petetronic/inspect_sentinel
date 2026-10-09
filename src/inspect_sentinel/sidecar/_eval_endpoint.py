from __future__ import annotations

import hmac
import json
from pathlib import Path
from typing import Any, Literal

from inspect_ai.model import ChatMessage
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from pydantic_core import to_jsonable_python
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .._context import EvalContext
from .._report import Failed
from ._handler import Handler, RegistrationError, RequestResult
from ._host import Record
from ._http import health_route, read_within

PATH = "/sentinel/v1"
"""The one path the endpoint serves. The `v1` is the version of its methods."""

# an eval's registration holds a sample's input, which is the largest thing sent
_MAX_BODY_BYTES = 4 * 1024 * 1024

# JSON-RPC 2.0's own codes
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
# this endpoint's, in the range JSON-RPC leaves to a server
_REGISTRATION_REFUSED = -32001
_UNKNOWN_RUN = -32002


class EvalDetails(BaseModel):
    """The task and the sample a run is: what a sentinel is given as `context.eval`."""

    task: str
    task_description: str | None = None
    sample_id: str | int
    epoch: int
    sample_description: str | None = None
    sample_input: str | list[dict[str, Any]] = Field(
        description="A string, or chat messages as inspect_ai writes them to JSON."
    )
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])


class RegisterRunParams(BaseModel):
    """Parameters of `register_run`."""

    run: str = Field(description="The name the run's model requests carry.")
    resume: bool = Field(
        default=False,
        description="Whether the eval is carrying on a run it began earlier.",
    )
    eval: EvalDetails


class RegisterRunResult(BaseModel):
    """Result of `register_run`."""

    run: str
    state: Literal["new", "resumed"]


class RequestResultParams(BaseModel):
    """Parameters of `request_result`."""

    id: str = Field(
        description="The id the eval put on the request, in the header the sidecar reads it from. That header is inspect_ai's `x-irid` unless the sidecar was started with another."
    )


class Recorded(BaseModel):
    """One thing recorded while a sentinel examined a step."""

    kind: Literal["record", "failed", "cancelled", "bypassed", "superseded"]
    path: str = Field(description="The instance path of the monitor or protocol.")
    factory: str = Field(description="Registry name of the instance's factory.")
    name: str = Field(description="The instance's name.")
    call: str | None = Field(
        default=None, description="The id of the tool call the step was about."
    )
    report: dict[str, Any] | None = Field(
        default=None,
        description="The observation or decision, for `record` and `superseded`.",
    )
    error: str | None = Field(
        default=None, description="What the monitor raised, for `failed`."
    )


class RequestResultResult(BaseModel):
    """Result of `request_result`."""

    id: str
    records: Literal["held", "dropped", "unknown"] = Field(
        description="`held` if the records follow, `dropped` if they are no longer kept, `unknown` if the request was never seen. Only `held` has the other fields."
    )
    state: Literal["judging", "awaiting_reply", "decided", "not_judged"] | None = Field(
        default=None,
        description="Where the latest attempt at the request stands. `judging`: the sentinel is at work on it. `awaiting_reply`: the request went on and the proxy has handed over no reply, which it never does for a call the provider failed. `decided`: `outcome` says what. `not_judged`: judging ended without a decision, and `reason` says why.",
    )
    reason: Literal["unreadable", "time_limit", "error", "provider_error"] | None = (
        Field(
            default=None,
            description="Why the request was not judged. `unreadable`: it or its reply couldn't be read. `time_limit`: the sentinel passed its time limit. `error`: the sentinel raised. `provider_error`: the provider's reply was an error, so there was nothing to judge.",
        )
    )
    outcome: Literal["continue", "reject", "terminate"] | None = Field(
        default=None,
        description="What the sentinel decided for the latest attempt at the request, where `state` is `decided`.",
    )
    message: str | None = Field(default=None, description="What the model is told.")
    recorded: list[Recorded] | None = None


class RunResultParams(BaseModel):
    """Parameters of `run_result`."""

    run: str


class RunResultResult(BaseModel):
    """Result of `run_result`."""

    run: str
    outcome: Literal["continue", "terminate"]
    store: dict[str, Any]


class _Call(BaseModel):
    jsonrpc: Literal["2.0"]
    method: str
    params: dict[str, Any] = Field(default_factory=dict[str, Any])
    id: int | str | None = None


_MESSAGES: TypeAdapter[list[ChatMessage]] = TypeAdapter(list[ChatMessage])

_METHODS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "register_run": (RegisterRunParams, RegisterRunResult),
    "request_result": (RequestResultParams, RequestResultResult),
    "run_result": (RunResultParams, RunResultResult),
}


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def eval_app(handler: Handler, *, token: str, results: bool = True) -> Starlette:
    """An ASGI app that answers evals: the sidecar's endpoint for them.

    An eval registers the sample a run is, and asks what became of a request and of a run. Every call is a JSON-RPC 2.0 request POSTed to `/sentinel/v1` with the token as a bearer credential. Nothing here loosens a decision: no method allows a call, clears a run's state or changes the sentinel.

    The app also answers a health check at `/health`, which asks for no token.

    Args:
        handler: The handler the proxy's requests and replies are put to, whose runs these methods read and brief.
        token: The credential a caller must present. An agent must not hold it.
        results: Whether `request_result` and `run_result` are served. A record holds each monitor's explanation, which tells an agent what the monitor looks for, so where a caller isn't trusted they are turned off and an eval can only register.

    Raises:
        ValueError: If `token` is empty.
    """
    if not token:
        raise ValueError("The endpoint for evals needs a token.")

    async def call(http: HttpRequest) -> Response:
        presented = http.headers.get("authorization", "")
        if not hmac.compare_digest(presented.encode(), f"Bearer {token}".encode()):
            return JSONResponse(
                {"error": "A bearer token is required."},
                status_code=401,
                headers={"www-authenticate": "Bearer"},
            )
        body = await read_within(http, _MAX_BODY_BYTES)
        if body is None:
            return JSONResponse(
                {"error": f"A call is at most {_MAX_BODY_BYTES} bytes."},
                status_code=413,
            )
        id: int | str | None = None
        try:
            try:
                parsed: object = json.loads(body)
            except ValueError:
                raise _RpcError(_PARSE_ERROR, "The body is not JSON.") from None
            try:
                asked = _Call.model_validate(parsed)
            except ValidationError:
                raise _RpcError(
                    _INVALID_REQUEST, "The body is not one JSON-RPC 2.0 request."
                ) from None
            id = asked.id
            if not results and asked.method != "register_run":
                raise _RpcError(
                    _METHOD_NOT_FOUND, f"There is no method {asked.method!r}."
                )
            result = _answer(handler, asked.method, asked.params)
        except _RpcError as ex:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": ex.code, "message": ex.message},
                }
            )
        # a call without an id is a notification, which is not answered
        if id is None:
            return Response(status_code=204)
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": id,
                "result": to_jsonable_python(result, exclude_none=True),
            }
        )

    return Starlette(routes=[Route(PATH, call, methods=["POST"]), health_route()])


def _answer(handler: Handler, method: str, params: dict[str, Any]) -> BaseModel:
    if method not in _METHODS:
        raise _RpcError(_METHOD_NOT_FOUND, f"There is no method {method!r}.")
    try:
        given = _METHODS[method][0].model_validate(params)
        # read here, since a sample's input may hold messages that don't parse
        details = _details(given.eval) if isinstance(given, RegisterRunParams) else None
    except ValidationError as ex:
        # where the parameters are wrong and how, never the values sent
        where = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in ex.errors(include_input=False)
        )
        raise _RpcError(_INVALID_PARAMS, f"The parameters don't fit: {where}") from None

    if isinstance(given, RegisterRunParams):
        assert details is not None
        try:
            state = handler.register(given.run, details, resume=given.resume)
        except RegistrationError as ex:
            raise _RpcError(_REGISTRATION_REFUSED, str(ex)) from None
        return RegisterRunResult(run=given.run, state=state)
    if isinstance(given, RequestResultParams):
        found = handler.request_result(given.id)
        if not isinstance(found, RequestResult):
            return RequestResultResult(id=given.id, records=found)
        return RequestResultResult(
            id=given.id,
            records="held",
            state=found.state,
            reason=found.reason,
            outcome=found.outcome,
            message=found.message,
            recorded=[_recorded(record) for record in found.records],
        )
    assert isinstance(given, RunResultParams)
    run = handler.run_result(given.run)
    if run is None:
        raise _RpcError(_UNKNOWN_RUN, f"Nothing is held for run {given.run!r}.")
    return RunResultResult(run=given.run, outcome=run.outcome, store=run.store)


def _details(given: EvalDetails) -> EvalContext:
    return EvalContext(
        task=given.task,
        task_description=given.task_description,
        sample_id=given.sample_id,
        epoch=given.epoch,
        sample_description=given.sample_description,
        sample_input=(
            given.sample_input
            if isinstance(given.sample_input, str)
            else _MESSAGES.validate_python(given.sample_input)
        ),
        metadata=given.metadata,
    )


def _recorded(record: Record) -> Recorded:
    detail = record.detail
    if isinstance(detail, str):
        return Recorded(
            kind=_kind(record),
            path=record.path,
            factory=record.factory,
            name=detail,
            call=record.call,
        )
    if isinstance(detail, Failed):
        return Recorded(
            kind=_kind(record),
            path=record.path,
            factory=record.factory,
            name=detail.name,
            call=record.call,
            error=f"{type(detail.error).__name__}: {detail.error}",
        )
    return Recorded(
        kind=_kind(record),
        path=record.path,
        factory=record.factory,
        name=detail.name,
        call=record.call,
        report=detail.report.model_dump(mode="json", exclude_none=True),
    )


def _kind(
    record: Record,
) -> Literal["record", "failed", "cancelled", "bypassed", "superseded"]:
    kind = record.kind
    if (
        kind == "record"
        or kind == "failed"
        or kind == "cancelled"
        or kind == "bypassed"
        or kind == "superseded"
    ):
        return kind
    raise ValueError(f"A record of an unknown kind: {kind!r}.")


def schema() -> dict[str, Any]:
    """The JSON Schema of each method's parameters and result.

    A call to method `m` has `params` that fit the definition named in `methods[m].params`, and is answered with a `result` that fits `methods[m].result`.
    """
    models = [model for pair in _METHODS.values() for model in pair]
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="Inspect Sentinel: the endpoint for evals",
        description=f"JSON-RPC 2.0 methods, POSTed to {PATH}. `methods` names the definitions that a method's `params` and its `result` fit.",
    )
    return {
        "$schema": GenerateJsonSchema.schema_dialect,
        **definitions,
        "methods": {
            name: {
                "params": {"$ref": f"#/$defs/{params.__name__}"},
                "result": {"$ref": f"#/$defs/{result.__name__}"},
            }
            for name, (params, result) in _METHODS.items()
        },
    }


SCHEMA_FILE = Path(__file__).with_name("eval_endpoint.schema.json")
"""Where `schema()` is kept, for a client to read without running this package."""


if __name__ == "__main__":
    # rewrites the schema file; run after changing a model above
    SCHEMA_FILE.write_text(json.dumps(schema(), indent=2) + "\n")
