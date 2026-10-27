from __future__ import annotations

import hmac
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ._handler import Handler, RequestResult
from ._http import read_within
from ._inspect_client import (
    SentinelField,
    WithheldTurn,
    record_json,
    sentinel_field,
    withheld_turn,
)
from ._json_rpc import (
    METHOD_NOT_FOUND,
    REGISTRATION_REFUSED,
    REGISTRATIONS_FULL,
    UNKNOWN_SAMPLE,
    UNKNOWN_TASK,
    RpcError,
    answered,
    failed,
    invalid_params,
    read_call,
)
from ._registration import (
    RegistrationError,
    RegistrationsFullError,
    SampleFacts,
    TaskRegistration,
    UnknownTaskError,
)
from ._state import NotJudged, Outcome

PATH = "/sentinel/v1"
"""Where the methods are served. The `v1` is the version of the methods."""

INFO_PATH = "/sentinel/info"
"""The probe, which has no version, since it is how an eval learns which versions there are."""

VERSIONS = [1]
"""The versions of the methods this sidecar serves."""

TOKEN_VARIABLE = "INSPECT_SENTINEL_EVAL_TOKEN"
"""The environment variable the endpoint's own token is read from, when it has a port of its own."""


class RegisterTaskParams(TaskRegistration):
    """Parameters of `register_task`: a task an eval runs, and the ids of its samples, sent at task init."""

    eval_id: str = Field(description="The eval's id, as its requests name it.")
    expiry: float | None = Field(
        default=None,
        gt=0,
        description="How long to keep the registration once no request has named the task, in seconds. A day if not given.",
    )


class RegisterTaskResult(BaseModel):
    """What the sidecar stored, or that the task was registered beforehand and nothing was stored."""

    eval_id: str
    task: str
    samples: int = Field(description="How many samples the registration named.")
    fixed: bool = Field(
        description="Whether the task was registered beforehand, in the tasks folder. Its sentinel is then given the folder's facts, the registration is not compared with them, and nothing is stored."
    )
    expiry: float | None = Field(
        default=None,
        description="How long the registration is kept once no request has named the task, in seconds: what was asked, or less if the sidecar keeps none that long. Absent when `fixed`, as a task registered beforehand is never dropped.",
    )


class CompleteRegistrationParams(SampleFacts):
    """Parameters of `complete_registration`: one sample of a registered task, sent before the sample's first epoch."""

    eval_id: str = Field(description="The eval's id, as its requests name it.")
    task: str = Field(description="The task's name, as `register_task` gave it.")


class CompleteRegistrationResult(BaseModel):
    """That the sample's facts were stored, or that the task was registered beforehand and nothing was stored."""

    eval_id: str
    task: str
    sample_id: str | int
    fixed: bool = Field(
        description="Whether the task was registered beforehand, in the tasks folder. Its sentinel is then given the folder's facts, the sample is not compared with them, and nothing is stored."
    )


class RequestResultParams(BaseModel):
    """Parameters of `request_result`: one model request, by the eval's id for it."""

    id: str = Field(description="The request's `x-irid`.")


class RequestResultResult(BaseModel):
    """What happened to the request. With `dropped` or `unknown` records, nothing else is said."""

    id: str
    records: Literal["held", "dropped", "unknown"] = Field(
        description="`held` if the sidecar has the request's records, `dropped` if it had them and no longer keeps them, `unknown` if it never saw the request."
    )
    state: Literal["judging", "awaiting_reply", "decided", "not_judged"] | None = Field(
        default=None, description="Where the latest attempt stands."
    )
    reason: NotJudged | None = Field(
        default=None, description="Why the latest attempt was not judged."
    )
    outcome: Outcome | None = Field(
        default=None, description="What the latest attempt came to, once decided."
    )
    message: str | None = Field(
        default=None, description="What the model is told, for a rejection."
    )
    recorded: list[dict[str, Any]] | None = Field(
        default=None,
        description="What each monitor and protocol recorded, every attempt's in order.",
    )
    withheld: WithheldTurn | None = Field(
        default=None,
        description="The model's turn, when the latest attempt was refused it because a call in it was rejected, or because a sentinel ended the run at the reply or at a call in it, as the `sentinel` field holds it. An eval puts it into its own copy of the conversation, as it would from the field.",
    )
    proposed: WithheldTurn | None = Field(
        default=None,
        description="The model's turn as it proposed it, when the latest attempt was handed it with calls taken out because they were rejected, as the `sentinel` field holds it.",
    )
    sentinel: SentinelField | None = Field(
        default=None,
        description="The `sentinel` field of the latest attempt, once decided, as the reply or the refusal carried it, so an eval whose proxy didn't pass it on acts on it as it would have. Trimmed, as the field is, where the sidecar doesn't send results.",
    )


class SampleResultParams(BaseModel):
    """Parameters of `sample_result`: one sample in one epoch, before it is scored."""

    sample_uuid: str = Field(
        description="The sample's uuid, as `x-inspect-context` gives it: one sample in one epoch, the same across the sample's retries."
    )


class SampleResultResult(BaseModel):
    """What the sample came to, and the sentinel's store."""

    sample_uuid: str
    outcome: Literal["proceed", "terminate"] = Field(
        description="`terminate` if a sentinel ended the sample."
    )
    store: dict[str, Any] = Field(
        description="The sentinel's store, by the keys the sentinel wrote, to copy into the sample's store."
    )


class EndSampleParams(BaseModel):
    """Parameters of `end_sample`: one sample in one epoch, once it has been scored."""

    sample_uuid: str = Field(
        description="The sample's uuid, as `x-inspect-context` gives it."
    )


class EndSampleResult(BaseModel):
    """Whether the sample's state was dropped."""

    sample_uuid: str
    state: Literal["ended", "unknown"] = Field(
        description="`ended`, or `unknown` for a sample the sidecar never saw."
    )


_METHODS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "register_task": (RegisterTaskParams, RegisterTaskResult),
    "complete_registration": (
        CompleteRegistrationParams,
        CompleteRegistrationResult,
    ),
    "request_result": (RequestResultParams, RequestResultResult),
    "sample_result": (SampleResultParams, SampleResultResult),
    "end_sample": (EndSampleParams, EndSampleResult),
}

_RESULTS = frozenset({"sample_result"})

_ERRORS: dict[int, tuple[str, str]] = {
    REGISTRATION_REFUSED: (
        "registration_refused",
        "The sidecar takes no tasks registered by an eval, and this task was not registered beforehand; or, for `complete_registration`, the sidecar takes only the samples a task's registration names, and this sample isn't one.",
    ),
    REGISTRATIONS_FULL: (
        "registrations_full",
        "The sidecar keeps as many task registrations as it takes, and takes another once a kept one expires. A failure, not a refusal: the deployment has not said that the task may not run.",
    ),
    UNKNOWN_SAMPLE: (
        "unknown_sample",
        "The sidecar holds nothing for the sample, so an eval never mistakes a forgotten sample for a clean one.",
    ),
    UNKNOWN_TASK: (
        "unknown_task",
        "The sidecar holds no registration of the task by the eval: it was never registered, or is no longer kept. The eval registers the task again, then completes the sample's registration.",
    ),
}


def eval_routes(
    handler: Handler,
    *,
    max_body_bytes: int,
    token: str | None = None,
    results: bool = True,
) -> list[Route]:
    """The routes of the endpoint for evals: the probe at `/sentinel/info`, and the methods at `/sentinel/v1`.

    An eval registers its tasks and completes each sample's registration before the sample runs, asks what became of a request and of a sample, and ends a sample once it is scored. Every call is a JSON-RPC 2.0 request. No method allows a call, clears a tally or changes the sentinel. A call larger than `max_body_bytes` is not read, and is answered with an HTTP 413.

    Args:
        handler: The handler the proxy's requests and replies are put to.
        max_body_bytes: The largest call that is read.
        token: A credential every call to the methods must present as a bearer token, when the endpoint has a port of its own. None where it shares the proxy's port and access control.
        results: Whether the results are sent: `sample_result`, and what each monitor and protocol recorded in `request_result`. A record holds each monitor's explanation, which tells an agent what the monitor looks for, so where an eval can't be trusted they are turned off. `request_result` is still served without them, since it is how an eval learns that a turn was refused or edited when its proxy didn't pass on the `sentinel` field.
    """

    async def info(http: HttpRequest) -> Response:
        return JSONResponse(
            {
                "service": "inspect_sentinel",
                "versions": VERSIONS,
                "release": handler.release,
            }
        )

    async def call(http: HttpRequest) -> Response:
        if token is not None:
            presented = http.headers.get("authorization", "")
            if not hmac.compare_digest(presented.encode(), f"Bearer {token}".encode()):
                return JSONResponse(
                    {"error": "A bearer token is required."},
                    status_code=401,
                    headers={"www-authenticate": "Bearer"},
                )
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
            if not results and asked.method in _RESULTS:
                raise RpcError(
                    METHOD_NOT_FOUND,
                    f"This sidecar doesn't send results, so there is no method {asked.method!r}.",
                )
            result = await _answer(handler, asked.method, asked.params, results)
        except RpcError as ex:
            return failed(id, ex)
        return answered(id, result)

    return [
        Route(INFO_PATH, info, methods=["GET"]),
        Route(PATH, call, methods=["POST"]),
    ]


async def _answer(
    handler: Handler, method: str, params: dict[str, Any], results: bool = True
) -> BaseModel:
    if method not in _METHODS:
        raise RpcError(METHOD_NOT_FOUND, f"There is no method {method!r}.")
    try:
        given = _METHODS[method][0].model_validate(params)
    except ValidationError as ex:
        raise invalid_params(ex) from None
    if isinstance(given, RegisterTaskParams):
        registration = TaskRegistration.model_validate(
            given.model_dump(include=set(TaskRegistration.model_fields))
        )
        try:
            granted = await handler.register_task(
                given.eval_id, registration, given.expiry
            )
        except RegistrationsFullError as ex:
            raise RpcError(REGISTRATIONS_FULL, str(ex)) from None
        except RegistrationError as ex:
            raise RpcError(REGISTRATION_REFUSED, str(ex)) from None
        return RegisterTaskResult(
            eval_id=given.eval_id,
            task=given.task,
            samples=len(given.sample_ids),
            fixed=granted is None,
            expiry=granted,
        )
    if isinstance(given, CompleteRegistrationParams):
        sample = SampleFacts.model_validate(
            given.model_dump(include=set(SampleFacts.model_fields))
        )
        try:
            fixed = await handler.complete_registration(
                given.eval_id, given.task, sample
            )
        except UnknownTaskError as ex:
            raise RpcError(UNKNOWN_TASK, str(ex)) from None
        except RegistrationError as ex:
            raise RpcError(REGISTRATION_REFUSED, str(ex)) from None
        return CompleteRegistrationResult(
            eval_id=given.eval_id,
            task=given.task,
            sample_id=given.sample_id,
            fixed=fixed,
        )
    if isinstance(given, RequestResultParams):
        found = await handler.request_result(given.id)
        if not isinstance(found, RequestResult):
            return RequestResultResult(id=given.id, records=found)
        latest = found.latest
        return RequestResultResult(
            id=given.id,
            records="held",
            state=latest.state,
            reason=latest.reason,
            outcome=latest.outcome,
            message=latest.message,
            recorded=(
                [record_json(record) for record in found.records] if results else None
            ),
            withheld=(
                withheld_turn(latest.withheld) if latest.withheld is not None else None
            ),
            proposed=(
                withheld_turn(latest.proposed) if latest.proposed is not None else None
            ),
            sentinel=(
                SentinelField.model_validate(
                    sentinel_field(
                        latest.judged,
                        latest.outcome,
                        trimmed=not (results and handler.eval_results),
                    )
                )
                if latest.state == "decided"
                and latest.judged is not None
                and latest.outcome is not None
                else None
            ),
        )
    if isinstance(given, SampleResultParams):
        run = await handler.sample_result(given.sample_uuid)
        if run is None:
            raise RpcError(
                UNKNOWN_SAMPLE,
                f"The sidecar holds nothing for the sample {given.sample_uuid!r}.",
            )
        return SampleResultResult(
            sample_uuid=given.sample_uuid, outcome=run.outcome, store=run.store
        )
    if isinstance(given, EndSampleParams):
        return EndSampleResult(
            sample_uuid=given.sample_uuid,
            state=await handler.end_sample(given.sample_uuid),
        )
    raise TypeError(f"No method takes {type(given).__name__}.")


def schema() -> dict[str, Any]:
    """The JSON Schema of each method's parameters and result, and the errors the methods may answer with.

    A call to method `m` has `params` that fit the definition named in `methods[m].params`, and is answered with a `result` that fits the definition named in `methods[m].result`, or with an error whose code is a key of `errors` or one of JSON-RPC's own.
    """
    models = {model: None for pair in _METHODS.values() for model in pair}
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="Inspect Sentinel: the endpoint for evals",
        description=f"JSON-RPC 2.0 methods, POSTed to {PATH}, with a probe at GET {INFO_PATH}. `methods` names the definitions that a method's `params` and `result` fit. `errors` lists the codes of the method errors of the endpoint's own.",
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
        "errors": {
            str(code): {"name": name, "description": description}
            for code, (name, description) in _ERRORS.items()
        },
    }


SCHEMA_FILE = Path(__file__).with_name("eval_endpoint.schema.json")
"""Where `schema()` is kept, for an eval's client to be written and tested against without running this package."""


if __name__ == "__main__":
    # rewrites the schema file; run after changing a model above
    SCHEMA_FILE.write_text(json.dumps(schema(), indent=2) + "\n")
