import argparse
import json
from dataclasses import replace
from typing import Any

import anyio
import httpx
import jsonschema
import pytest
from inspect_ai.util import StoreModel
from starlette.applications import Starlette

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol, ProtocolGroup
from inspect_sentinel.sidecar import (
    Handler,
    Refuse,
    Replace,
    Request,
    UnreadableError,
    _handler,
    _state,
)
from inspect_sentinel.sidecar._eval_endpoint import (
    INFO_PATH,
    PATH,
    SCHEMA_FILE,
    TOKEN_VARIABLE,
    eval_routes,
    schema,
)
from inspect_sentinel.sidecar._json_rpc import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    REGISTRATION_REFUSED,
    REGISTRATIONS_FULL,
    UNKNOWN_SAMPLE,
    UNKNOWN_TASK,
)
from inspect_sentinel.sidecar._registration import FileTask, UnregisteredSampleError
from inspect_sentinel.sidecar._serve import add_arguments, apps
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import (
    inspect_headers,
    message,
    reply,
    request,
    weather,
    weather_rule,
)

_SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text())

_REGISTRATION: dict[str, Any] = {
    "eval_id": "eval_789",
    "task": "philadelphia",
    "task_description": None,
    "metadata": {},
    "expiry": 604800,
    "sample_ids": [1],
    "epochs": 2,
}

_COMPLETION: dict[str, Any] = {
    "eval_id": "eval_789",
    "task": "philadelphia",
    "sample_id": 1,
    "sample_description": None,
    "sample_input": "File today's weather report for Philadelphia.",
    "metadata": {"desk": "weather"},
}


def _client(handler: Handler, **options: Any) -> httpx.AsyncClient:
    app = Starlette(routes=eval_routes(handler, max_body_bytes=1024 * 1024, **options))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
    )


async def _call(
    client: httpx.AsyncClient, method: str, params: dict[str, Any], **headers: str
) -> dict[str, Any]:
    answered = await client.post(
        PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers=headers,
    )
    assert answered.status_code == 200
    body: dict[str, Any] = answered.json()
    if "result" in body:
        assert _fits(_SCHEMA["methods"][method]["result"], body["result"])
    return body


def _fits(part: dict[str, Any], value: Any) -> bool:
    try:
        jsonschema.validate(
            value, {**_SCHEMA, **part}, cls=jsonschema.Draft202012Validator
        )
    except jsonschema.ValidationError:
        return False
    return True


def _asked(id: str, irid: str = "irid-1", **sample: Any) -> Request:
    # a request an Inspect eval sends, with its id for the request
    headers = inspect_headers(**sample) | {"x-irid": irid}
    return replace(request(id), headers=headers)


@pytest.mark.anyio
async def test_the_probe_says_which_versions_are_served() -> None:
    async with _client(Handler(weather_rule())) as client:
        answered = await client.get(INFO_PATH)

    assert answered.json() == {
        "service": "inspect_sentinel",
        "versions": [1],
        "release": "whole",
    }


@pytest.mark.anyio
async def test_the_probe_says_how_replies_are_released() -> None:
    async with _client(Handler(weather_rule(), release="early")) as client:
        answered = await client.get(INFO_PATH)

    assert answered.json()["release"] == "early"


@pytest.mark.anyio
async def test_a_task_is_registered() -> None:
    async with _client(Handler(weather_rule(), task_expiry=86400)) as client:
        answered = await _call(client, "register_task", _REGISTRATION)

    assert answered["result"] == {
        "eval_id": "eval_789",
        "task": "philadelphia",
        "samples": 1,
        "fixed": False,
        # no longer than the sidecar keeps any
        "expiry": 86400,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("runtime_task_registration", [True, False])
async def test_a_task_registered_beforehand_is_already_registered(
    runtime_task_registration: bool,
) -> None:
    task = FileTask.model_validate({"task": "philadelphia", "samples": []})
    handler = Handler(
        weather_rule(),
        tasks={"philadelphia": task},
        runtime_task_registration=runtime_task_registration,
    )

    async with _client(handler) as client:
        answered = await _call(client, "register_task", _REGISTRATION)

    # never dropped, so no expiry
    assert answered["result"] == {
        "eval_id": "eval_789",
        "task": "philadelphia",
        "samples": 1,
        "fixed": True,
    }


@pytest.mark.anyio
async def test_a_sidecar_that_takes_no_registrations_refuses_another_task() -> None:
    async with _client(
        Handler(weather_rule(), runtime_task_registration=False)
    ) as client:
        answered = await _call(client, "register_task", _REGISTRATION)

    assert answered["error"]["code"] == REGISTRATION_REFUSED


@pytest.mark.anyio
async def test_a_registration_past_the_limit_fails_with_its_own_error() -> None:
    # anything that can reach the endpoint can register, so what is kept is
    # bounded, and nothing kept is dropped to make room
    async with _client(Handler(weather_rule(), max_task_registrations=2)) as client:
        kept = [
            await _call(client, "register_task", _REGISTRATION | {"eval_id": id})
            for id in ["eval_1", "eval_2"]
        ]
        refused = await _call(
            client, "register_task", _REGISTRATION | {"eval_id": "eval_3"}
        )
        again = await _call(
            client, "register_task", _REGISTRATION | {"eval_id": "eval_1"}
        )

    assert all("result" in answered for answered in kept)
    # a failure, which Inspect's `optional` covers, and not the refusal of a
    # sidecar that takes no registrations
    assert refused["error"]["code"] == REGISTRATIONS_FULL
    assert "already keeps the 2 tasks" in refused["error"]["message"]
    # a registration of a task kept already replaces it
    assert "result" in again


@pytest.mark.anyio
@pytest.mark.parametrize(
    "params",
    [
        {k: v for k, v in _REGISTRATION.items() if k != "eval_id"},
        _REGISTRATION | {"expiry": 0},
        _REGISTRATION | {"epochs": 0},
    ],
    ids=["no eval id", "no time to keep it", "no epochs"],
)
async def test_a_registration_that_doesnt_fit_is_an_error(
    params: dict[str, Any],
) -> None:
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(client, "register_task", params)

    assert answered["error"]["code"] == INVALID_PARAMS


@pytest.mark.anyio
async def test_a_samples_registration_is_completed() -> None:
    async with _client(Handler(weather_rule())) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(client, "complete_registration", _COMPLETION)

    assert answered["result"] == {
        "eval_id": "eval_789",
        "task": "philadelphia",
        "sample_id": 1,
        "fixed": False,
    }


@pytest.mark.anyio
async def test_a_samples_registration_for_a_task_not_registered_is_its_own_error() -> (
    None
):
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(client, "complete_registration", _COMPLETION)

    # so an eval whose registration expired knows to register the task again
    assert answered["error"]["code"] == UNKNOWN_TASK
    assert "Register the task" in answered["error"]["message"]


@pytest.mark.anyio
async def test_a_samples_registration_for_a_task_registered_beforehand_stores_nothing() -> (
    None
):
    task = FileTask.model_validate({"task": "philadelphia", "samples": []})
    handler = Handler(weather_rule(), tasks={"philadelphia": task})

    async with _client(handler) as client:
        answered = await _call(client, "complete_registration", _COMPLETION)

    assert answered["result"]["fixed"] is True


@pytest.mark.anyio
async def test_a_sidecar_that_takes_only_registered_samples_refuses_another() -> None:
    async with _client(Handler(weather_rule(), strictness="registered_only")) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(
            client, "complete_registration", _COMPLETION | {"sample_id": 2}
        )

    assert answered["error"]["code"] == REGISTRATION_REFUSED
    assert "doesn't name the sample 2" in answered["error"]["message"]


@pytest.mark.anyio
async def test_a_samples_registration_that_doesnt_fit_is_an_error() -> None:
    without_input = {k: v for k, v in _COMPLETION.items() if k != "sample_input"}

    async with _client(Handler(weather_rule())) as client:
        await _call(client, "register_task", _REGISTRATION)
        answered = await _call(client, "complete_registration", without_input)

    assert answered["error"]["code"] == INVALID_PARAMS


async def _request_result(handler: Handler, irid: str = "irid-1") -> dict[str, Any]:
    async with _client(handler) as client:
        answered = await _call(client, "request_result", {"id": irid})
    result: dict[str, Any] = answered["result"]
    return result


@pytest.mark.anyio
async def test_a_request_awaiting_its_reply() -> None:
    handler = Handler(weather_rule())
    await handler.request(_asked("x"))

    assert await _request_result(handler) == {
        "id": "irid-1",
        "records": "held",
        "state": "awaiting_reply",
        "recorded": [],
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "city", "outcome", "message"),
    [
        ("reject", "Cairo", "proceed", None),
        ("reject", "Tokyo", "reject", "Tokyo is not allowed."),
        ("terminate", "Tokyo", "terminate", None),
    ],
)
async def test_a_request_whose_reply_was_judged_is_decided(
    action: str, city: str, outcome: str, message: str | None
) -> None:
    handler = Handler(weather_rule(action, "Tokyo"))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message(city)))

    result = await _request_result(handler)

    assert (result["state"], result["outcome"], result.get("message")) == (
        "decided",
        outcome,
        message,
    )
    assert [record["call"] for record in result["recorded"]] == ["toolu_1"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "outcome"), [("reject", "reject"), ("terminate", "terminate")]
)
async def test_a_request_of_a_run_no_eval_names_is_decided_as_judged(
    action: str, outcome: str
) -> None:
    # A run no eval names, ended at its first turn, is not remembered as
    # ended, so what was decided is told from the judging itself.
    handler = Handler(weather_rule(action, "Tokyo"))
    await handler.request(replace(request("x"), headers={"x-irid": "irid-1"}))
    await handler.reply(reply(_message("Tokyo")))

    result = await _request_result(handler)

    assert (result["state"], result["outcome"]) == ("decided", outcome)
    # the field is only ever an Inspect client's
    assert "sentinel" not in result


def _message(city: str) -> bytes:
    return message(weather(city))


@pytest.mark.anyio
async def test_a_request_the_provider_failed_was_not_judged() -> None:
    handler = Handler(weather_rule())
    await handler.request(_asked("x"))
    await handler.reply(reply(b"overloaded", status=529))

    result = await _request_result(handler)

    assert (result["state"], result["reason"]) == ("not_judged", "provider_error")
    assert "sentinel" not in result


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "city"),
    [("reject", "Oslo"), ("reject", "Tokyo"), ("terminate", "Tokyo")],
    ids=["proceed", "reject", "terminate"],
)
async def test_request_result_gives_the_field_the_client_was_sent(
    action: str, city: str
) -> None:
    handler = Handler(weather_rule(action, city))
    await handler.request(_asked("x"))

    answer = await handler.reply(reply(_message("Tokyo")))

    # as when the proxy didn't pass the field on
    if isinstance(answer, Refuse):
        sent = answer.sentinel
    else:
        assert isinstance(answer, Replace)
        sent = json.loads(answer.body)["sentinel"]
    result = await _request_result(handler)
    assert result["sentinel"] == sent
    assert result["sentinel"]["steps"][-1]["recorded"]


@pytest.mark.anyio
async def test_request_result_gives_the_field_of_a_request_refused() -> None:
    handler = Handler(weather_rule("terminate", "Tokyo"))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    # the run has ended, so the next request is refused before the model
    answer = await handler.request(_asked("y", irid="irid-2"))

    assert isinstance(answer, Refuse)
    result = await _request_result(handler, "irid-2")
    assert result["sentinel"] == answer.sentinel
    assert result["sentinel"]["outcome"] == "terminate"


@pytest.mark.anyio
async def test_request_result_trims_the_field_where_results_are_not_sent() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    async with _client(handler, results=False) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"})

    field = answered["result"]["sentinel"]
    assert field["trimmed"] is True
    assert all("recorded" not in step for step in field["steps"])
    assert set(field["steps"][-1]["decision"]) <= {"action", "message", "modified"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "options",
    [
        {
            "tasks": {
                "philadelphia": FileTask.model_validate(
                    {"task": "philadelphia", "samples": []}
                )
            }
        },
        {"strictness": "known_tasks"},
    ],
    ids=["a sample the tasks folder doesn't hold", "a task the sidecar doesn't hold"],
)
async def test_a_request_the_sidecar_cant_place_was_not_judged(
    options: dict[str, Any],
) -> None:
    handler = Handler(weather_rule(), **options)

    with pytest.raises(UnregisteredSampleError):
        await handler.request(_asked("x"))

    # how an eval learns that the sample can't run there, and not to retry it
    result = await _request_result(handler)
    assert (result["state"], result["reason"]) == ("not_judged", "unregistered")


@pytest.mark.anyio
async def test_a_request_that_cant_be_read_was_not_judged() -> None:
    handler = Handler(weather_rule())
    unreadable = replace(_asked("x"), body=b"not json")

    with pytest.raises(UnreadableError):
        await handler.request(unreadable)

    result = await _request_result(handler)
    assert (result["state"], result["reason"]) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_a_request_whose_inspect_header_cant_be_read_was_not_judged() -> None:
    handler = Handler(weather_rule())
    headers = {"x-inspect-context": "{not json", "x-irid": "irid-1"}

    # refused, so a proxy that lets through what can't be judged doesn't
    answer = await handler.request(replace(request("x"), headers=headers))

    assert isinstance(answer, Refuse)
    result = await _request_result(handler)
    assert (result["state"], result["reason"]) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_a_retried_request_holds_every_attempts_records() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"), reject_status=503)
    await handler.request(_asked("x1"))
    await handler.reply(reply(_message("Tokyo"), id="x1"))
    # the client retries the request under the same id
    await handler.request(_asked("x2"))
    await handler.reply(reply(_message("Oslo"), id="x2"))

    result = await _request_result(handler)

    assert (result["state"], result["outcome"]) == ("decided", "proceed")
    assert [record["report"]["action"] for record in result["recorded"]] == [
        "reject",
        "proceed",
    ]


@pytest.mark.anyio
async def test_a_request_never_seen_is_unknown() -> None:
    handler = Handler(weather_rule())
    # a request with no id of the eval's is not one it can ask about
    await handler.request(request("x"))

    assert await _request_result(handler) == {"id": "irid-1", "records": "unknown"}


class _Clock:
    now = 0.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.now


@pytest.mark.anyio
async def test_a_request_whose_records_were_dropped_says_so_for_a_while(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = Handler(weather_rule(), state=MemoryState(expiry=100))
    await handler.request(_asked("x"))

    _Clock.now = 150.0
    dropped = await _request_result(handler)
    _Clock.now = 300.0
    forgotten = await _request_result(handler)

    assert dropped == {"id": "irid-1", "records": "dropped"}
    assert forgotten == {"id": "irid-1", "records": "unknown"}


class _Tally(StoreModel):
    lookups: int = 0


@protocol
def counts_lookups() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        context.store_as(_Tally).lookups += 1
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_run_says_what_it_came_to_and_its_store() -> None:
    handler = Handler(counts_lookups())
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Oslo")))

    async with _client(handler) as client:
        answered = await _call(client, "sample_result", {"sample_uuid": "sample_123"})

    result = answered["result"]
    assert result["sample_uuid"] == "sample_123"
    assert result["outcome"] == "proceed"
    # keyed as the sentinel wrote it, by the store model and its field
    assert [
        value for key, value in result["store"].items() if key.endswith("lookups")
    ] == [1]


@pytest.mark.anyio
async def test_a_sample_a_sentinel_ended_says_so() -> None:
    handler = Handler(weather_rule("terminate", "Tokyo"))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    async with _client(handler) as client:
        answered = await _call(client, "sample_result", {"sample_uuid": "sample_123"})

    assert answered["result"]["outcome"] == "terminate"


@pytest.mark.anyio
async def test_a_run_no_eval_names_cant_be_reached_by_its_opening() -> None:
    # Such a run is keyed by a digest of how its conversation opens, which
    # anyone who knows the opening can work out.
    handler = Handler(counts_lookups())
    asked = request("x")
    await handler.request(asked)
    await handler.reply(reply(_message("Tokyo")))
    opening = _handler._opening(
        await _handler.read_opening(asked.provider, asked.body, asked.format)
    )

    async with _client(handler) as client:
        result = await _call(client, "sample_result", {"sample_uuid": opening})
        ended = await _call(client, "end_sample", {"sample_uuid": opening})
    later = await handler.request(request("y"))

    assert result["error"]["code"] == UNKNOWN_SAMPLE
    assert ended["result"]["state"] == "unknown"
    assert not isinstance(later, Refuse)


@pytest.mark.anyio
async def test_a_sample_never_seen_has_no_result() -> None:
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(client, "sample_result", {"sample_uuid": "sample_123"})

    assert answered["error"]["code"] == UNKNOWN_SAMPLE


@pytest.mark.anyio
async def test_a_sample_its_eval_ended_is_dropped_and_refused_from_then_on() -> None:
    handler = Handler(counts_lookups())
    await handler.request(_asked("x"))

    async with _client(handler) as client:
        ended = await _call(client, "end_sample", {"sample_uuid": "sample_123"})
        again = await _call(client, "end_sample", {"sample_uuid": "sample_123"})
        never = await _call(client, "end_sample", {"sample_uuid": "sample_999"})
        never_again = await _call(client, "end_sample", {"sample_uuid": "sample_999"})
        result = await _call(client, "sample_result", {"sample_uuid": "sample_123"})
    later = await handler.request(_asked("y", irid="irid-2"))
    unseen = await handler.request(_asked("z", irid="irid-3", sample_uuid="sample_999"))

    assert [
        answered["result"]["state"] for answered in (ended, again, never, never_again)
    ] == ["ended", "ended", "unknown", "unknown"]
    assert result["error"]["code"] == UNKNOWN_SAMPLE
    assert isinstance(later, Refuse)
    assert later.sentinel is not None
    assert later.sentinel["outcome"] == "terminate"
    # a run the sidecar never saw is left as it was, not refused from then on
    assert not isinstance(unseen, Refuse)


@pytest.mark.anyio
async def test_results_can_be_withheld() -> None:
    async with _client(Handler(weather_rule()), results=False) as client:
        withheld = await _call(client, "sample_result", {"sample_uuid": "sample_123"})
        registered = await _call(client, "register_task", _REGISTRATION)

    assert withheld["error"]["code"] == METHOD_NOT_FOUND
    assert "doesn't send results" in withheld["error"]["message"]
    assert "result" in registered


@pytest.mark.anyio
async def test_a_refused_turn_is_told_without_the_results() -> None:
    # how an eval whose proxy dropped the `sentinel` field learns of it
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    async with _client(handler, results=False) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"})

    result = answered["result"]
    assert (result["outcome"], result["message"]) == ("reject", "Tokyo is not allowed.")
    assert result["withheld"] is not None
    assert "recorded" not in result


@pytest.mark.anyio
async def test_a_port_of_its_own_needs_its_own_token() -> None:
    async with _client(Handler(weather_rule()), token="s3cret") as client:
        without = await client.post(PATH, json={})
        wrong = await client.post(
            PATH, json={}, headers={"authorization": "Bearer nope"}
        )
        right = await _call(
            client,
            "end_sample",
            {"sample_uuid": "sample_123"},
            authorization="Bearer s3cret",
        )
        probe = await client.get(INFO_PATH)

    assert (without.status_code, wrong.status_code) == (401, 401)
    assert right["result"]["state"] == "unknown"
    # the probe asks for no token
    assert probe.status_code == 200


@pytest.mark.anyio
async def test_a_call_larger_than_the_limit_is_not_read() -> None:
    app = Starlette(routes=eval_routes(Handler(weather_rule()), max_body_bytes=10))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
    ) as client:
        answered = await client.post(PATH, json=_REGISTRATION)

    assert answered.status_code == 413


def _arguments(*given: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(["--sentinel", "x", *given])


def _paths(app: Starlette) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_both_endpoints_share_a_port_by_default() -> None:
    served = apps(Handler(weather_rule()), _arguments())

    assert [(port, _paths(app)) for app, port in served] == [
        (8900, {"/proxy/v1", "/health", INFO_PATH, PATH})
    ]


def test_the_endpoint_for_evals_can_have_a_port_and_a_token_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_VARIABLE, "s3cret")

    served = apps(Handler(weather_rule()), _arguments("--eval-port", "8901"))

    assert [(port, _paths(app)) for app, port in served] == [
        (8900, {"/proxy/v1", "/health"}),
        (8901, {"/health", INFO_PATH, PATH}),
    ]


def test_a_port_of_its_own_without_a_token_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKEN_VARIABLE, raising=False)

    with pytest.raises(ValueError, match=TOKEN_VARIABLE):
        apps(Handler(weather_rule()), _arguments("--eval-port", "8901"))


def test_schema_file_is_what_the_models_say() -> None:
    # rewrite it with: python -m inspect_sentinel.sidecar._eval_endpoint
    assert _SCHEMA == schema()


def test_calls_in_these_tests_fit_the_schema_file() -> None:
    methods = _SCHEMA["methods"]
    assert _fits(methods["register_task"]["params"], _REGISTRATION)
    assert _fits(methods["complete_registration"]["params"], _COMPLETION)
    assert _fits(methods["request_result"]["params"], {"id": "irid-1"})
    assert not _fits(methods["request_result"]["params"], {})


@pytest.mark.anyio
async def test_a_reply_in_flight_when_its_eval_ended_the_run_is_refused() -> None:
    handler = Handler(counts_lookups())
    await handler.request(_asked("x"))
    await handler.end_sample("sample_123")

    answer = await handler.reply(reply(_message("Oslo")))

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert answer.sentinel["outcome"] == "terminate"
    # the run is not made again
    assert await handler.sample_result("sample_123") is None
    result = await _request_result(handler)
    assert (result["state"], result["outcome"]) == ("decided", "terminate")


# made in each test, as an event belongs to the event loop it is made in:
# one that lets the request be judged, one that lets the reply be, and one
# set once the reply's judging has begun
_gates: list[anyio.Event] = []


@protocol
def waits_to_be_told() -> ProtocolGroup:
    async def before(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        await _gates[0].wait()
        return Decision.proceed()

    async def call(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        _gates[2].set()
        await _gates[1].wait()
        return Decision.proceed()

    return ProtocolGroup(before, call)


@pytest.mark.anyio
async def test_a_request_judged_while_its_eval_ended_the_run_is_refused() -> None:
    _gates[:] = [anyio.Event(), anyio.Event(), anyio.Event()]
    handler = Handler(waits_to_be_told())
    answers: list[Any] = []

    async def ask() -> None:
        answers.append(await handler.request(_asked("x")))

    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        await anyio.wait_all_tasks_blocked()
        await handler.end_sample("sample_123")
        _gates[0].set()

    (answer,) = answers
    assert isinstance(answer, Refuse)
    assert await handler.sample_result("sample_123") is None


@pytest.mark.anyio
async def test_a_reply_being_judged_is_judging() -> None:
    _gates[:] = [anyio.Event(), anyio.Event(), anyio.Event()]
    handler = Handler(waits_to_be_told())
    states: list[str] = []

    async def ask() -> None:
        await handler.request(_asked("x"))
        await handler.reply(reply(_message("Oslo")))

    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        await anyio.wait_all_tasks_blocked()
        states.append((await _request_result(handler))["state"])
        _gates[0].set()
        await _gates[2].wait()
        states.append((await _request_result(handler))["state"])
        _gates[1].set()

    states.append((await _request_result(handler))["state"])
    # judging the request, then judging the reply, then decided
    assert states == ["judging", "judging", "decided"]


@pytest.mark.anyio
async def test_a_rejected_turn_is_given_back_by_request_result() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(_asked("x"))
    await handler.reply(reply(_message("Tokyo")))

    result = await _request_result(handler)

    withheld = result["withheld"]
    assert withheld["message"]["role"] == "assistant"
    assert withheld["results"] == {"toolu_1": "Tokyo is not allowed."}


@pytest.mark.anyio
async def test_a_turn_with_a_call_taken_out_is_given_back_by_request_result() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(_asked("x"))
    await handler.reply(
        reply(message(weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")))
    )

    result = await _request_result(handler)

    # the turn went on without Tokyo
    assert result["outcome"] == "proceed"
    assert "withheld" not in result
    proposed = result["proposed"]
    assert [call["id"] for call in proposed["message"]["tool_calls"]] == [
        "toolu_1",
        "toolu_2",
    ]
    assert proposed["results"] == {"toolu_1": "Tokyo is not allowed."}
