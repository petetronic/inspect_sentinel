import json
from dataclasses import replace
from typing import Any

import anyio
import httpx
import jsonschema
import pytest
from inspect_ai.util import StoreModel

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse, Reply, Request, _handler
from inspect_sentinel.sidecar._eval_endpoint import (
    PATH,
    SCHEMA_FILE,
    eval_app,
    schema,
)
from tests.sidecar._traffic import message, reply, request, weather, weather_rule

TOKEN = "an-eval's-token"

DETAILS: dict[str, Any] = {
    "task": "forecast",
    "sample_id": 7,
    "epoch": 2,
    "sample_input": "Report the weather in Oslo.",
    "metadata": {"desk": "weather"},
}


def _client(handler: Handler, token: str | None = TOKEN) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=eval_app(handler, token=TOKEN)),
        base_url="http://sidecar",
        headers={"authorization": f"Bearer {token}"} if token is not None else {},
    )


async def _call(
    client: httpx.AsyncClient, method: str, params: dict[str, Any]
) -> dict[str, Any]:
    answered = await client.post(
        PATH, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    )
    assert answered.status_code == 200
    body: dict[str, Any] = answered.json()
    assert (body["jsonrpc"], body["id"]) == ("2.0", 1)
    return body


async def _lookup(handler: Handler, city: str, id: str = "x", **changes: Any) -> object:
    """Hand the handler a request and the model's reply that looks up `city`."""
    await handler.request(replace(request(id), **changes))
    return await handler.reply(reply(message(weather(city)), id=id))


class Tally(StoreModel):
    lookups: int = 0


@protocol
def counts_lookups() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        context.store_as(Tally).lookups += 1
        return Decision.proceed()

    return decide


@protocol
def reports_the_sample() -> Protocol:
    """Reject every call, telling the client the sample the sentinel was given."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        sample = context.eval
        if sample is None:
            return Decision.reject("sample", message="no sample")
        return Decision.reject(
            "sample",
            message=f"{sample.task}|{sample.sample_id}|{sample.epoch}|{sample.sample_input_text}|{sample.metadata}",
        )

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sample_input", "text"),
    [
        ("Report the weather in Oslo.", "Report the weather in Oslo."),
        ([{"role": "user", "content": "Report it."}], "Report it."),
    ],
)
async def test_registered_sample_is_what_the_sentinel_is_given(
    sample_input: object, text: str
) -> None:
    handler = Handler(reports_the_sample())
    async with _client(handler) as client:
        registered = await _call(
            client,
            "register_run",
            {"run": "run-1", "eval": DETAILS | {"sample_input": sample_input}},
        )

    assert registered["result"] == {"run": "run-1", "state": "new"}
    told = await _lookup(handler, "Oslo", run="run-1")
    assert isinstance(told, Refuse)
    assert told.message.endswith(f"forecast|7|2|{text}|{{'desk': 'weather'}}")
    # a run nobody registered is given no sample
    other = await _lookup(handler, "Oslo", "y", run="run-2")
    assert isinstance(other, Refuse)
    assert other.message.endswith("no sample")


@pytest.mark.anyio
async def test_same_registration_again_changes_nothing() -> None:
    handler = Handler(weather_rule())
    params = {"run": "run-1", "eval": DETAILS}
    async with _client(handler) as client:
        first = await _call(client, "register_run", params)
        await _lookup(handler, "Oslo", run="run-1")
        # the run has been judged by now, and the registration is still the same
        again = await _call(client, "register_run", params)

    assert first["result"] == again["result"] == {"run": "run-1", "state": "new"}


@pytest.mark.anyio
async def test_other_details_for_a_registered_run_are_refused() -> None:
    handler = Handler(weather_rule())
    async with _client(handler) as client:
        await _call(client, "register_run", {"run": "run-1", "eval": DETAILS})
        changed = await _call(
            client,
            "register_run",
            {"run": "run-1", "eval": DETAILS | {"metadata": {"desk": "sport"}}},
        )

    assert changed["error"]["code"] == -32001
    assert "run-1" in changed["error"]["message"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("judged", "resume", "expected"),
    [
        pytest.param(True, True, "resumed", id="carrying on a run that is held"),
        pytest.param(False, True, "new", id="carrying on a run that isn't held"),
        pytest.param(False, False, "new", id="a new run"),
        pytest.param(True, False, None, id="a judged run registered as new"),
    ],
)
async def test_new_or_a_resume(
    judged: bool, resume: bool, expected: str | None
) -> None:
    handler = Handler(weather_rule())
    if judged:
        await _lookup(handler, "Oslo", run="run-1")
    async with _client(handler) as client:
        answered = await _call(
            client, "register_run", {"run": "run-1", "resume": resume, "eval": DETAILS}
        )

    if expected is None:
        assert answered["error"]["code"] == -32001
    else:
        assert answered["result"] == {"run": "run-1", "state": expected}


@pytest.mark.anyio
async def test_request_result_says_what_was_decided_and_recorded() -> None:
    handler = Handler(weather_rule())
    await _lookup(handler, "Tokyo", headers={"x-irid": "irid-1"})
    async with _client(handler) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"})

    result = answered["result"]
    assert (result["id"], result["records"]) == ("irid-1", "held")
    assert (result["outcome"], result["message"]) == ("reject", "Tokyo is not allowed.")
    [record] = result["recorded"]
    assert (record["kind"], record["name"], record["call"]) == (
        "record",
        "weather_rule",
        "toolu_1",
    )
    assert record["report"]["action"] == "reject"
    # for the log, and never what the model is told
    assert record["report"]["explanation"] == "lookup of Tokyo"


@protocol
def takes_forever() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        await anyio.sleep_forever()
        return None

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sentinel", "time_limit", "asked", "answered", "expected"),
    [
        pytest.param(
            weather_rule(),
            None,
            request(),
            None,
            {"state": "awaiting_reply"},
            id="the request went on and no reply was handed over",
        ),
        pytest.param(
            weather_rule(),
            None,
            request(),
            reply(message(weather("Oslo"))),
            {"state": "decided", "outcome": "continue"},
            id="decided",
        ),
        pytest.param(
            weather_rule(),
            None,
            replace(request(), provider="a provider nobody reads"),
            None,
            {"state": "not_judged", "reason": "unreadable"},
            id="a request that can't be read",
        ),
        pytest.param(
            weather_rule(),
            None,
            request(),
            reply(b"not a reply"),
            {"state": "not_judged", "reason": "unreadable"},
            id="a reply that can't be read",
        ),
        pytest.param(
            takes_forever(),
            0.05,
            request(),
            reply(message(weather("Oslo"))),
            {"state": "not_judged", "reason": "time_limit"},
            id="the sentinel passed its time limit",
        ),
        pytest.param(
            weather_rule("break"),
            None,
            request(),
            reply(message(weather("Tokyo"))),
            {"state": "not_judged", "reason": "error"},
            id="the sentinel raised",
        ),
        pytest.param(
            weather_rule(),
            None,
            request(),
            reply(b"overloaded", status=529),
            {"state": "not_judged", "reason": "provider_error"},
            id="the provider's reply was an error",
        ),
    ],
)
async def test_request_result_says_where_a_request_stands(
    sentinel: Protocol,
    time_limit: float | None,
    asked: Request,
    answered: Reply | None,
    expected: dict[str, str],
) -> None:
    handler = Handler(sentinel, time_limit=time_limit)
    try:
        await handler.request(replace(asked, headers={"x-irid": "irid-1"}))
        if answered is not None:
            await handler.reply(answered)
    except Exception:
        # what went wrong is what the eval is then told
        pass
    async with _client(handler) as client:
        result = (await _call(client, "request_result", {"id": "irid-1"}))["result"]

    told = {k: v for k, v in result.items() if k in ("state", "reason", "outcome")}
    assert told == expected


@pytest.mark.anyio
async def test_request_sent_again_under_one_id_is_one_result() -> None:
    handler = Handler(weather_rule())
    await _lookup(handler, "Tokyo", "first", headers={"x-irid": "irid-1"})
    await _lookup(handler, "Oslo", "second", headers={"x-irid": "irid-1"})
    async with _client(handler) as client:
        answered = await _call(client, "request_result", {"id": "irid-1"})

    result = answered["result"]
    # the latest attempt's outcome, and every attempt's records
    assert result["outcome"] == "continue"
    assert "message" not in result
    assert [record["report"]["action"] for record in result["recorded"]] == [
        "reject",
        "continue",
    ]


@pytest.mark.anyio
async def test_request_never_seen_and_request_no_longer_kept_are_told_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_handler, "_KEPT", 1)
    handler = Handler(weather_rule())
    await _lookup(handler, "Oslo", "first", headers={"x-irid": "irid-1"})
    await _lookup(handler, "Oslo", "second", headers={"x-irid": "irid-2"})
    async with _client(handler) as client:
        dropped = await _call(client, "request_result", {"id": "irid-1"})
        unknown = await _call(client, "request_result", {"id": "irid-9"})

    assert dropped["result"] == {"id": "irid-1", "records": "dropped"}
    assert unknown["result"] == {"id": "irid-9", "records": "unknown"}


@pytest.mark.anyio
async def test_run_result_is_the_store_and_whether_the_run_was_ended() -> None:
    handler = Handler([counts_lookups(), weather_rule("terminate")])
    await _lookup(handler, "Oslo", "first", run="run-1")
    async with _client(handler) as client:
        going = await _call(client, "run_result", {"run": "run-1"})
        await _lookup(handler, "Tokyo", "second", run="run-1")
        ended = await _call(client, "run_result", {"run": "run-1"})
        unknown = await _call(client, "run_result", {"run": "run-9"})

    assert going["result"]["outcome"] == "continue"
    # as the sentinel's own instance wrote it: `store_as` keys a model by the
    # instance that asked for it
    assert going["result"]["store"] == {
        "Tally:counts_lookups:instance": "counts_lookups",
        "Tally:counts_lookups:lookups": 1,
    }
    assert ended["result"]["outcome"] == "terminate"
    assert ended["result"]["store"]["Tally:counts_lookups:lookups"] == 2
    assert unknown["error"]["code"] == -32002


@pytest.mark.anyio
@pytest.mark.parametrize("token", [None, "", "another-token", TOKEN + "x"])
async def test_caller_without_the_token_is_turned_away(token: str | None) -> None:
    handler = Handler(weather_rule())
    async with _client(handler, token) as client:
        answered = await client.post(
            PATH,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "register_run",
                "params": {"run": "run-1", "eval": DETAILS},
            },
        )

    assert answered.status_code == 401
    assert handler.run_result("run-1") is None


@pytest.mark.anyio
async def test_health_check_asks_for_no_token() -> None:
    async with _client(Handler(weather_rule()), token=None) as client:
        answered = await client.get("/health")

    assert answered.status_code == 200
    assert answered.text == "ok"


@pytest.mark.anyio
async def test_results_can_be_withheld_and_registration_still_taken() -> None:
    handler = Handler(weather_rule())
    await _lookup(handler, "Tokyo", run="run-1", headers={"x-irid": "irid-1"})
    app = eval_app(handler, token=TOKEN, results=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://sidecar",
        headers={"authorization": f"Bearer {TOKEN}"},
    ) as client:
        request_result = await _call(client, "request_result", {"id": "irid-1"})
        run_result = await _call(client, "run_result", {"run": "run-1"})
        registered = await _call(
            client, "register_run", {"run": "run-2", "eval": DETAILS}
        )

    assert request_result["error"]["code"] == run_result["error"]["code"] == -32601
    assert registered["result"] == {"run": "run-2", "state": "new"}


def test_endpoint_is_not_made_without_a_token() -> None:
    with pytest.raises(ValueError, match="token"):
        eval_app(Handler(weather_rule()), token="")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "code", "id"),
    [
        pytest.param(b"not json", -32700, None, id="not JSON"),
        pytest.param(b"[]", -32600, None, id="a batch"),
        pytest.param(
            json.dumps({"id": 1, "method": "run_result"}).encode(),
            -32600,
            None,
            id="not JSON-RPC 2.0",
        ),
        pytest.param(
            json.dumps({"jsonrpc": "2.0", "id": 5, "method": "end_run"}).encode(),
            -32601,
            5,
            id="no such method",
        ),
        pytest.param(
            json.dumps(
                {"jsonrpc": "2.0", "id": 6, "method": "run_result", "params": {}}
            ).encode(),
            -32602,
            6,
            id="a parameter missing",
        ),
    ],
)
async def test_call_that_cannot_be_read_is_a_json_rpc_error(
    body: bytes, code: int, id: int | None
) -> None:
    async with _client(Handler(weather_rule())) as client:
        answered = await client.post(PATH, content=body)

    assert answered.status_code == 200
    assert (answered.json()["id"], answered.json()["error"]["code"]) == (id, code)


@pytest.mark.anyio
async def test_parameters_that_do_not_fit_are_named_and_not_repeated() -> None:
    secret = "the sample's input, which is nobody else's to read"
    async with _client(Handler(weather_rule())) as client:
        answered = await _call(
            client,
            "register_run",
            {"run": "run-1", "eval": DETAILS | {"epoch": secret}},
        )

    assert answered["error"]["code"] == -32602
    assert "eval.epoch" in answered["error"]["message"]
    assert secret not in answered["error"]["message"]


@pytest.mark.anyio
async def test_notification_is_carried_out_and_not_answered() -> None:
    handler = Handler(weather_rule())
    async with _client(handler) as client:
        answered = await client.post(
            PATH,
            json={
                "jsonrpc": "2.0",
                "method": "register_run",
                "params": {"run": "run-1", "eval": DETAILS},
            },
        )

    assert (answered.status_code, answered.content) == (204, b"")
    assert handler.run_result("run-1") is not None


@pytest.mark.anyio
async def test_call_larger_than_the_limit_is_turned_away() -> None:
    async with _client(Handler(weather_rule())) as client:
        answered = await client.post(PATH, content=b" " * (4 * 1024 * 1024 + 1))

    assert answered.status_code == 413


def test_schema_file_is_what_the_models_say() -> None:
    # rewrite it with: python -m inspect_sentinel.sidecar._eval_endpoint
    assert json.loads(SCHEMA_FILE.read_text()) == schema()


@pytest.mark.anyio
async def test_params_and_results_fit_the_schema_file() -> None:
    described = json.loads(SCHEMA_FILE.read_text())

    def fits(kind: str, method: str, value: object) -> None:
        jsonschema.validate(
            value,
            {**described, **described["methods"][method][kind]},
            cls=jsonschema.Draft202012Validator,
        )

    handler = Handler([counts_lookups(), weather_rule()])
    await _lookup(handler, "Tokyo", run="run-1", headers={"x-irid": "irid-1"})
    calls = [
        ("register_run", {"run": "run-1", "resume": True, "eval": DETAILS}),
        ("request_result", {"id": "irid-1"}),
        ("request_result", {"id": "irid-9"}),
        ("run_result", {"run": "run-1"}),
    ]
    async with _client(handler) as client:
        for method, params in calls:
            fits("params", method, params)
            fits("result", method, (await _call(client, method, params))["result"])
