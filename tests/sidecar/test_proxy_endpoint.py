import base64
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import anyio
import httpx
import jsonschema
import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler
from inspect_sentinel.sidecar._json_rpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SENTINEL_FAILED,
    TIME_LIMIT,
    UNREADABLE,
    UNREGISTERED,
)
from inspect_sentinel.sidecar._proxy_endpoint import (
    PATH,
    SCHEMA_FILE,
    proxy_app,
    schema,
)
from inspect_sentinel.sidecar._registration import FileTask
from tests.sidecar._traffic import (
    JSON,
    message,
    ran,
    storm_rule,
    weather,
    weather_rule,
)

# the methods' own description of their parameters and results, which the part
# that runs inside a proxy is written against
_SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text())

_REQUEST: dict[str, Any] = {
    "id": "x",
    "provider": "anthropic",
    "model": "claude-test",
    "user": "user-1",
    "headers": {},
    "body": {"messages": [{"role": "user", "content": "What is the weather?"}]},
}

_REFUSED: dict[str, Any] = {
    "action": "refuse",
    "status": 400,
    "message": "A sentinel rejected the call to get_weather: Tokyo is not allowed.",
}


def _reply(body: bytes, **changes: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "id": "x",
        "provider": "anthropic",
        "model": "claude-test",
        "user": "user-1",
        "status": 200,
        "content_type": JSON,
        "body": base64.b64encode(body).decode(),
    }
    return params | changes


def _fits(part: dict[str, Any], value: Any) -> bool:
    try:
        jsonschema.validate(
            value, {**_SCHEMA, **part}, cls=jsonschema.Draft202012Validator
        )
    except jsonschema.ValidationError:
        return False
    return True


def _client(handler: Handler | None = None, **limits: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=proxy_app(handler or Handler(weather_rule()), **limits)
        ),
        base_url="http://sidecar",
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("city", "expected"), [("Oslo", {"action": "pass"}), ("Tokyo", _REFUSED)]
)
async def test_request_and_reply_are_judged(
    city: str, expected: dict[str, Any]
) -> None:
    async with _client() as client:
        asked = await _call(client, "judge_request", _REQUEST)
        answered = await _call(client, "judge_reply", _reply(message(weather(city))))

    assert asked == {"jsonrpc": "2.0", "id": 1, "result": {"action": "pass"}}
    assert answered["result"] == expected
    assert _fits(_SCHEMA["methods"]["judge_request"]["result"], asked["result"])
    assert _fits(_SCHEMA["methods"]["judge_reply"]["result"], answered["result"])


_FROM_INSPECT = {
    # a proxy may pass the headers on as the client wrote them
    "X-Inspect-Context": json.dumps(
        {
            "version": 1,
            "eval_id": "eval_789",
            "task": "philadelphia",
            "sample_id": 1,
            "epoch": 1,
            "sample_uuid": "sample_123",
            "attempt": "attempt_1",
        }
    ),
}


@pytest.mark.anyio
@pytest.mark.parametrize("city", ["Oslo", "Tokyo"])
async def test_an_inspect_client_is_told_what_was_decided(city: str) -> None:
    async with _client() as client:
        asked = await _call(
            client, "judge_request", _REQUEST | {"headers": _FROM_INSPECT}
        )
        answered = await _call(client, "judge_reply", _reply(message(weather(city))))

    assert asked["result"] == {"action": "pass"}
    result = answered["result"]
    assert _fits(_SCHEMA["methods"]["judge_reply"]["result"], result)
    if city == "Oslo":
        assert result["action"] == "replace"
        field = json.loads(base64.b64decode(result["body"]))["sentinel"]
    else:
        assert result["action"] == "refuse"
        field = result["sentinel"]
    assert [step["decision"]["action"] for step in field["steps"]] == [
        "proceed" if city == "Oslo" else "reject",
    ]


@pytest.mark.anyio
async def test_request_that_carries_on_from_a_rejected_turn_is_replaced() -> None:
    async with _client() as client:
        await _call(client, "judge_request", _REQUEST)
        await _call(client, "judge_reply", _reply(message(weather("Tokyo"))))
        again = await _call(client, "judge_request", _REQUEST | {"id": "y"})

    replaced = again["result"]
    assert replaced["action"] == "replace"
    # the turn the client was never handed, and what the model is told of it
    assert [entry["role"] for entry in replaced["body"]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert _fits(_SCHEMA["methods"]["judge_request"]["result"], replaced)


@pytest.mark.anyio
async def test_body_is_read_in_the_format_the_proxy_names() -> None:
    # the provider's name alone would find no reader
    named = {"provider": "bedrock", "format": "anthropic-messages"}

    async with _client() as client:
        asked = await _call(client, "judge_request", _REQUEST | named)
        answered = await _call(
            client, "judge_reply", _reply(message(weather("Tokyo"))) | named
        )

    assert asked["result"] == {"action": "pass"}
    assert answered["result"] == _REFUSED
    assert _fits(_SCHEMA["methods"]["judge_request"]["params"], _REQUEST | named)


@protocol
def reports_the_conversation() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("run", message=step.conversation)

    return decide


@pytest.mark.anyio
async def test_run_is_the_sample_the_eval_names() -> None:
    async with _client(Handler(reports_the_conversation())) as client:
        await _call(client, "judge_request", _REQUEST | {"headers": _FROM_INSPECT})
        answered = await _call(client, "judge_reply", _reply(message(weather("Oslo"))))

    assert re.search(r": sample_123/[0-9a-f]{16}$", answered["result"]["message"])


@pytest.mark.anyio
async def test_request_can_be_refused() -> None:
    body = {"messages": [*_REQUEST["body"]["messages"], *ran("Oslo", "storm")]}

    async with _client(Handler(storm_rule())) as client:
        answered = await _call(client, "judge_request", _REQUEST | {"body": body})

    assert answered["result"] == {
        "action": "refuse",
        "status": 400,
        "message": "A sentinel ended this run at the result of the call to get_weather.",
    }


@pytest.mark.anyio
async def test_run_that_a_sentinel_ended_is_refused_from_then_on() -> None:
    named = {"headers": _FROM_INSPECT}
    refused = {
        "action": "refuse",
        "status": 400,
        "message": "A sentinel ended this run at the call to get_weather.",
    }

    async with _client(Handler(weather_rule("terminate", "Tokyo"))) as client:
        await _call(client, "judge_request", _REQUEST | named)
        ended = await _call(client, "judge_reply", _reply(message(weather("Tokyo"))))
        later = await _call(client, "judge_request", _REQUEST | named | {"id": "y"})
        other = await _call(client, "judge_request", _REQUEST | {"id": "z"})

    assert [
        {key: value for key, value in result.items() if key != "sentinel"}
        for result in (ended["result"], later["result"])
    ] == [refused, refused]
    assert later["result"]["sentinel"]["outcome"] == "terminate"
    assert other["result"] == {"action": "pass"}


@pytest.mark.anyio
async def test_reply_to_a_request_that_was_never_put_is_not_judged() -> None:
    async with _client() as client:
        answered = await _call(
            client, "judge_reply", _reply(message(weather("Oslo")), id="unseen")
        )

    assert answered["error"]["code"] == UNREADABLE


@pytest.mark.anyio
async def test_health_check_is_answered() -> None:
    async with _client() as client:
        answered = await client.get("/health")

    assert (answered.status_code, answered.text) == (200, "ok")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sent", "code"),
    [
        pytest.param(b"nope", PARSE_ERROR, id="not JSON"),
        pytest.param(b'{"method": "judge_request"}', INVALID_REQUEST, id="not a call"),
        pytest.param(
            json.dumps(
                {"jsonrpc": "2.0", "method": "judge_request", "params": _REQUEST}
            ).encode(),
            INVALID_REQUEST,
            id="no id to answer to",
        ),
    ],
)
async def test_body_that_is_not_a_call_is_an_error(sent: bytes, code: int) -> None:
    async with _client() as client:
        answered = await client.post(PATH, content=sent)

    assert answered.status_code == 200
    assert answered.json()["error"]["code"] == code


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "params", "code", "says"),
    [
        pytest.param("judge", _REQUEST, METHOD_NOT_FOUND, "'judge'", id="no method"),
        pytest.param(
            "judge_request",
            {k: v for k, v in _REQUEST.items() if k != "model"},
            INVALID_PARAMS,
            "model",
            id="a parameter missing",
        ),
        pytest.param(
            "judge_request",
            _REQUEST | {"format": "gemini-generate-content"},
            INVALID_PARAMS,
            "format",
            id="a format this does not read",
        ),
        pytest.param(
            "judge_reply",
            _reply(b"") | {"body": "not base64!"},
            INVALID_PARAMS,
            "base64",
            id="a reply not base64",
        ),
        pytest.param(
            "judge_reply",
            _reply(b"") | {"body": "\u2603"},
            INVALID_PARAMS,
            "base64",
            id="a reply not even ASCII",
        ),
    ],
)
async def test_call_that_does_not_fit_is_an_error(
    method: str, params: dict[str, Any], code: int, says: str
) -> None:
    async with _client() as client:
        answered = await _call(client, method, params)

    assert answered["error"]["code"] == code
    assert says in answered["error"]["message"]
    # where the parameters are wrong, never the values sent
    assert "gemini" not in answered["error"]["message"]


@protocol
def never_finishes() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        await anyio.sleep_forever()
        return None

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("handler", "reply", "code", "says"),
    [
        pytest.param(
            Handler(weather_rule()),
            _reply(b'{"type": "message", "content": [{"type": "hologram"}]}'),
            UNREADABLE,
            "couldn't be read",
            id="a reply that can't be read",
        ),
        pytest.param(
            Handler(weather_rule("break", "Tokyo")),
            _reply(message(weather("Tokyo"))),
            SENTINEL_FAILED,
            "The sentinel failed",
            id="a sentinel that raises",
        ),
        pytest.param(
            Handler(never_finishes(), time_limit=0.05),
            _reply(message(weather("Oslo"))),
            TIME_LIMIT,
            "within 0.05 seconds",
            id="a sentinel that takes too long",
        ),
    ],
)
async def test_reply_that_was_not_judged_is_an_error_and_not_a_refusal(
    handler: Handler, reply: dict[str, Any], code: int, says: str
) -> None:
    # the proxy's side says whether this refuses the call or lets it through
    async with _client(handler) as client:
        await _call(client, "judge_request", _REQUEST)
        answered = await _call(client, "judge_reply", reply)

    assert "result" not in answered
    assert answered["error"]["code"] == code
    assert says in answered["error"]["message"]
    # why a reply couldn't be read may quote it, so the proxy is not told
    assert "hologram" not in answered["error"]["message"]


@pytest.mark.anyio
@pytest.mark.parametrize("declared", [True, False], ids=["length declared", "chunked"])
@pytest.mark.parametrize(("over", "status"), [(0, 200), (1, 413)])
async def test_call_larger_than_the_limit_is_not_read(
    declared: bool, over: int, status: int
) -> None:
    sent = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "judge_request", "params": _REQUEST}
    ).encode()
    handler = Handler(weather_rule())

    async def chunks() -> AsyncIterator[bytes]:
        yield sent[:10]
        yield sent[10:]

    async with _client(handler, max_body_bytes=len(sent) - over) as client:
        answered = await client.post(PATH, content=sent if declared else chunks())

    assert answered.status_code == status
    # a call that was too large never reached the handler
    assert (await handler.records(_REQUEST["id"]) is None) == bool(over)


def test_schema_file_is_what_the_models_say() -> None:
    # rewrite it with: python -m inspect_sentinel.sidecar._proxy_endpoint
    assert _SCHEMA == schema()


@pytest.mark.anyio
async def test_a_sample_not_registered_beforehand_is_not_judged() -> None:
    task = FileTask.model_validate(
        {"task": "philadelphia", "samples": [{"sample_id": 7, "sample_input": "Hi."}]}
    )
    handler = Handler(weather_rule(), tasks={"philadelphia": task})

    async with _client(handler) as client:
        answered = await _call(
            client, "judge_request", _REQUEST | {"headers": _FROM_INSPECT}
        )

    assert answered["error"]["code"] == UNREGISTERED
    assert "sample 1" in answered["error"]["message"]


def test_schema_file_says_a_call_not_judged_is_refused() -> None:
    errors = _SCHEMA["errors"]

    assert set(errors) == {
        str(UNREADABLE),
        str(SENTINEL_FAILED),
        str(TIME_LIMIT),
        str(UNREGISTERED),
    }
    assert all("refuses the call" in error["description"] for error in errors.values())


def test_calls_in_these_tests_fit_the_schema_file() -> None:
    methods = _SCHEMA["methods"]
    assert _fits(methods["judge_request"]["params"], _REQUEST)
    assert _fits(methods["judge_reply"]["params"], _reply(b"{}"))
    assert not _fits(methods["judge_request"]["params"], _reply(b"{}"))
