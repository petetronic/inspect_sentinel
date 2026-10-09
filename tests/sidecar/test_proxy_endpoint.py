import base64
import json
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
from inspect_sentinel.sidecar import Answer, Handler, Pass, Refuse, Replace
from inspect_sentinel.sidecar._proxy_endpoint import (
    SCHEMA_FILE,
    _answer,
    proxy_app,
    schema,
)
from tests.sidecar._traffic import (
    JSON,
    message,
    ran,
    reports_the_proxy,
    reports_the_run,
    storm_rule,
    weather,
    weather_rule,
)

# the endpoint's own description of its messages and answers, which the part
# that runs inside a proxy is written against
_SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text())

_REQUEST: dict[str, Any] = {
    "version": 1,
    "phase": "request",
    "id": "x",
    "provider": "anthropic",
    "model": "claude-test",
    "user": "user-1",
    "run": None,
    "headers": {},
    "body": {"messages": [{"role": "user", "content": "What is the weather?"}]},
}


def _reply(body: bytes, **changes: Any) -> dict[str, Any]:
    exchange: dict[str, Any] = {
        "version": 1,
        "phase": "reply",
        "id": "x",
        "provider": "anthropic",
        "model": "claude-test",
        "user": "user-1",
        "status": 200,
        "content_type": JSON,
        "body": base64.b64encode(body).decode(),
    }
    return exchange | changes


def _fits(definition: str, value: Any) -> bool:
    schema = {**_SCHEMA, "$ref": f"#/$defs/{definition}"}
    try:
        jsonschema.validate(value, schema, cls=jsonschema.Draft202012Validator)
    except jsonschema.ValidationError:
        return False
    return True


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=proxy_app(Handler(weather_rule())), raise_app_exceptions=False
        ),
        base_url="http://sidecar",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("city", "expected"),
    [
        ("Oslo", {"action": "pass"}),
        (
            "Tokyo",
            {
                "action": "refuse",
                "status": 400,
                "message": "A sentinel rejected the call to get_weather: Tokyo is not allowed.",
                "headers": {"x-sentinel-decision": "reject"},
            },
        ),
    ],
)
async def test_exchange_is_answered(city: str, expected: dict[str, Any]) -> None:
    async with _client() as client:
        asked = await client.post("/", json=_REQUEST)
        answered = await client.post("/", json=_reply(message(weather(city))))

    assert (asked.status_code, asked.json()) == (200, {"action": "pass"})
    assert (answered.status_code, answered.json()) == (200, expected)


@protocol
def never_finishes() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        await anyio.sleep_forever()
        return None

    return decide


@pytest.mark.anyio
async def test_reply_not_judged_within_the_time_limit_is_a_gateway_timeout() -> None:
    app = proxy_app(Handler(never_finishes(), time_limit=0.05))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
    ) as client:
        asked = await client.post("/", json=_REQUEST)
        answered = await client.post("/", json=_reply(message(weather("Oslo"))))

    assert asked.status_code == 200
    assert answered.status_code == 504
    assert "within 0.05 seconds" in answered.json()["error"]


@pytest.mark.anyio
@pytest.mark.parametrize("declared", [True, False], ids=["length declared", "chunked"])
@pytest.mark.parametrize(("over", "status"), [(0, 200), (1, 413)])
async def test_message_larger_than_the_limit_is_not_read(
    declared: bool, over: int, status: int
) -> None:
    sent = json.dumps(_REQUEST).encode()
    handler = Handler(weather_rule())

    async def chunks() -> AsyncIterator[bytes]:
        yield sent[:10]
        yield sent[10:]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=proxy_app(handler, max_body_bytes=len(sent) - over)
        ),
        base_url="http://sidecar",
    ) as client:
        answered = await client.post("/", content=sent if declared else chunks())

    assert answered.status_code == status
    # a message that was too large never reached the handler
    assert (handler.records(_REQUEST["id"]) is None) == bool(over)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exchange",
    [
        pytest.param({"phase": "neither"}, id="unknown phase"),
        pytest.param({k: v for k, v in _REQUEST.items() if k != "id"}, id="no id"),
        pytest.param(_reply(b"") | {"body": "not base64!"}, id="body not base64"),
    ],
)
async def test_malformed_exchange_is_a_bad_request(exchange: dict[str, Any]) -> None:
    async with _client() as client:
        answered = await client.post("/", content=json.dumps(exchange))

    assert answered.status_code == 400
    assert "error" in answered.json()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exchange",
    [
        pytest.param(_REQUEST | {"version": 2}, id="a later version"),
        pytest.param(
            {k: v for k, v in _REQUEST.items() if k != "version"}, id="no version"
        ),
    ],
)
async def test_another_version_is_a_bad_request(exchange: dict[str, Any]) -> None:
    async with _client() as client:
        answered = await client.post("/", json=exchange)

    assert answered.status_code == 400
    assert "version" in answered.json()["error"]


def test_schema_file_is_what_the_models_say() -> None:
    # rewrite it with: python -m inspect_sentinel.sidecar._proxy_endpoint
    assert _SCHEMA == schema()


def test_messages_in_these_tests_fit_the_schema_file() -> None:
    assert _fits("RequestMessage", _REQUEST)
    assert _fits("ReplyMessage", _reply(b"{}"))
    assert not _fits("RequestMessage", _reply(b"{}"))


@pytest.mark.parametrize(
    ("answer", "reply", "definition"),
    [
        (Pass(), False, "PassAnswer"),
        (Pass(), True, "PassAnswer"),
        (Replace(body=b'{"messages": []}'), False, "ReplaceAnswer"),
        (Replace(body=b"event: ping\n\n"), True, "ReplaceAnswer"),
        (
            Refuse(
                status=400,
                message="Not allowed.",
                headers={"x-sentinel-decision": "reject"},
            ),
            False,
            "RefuseAnswer",
        ),
        (
            Refuse(
                status=503,
                message="Not allowed.",
                headers={"x-sentinel-decision": "reject"},
            ),
            True,
            "RefuseAnswer",
        ),
    ],
)
def test_answers_fit_the_schema_file(
    answer: Answer, reply: bool, definition: str
) -> None:
    assert _fits(definition, _answer(answer, reply=reply))


def test_headers_a_handler_names_go_with_an_answer_that_lets_a_reply_through() -> None:
    added = {"x-checked": "yes"}

    passed = _answer(Pass(headers=added), reply=True)
    replaced = _answer(Replace(body=b"{}", headers=added), reply=True)

    assert (passed["headers"], replaced["headers"]) == (added, added)
    assert _fits("PassAnswer", passed) and _fits("ReplaceAnswer", replaced)


@pytest.mark.anyio
async def test_health_check_is_answered() -> None:
    async with _client() as client:
        answered = await client.get("/health")

    assert answered.status_code == 200
    assert answered.text == "ok"


@pytest.mark.anyio
async def test_exchange_that_cannot_be_judged_is_a_server_error() -> None:
    # the proxy's side says whether this refuses the exchange or lets it through
    async with _client() as client:
        answered = await client.post(
            "/", json=_reply(message(weather("Oslo")), id="unseen")
        )

    assert answered.status_code == 500


@pytest.mark.anyio
async def test_request_can_be_refused() -> None:
    handler = Handler(storm_rule())
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app(handler)),
        base_url="http://sidecar",
    )
    body = {"messages": [*_REQUEST["body"]["messages"], *ran("Oslo", "storm")]}

    async with client:
        answered = await client.post("/", json=_REQUEST | {"body": body})

    assert (answered.status_code, answered.json()) == (
        200,
        {
            "action": "refuse",
            "status": 400,
            "message": "A sentinel ended this run at the result of the call to get_weather.",
            "headers": {"x-sentinel-decision": "terminate"},
        },
    )


@pytest.mark.anyio
async def test_run_is_the_one_the_proxy_names() -> None:
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app(Handler(reports_the_run()))),
        base_url="http://sidecar",
    )
    named = {"run": "sample-uuid", "headers": {"x-hawk-job-id": "job-1"}}

    async with client:
        await client.post("/", json=_REQUEST | named)
        answered = await client.post("/", json=_reply(message(weather("Oslo"))))

    assert answered.json()["message"].endswith(": sample-uuid|None")


@pytest.mark.anyio
async def test_run_that_a_sentinel_ended_is_refused_from_then_on() -> None:
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=proxy_app(Handler(weather_rule("terminate", "Tokyo")))
        ),
        base_url="http://sidecar",
    )
    named = {"run": "sample-uuid"}

    async with client:
        await client.post("/", json=_REQUEST | named)
        ended = await client.post("/", json=_reply(message(weather("Tokyo"))))
        later = await client.post("/", json=_REQUEST | named | {"id": "y"})
        other = await client.post("/", json=_REQUEST | {"id": "z"})

    refused = {
        "action": "refuse",
        "status": 400,
        "message": "A sentinel ended this run at the call to get_weather.",
        "headers": {"x-sentinel-decision": "terminate"},
    }
    assert (ended.json(), later.json()) == (refused, refused)
    assert other.json() == {"action": "pass"}


@pytest.mark.anyio
async def test_sentinel_is_given_the_headers_the_proxy_passed_on() -> None:
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app(Handler(reports_the_proxy()))),
        base_url="http://sidecar",
    )
    told = {"headers": {"x-hawk-job-type": "scan", "x-irid": "request-1"}}

    async with client:
        await client.post("/", json=_REQUEST | told)
        answered = await client.post("/", json=_reply(message(weather("Oslo"))))

    assert answered.json()["message"].endswith(
        ": anthropic|claude-test|user-1|[('x-hawk-job-type', 'scan'), ('x-irid', 'request-1')]"
    )
