import json
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from inspect_sentinel.sidecar import UnreadableError
from inspect_sentinel.sidecar._wire import read_reply, read_request, stream_events
from tests.sidecar._traffic import JSON, STREAM, message, recorded, weather


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "stop_reason", "locations", "text"),
    [
        ("single-tool-call", "tool_calls", ["Paris"], ""),
        ("parallel-tool-calls", "tool_calls", ["Paris", "Tokyo", "Lima"], "I'll check"),
        ("text-then-tool", "tool_calls", ["Oslo"], "I'm going to check"),
        ("text-only", "stop", [], "Hello there!"),
    ],
)
async def test_recorded_stream_is_read(
    name: str, stop_reason: str, locations: list[str], text: str
) -> None:
    output = await read_reply("anthropic", STREAM, recorded(name))

    assert output.stop_reason == stop_reason
    calls = output.message.tool_calls or []
    assert [call.function for call in calls] == ["get_weather"] * len(locations)
    assert [call.arguments["location"] for call in calls] == locations
    assert output.message.text.startswith(text)


@pytest.mark.anyio
@pytest.mark.parametrize("line_end", [b"\r\n", b"\r"], ids=["CRLF", "CR"])
async def test_stream_is_read_whatever_ends_its_lines(line_end: bytes) -> None:
    as_recorded = recorded("parallel-tool-calls")
    assert b"\r" not in as_recorded

    body = as_recorded.replace(b"\n", line_end)
    output = await read_reply("anthropic", STREAM, body)

    calls = output.message.tool_calls or []
    assert [call.arguments["location"] for call in calls] == ["Paris", "Tokyo", "Lima"]
    assert output.message.text.startswith("I'll check")


@pytest.mark.parametrize("code_point", [0x2028, 0x2029, 0x85])
def test_line_separator_inside_a_string_does_not_end_the_line(code_point: int) -> None:
    event = {"type": "text", "text": f"before{chr(code_point)}after"}
    data = json.dumps(event, ensure_ascii=False)
    assert chr(code_point) in data

    assert list(stream_events(f"data: {data}\n\n".encode())) == [event]


@pytest.mark.anyio
async def test_unstreamed_reply_is_read() -> None:
    output = await read_reply(
        "anthropic",
        JSON,
        message({"type": "text", "text": "Checking."}, weather("Oslo")),
    )

    assert output.message.text == "Checking."
    assert [call.arguments for call in output.message.tool_calls or []] == [
        {"location": "Oslo"}
    ]


@pytest.mark.anyio
async def test_unknown_field_value_is_not_an_error() -> None:
    body = json.loads(message(weather("Oslo")))
    body["usage"]["service_tier"] = "a tier this SDK has never heard of"

    output = await read_reply("anthropic", JSON, json.dumps(body).encode())

    assert len(output.message.tool_calls or []) == 1


def _stream(*events: dict[str, Any]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


_START: dict[str, Any] = {
    "type": "message_start",
    "message": json.loads(message(stop_reason="end_turn")) | {"stop_reason": None},
}
_UNKNOWN_BLOCK: dict[str, Any] = {
    "type": "content_block_start",
    "index": 0,
    "content_block": {"type": "future_tool_use", "id": "f1", "name": "x", "input": {}},
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content_type", "body", "says"),
    [
        (JSON, b"not json", "not JSON"),
        (JSON, b"[]", "not a JSON object"),
        (
            JSON,
            message({"type": "future_tool_use", "id": "f1", "name": "x", "input": {}}),
            "'future_tool_use'",
        ),
        (
            STREAM,
            _stream(_START, _UNKNOWN_BLOCK, {"type": "message_stop"}),
            "'future_tool_use'",
        ),
        (STREAM, _stream(_START), "not a complete message"),
        (STREAM, b"", "not a complete message"),
        (STREAM, b"data: {oops\n\n", "not JSON"),
        (
            STREAM,
            _stream(_START, {"type": "error", "error": {"type": "overloaded_error"}}),
            "error from the provider",
        ),
    ],
)
async def test_reply_that_cannot_be_read(
    content_type: str, body: bytes, says: str
) -> None:
    with pytest.raises(UnreadableError, match=says):
        await read_reply("anthropic", content_type, body)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Be brief.", "Be brief."),
        (
            [
                {"type": "text", "text": "Be brief."},
                {"type": "text", "text": "Be kind."},
            ],
            "Be brief.\n\nBe kind.",
        ),
        (None, None),
    ],
)
async def test_request_is_read(system: object, expected: str | None) -> None:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "Hello"}]}
    if system is not None:
        body["system"] = system

    messages = await read_request("anthropic", json.dumps(body).encode())

    roles = [type(m) for m in messages]
    assert roles == ([ChatMessageSystem] if expected else []) + [ChatMessageUser]
    assert messages[-1].text == "Hello"
    if expected:
        assert messages[0].text == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("provider", "body", "says"),
    [
        ("anthropic", b"{}", "no list of messages"),
        ("anthropic", b"nope", "not JSON"),
        ("somebody-else", b"{}", "no reader for provider 'somebody-else'"),
    ],
)
async def test_request_that_cannot_be_read(
    provider: str, body: bytes, says: str
) -> None:
    with pytest.raises(UnreadableError, match=says):
        await read_request(provider, body)
