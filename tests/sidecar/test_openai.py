import json
from collections.abc import Sequence
from typing import Any, cast

import pytest

from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
    UnreadableError,
)
from inspect_sentinel.sidecar._wire import read_reply, read_request
from tests.sidecar._traffic import DATA, JSON, STREAM, storm_rule, weather_rule

# Replies recorded from the OpenAI API, each calling get_weather for Paris,
# Tokyo and Lima at once.
RECORDED = {
    "chat": "openai-chat.json",
    "chat stream": "openai-chat.sse",
    "responses": "openai-responses.json",
    "responses stream": "openai-responses.sse",
}


def recorded(name: str) -> tuple[str, bytes]:
    file = RECORDED[name]
    return (STREAM if file.endswith(".sse") else JSON), (DATA / file).read_bytes()


def request(api: str, id: str = "x", then: Sequence[dict[str, Any]] = ()) -> Request:
    opening = {"role": "user", "content": "What is the weather?"}
    body: dict[str, Any] = (
        {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                opening,
                *then,
            ],
        }
        if api == "chat"
        else {
            "model": "gpt-test",
            "instructions": "You are helpful.",
            "input": [opening, *then],
            "store": False,
        }
    )
    return Request(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        body=json.dumps(body).encode(),
    )


def reply(name: str, id: str = "x") -> Reply:
    content_type, body = recorded(name)
    return Reply(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        status=200,
        content_type=content_type,
        body=body,
    )


def ran(api: str, result: str) -> list[dict[str, Any]]:
    """A weather lookup the model made and the client's result for it, as a later request holds them."""
    arguments = json.dumps({"location": "Oslo"})
    if api == "chat":
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": arguments},
        }
        return [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": result},
        ]
    return [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": arguments,
        },
        {"type": "function_call_output", "call_id": "call_1", "output": result},
    ]


def _half(body: bytes) -> bytes:
    frames = body.split(b"\n\n")
    return b"\n\n".join(frames[: len(frames) // 2]) + b"\n\n"


def _changed(name: str, **changes: Any) -> bytes:
    return json.dumps(json.loads(recorded(name)[1]) | changes).encode()


@pytest.mark.anyio
@pytest.mark.parametrize("name", list(RECORDED))
async def test_recorded_reply_is_read(name: str) -> None:
    output = await read_reply("openai", *recorded(name))

    assert output.stop_reason == "tool_calls"
    calls = output.message.tool_calls or []
    assert [call.function for call in calls] == ["get_weather"] * 3
    # the model wrote "Tokyo" in one reply and "Tokyo, Japan" in another
    assert [str(call.arguments["location"]).split(",")[0] for call in calls] == [
        "Paris",
        "Tokyo",
        "Lima",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content_type", "body", "match"),
    [
        pytest.param(
            STREAM,
            _half(recorded("chat stream")[1]),
            "not a complete message",
            id="chat stream cut short",
        ),
        pytest.param(
            STREAM,
            _half(recorded("responses stream")[1]),
            "not a complete response",
            id="responses stream cut short",
        ),
        pytest.param(
            STREAM,
            b'data: {"type": "error", "message": "overloaded"}\n\n',
            "error from the provider",
            id="stream that ends in an error",
        ),
        pytest.param(
            JSON,
            _changed("chat", choices=json.loads(recorded("chat")[1])["choices"] * 2),
            "more than one choice",
            id="two choices",
        ),
        pytest.param(
            JSON,
            _changed("responses", output=[{"type": "not_known_yet", "id": "x"}]),
            "can't be read",
            id="output of a kind inspect_ai doesn't read",
        ),
        pytest.param(
            JSON, _changed("chat", object="list"), "'list' reply", id="another API"
        ),
        pytest.param(JSON, b"[]", "not a JSON object", id="not an object"),
    ],
)
async def test_reply_that_cannot_be_read(
    content_type: str, body: bytes, match: str
) -> None:
    with pytest.raises(UnreadableError, match=match):
        await read_reply("openai", content_type, body)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "roles"),
    [
        pytest.param(
            json.loads(request("chat", then=ran("chat", "rain")).body),
            ["system", "user", "assistant", "tool"],
            id="chat",
        ),
        pytest.param(
            json.loads(request("responses", then=ran("responses", "rain")).body),
            ["system", "user", "assistant", "tool"],
            id="responses",
        ),
        pytest.param(
            {"model": "gpt-test", "input": "What is the weather?"},
            ["user"],
            id="responses, a lone message as a string",
        ),
    ],
)
async def test_request_is_read(body: dict[str, Any], roles: list[str]) -> None:
    messages = await read_request("openai", json.dumps(body).encode())

    assert [message.role for message in messages] == roles
    assert messages[-1].text == ("rain" if roles[-1] == "tool" else body["input"])


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
            {"input": "And then?", "previous_response_id": "resp_1"},
            "the provider keeps",
            id="continues a stored response",
        ),
        pytest.param(
            {"input": "And then?", "conversation": "conv_1"},
            "the provider keeps",
            id="continues a stored conversation",
        ),
        pytest.param(
            {"input": "Take your time.", "background": True},
            "fetched later",
            id="answered in the background",
        ),
        pytest.param({"prompt": "Once upon a time"}, "no reader", id="completions"),
        pytest.param({"messages": "hello"}, "no list of messages", id="no messages"),
    ],
)
async def test_request_that_cannot_be_read(body: dict[str, Any], match: str) -> None:
    with pytest.raises(UnreadableError, match=match):
        await read_request("openai", json.dumps(body | {"model": "gpt-test"}).encode())


def _conversation(api: str, body: bytes) -> list[dict[str, Any]]:
    return cast(
        "list[dict[str, Any]]",
        json.loads(body)["messages" if api == "chat" else "input"],
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("api", "kinds", "told"),
    [
        (
            "chat",
            ["assistant", "tool", "tool", "tool"],
            # OpenAI's formats have no flag for a failed call, so inspect_ai
            # says it in the text
            [
                "Error: " + text
                for text in ("not run", "Tokyo is not allowed.", "not run")
            ],
        ),
        (
            "responses",
            ["reasoning", *["function_call"] * 3, *["function_call_output"] * 3],
            ["not run", "Tokyo is not allowed.", "not run"],
        ),
    ],
)
async def test_rejection_is_added_in_the_format_of_the_request(
    api: str, kinds: list[str], told: list[str]
) -> None:
    handler = Handler(weather_rule("reject", "Tokyo"), reject_status=503)
    first = request(api, "a")
    await handler.request(first)

    refused = await handler.reply(reply(api, "a"))
    again = await handler.request(request(api, "b"))

    assert refused == Refuse(
        503,
        "A sentinel rejected the call to get_weather: Tokyo is not allowed.",
        {"x-sentinel-decision": "reject"},
    )
    assert isinstance(again, Replace)
    sent, added = _conversation(api, first.body), _conversation(api, again.body)
    assert added[: len(sent)] == sent
    entries = added[len(sent) :]
    assert [entry.get("type") or entry["role"] for entry in entries] == kinds
    results = [entry.get("content") or entry.get("output") for entry in entries][-3:]
    not_run = (
        "This call was not run, because another call in the same turn was rejected."
    )
    assert results == [text.replace("not run", not_run) for text in told]
    # everything else in the request is as the client sent it
    assert {k: v for k, v in json.loads(again.body).items() if k != "messages"} | {
        "input": None
    } == {k: v for k, v in json.loads(first.body).items() if k != "messages"} | {
        "input": None
    }


@pytest.mark.anyio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            "storm",
            Refuse(
                400,
                "A sentinel ended this run at the result of the call to get_weather.",
                {"x-sentinel-decision": "terminate"},
            ),
        ),
        ("rain", Pass()),
    ],
)
async def test_result_is_judged(api: str, result: str, expected: Pass | Refuse) -> None:
    handler = Handler(storm_rule())

    answer = await handler.request(request(api, then=ran(api, result)))

    assert answer == expected
