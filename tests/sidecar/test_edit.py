import json
from dataclasses import replace
from typing import Any

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.tool import ToolCall

from inspect_sentinel.sidecar._openai import _check_responses_stream
from inspect_sentinel.sidecar._wire import (
    Edits,
    UnreadableError,
    edit_reply,
    read_reply,
    written_events,
)
from tests.sidecar._traffic import (
    JSON,
    STREAM,
    anthropic_recorded,
    message,
    openai_recorded,
    weather,
)

# Every recorded reply that holds three calls, Paris, Tokyo and Lima, in each
# API, streamed and not: (provider, content type, body).
_REPLIES: dict[str, tuple[str, str, bytes]] = {
    "anthropic": (
        "anthropic",
        JSON,
        message(
            {"type": "text", "text": "Looking up all three."},
            weather("Paris", "toolu_1"),
            weather("Tokyo", "toolu_2"),
            weather("Lima", "toolu_3"),
        ),
    ),
    "anthropic stream": (
        "anthropic",
        STREAM,
        anthropic_recorded("parallel-tool-calls"),
    ),
    **{
        f"openai {name}": ("openai", *openai_recorded(name))
        for name in ("chat", "chat stream", "responses", "responses stream")
    },
}


async def _read(name: str) -> tuple[str, str, bytes, ModelOutput]:
    provider, content_type, body = _REPLIES[name]
    return provider, content_type, body, await read_reply(provider, content_type, body)


def _modified(output: ModelOutput, index: int, location: str) -> ToolCall:
    call = (output.message.tool_calls or [])[index]
    return replace(call, arguments={"location": location})


# which calls to take out and which to change, by their place in the reply
_CASES: dict[str, tuple[list[int], dict[int, str]]] = {
    "first taken out": ([0], {}),
    "middle taken out": ([1], {}),
    "last taken out": ([2], {}),
    "two taken out": ([0, 2], {}),
    "one changed": ([], {1: "Oslo"}),
    "one changed, one taken out": ([0], {2: "Oslo"}),
}


@pytest.mark.anyio
@pytest.mark.parametrize("name", _REPLIES)
@pytest.mark.parametrize("case", _CASES)
async def test_an_edited_reply_reads_back_with_only_the_edits(
    name: str, case: str
) -> None:
    provider, content_type, body, output = await _read(name)
    calls = output.message.tool_calls or []
    removed, changed = _CASES[case]
    edits = Edits(
        frozenset(calls[index].id for index in removed),
        {
            calls[index].id: _modified(output, index, location)
            for index, location in changed.items()
        },
    )

    edited = await edit_reply(provider, content_type, body, None, edits, output)

    read = await read_reply(provider, content_type, edited)
    assert [(call.id, call.arguments) for call in read.message.tool_calls or []] == [
        (call.id, edits.modified.get(call.id, call).arguments)
        for call in calls
        if call.id not in edits.removed
    ]
    assert read.message.text == output.message.text
    assert read.stop_reason == output.stop_reason


@pytest.mark.anyio
@pytest.mark.parametrize(
    "name", [name for name, (_, kind, _) in _REPLIES.items() if kind == STREAM]
)
async def test_a_stream_keeps_the_events_an_edit_doesnt_touch(name: str) -> None:
    provider, content_type, body, output = await _read(name)
    last = (output.message.tool_calls or [])[-1]

    edited = await edit_reply(
        provider, content_type, body, None, Edits(frozenset({last.id})), output
    )

    # every event before the last call's first goes out as it was written
    before = [event.lines for event in written_events(body)]
    after = [event.lines for event in written_events(edited)]
    first = next(
        index
        for index, lines in enumerate(before)
        if any(last.id in line for line, _ in lines)
    )
    assert after[:first] == before[:first]


@pytest.mark.anyio
@pytest.mark.parametrize("name", _REPLIES)
async def test_an_edit_of_a_call_the_reply_doesnt_hold_is_refused(name: str) -> None:
    provider, content_type, body, output = await _read(name)

    with pytest.raises(UnreadableError, match="no call"):
        await edit_reply(
            provider,
            content_type,
            body,
            None,
            Edits(frozenset({"call_absent"})),
            output,
        )


@pytest.mark.anyio
async def test_a_responses_call_that_is_not_a_function_call_cant_be_changed() -> None:
    reply: dict[str, Any] = json.loads(openai_recorded("responses")[1])
    reply["output"].append(
        {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_custom",
            "name": "shell",
            "input": "ls",
            "status": "completed",
        }
    )
    body = json.dumps(reply).encode()
    output = await read_reply("openai", JSON, body)
    custom = ToolCall(id="call_custom", function="shell", arguments={"input": "pwd"})

    with pytest.raises(UnreadableError, match="not a function call"):
        await edit_reply(
            "openai", JSON, body, None, Edits(modified={"call_custom": custom}), output
        )


def _responses_events() -> list[dict[str, Any]]:
    _, body = openai_recorded("responses stream")
    return [event.data for event in written_events(body) if event.data is not None]


def _check(events: list[dict[str, Any]]) -> None:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    _check_responses_stream(written_events(body))


def test_a_responses_stream_that_holds_together_passes() -> None:
    _check(_responses_events())


def test_a_responses_stream_whose_deltas_disagree_with_its_output_is_refused() -> None:
    events = _responses_events()
    delta = next(
        event
        for event in events
        if event["type"] == "response.function_call_arguments.delta"
    )
    delta["delta"] = '{"location": "Atlantis"}'

    with pytest.raises(UnreadableError, match="different arguments"):
        _check(events)


def test_a_responses_stream_with_an_item_out_of_place_is_refused() -> None:
    events = _responses_events()
    for event in events:
        if event.get("type") == "response.function_call_arguments.delta":
            event["output_index"] = 9999

    with pytest.raises(UnreadableError):
        _check(events)


@pytest.mark.anyio
async def test_a_chat_stream_call_without_an_index_is_refused() -> None:
    _, body = openai_recorded("chat stream")
    output = await read_reply("openai", STREAM, body)
    first = (output.message.tool_calls or [])[0]
    broken = body.replace(b'"index":0,"id"', b'"id"', 1)

    with pytest.raises(UnreadableError, match="no index"):
        await edit_reply(
            "openai", STREAM, broken, None, Edits(frozenset({first.id})), output
        )
