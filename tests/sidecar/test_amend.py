import json
from dataclasses import replace
from typing import Any

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall

from inspect_sentinel.sidecar._inspect_ai import noting
from inspect_sentinel.sidecar._wire import (
    Edited,
    Format,
    UnreadableError,
    _api_of,  # pyright: ignore[reportPrivateUsage]
    amend_request,
    amendments,
    read_reply_noted,
    read_request,
    rejection,
)
from tests.sidecar._traffic import JSON, message, openai_recorded, weather

# A reply in each API whose calls are Paris, Tokyo and Lima: (provider, body).
_REPLIES: dict[Format, tuple[str, bytes]] = {
    "anthropic-messages": (
        "anthropic",
        message(
            {"type": "text", "text": "Looking up all three."},
            weather("Paris", "toolu_1"),
            weather("Tokyo", "toolu_2"),
            weather("Lima", "toolu_3"),
        ),
    ),
    "openai-chat": ("openai", openai_recorded("chat")[1]),
    "openai-responses": ("openai", openai_recorded("responses")[1]),
}

_ASKED = "What is the weather in Paris, Tokyo and Lima?"


class _Turn:
    """A turn whose second call, Tokyo, was taken out, and whose third, Lima, was changed to Oslo."""

    def __init__(
        self,
        format: Format,
        provider: str,
        proposed: ChatMessageAssistant,
        edited: Edited,
    ) -> None:
        self.format: Format = format
        self.provider = provider
        self.proposed = proposed
        self.edited = edited
        paris, tokyo, lima = proposed.tool_calls or []
        self.paris, self.tokyo, self.lima = paris, tokyo, lima
        self.oslo = edited.modified[lima.id]


async def _turn(format: Format) -> _Turn:
    provider, body = _REPLIES[format]
    output, noted = await read_reply_noted(provider, JSON, body)
    proposed = output.message
    _, tokyo, lima = proposed.tool_calls or []
    oslo = replace(lima, arguments={"location": "Oslo"})
    taken_out = {tokyo.id: "Not Tokyo."}
    edited = Edited(
        sent=1,
        mark="m",
        order=tuple(call.id for call in proposed.tool_calls or []),
        taken_out=taken_out,
        originals={lima.id: lima},
        modified={lima.id: oslo},
        amendments=await amendments(format, proposed, taken_out, [lima.id], noted),
    )
    return _Turn(format, provider, proposed, edited)


async def _request(
    turn: _Turn, calls: list[ToolCall] | None, results: list[ChatMessageTool]
) -> bytes:
    """A request a client sends after the turn: the question, then the turn as the client holds it with `calls`, then `results`. No turn when `calls` is None."""
    entries: list[Any] = [{"role": "user", "content": _ASKED}]
    if calls is not None:
        held = turn.proposed.model_copy(update={"tool_calls": calls})
        # written as a client with no notes of its own writes it
        with noting():
            entries += await _api_of(turn.format).rejected_turn(held, results)
    if turn.format == "openai-responses":
        return json.dumps({"model": "gpt-5-mini", "input": entries}).encode()
    request: dict[str, Any] = {"model": "m", "messages": entries}
    if turn.format == "anthropic-messages":
        request["max_tokens"] = 1024
    return json.dumps(request).encode()


def _ran(call: ToolCall, text: str = "Sunny.") -> ChatMessageTool:
    return ChatMessageTool(content=text, tool_call_id=call.id, function=call.function)


async def _amended(turn: _Turn, body: bytes) -> bytes | None:
    amended = await amend_request(
        turn.provider, body, edited=[turn.edited], format=turn.format
    )
    return amended.body


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
async def test_a_turn_held_as_handed_has_the_models_calls_put_back(
    format: Format,
) -> None:
    turn = await _turn(format)
    body = await _request(
        turn, [turn.paris, turn.oslo], [_ran(turn.paris), _ran(turn.oslo)]
    )

    amended = await _amended(turn, body)

    assert amended is not None
    messages = await read_request(turn.provider, amended, format)
    assistant = next(m for m in messages if isinstance(m, ChatMessageAssistant))
    # every call the model proposed, in its order, with its own arguments
    assert [(call.id, call.arguments) for call in assistant.tool_calls or []] == [
        (call.id, call.arguments) for call in turn.proposed.tool_calls or []
    ]
    told = {m.tool_call_id: m for m in messages if isinstance(m, ChatMessageTool)}
    assert list(told) == list(turn.edited.order)
    assert "Not Tokyo." in told[turn.tokyo.id].text
    assert told[turn.paris.id].text == "Sunny."
    assert told[turn.lima.id].text == "Sunny."


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
async def test_a_turn_the_client_restored_has_nothing_added(format: Format) -> None:
    turn = await _turn(format)
    body = await _request(
        turn,
        [turn.paris, turn.tokyo, turn.lima],
        [_ran(turn.paris), rejection(turn.tokyo, "Not Tokyo."), _ran(turn.lima)],
    )

    amended = await amend_request(
        turn.provider, body, edited=[turn.edited], format=turn.format
    )
    assert amended.body is None
    # the rejection the client put back is the sidecar's, not a result
    assert amended.restored == {turn.tokyo.id}


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
async def test_a_turn_restored_in_part_has_the_rest_put_back(format: Format) -> None:
    turn = await _turn(format)
    # the taken-out call put back, without its result, and the changed call
    # as it ran
    body = await _request(
        turn,
        [turn.paris, turn.tokyo, turn.oslo],
        [_ran(turn.paris), _ran(turn.oslo)],
    )

    amended = await _amended(turn, body)

    assert amended is not None
    messages = await read_request(turn.provider, amended, format)
    assistant = next(m for m in messages if isinstance(m, ChatMessageAssistant))
    assert [call.arguments for call in assistant.tool_calls or []] == [
        call.arguments for call in turn.proposed.tool_calls or []
    ]
    told = [m.tool_call_id for m in messages if isinstance(m, ChatMessageTool)]
    assert told == list(turn.edited.order)


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
async def test_a_request_without_the_turn_is_left_as_it_is(format: Format) -> None:
    turn = await _turn(format)
    # a retry of the request the turn was handed for
    body = await _request(turn, None, [])

    assert await _amended(turn, body) is None


def _unreadable_cases(
    turn: _Turn,
) -> dict[str, tuple[list[ToolCall], list[ChatMessageTool]]]:
    extra = replace(turn.paris, id="call_extra")
    rome = replace(turn.lima, arguments={"location": "Rome"})
    lisbon = replace(turn.tokyo, arguments={"location": "Lisbon"})
    return {
        "calls out of order": (
            [turn.oslo, turn.paris],
            [_ran(turn.oslo), _ran(turn.paris)],
        ),
        "a call the turn didn't have": (
            [turn.paris, turn.oslo, extra],
            [_ran(turn.paris), _ran(turn.oslo), _ran(extra)],
        ),
        "a handed call missing": ([turn.paris], [_ran(turn.paris)]),
        "a changed call with other arguments": (
            [turn.paris, rome],
            [_ran(turn.paris), _ran(rome)],
        ),
        "a taken-out call with other arguments": (
            [turn.paris, lisbon, turn.oslo],
            [_ran(turn.paris), _ran(turn.oslo)],
        ),
        "a taken-out call that ran": (
            [turn.paris, turn.tokyo, turn.oslo],
            [_ran(turn.paris), _ran(turn.tokyo), _ran(turn.oslo)],
        ),
    }


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
@pytest.mark.parametrize(
    ("case", "why"),
    [
        ("calls out of order", "out of order"),
        ("a call the turn didn't have", "a call it didn't have"),
        ("a handed call missing", "without a call it was handed"),
        ("a changed call with other arguments", "neither the model's nor"),
        ("a taken-out call with other arguments", "not as the model proposed it"),
        ("a taken-out call that ran", "never ran"),
    ],
)
async def test_a_turn_held_otherwise_is_unreadable(
    format: Format, case: str, why: str
) -> None:
    turn = await _turn(format)
    calls, results = _unreadable_cases(turn)[case]
    body = await _request(turn, calls, results)

    with pytest.raises(UnreadableError, match=why):
        await _amended(turn, body)


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
async def test_a_call_of_the_turn_elsewhere_is_unreadable(format: Format) -> None:
    turn = await _turn(format)
    body = json.loads(
        await _request(
            turn, [turn.paris, turn.oslo], [_ran(turn.paris), _ran(turn.oslo)]
        )
    )
    key = "input" if format == "openai-responses" else "messages"
    # the turn moved one entry later than where it was handed
    body[key].insert(1, {"role": "user", "content": "And one more thing."})

    with pytest.raises(UnreadableError, match="where the turn isn't"):
        await _amended(turn, json.dumps(body).encode())


@pytest.mark.anyio
async def test_a_refused_turn_after_an_edited_one_goes_after_it() -> None:
    turn = await _turn("anthropic-messages")
    body = await _request(
        turn, [turn.paris, turn.oslo], [_ran(turn.paris), _ran(turn.oslo)]
    )
    refused, noted = await read_reply_noted(
        "anthropic", JSON, message(weather("Rome", "toolu_9"))
    )

    # the client's next request was refused whole, after the edited turn and
    # its results, which are its second and third entries
    amended = await amend_request(
        "anthropic",
        body,
        inserted={3: [(refused.message, {"toolu_9": "Not Rome."}, noted)]},
        edited=[turn.edited],
        format="anthropic-messages",
    )

    assert amended.body is not None
    messages = await read_request("anthropic", amended.body, "anthropic-messages")
    calls = [
        [call.id for call in m.tool_calls or []]
        for m in messages
        if isinstance(m, ChatMessageAssistant)
    ]
    assert calls == [list(turn.edited.order), ["toolu_9"]]
    told = [m.tool_call_id for m in messages if isinstance(m, ChatMessageTool)]
    assert told == [*turn.edited.order, "toolu_9"]


_PNG = "iVBORw0KGgo="


def _with_an_image(format: Format, result: dict[str, Any]) -> None:
    """Add an image beside the text of a result as written in `format`."""
    if format == "anthropic-messages":
        result["content"] = [
            {"type": "text", "text": result["content"]},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _PNG},
            },
        ]
    elif format == "openai-chat":
        result["content"] = [
            {"type": "text", "text": result["content"]},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_PNG}"},
            },
        ]
    else:
        result["output"] = [
            {"type": "input_text", "text": result["output"]},
            {"type": "input_image", "image_url": f"data:image/png;base64,{_PNG}"},
        ]


def _result_for(format: Format, body: dict[str, Any], id: str) -> dict[str, Any]:
    entries = body["input" if format == "openai-responses" else "messages"]
    for entry in entries:
        candidates = (
            entry["content"]
            if format == "anthropic-messages" and isinstance(entry["content"], list)
            else [entry]
        )
        for candidate in candidates:
            if id in (
                candidate.get("tool_use_id"),
                candidate.get("tool_call_id"),
                candidate.get("call_id") if "output" in candidate else None,
            ):
                return candidate
    raise AssertionError(f"no result for {id}")


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
async def test_a_restored_rejection_with_anything_added_is_unreadable(
    format: Format,
) -> None:
    turn = await _turn(format)
    body = json.loads(
        await _request(
            turn,
            [turn.paris, turn.tokyo, turn.lima],
            [_ran(turn.paris), rejection(turn.tokyo, "Not Tokyo."), _ran(turn.lima)],
        )
    )
    # the rejection's own text, and an image beside it
    _with_an_image(format, _result_for(format, body, turn.tokyo.id))

    with pytest.raises(UnreadableError, match="never ran"):
        await _amended(turn, json.dumps(body).encode())


@pytest.mark.anyio
@pytest.mark.parametrize("format", _REPLIES)
async def test_a_changed_call_under_another_function_is_unreadable(
    format: Format,
) -> None:
    turn = await _turn(format)
    other = replace(turn.oslo, function="get_time")
    body = await _request(turn, [turn.paris, other], [_ran(turn.paris), _ran(other)])

    with pytest.raises(UnreadableError, match="another kind or function"):
        await _amended(turn, body)


def _thinking(text: str, signature: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": text, "signature": signature}


async def _interleaved() -> tuple[Edited, list[dict[str, Any]]]:
    """A turn that thinks before each of its calls, whose first call, Tokyo, was taken out; and the blocks a client was handed of it."""
    first, second = _thinking("Tokyo first.", "sig_1"), _thinking("Now Paris.", "sig_2")
    tokyo, paris = weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")
    output, noted = await read_reply_noted(
        "anthropic", JSON, message(first, tokyo, second, paris)
    )
    taken_out = {"toolu_1": "Not Tokyo."}
    edited = Edited(
        sent=1,
        mark="m",
        order=("toolu_1", "toolu_2"),
        taken_out=taken_out,
        originals={},
        modified={},
        amendments=await amendments(
            "anthropic-messages", output.message, taken_out, (), noted
        ),
    )
    return edited, [first, second, paris]


def _anthropic_request(content: list[dict[str, Any]]) -> bytes:
    told = {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Sunny."}
    return json.dumps(
        {
            "model": "m",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": _ASKED},
                {"role": "assistant", "content": content},
                {"role": "user", "content": [told]},
            ],
        }
    ).encode()


def _blocks(body: bytes) -> list[str]:
    return [
        block.get("signature") or block["id"]
        for block in json.loads(body)["messages"][1]["content"]
    ]


@pytest.mark.anyio
async def test_a_call_goes_back_between_the_thinking_it_parted() -> None:
    edited, handed = await _interleaved()

    amended = await amend_request(
        "anthropic", _anthropic_request(handed), edited=[edited]
    )

    # thinking blocks the call stood between are kept apart, as they came
    assert amended.body is not None
    assert _blocks(amended.body) == ["sig_1", "toolu_1", "sig_2", "toolu_2"]


@pytest.mark.anyio
async def test_a_call_goes_back_before_the_next_if_its_thinking_was_left_out() -> None:
    edited, handed = await _interleaved()

    # a client may leave out the thinking of an earlier turn
    amended = await amend_request(
        "anthropic", _anthropic_request(handed[2:]), edited=[edited]
    )

    assert amended.body is not None
    assert _blocks(amended.body) == ["toolu_1", "toolu_2"]
