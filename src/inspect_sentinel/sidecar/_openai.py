from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ModelOutput,
    messages_from_openai,
    messages_from_openai_responses,
    messages_to_openai,
    messages_to_openai_responses,
    model_output_from_openai,
    model_output_from_openai_responses,
)

from ._wire import RequestApi, UnreadableError, json_object, stream_events

# what inspect_ai's converters and the SDK's types raise on content they can't
# read; pydantic's ValidationError is a ValueError
_UNREAD = (ValueError, TypeError, KeyError, NotImplementedError)


async def read_reply(content_type: str, body: bytes) -> ModelOutput:
    """Read a Chat Completions or Responses reply, streamed or not."""
    if content_type.startswith("text/event-stream"):
        events = list(stream_events(body))
        chat = any(event.get("object") == "chat.completion.chunk" for event in events)
        reply = _chat_stream(events) if chat else _responses_stream(events)
    else:
        reply = json_object(body, "reply")
        kind = reply.get("object")
        if kind not in ("chat.completion", "response"):
            raise UnreadableError(f"There is no reader for an OpenAI {kind!r} reply.")
        chat = kind == "chat.completion"
    try:
        output = await (
            model_output_from_openai(reply)
            if chat
            else model_output_from_openai_responses(reply)
        )
    except _UNREAD as ex:
        raise UnreadableError("The reply holds content that can't be read.") from ex
    if len(output.choices) > 1:
        raise UnreadableError(
            "The reply holds more than one choice, and only the first would be judged."
        )
    return output


async def _read_chat(request: dict[str, Any]) -> list[ChatMessage]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise UnreadableError("The request has no list of messages.")
    try:
        return await messages_from_openai(cast(Any, messages), _model(request))
    except _UNREAD as ex:
        raise UnreadableError("The request holds a message that can't be read.") from ex


async def _read_responses(request: dict[str, Any]) -> list[ChatMessage]:
    if (
        request.get("previous_response_id") is not None
        or request.get("conversation") is not None
    ):
        raise UnreadableError(
            "The request continues a conversation the provider keeps, which this sidecar is not shown."
        )
    if request.get("background"):
        raise UnreadableError(
            "The request asks for a reply that is fetched later, which this sidecar would not be shown."
        )
    entries: object = request.get("input")
    if isinstance(entries, str):
        entries = [{"role": "user", "content": entries}]
    if not isinstance(entries, list):
        raise UnreadableError("The request has no input.")
    try:
        messages = await messages_from_openai_responses(
            cast(Any, entries), _model(request)
        )
    except _UNREAD as ex:
        raise UnreadableError("The request holds input that can't be read.") from ex
    instructions = request.get("instructions")
    if isinstance(instructions, str) and instructions:
        return [ChatMessageSystem(content=instructions), *messages]
    return messages


async def _chat_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    return cast("list[Any]", await messages_to_openai([message, *results]))


async def _responses_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    # rendered by inspect_ai, so a reasoning item goes back with the calls it
    # led to
    return cast("list[Any]", await messages_to_openai_responses([message, *results]))


def _as_sent(entry: object) -> object:
    return entry


CHAT = RequestApi("messages", _read_chat, _chat_turn, _as_sent)
"""OpenAI's Chat Completions API."""

RESPONSES = RequestApi("input", _read_responses, _responses_turn, _as_sent)
"""OpenAI's Responses API."""


def _model(request: dict[str, Any]) -> str | None:
    model = request.get("model")
    return model if isinstance(model, str) else None


def _chat_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    # the SDK's own accumulator, which is what inspect_ai's provider relies on
    from openai.lib.streaming.chat import ChatCompletionStreamState
    from openai.types.chat import ChatCompletionChunk

    state = ChatCompletionStreamState()
    try:
        for event in events:
            if "error" in event:
                raise UnreadableError(
                    "The reply stream ended in an error from the provider."
                )
            state.handle_chunk(ChatCompletionChunk.model_validate(event))
        snapshot = state.current_completion_snapshot
    except _UNREAD + (AssertionError,) as ex:
        raise UnreadableError("The reply stream can't be read.") from ex
    reply = snapshot.model_dump()
    # a choice has no finish reason until the chunk that ends it
    choices = cast("list[dict[str, Any]]", reply.get("choices") or [])
    if not choices or any(choice.get("finish_reason") is None for choice in choices):
        raise UnreadableError("The reply stream is not a complete message.")
    return reply


def _responses_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        kind = event.get("type")
        if kind in ("response.failed", "error"):
            raise UnreadableError(
                "The reply stream ended in an error from the provider."
            )
        # the event that ends the stream holds the whole response
        if kind in ("response.completed", "response.incomplete"):
            response = event.get("response")
            if isinstance(response, dict):
                return cast("dict[str, Any]", response)
    raise UnreadableError("The reply stream is not a complete response.")
