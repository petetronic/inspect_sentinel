from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    messages_from_anthropic,
    model_output_from_anthropic,
)

from ._wire import RequestApi, UnreadableError, json_object, stream_events

# The content kinds inspect_ai's Anthropic converter reads. It drops any other
# kind without a word, so a reply holding one is refused here as unreadable
# rather than judged with part of it unseen.
_BLOCKS = frozenset(
    {
        "text",
        "thinking",
        "redacted_thinking",
        "tool_use",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "mcp_tool_use",
        "mcp_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "compaction",
        "fallback",
    }
)


async def read_reply(content_type: str, body: bytes) -> ModelOutput:
    """Read a Messages API reply, streamed or not."""
    if content_type.startswith("text/event-stream"):
        message = _stream(body)
    else:
        from anthropic._models import construct_type
        from anthropic.types import Message

        # parsed as the SDK parses a reply, so a field value this SDK version
        # doesn't know isn't an error
        message = construct_type(type_=Message, value=json_object(body, "reply"))
    blocks = cast("list[Any]", getattr(message, "content", None) or [])
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind not in _BLOCKS:
            raise UnreadableError(
                f"The reply holds content of a kind that can't be read: {kind!r}."
            )
    return await model_output_from_anthropic(cast(Any, message))


async def _read(request: dict[str, Any]) -> list[ChatMessage]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise UnreadableError("The request has no list of messages.")
    return await messages_from_anthropic(
        cast(Any, messages), _system(request.get("system"))
    )


async def _rejected_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    # inspect_ai has no public function that writes a message in Anthropic's
    # format, as it has for OpenAI's, so this is its provider's own
    from inspect_ai.model._providers.anthropic import (  # pyright: ignore[reportMissingTypeStubs]
        message_param,
    )

    blocks: list[Any] = []
    for result in results:
        rendered = cast("dict[str, Any]", await message_param(result))
        blocks.extend(cast("list[Any]", rendered["content"]))
    # rendered by inspect_ai, so a thinking block keeps its signature and a
    # call its id
    return [await message_param(message), {"role": "user", "content": blocks}]


def _settled(message: object) -> object:
    # inspect_ai's provider moves its cache marks to the latest messages on
    # every request, and writes a text it marks as a block
    if not isinstance(message, dict):
        return message
    entry = cast("dict[str, Any]", message)
    content: object = entry.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content = [
            {
                k: v
                for k, v in cast("dict[str, Any]", block).items()
                if k != "cache_control"
            }
            if isinstance(block, dict)
            else block
            for block in cast("list[object]", content)
        ]
    return {**entry, "content": content}


MESSAGES = RequestApi("messages", _read, _rejected_turn, _settled)
"""Anthropic's Messages API."""


def _system(system: object) -> str | None:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        texts = [
            str(cast("dict[str, Any]", block).get("text", ""))
            for block in cast("list[object]", system)
            if isinstance(block, dict)
        ]
        return "\n\n".join(texts) or None
    return None


def _stream(body: bytes) -> object:
    # the SDK's own accumulator, which is what inspect_ai's provider relies on.
    # It is in a private module of the SDK and its signature has changed
    # between releases: newer ones take the caller's `json_bufs`.
    from anthropic._models import construct_type
    from anthropic.lib.streaming import (
        _messages,  # pyright: ignore[reportPrivateUsage]
    )
    from anthropic.types import RawMessageStreamEvent

    accumulate = cast(Any, _messages.accumulate_event)
    extra: dict[str, Any] = (
        {"json_bufs": {}}
        if "json_bufs" in inspect.signature(accumulate).parameters
        else {}
    )
    snapshot: Any = None
    stopped = False
    for event in stream_events(body):
        kind = event.get("type")
        if kind == "ping":
            continue
        if kind == "error":
            raise UnreadableError(
                "The reply stream ended in an error from the provider."
            )
        stopped = stopped or kind == "message_stop"
        snapshot = accumulate(
            event=construct_type(type_=RawMessageStreamEvent, value=event),
            current_snapshot=snapshot,
            **extra,
        )
    if snapshot is None or not stopped:
        raise UnreadableError("The reply stream is not a complete message.")
    return cast(object, snapshot)
