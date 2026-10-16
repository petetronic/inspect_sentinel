from __future__ import annotations

import inspect
import json
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

from ._inspect_ai import anthropic_message
from ._wire import (
    Edits,
    Spliced,
    StreamEvent,
    UnreadableError,
    check_found,
    check_internal,
    check_internal_kept,
    json_object,
    made_event,
    stream_events,
    write_events,
    written_events,
)

# The content kinds that inspect_ai's Anthropic converter reads, in a reply or
# in an assistant turn of a request. It drops any other kind without an error,
# so a body holding one is unreadable here, rather than judged with part of it
# unseen. Kept in step with content_and_tool_calls_from_assistant_content_blocks()
# in inspect_ai's src/inspect_ai/model/_providers/anthropic.py.
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

# The kinds of content delta that the SDK's stream accumulator applies. It
# drops any other kind without an error, such as the delta that carries a
# compaction block's summary, so a stream holding one is unreadable here.
# Kept in step with accumulate_event() in the SDK's
# anthropic/lib/streaming/_messages.py.
_DELTAS = frozenset(
    {
        "text_delta",
        "input_json_delta",
        "citations_delta",
        "thinking_delta",
        "signature_delta",
    }
)


async def read_reply(content_type: str, body: bytes) -> ModelOutput:
    if content_type.startswith("text/event-stream"):
        message = _stream(body)
    else:
        message = _message(json_object(body, "reply"))
    _check_blocks(getattr(message, "content", None), "reply")
    blocks = cast("list[object]", getattr(message, "content", None) or [])
    texts = [getattr(block, "text", None) for block in blocks]
    check_internal(texts, "reply")
    try:
        # an object, not a dict, so that the converter doesn't validate it again
        output = await model_output_from_anthropic(cast(Any, message))
    except Exception as ex:
        # the converter raises errors of many kinds on content it can't read
        raise UnreadableError("The reply holds content that can't be read.") from ex
    check_internal_kept(texts, [output.message] if output.choices else [], "reply")
    return output


async def read_request(request: dict[str, Any]) -> list[ChatMessage]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise UnreadableError("The request has no list of messages.")
    texts: list[object] = []
    for entry in cast("list[object]", messages):
        if not isinstance(entry, dict):
            continue
        entry = cast("dict[str, Any]", entry)
        if entry.get("role") == "assistant" and isinstance(entry.get("content"), list):
            _check_blocks(entry["content"], "request")
        _check_documents(entry.get("content"))
        texts.extend(_texts(entry))
    check_internal(texts, "request")
    try:
        read = await messages_from_anthropic(
            cast(Any, messages), _system(request.get("system"))
        )
    except Exception as ex:
        # the converter raises errors of many kinds on content it can't read
        raise UnreadableError("The request holds a message that can't be read.") from ex
    check_internal_kept(texts, read, "request")
    return read


async def rejected_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    # written by inspect_ai, so a thinking block keeps its signature and a call
    # its id
    blocks: list[Any] = []
    for result in results:
        blocks.extend(cast("list[Any]", (await anthropic_message(result))["content"]))
    return [await anthropic_message(message), {"role": "user", "content": blocks}]


def edit_reply(content_type: str, body: bytes, edits: Edits) -> bytes:
    """A reply with the `tool_use` blocks of `edits` taken out or given new input, streamed or not.

    Raises:
        UnreadableError: If the reply doesn't hold every call of `edits` as a `tool_use` block, or is a stream whose `message_start` already holds content.
    """
    if content_type.startswith("text/event-stream"):
        return _edit_stream(written_events(body), edits)
    reply = json_object(body, "reply")
    content: object = reply.get("content")
    if not isinstance(content, list):
        raise UnreadableError("The reply has no list of content.")
    found: list[str] = []
    blocks: list[object] = []
    for block in cast("list[object]", content):
        call = _tool_use(block)
        if call is None:
            blocks.append(block)
            continue
        if call in edits.removed:
            found.append(call)
            continue
        if call in edits.modified:
            found.append(call)
            block = {
                **cast("dict[str, Any]", block),
                "input": edits.modified[call].arguments,
            }
        blocks.append(block)
    check_found(found, edits)
    reply["content"] = blocks
    return json.dumps(reply).encode()


def _tool_use(block: object) -> str | None:
    # the id of a call the client runs, which inspect_ai reads as a tool call
    entry = cast("dict[str, Any]", block) if isinstance(block, dict) else {}
    if entry.get("type") == "tool_use":
        id: object = entry.get("id")
        return id if isinstance(id, str) else None
    return None


def _edit_stream(events: list[StreamEvent], edits: Edits) -> bytes:
    # each block is told apart by its index, which the later blocks' events
    # are moved down by for each block taken out before them
    removed: set[int] = set()
    modified: dict[int, str] = {}
    found: list[str] = []
    for event in events:
        data = event.data or {}
        if data.get("type") == "message_start" and (
            cast("dict[str, Any]", data.get("message") or {}).get("content")
        ):
            raise UnreadableError(
                "The reply stream holds content in its message_start, which this sidecar can't change."
            )
        if data.get("type") != "content_block_start":
            continue
        call = _tool_use(data.get("content_block"))
        index: object = data.get("index")
        if call is not None and call in edits.targets and isinstance(index, int):
            found.append(call)
            if call in edits.removed:
                removed.add(index)
            else:
                modified[index] = call
    check_found(found, edits)
    written: list[StreamEvent] = []
    for event in events:
        data = event.data or {}
        index = data.get("index")
        kind = str(data.get("type", ""))
        if not kind.startswith("content_block_") or not isinstance(index, int):
            written.append(event)
            continue
        if index in removed:
            continue
        if index in modified and kind == "content_block_delta":
            # its input goes out whole, in one delta after its start
            continue
        moved = index - sum(1 for taken in removed if taken < index)
        if moved != index:
            data["index"] = moved
            event.changed = True
        if index in modified and kind == "content_block_start":
            block = cast("dict[str, Any]", data["content_block"])
            data["content_block"] = {**block, "input": {}}
            event.changed = True
            written.append(event)
            arguments = edits.modified[modified[index]].arguments
            delta = {"type": "input_json_delta", "partial_json": json.dumps(arguments)}
            written.append(
                made_event(
                    event,
                    {"type": "content_block_delta", "index": moved, "delta": delta},
                )
            )
            continue
        written.append(event)
    return write_events(written)


def turn_at(entries: list[Any], sent: int) -> Spliced | None:
    # the assistant message at `sent`, and the user message after it that
    # holds the results of its calls
    turn: object = entries[sent] if sent < len(entries) else None
    if not isinstance(turn, dict):
        return None
    message = cast("dict[str, Any]", turn)
    content: object = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, list):
        return None
    after: object = entries[sent + 1] if sent + 1 < len(entries) else None
    following = cast("dict[str, Any]", after) if isinstance(after, dict) else None
    results: list[Any] = []
    if following is not None and following.get("role") == "user":
        said: object = following.get("content")
        if isinstance(said, list):
            results = list(cast("list[Any]", said))
        elif isinstance(said, str) and said:
            results = [{"type": "text", "text": said}]
    else:
        following = None

    def rebuild(calls: list[Any], told: list[Any]) -> list[Any]:
        written: list[Any] = [{**message, "content": calls}]
        if following is not None:
            written.append({**following, "content": told})
        elif told:
            written.append({"role": "user", "content": told})
        return written

    return Spliced(
        sent,
        sent + (2 if following is not None else 1),
        list(cast("list[Any]", content)),
        results,
        rebuild,
    )


def call_ids(entry: object) -> list[str]:
    message = cast("dict[str, Any]", entry) if isinstance(entry, dict) else {}
    content: object = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, list):
        return []
    return [id for block in cast("list[object]", content) if (id := _tool_use(block))]


def call_of(element: object) -> tuple[str | None, str | None, object]:
    id = _tool_use(element)
    if id is None:
        return None, None, None
    block = cast("dict[str, Any]", element)
    return id, block.get("name"), block.get("input")


def result_of(element: object) -> tuple[str | None, str, bool]:
    block = cast("dict[str, Any]", element) if isinstance(element, dict) else {}
    if block.get("type") != "tool_result":
        return None, "", False
    content: object = block.get("content")
    text = (
        "".join(
            str(cast("dict[str, Any]", part).get("text", ""))
            for part in cast("list[object]", content)
            if isinstance(part, dict)
        )
        if isinstance(content, list)
        else str(content or "")
    )
    return block.get("tool_use_id"), text, bool(block.get("is_error"))


def pick(rendered: list[Any], id: str) -> tuple[Any, tuple[Any, ...]]:
    # a call's block, and its result's, from a turn inspect_ai wrote
    blocks = cast("list[Any]", cast("dict[str, Any]", rendered[0])["content"])
    call = next(block for block in blocks if _tool_use(block) == id)
    told: list[Any] = []
    for message in rendered[1:]:
        content: object = cast("dict[str, Any]", message).get("content")
        if isinstance(content, list):
            told += [
                block
                for block in cast("list[Any]", content)
                if result_of(block)[0] == id
            ]
    return call, tuple(told)


def settled(entry: object) -> object:
    # inspect_ai's Anthropic provider moves its cache marks to the latest
    # messages on every request, which writes a text it marks as a block
    if not isinstance(entry, dict):
        return entry
    message = cast("dict[str, Any]", entry)
    content: object = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content = [
            {
                key: value
                for key, value in cast("dict[str, Any]", block).items()
                if key != "cache_control"
            }
            if isinstance(block, dict)
            else block
            for block in cast("list[object]", content)
        ]
    return {**message, "content": content}


def _kind(block: object) -> object:
    if isinstance(block, dict):
        return cast("dict[str, Any]", block).get("type")
    return getattr(block, "type", None)


def _check_blocks(content: object, what: str) -> None:
    for block in cast("list[object]", content or []):
        kind = _kind(block)
        if kind not in _BLOCKS:
            raise UnreadableError(
                f"The {what} holds content of a kind that can't be read: {kind!r}."
            )


def _texts(message: dict[str, Any]) -> list[object]:
    # The texts inspect_ai's Anthropic reader takes <content-internal> spans
    # out of: text blocks, wherever they are, and an assistant's text given
    # as a string. A user's string, a tool result's string and a document of
    # plain text are read whole. See content_block_to_content() in
    # inspect_ai's src/inspect_ai/agent/_bridge/anthropic_api_impl.py.
    content: object = message.get("content")
    if isinstance(content, str):
        return [content] if message.get("role") == "assistant" else []
    return _block_texts(content)


def _block_texts(blocks: object) -> list[object]:
    # text blocks, and those in a tool result or a document made of blocks
    texts: list[object] = []
    for block in cast("list[object]", blocks if isinstance(blocks, list) else []):
        if not isinstance(block, dict):
            continue
        block = cast("dict[str, Any]", block)
        if block.get("type") == "text":
            texts.append(block.get("text"))
        elif block.get("type") == "tool_result":
            texts.extend(_block_texts(block.get("content")))
        elif block.get("type") == "document":
            source: object = block.get("source")
            if isinstance(source, dict):
                texts.extend(
                    _block_texts(cast("dict[str, Any]", source).get("content"))
                )
    return texts


def _check_documents(content: object) -> None:
    # inspect_ai's Anthropic converter reads only the first block of a document
    # whose source is a list of blocks, and drops the rest, and drops a
    # document's title and context, which the model is sent too, all without
    # an error. See content_block_to_content() in inspect_ai's
    # src/inspect_ai/agent/_bridge/anthropic_api_impl.py.
    if not isinstance(content, list):
        return
    for block in cast("list[object]", content):
        if not isinstance(block, dict):
            continue
        block = cast("dict[str, Any]", block)
        if block.get("type") == "tool_result":
            _check_documents(block.get("content"))
        if block.get("type") != "document":
            continue
        if block.get("title") or block.get("context"):
            raise UnreadableError(
                "The request holds a document with a title or context, which would not be read."
            )
        source = block.get("source")
        if not isinstance(source, dict):
            continue
        parts = cast("dict[str, Any]", source).get("content")
        if isinstance(parts, list) and len(cast("list[object]", parts)) > 1:
            raise UnreadableError(
                "The request holds a document of more than one block, of which only the first would be read."
            )


def _message(reply: dict[str, Any]) -> object:
    # built as the SDK builds a reply it receives, so that a field value this
    # SDK version doesn't know yet is kept rather than an error
    from anthropic._models import construct_type
    from anthropic.types import Message

    return construct_type(type_=Message, value=reply)


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
    # The SDK's own accumulator, which inspect_ai's Anthropic provider also
    # relies on. inspect_ai has no public function that turns a stream into a
    # ModelOutput, so the provider's own repairs on top of the accumulator
    # (compaction content and the code execution container) are not applied
    # here. The accumulator is in a private module of the SDK, and newer
    # releases take the caller's `json_bufs`.
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
        if kind == "content_block_delta":
            delta: object = event.get("delta")
            delta_kind = _kind(delta)
            if delta_kind not in _DELTAS:
                raise UnreadableError(
                    f"The reply stream holds content of a kind that can't be read: {delta_kind!r}."
                )
        try:
            snapshot = accumulate(
                event=construct_type(type_=RawMessageStreamEvent, value=event),
                current_snapshot=snapshot,
                **extra,
            )
        except Exception as ex:
            # the accumulator raises errors of many kinds on events out of order
            raise UnreadableError("The reply stream can't be read.") from ex
    if snapshot is None or not stopped:
        raise UnreadableError("The reply stream is not a complete message.")
    return cast(object, snapshot)
