from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallError, ToolChoice, ToolInfo
from pydantic import JsonValue

Format: TypeAlias = Literal["anthropic-messages", "openai-chat", "openai-responses"]
"""Which API a request or a reply is written in. A provider can have more than one, and a proxy can serve one provider's model through another provider's API."""

# the line endings an event stream may use. str.splitlines breaks at more,
# some of which JSON allows unescaped inside a string
_LINE_END = re.compile(r"\r\n|\r|\n")


Turn: TypeAlias = tuple[ChatMessageAssistant, Mapping[str, str], JsonValue | None]
"""A rejected turn: the model's message, what the model is told happened to each of its calls by call id, and what inspect_ai's converters noted when the reply was read."""


class UnreadableError(Exception):
    """A request or reply couldn't be read into Inspect's types, so it can't be judged."""


async def read_request(
    provider: str,
    body: bytes,
    format: Format | None = None,
    *,
    start: JsonValue | None = None,
) -> list[ChatMessage]:
    """Read a request body into the messages the model is being sent.

    Args:
        provider: Whose API the body is written for, e.g. `anthropic`.
        body: The request body, in the provider's own format.
        format: Which API the body is written in, if the proxy said. It is then read as that, whatever the provider.
        start: What was noted while reading the conversation's replies, as `read_reply_noted` hands it back. A request holds those replies, and Anthropic can give a server tool's result in a later reply than the one that started it, which can't be read without it.

    Raises:
        UnreadableError: If there is no reader for the provider's API, or the body isn't what that API takes.
    """
    from ._inspect_ai import noting

    request = json_object(body, "request")
    api = _api(provider, request, format)
    with noting(start):
        messages = await api.read(request)
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            _check_parsed(message, "request")
    return messages


async def read_opening(
    provider: str, body: bytes, format: Format | None = None
) -> list[ChatMessage]:
    """Read how a request's conversation opens: its system and developer messages, and its first user message.

    It is read without anything noted from the conversation's replies, which none of those messages need, so it can say which conversation the request continues before the rest is read.

    Raises:
        UnreadableError: As `read_request` does.
    """
    from ._inspect_ai import noting

    request = json_object(body, "request")
    api = _api(provider, request, format)
    opening: list[Any] = []
    user = False
    for entry in _conversation(request, api):
        role = (
            cast("dict[str, Any]", entry).get("role")
            if isinstance(entry, dict)
            else None
        )
        if role in ("system", "developer") or (role == "user" and not user):
            opening.append(entry)
            user = user or role == "user"
    with noting():
        return await api.read(request | {api.conversation: opening})


async def read_reply(
    provider: str, content_type: str, body: bytes, format: Format | None = None
) -> ModelOutput:
    """Read a reply body, streamed or not, into the model's output.

    Args:
        provider: Whose API the body is written in, e.g. `anthropic`.
        content_type: The reply's `content-type`, which says whether the body is a stream.
        body: The whole reply body, in the provider's own format.
        format: Which API the body is written in, if the proxy said. It is then read as that, whatever the provider.

    Raises:
        UnreadableError: If there is no reader for the provider's API, the body isn't a complete reply, or it holds content that can't be read.
    """
    output, _ = await read_reply_noted(provider, content_type, body, format)
    return output


async def read_reply_noted(
    provider: str,
    content_type: str,
    body: bytes,
    format: Format | None = None,
    *,
    start: JsonValue | None = None,
) -> tuple[ModelOutput, JsonValue | None]:
    """Read a reply body, as `read_reply` does, with what inspect_ai's converters noted while reading it.

    What they noted is what reading the conversation's next reply needs, and what writing this reply's message back into a request needs, to put back each item as the provider wrote it.

    Args:
        provider: Whose API the body is written in.
        content_type: The reply's `content-type`.
        body: The whole reply body.
        format: Which API the body is written in, if the proxy said.
        start: What was noted while reading the conversation's earlier replies. Anthropic can start a server tool in one reply and give its result in the next, which can't be read without it.
    """
    from ._inspect_ai import noting

    with noting(start) as noted:
        output = await _read_reply(provider, content_type, body, format)
    if output.choices:
        _check_parsed(output.message, "reply")
    return output, noted.value


def _check_parsed(message: ChatMessageAssistant, what: str) -> None:
    # inspect_ai reads a call whose arguments it can't parse, such as JSON
    # nested deeper than it allows, as one with no arguments and an error,
    # while the client is handed the arguments as they were
    for call in message.tool_calls or []:
        if call.parse_error is not None:
            raise UnreadableError(
                f"The {what} holds a call to {call.function} whose arguments can't be read."
            )


async def _read_reply(
    provider: str, content_type: str, body: bytes, format: Format | None
) -> ModelOutput:
    if format == "anthropic-messages" or (format is None and provider == "anthropic"):
        from . import _anthropic

        return await _anthropic.read_reply(content_type, body)
    if format is not None or provider == "openai":
        from . import _openai

        return await _openai.read_reply(
            content_type, body, None if format is None else format == "openai-chat"
        )
    raise UnreadableError(f"There is no reader for provider {provider!r}.")


@dataclass(frozen=True)
class Offered:
    """What a request offers the model besides its messages."""

    tools: list[ToolInfo]
    tool_choice: ToolChoice
    config: GenerateConfig


def offered(provider: str, body: bytes, format: Format | None = None) -> Offered:
    """The tools, tool choice and configuration a request sends the model.

    Raises:
        UnreadableError: If there is no reader for the provider's API, or the request's tools, tool choice or settings can't be read.
    """
    from ._inspect_ai import noting

    request = json_object(body, "request")
    api = _api(provider, request, format)
    try:
        with noting():
            tools, tool_choice, config = api.offered(request)
    except UnreadableError:
        raise
    except Exception as ex:
        # inspect_ai's readers raise errors of many kinds on what they can't read
        raise UnreadableError(
            "The request's tool choice or settings can't be read."
        ) from ex
    if tool_choice is None:
        # what inspect_ai's own generate sends when it is given none
        tool_choice = "auto" if tools else "none"
    return Offered(tools, tool_choice, config)


async def add_rejections(
    provider: str,
    body: bytes,
    rejected: Mapping[int, Sequence[Turn]],
    format: Format | None = None,
) -> bytes:
    """Add rejected turns to a later request of their conversation, so the model is told.

    Each rejected turn, the model's message followed by a result for each of its calls saying why it didn't run, goes where the client would have put it had it been handed the reply: after the entries the client had sent when the turn was rejected. Nothing else in the request is changed, though the body is written out again.

    Args:
        provider: Whose API the body is written for.
        body: The request body.
        rejected: The rejected turns in order, by how many entries the client's conversation held when each was rejected.
        format: Which API the body is written in, if the proxy said.
    """
    from ._inspect_ai import noting

    request = json_object(body, "request")
    api = _api(provider, request, format)
    entries = _conversation(request, api)
    sent: list[Any] = []
    for count in range(len(entries) + 1):
        for message, told, noted in rejected.get(count, ()):
            # as inspect_ai writes a rejected call's result: no content, and
            # the reason as an approval error, which each format writes out
            results = [
                ChatMessageTool(
                    content="",
                    tool_call_id=call.id,
                    function=call.function,
                    error=ToolCallError("approval", told[call.id]),
                )
                for call in message.tool_calls or []
            ]
            # from what was noted when the turn was read, and only that,
            # so each item goes back as the provider wrote it for this run
            with noting(noted):
                sent.extend(await api.rejected_turn(message, results))
        if count < len(entries):
            sent.append(entries[count])
    request[api.conversation] = sent
    return json.dumps(request).encode()


def call_texts(
    provider: str, body: bytes, format: Format | None = None
) -> dict[str, list[str]]:
    """What the model said with each tool call in a request, as the client wrote it, by call id, in the order the calls come, where reading the request can change it.

    inspect_ai's Chat Completions reader merges a turn some scaffolds split, assistant(calls) -> tool -> assistant(text), moving the later text into the turn that made the calls. What the model said with a call is then its own words as the client wrote them, so a summary that follows a result doesn't make the result new. Other formats are read as written, and have nothing here.
    """
    request = json_object(body, "request")
    if _request_format(provider, request, format) != "openai-chat":
        return {}
    said: dict[str, list[str]] = {}
    for entry in _conversation(request, _api(provider, request, format)):
        if not isinstance(entry, dict):
            continue
        message = cast("dict[str, Any]", entry)
        calls: object = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            continue
        # a refusal too, which inspect_ai reads as the text when there is no
        # content
        written = json.dumps(
            [message.get("content"), message.get("refusal")], sort_keys=True
        )
        for call in cast("list[object]", calls):
            if isinstance(call, dict):
                call_id = str(cast("dict[str, Any]", call).get("id"))
                said.setdefault(call_id, []).append(written)
    return said


def request_marks(
    provider: str, body: bytes, format: Format | None = None
) -> list[str]:
    """A mark for each length of a request's conversation, to tell whether a later request continues from it.

    Two requests whose marks agree at `n` open with the same `n` entries. A mark leaves out what a client changes from one request to the next while the conversation stays the same.

    Returns:
        One mark for each count of entries from none to all of them, so one more mark than there are entries.
    """
    request = json_object(body, "request")
    api = _api(provider, request, format)
    running = hashlib.sha256()
    marks = [running.hexdigest()]
    for entry in _conversation(request, api):
        running.update(json.dumps(api.settled(entry), sort_keys=True).encode())
        running.update(bytes(1))
        marks.append(running.hexdigest())
    return marks


# inspect_ai's providers write an item's `internal` into its text in this
# tag, and its readers take every such span out of a text but read only the
# first. See parse_content_with_internal() in inspect_ai's
# src/inspect_ai/model/_internal.py.
_INTERNAL = re.compile(r"<content-internal>.*?</content-internal>", re.DOTALL)


def check_internal(texts: Iterable[object], what: str) -> None:
    """Refuse a body in which inspect_ai's readers would take text out unread.

    Args:
        texts: The texts of the body that inspect_ai's readers take `<content-internal>` spans out of. Each reader passes its own: the others are read whole, and a span in them is kept.
        what: `request` or `reply`, for the error.

    Raises:
        UnreadableError: If one of `texts` holds more than one span, as only the first is read and the others are dropped, while the model is sent them all.
    """
    for text in texts:
        if isinstance(text, str) and len(_INTERNAL.findall(text)) > 1:
            raise UnreadableError(
                f"The {what} holds text with more than one <content-internal> span, of which only the first would be read."
            )


def check_internal_kept(
    texts: Iterable[object], messages: Iterable[ChatMessage], what: str
) -> None:
    """Refuse a body whose `<content-internal>` values inspect_ai's readers took out of its texts but kept nowhere.

    A reader keeps a span's value on the part it reads the text as, but some read no part for a text that holds nothing else, or keep the text and drop the value.

    Args:
        texts: The texts the readers take spans out of, as `check_internal` is given them.
        messages: What the readers read the body as.
        what: `request` or `reply`, for the error.

    Raises:
        UnreadableError: If the value of a span in `texts` is on no part of `messages`.
    """
    from ._inspect_ai import internal_key, internal_text

    sent = {
        read[0]
        for text in texts
        if isinstance(text, str) and (read := internal_text(text)) is not None
    }
    if not sent:
        return
    kept = {
        internal_key(part.internal)
        for message in messages
        if not isinstance(message.content, str)
        for part in message.content
        if getattr(part, "internal", None) is not None
    }
    if sent - kept:
        raise UnreadableError(
            f"The {what} holds a <content-internal> value that would be dropped unread."
        )


def json_object(body: bytes, what: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(body)
    except ValueError as ex:
        raise UnreadableError(f"The {what} body is not JSON.") from ex
    if not isinstance(parsed, dict):
        raise UnreadableError(f"The {what} body is not a JSON object.")
    return cast("dict[str, Any]", parsed)


@dataclass
class StreamEvent:
    """One event of a reply stream as it was written: its lines, each with the line end after it, and its data read as JSON."""

    lines: list[tuple[str, str]]
    data: dict[str, Any] | None
    """The event's data, or None if it has none that is a JSON object, as for OpenAI's `[DONE]`."""
    changed: bool = False
    """Whether `data` was changed, so that the event is written out again."""


# a line end, kept by the split
_LINE_END_KEPT = re.compile(r"(\r\n|\r|\n)")


def written_events(body: bytes) -> list[StreamEvent]:
    """A reply stream's events as they were written, so that the events left unchanged go out again byte for byte.

    Raises:
        UnreadableError: If the stream is not UTF-8.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as ex:
        raise UnreadableError("The reply stream is not UTF-8.") from ex
    # each line with the line end after it, the last with none
    parts = _LINE_END_KEPT.split(text)
    lines = list(zip(parts[::2], [*parts[1::2], ""], strict=True))
    events: list[StreamEvent] = []
    held: list[tuple[str, str]] = []
    for line, end in lines:
        held.append((line, end))
        # a blank line ends an event, and so does the end of the stream
        if not line and any(written for written, _ in held[:-1]):
            events.append(_event(held))
            held = []
    if held:
        events.append(_event(held))
    return events


def _event(lines: list[tuple[str, str]]) -> StreamEvent:
    payload = "\n".join(
        line[5:].lstrip(" ") for line, _ in lines if line.startswith("data:")
    )
    try:
        data: object = json.loads(payload) if payload else None
    except ValueError:
        # OpenAI's chat stream ends with an event that is not JSON
        data = None
    return StreamEvent(
        lines, cast("dict[str, Any]", data) if isinstance(data, dict) else None
    )


def made_event(like: StreamEvent, data: dict[str, Any]) -> StreamEvent:
    """A new event, written as `like` is: with an `event:` line naming its type if `like` has one, and `like`'s line end."""
    end = like.lines[0][1] or "\n"
    lines: list[tuple[str, str]] = []
    if any(line.startswith("event:") for line, _ in like.lines):
        lines.append((f"event: {data.get('type')}", end))
    lines += [(f"data: {json.dumps(data)}", end), ("", end)]
    return StreamEvent(lines, data)


def write_events(events: Iterable[StreamEvent]) -> bytes:
    """A reply stream from its events. An event whose data changed has its data written on one line; every other line goes out as it was."""
    written: list[str] = []
    for event in events:
        first = True
        for line, end in event.lines:
            if not (event.changed and line.startswith("data:")):
                written.append(line + end)
            elif first:
                written.append(f"data: {json.dumps(event.data)}{end}")
                first = False
    return "".join(written).encode()


@dataclass(frozen=True)
class Edits:
    """What to change in a reply's tool calls before the client is handed it."""

    removed: frozenset[str] = frozenset()
    """The ids of the calls to take out."""
    modified: Mapping[str, ToolCall] = field(default_factory=dict[str, ToolCall])
    """The call to hand out in place of each call, by id. Only its arguments differ from the model's."""

    @property
    def targets(self) -> frozenset[str]:
        """The ids of every call that is changed or taken out."""
        return self.removed | self.modified.keys()


def check_found(found: Iterable[str], edits: Edits) -> None:
    """Refuse an edit that didn't find every call it changes in the reply.

    Raises:
        UnreadableError: If a call of `edits` was not found.
    """
    missing = edits.targets - set(found)
    if missing:
        raise UnreadableError(
            f"The reply has no call this sidecar can change with the id {sorted(missing)[0]!r}."
        )


async def edit_reply(
    provider: str,
    content_type: str,
    body: bytes,
    format: Format | None,
    edits: Edits,
    output: ModelOutput,
    *,
    start: JsonValue | None = None,
) -> bytes:
    """Take tool calls out of a reply and change the arguments of others, as the provider wrote it, streamed or not.

    The reply is edited as written, so every field inspect_ai doesn't read is kept, and in a stream every event the edit doesn't touch goes out byte for byte. The edited reply is then read back, and must read as the reply did with only the edits made.

    Args:
        provider: Whose API the body is written in.
        content_type: The reply's `content-type`, which says whether the body is a stream.
        body: The whole reply body.
        format: Which API the body is written in, if the proxy said.
        edits: The calls to take out, and the calls to change.
        output: The reply as it was read.
        start: What was noted while reading the conversation's earlier replies, as the reply was read with.

    Raises:
        UnreadableError: If a call can't be changed as asked, or the edited reply doesn't read back as the reply with the edits made.
    """
    from ._inspect_ai import noting

    if format == "anthropic-messages" or (format is None and provider == "anthropic"):
        from . import _anthropic

        edited = _anthropic.edit_reply(content_type, body, edits)
    elif format is not None or provider == "openai":
        from . import _openai

        edited = _openai.edit_reply(
            content_type,
            body,
            None if format is None else format == "openai-chat",
            edits,
        )
    else:
        raise UnreadableError(f"There is no reader for provider {provider!r}.")
    with noting(start):
        read = await _read_reply(provider, content_type, edited, format)
    _check_edited(output, read, edits)
    return edited


def _check_edited(output: ModelOutput, edited: ModelOutput, edits: Edits) -> None:
    # The edited reply as read must be the reply as read, with the calls taken
    # out and changed and nothing else different. A message is given a new id
    # each time it is read.
    if not output.choices or len(edited.choices) != 1:
        raise UnreadableError("The edited reply doesn't read as one message.")
    before, after = output.message, edited.message
    intended = [
        (call.id, call.function, edits.modified.get(call.id, call).arguments)
        for call in before.tool_calls or []
        if call.id not in edits.removed
    ]
    made = [(call.id, call.function, call.arguments) for call in after.tool_calls or []]
    unchanged = {"id", "tool_calls"}
    if (
        made != intended
        or after.model_dump(exclude=unchanged) != before.model_dump(exclude=unchanged)
        or edited.stop_reason != output.stop_reason
    ):
        raise UnreadableError("The edited reply doesn't read back as intended.")


def stream_events(body: bytes) -> Iterator[dict[str, Any]]:
    held: list[str] = []
    # a blank line ends an event, and so does the end of the stream
    for line in [*_LINE_END.split(body.decode("utf-8", errors="replace")), ""]:
        if line.startswith("data:"):
            held.append(line[5:].lstrip(" "))
            continue
        if line:
            continue
        data, held = "\n".join(held), []
        # OpenAI's chat stream ends with an event that is not JSON
        if not data or data == "[DONE]":
            continue
        try:
            event: object = json.loads(data)
        except ValueError as ex:
            raise UnreadableError(
                "The reply stream holds an event that is not JSON."
            ) from ex
        if not isinstance(event, dict):
            raise UnreadableError(
                "The reply stream holds an event that is not a JSON object."
            )
        yield cast("dict[str, Any]", event)


@dataclass(frozen=True)
class _Api:
    # how one provider API writes a request
    conversation: str
    read: Callable[[dict[str, Any]], Awaitable[list[ChatMessage]]]
    offered: Callable[
        [dict[str, Any]], tuple[list[ToolInfo], ToolChoice | None, GenerateConfig]
    ]
    rejected_turn: Callable[
        [ChatMessageAssistant, Sequence[ChatMessageTool]], Awaitable[list[Any]]
    ]
    settled: Callable[[object], object]


def _api(provider: str, request: dict[str, Any], format: Format | None) -> _Api:
    from . import _inspect_ai

    match _request_format(provider, request, format):
        case "anthropic-messages":
            from . import _anthropic

            return _Api(
                "messages",
                _anthropic.read_request,
                _inspect_ai.anthropic_offered,
                _anthropic.rejected_turn,
                _anthropic.settled,
            )
        case "openai-chat":
            from . import _openai

            return _Api(
                "messages",
                _openai.read_chat_request,
                _inspect_ai.chat_offered,
                _openai.chat_rejected_turn,
                _as_sent,
            )
        case "openai-responses":
            from . import _openai

            return _Api(
                "input",
                _openai.read_responses_request,
                _inspect_ai.responses_offered,
                _openai.responses_rejected_turn,
                _as_sent,
            )


def _as_sent(entry: object) -> object:
    return entry


def _conversation(request: dict[str, Any], api: _Api) -> list[Any]:
    entries: object = request.get(api.conversation)
    if isinstance(entries, str):
        # the Responses API takes a lone user message as a string
        return [{"role": "user", "content": entries}]
    return cast("list[Any]", entries) if isinstance(entries, list) else []


def _request_format(
    provider: str, request: dict[str, Any], format: Format | None
) -> Format:
    if format is not None:
        return format
    if provider == "anthropic":
        return "anthropic-messages"
    if provider == "openai":
        # the proxy named the provider, and not which of its APIs a request is for
        if "messages" in request:
            return "openai-chat"
        if "input" in request:
            return "openai-responses"
        raise UnreadableError("There is no reader for this OpenAI API.")
    raise UnreadableError(f"There is no reader for provider {provider!r}.")
