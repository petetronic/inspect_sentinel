from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
)
from inspect_ai.tool import ToolCallError

# the line endings an event stream may use. str.splitlines breaks at more,
# some of which JSON allows unescaped inside a string
_LINE_END = re.compile(r"\r\n|\r|\n")


class UnreadableError(Exception):
    """A request or reply couldn't be read into Inspect's types, so it can't be judged."""


@dataclass(frozen=True)
class RequestApi:
    """How one provider API writes a request."""

    conversation: str
    """The request field that holds the conversation."""

    read: Callable[[dict[str, Any]], Awaitable[list[ChatMessage]]]
    """Read a request into the messages the model is being sent."""

    rejected_turn: Callable[
        [ChatMessageAssistant, Sequence[ChatMessageTool]], Awaitable[list[Any]]
    ]
    """Write the model's message and a result for each of its calls as entries of the conversation."""

    settled: Callable[[object], object]
    """An entry of the conversation without what a client changes from one request to the next while the conversation stays the same."""


async def read_request(provider: str, body: bytes) -> list[ChatMessage]:
    """Read a request body into the messages the model is being sent.

    Args:
        provider: Whose API the body is written for.
        body: The request body.

    Raises:
        UnreadableError: If there is no reader for the provider's API, or the body isn't what that API takes.
    """
    request = json_object(body, "request")
    return await _request_api(provider, request).read(request)


async def read_reply(provider: str, content_type: str, body: bytes) -> ModelOutput:
    """Read a reply body, streamed or not, into the model's output.

    Args:
        provider: Whose API the body is written in.
        content_type: The reply's `content-type`.
        body: The whole reply body.

    Raises:
        UnreadableError: If there is no reader for the provider's API, the body isn't a complete reply, or it holds content that can't be read.
    """
    if provider == "anthropic":
        from . import _anthropic

        return await _anthropic.read_reply(content_type, body)
    if provider == "openai":
        from . import _openai

        return await _openai.read_reply(content_type, body)
    raise UnreadableError(f"There is no reader for provider {provider!r}.")


async def add_rejections(
    provider: str,
    body: bytes,
    rejected: Mapping[int, Sequence[tuple[ChatMessageAssistant, dict[str, str]]]],
) -> bytes:
    """Add the turns that were rejected to a later request of their conversation, so the model is told.

    For each rejected turn, the model's own message and a result for each call in it saying why it didn't run are put where the client would have put them had it been handed the reply: after the entries the client had sent when the turn was rejected. Nothing else in the request is changed.

    Args:
        provider: Whose API the body is written for.
        body: The request body, which continues the conversation the turns were rejected in.
        rejected: The rejected turns in order, by how many entries the client's conversation held when they were rejected. A turn is the model's message, and for each call id in it what the model is told happened to that call.

    Returns:
        The body with the entries added.

    Raises:
        UnreadableError: If there is no reader for the provider's API, or the body isn't what that API takes.
    """
    request = json_object(body, "request")
    api = _request_api(provider, request)
    entries = _conversation(request, api)
    sent: list[Any] = []
    for count in range(len(entries) + 1):
        for message, results in rejected.get(count, ()):
            told = [
                ChatMessageTool(
                    content=results[call.id],
                    tool_call_id=call.id,
                    function=call.function,
                    error=ToolCallError("approval", results[call.id]),
                )
                for call in message.tool_calls or []
            ]
            sent.extend(await api.rejected_turn(message, told))
        if count < len(entries):
            sent.append(entries[count])
    request[api.conversation] = sent
    return json.dumps(request).encode()


def request_marks(provider: str, body: bytes) -> list[str]:
    """A mark for each length of a request's conversation, for telling later whether another request continues from it.

    Two requests whose marks agree at `n` open with the same `n` entries. A mark leaves out what a client changes from one request to the next while the conversation stays the same, such as where it asks for the prompt to be cached.

    Args:
        provider: Whose API the body is written for.
        body: The request body.

    Returns:
        One mark for each count of entries from none to all of them, so the list is one longer than the request's conversation.

    Raises:
        UnreadableError: If there is no reader for the provider's API, or the body isn't what that API takes.
    """
    request = json_object(body, "request")
    api = _request_api(provider, request)
    running = hashlib.sha256()
    marks = [running.hexdigest()]
    for entry in _conversation(request, api):
        running.update(json.dumps(api.settled(entry), sort_keys=True).encode())
        running.update(bytes(1))
        marks.append(running.hexdigest())
    return marks


def json_object(body: bytes, what: str) -> dict[str, Any]:
    """Parse a body that must be a JSON object."""
    try:
        parsed: object = json.loads(body)
    except ValueError as ex:
        raise UnreadableError(f"The {what} body is not JSON.") from ex
    if not isinstance(parsed, dict):
        raise UnreadableError(f"The {what} body is not a JSON object.")
    return cast("dict[str, Any]", parsed)


def stream_events(body: bytes) -> Iterator[dict[str, Any]]:
    """The JSON object of each event in a server-sent event stream, in order."""
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


def _request_api(provider: str, request: dict[str, Any]) -> RequestApi:
    if provider == "anthropic":
        from . import _anthropic

        return _anthropic.MESSAGES
    if provider == "openai":
        from . import _openai

        # the proxy names the provider, not which of its APIs a request is for
        if "messages" in request:
            return _openai.CHAT
        if "input" in request:
            return _openai.RESPONSES
        raise UnreadableError("There is no reader for this OpenAI API.")
    raise UnreadableError(f"There is no reader for provider {provider!r}.")


def _conversation(request: dict[str, Any], api: RequestApi) -> list[Any]:
    entries: object = request.get(api.conversation)
    if isinstance(entries, str):
        # the Responses API takes a lone user message as a string
        return [{"role": "user", "content": entries}]
    return cast("list[Any]", entries) if isinstance(entries, list) else []
