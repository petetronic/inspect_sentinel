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

from ._inspect_ai import internal_text, is_assistant_item, without_reasoning
from ._wire import (
    UnreadableError,
    check_internal,
    check_internal_kept,
    json_object,
    stream_events,
)


async def read_reply(
    content_type: str, body: bytes, chat: bool | None = None
) -> ModelOutput:
    if content_type.startswith("text/event-stream"):
        events = list(stream_events(body))
        if chat is None:
            chat = any(
                event.get("object") == "chat.completion.chunk" for event in events
            )
        if chat:
            _check_choices(events, "delta")
        reply = _chat_stream(events) if chat else _responses_stream(events)
    else:
        parsed = json_object(body, "reply")
        if chat is None:
            kind = parsed.get("object")
            if kind not in ("chat.completion", "response"):
                raise UnreadableError(
                    f"There is no reader for an OpenAI {kind!r} reply."
                )
            chat = kind == "chat.completion"
        if chat:
            _check_choices([parsed], "message")
        reply = _completion(parsed) if chat else _response(parsed)
    if chat:
        for choice in cast("list[Any]", getattr(reply, "choices", None) or []):
            said = getattr(choice, "message", None)
            if getattr(said, "content", None) and getattr(said, "refusal", None):
                raise UnreadableError(
                    "The reply holds text and a refusal both, of which only the refusal would be read."
                )
            calls = getattr(said, "tool_calls", None)
            _check_calls(calls)
            for call in cast("list[Any]", calls or []):
                _check_arguments(getattr(call.function, "arguments", None), "reply")
    else:
        for item in cast("list[Any]", getattr(reply, "output", None) or []):
            if getattr(item, "type", None) == "function_call":
                _check_arguments(getattr(item, "arguments", None), "reply")
    try:
        # objects, not dicts, so that the converters don't validate them again
        output = await (
            model_output_from_openai(cast(Any, reply))
            if chat
            else model_output_from_openai_responses(cast(Any, reply))
        )
    except Exception as ex:
        # the converters raise errors of many kinds on content they can't read
        raise UnreadableError("The reply holds content that can't be read.") from ex
    if len(output.choices) > 1:
        raise UnreadableError(
            "The reply holds more than one choice, and only the first would be judged."
        )
    return output


async def read_chat_request(request: dict[str, Any]) -> list[ChatMessage]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise UnreadableError("The request has no list of messages.")
    if request.get("functions") is not None or request.get("function_call") is not None:
        # declared as `tools` and `tool_choice` were before them, which
        # inspect_ai's readers don't read, so a step would be told the model
        # is offered no tools
        raise UnreadableError(
            "The request offers tools in the legacy `functions` field, which can't be read."
        )
    texts: list[object] = []
    for entry in cast("list[object]", messages):
        if isinstance(entry, dict):
            message = cast("dict[str, Any]", entry)
            _check_calls(message.get("tool_calls"))
            for call in cast("list[object]", message.get("tool_calls") or []):
                function: object = (
                    cast("dict[str, Any]", call).get("function")
                    if isinstance(call, dict)
                    else None
                )
                if isinstance(function, dict):
                    _check_arguments(
                        cast("dict[str, Any]", function).get("arguments"), "request"
                    )
            if message.get("function_call") or message.get("role") == "function":
                raise UnreadableError(
                    "The request holds a call in the legacy `function_call` field, or its result, which can't be read."
                )
            # inspect_ai's reader drops the model's earlier audio, which the
            # model is sent by its id; a user's input_audio is read
            if message.get("role") == "assistant" and message.get("audio"):
                raise UnreadableError(
                    "The request holds audio from the model, which can't be read."
                )
            texts.extend(_chat_texts(message))
            if (
                message.get("role") == "assistant"
                # the refusal is read only when there is no content at all
                and message.get("content") is not None
                and message.get("refusal")
            ):
                raise UnreadableError(
                    "The request holds an assistant's text and refusal both, of which only the text would be read."
                )
    check_internal(texts, "request")
    _check_merged(cast("list[object]", messages))
    try:
        read = await messages_from_openai(cast(Any, messages), _model(request))
    except Exception as ex:
        # the converters raise errors of many kinds on content they can't read
        raise UnreadableError("The request holds a message that can't be read.") from ex
    _check_results(cast("list[object]", messages), read)
    check_internal_kept(texts, read, "request")
    return read


def _check_results(messages: list[object], read: list[ChatMessage]) -> None:
    # inspect_ai's Chat Completions converter takes reasoning tags out of a
    # tool result written as a string, and keeps only the rest of it, while
    # the model is sent all of it. A result read as anything but what was
    # sent is unreadable, rather than judged with part of it unseen.
    sent = [
        cast("dict[str, Any]", entry).get("content")
        for entry in messages
        if isinstance(entry, dict)
        and cast("dict[str, Any]", entry).get("role") == "tool"
    ]
    results = [message for message in read if isinstance(message, ChatMessageTool)]
    for content, result in zip(sent, results, strict=False):
        if isinstance(content, str) and result.text != content:
            raise UnreadableError(
                "The request holds a tool result that would be read as less than the model is sent."
            )


async def read_responses_request(request: dict[str, Any]) -> list[ChatMessage]:
    if (
        request.get("previous_response_id") is not None
        or request.get("conversation") is not None
    ):
        raise UnreadableError(
            "The request continues a conversation the provider keeps, which this sidecar is not shown."
        )
    if request.get("prompt") is not None:
        raise UnreadableError(
            "The request names a prompt the provider keeps, whose text this sidecar is not shown."
        )
    if request.get("background"):
        raise UnreadableError(
            "The request asks for a reply that is fetched later, which this sidecar would not be shown."
        )
    entries: object = request.get("input")
    if isinstance(entries, str):
        # the Responses API takes a lone user message as a string
        entries = [{"role": "user", "content": entries}]
    if not isinstance(entries, list):
        raise UnreadableError("The request has no input.")
    _check_parts(cast("list[object]", entries))
    _check_duplicates(cast("list[object]", entries))
    texts = [
        text
        for entry in cast("list[object]", entries)
        if isinstance(entry, dict)
        and cast("dict[str, Any]", entry).get("role") == "assistant"
        for text in _assistant_texts(cast("dict[str, Any]", entry))
    ]
    check_internal(texts, "request")
    for entry in cast("list[object]", entries):
        if (
            isinstance(entry, dict)
            and cast("dict[str, Any]", entry).get("type") == "function_call"
        ):
            _check_arguments(cast("dict[str, Any]", entry).get("arguments"), "request")
    try:
        messages = await messages_from_openai_responses(
            cast(Any, entries), _model(request)
        )
    except Exception as ex:
        # the converters raise errors of many kinds on content they can't read
        raise UnreadableError("The request holds input that can't be read.") from ex
    check_internal_kept(texts, messages, "request")
    instructions = request.get("instructions")
    if isinstance(instructions, str) and instructions:
        return [ChatMessageSystem(content=instructions), *messages]
    return messages


# what inspect_ai's Responses reader keeps of a tool result given as parts;
# it leaves out a part of any other kind
_RESULT_PARTS = frozenset({"input_text", "input_image"})


def _check_parts(entries: list[object]) -> None:
    # inspect_ai's Responses reader leaves out a tool result's file, and reads
    # an image or file given by a file id, which it can't fetch, as an empty
    # one, while the model is sent all of it. A part read as less than was
    # sent is unreadable, rather than judged with part of it unseen.
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = cast("dict[str, Any]", entry)
        result = item.get("type") in ("function_call_output", "custom_tool_call_output")
        parts: object = item.get("output" if result else "content")
        if not isinstance(parts, list):
            continue
        for part in cast("list[object]", parts):
            if not isinstance(part, dict):
                continue
            kind = cast("dict[str, Any]", part).get("type")
            if (result and kind not in _RESULT_PARTS) or not _whole(
                cast("dict[str, Any]", part)
            ):
                raise UnreadableError(
                    f"The request holds {'a tool result' if result else 'a message'} with an {kind!r} part that would be read as less than the model is sent."
                )


def _whole(part: dict[str, Any]) -> bool:
    # whether a part carries its content, rather than naming a file the
    # provider keeps
    match part.get("type"):
        case "input_image":
            source = part.get("image_url")
        case "input_file":
            source = part.get("file_data")
        case _:
            return True
    return isinstance(source, str) and source != ""


async def chat_rejected_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    return cast("list[Any]", await messages_to_openai([message, *results]))


async def responses_rejected_turn(
    message: ChatMessageAssistant, results: Sequence[ChatMessageTool]
) -> list[Any]:
    # written by inspect_ai, so a reasoning item goes back with the calls it
    # led to
    written = cast("list[Any]", await messages_to_openai_responses([message, *results]))
    # A computer call's output holds only a screenshot, so inspect_ai writes
    # a blank one and the model isn't told why the call didn't run. It is
    # told in a message after the outputs.
    computer = {
        cast("dict[str, Any]", entry).get("call_id")
        for entry in written
        if isinstance(entry, dict)
        and cast("dict[str, Any]", entry).get("type") == "computer_call_output"
    }
    told = [
        f"{result.tool_call_id}: {result.error.message if result.error else result.text}"
        for result in results
        if result.tool_call_id in computer
    ]
    if told:
        written.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "\n".join(told)}],
            }
        )
    return written


def _chat_texts(message: dict[str, Any]) -> list[object]:
    # The texts inspect_ai's Chat Completions reader takes <content-internal>
    # spans out of: a text part of any message, and an assistant's text given
    # as a string, once the reasoning in tags is taken out of it, as a span in
    # the reasoning stays there. Any other message's string is read whole. See
    # messages_from_openai() and content_from_openai() in inspect_ai's
    # src/inspect_ai/model/_openai.py.
    content: object = message.get("content")
    if isinstance(content, str):
        if message.get("role") != "assistant":
            return []
        return [without_reasoning(content)]
    return [
        cast("dict[str, Any]", part).get("text")
        for part in cast("list[object]", content if isinstance(content, list) else [])
        if isinstance(part, dict) and _is_text(cast("dict[str, Any]", part))
    ]


def _is_text(part: dict[str, Any]) -> bool:
    # a text part, or one written as {"text": ...} alone, which the reader
    # takes for a text part too
    return part.get("type") == "text" or ("type" not in part and set(part) == {"text"})


def _assistant_texts(entry: dict[str, Any]) -> list[object]:
    # the texts of an assistant's item in a Responses request, which
    # inspect_ai's reader takes <content-internal> spans out of; a refusal is
    # read whole
    content: object = entry.get("content")
    if isinstance(content, str):
        return [content]
    return [
        cast("dict[str, Any]", part).get("text")
        for part in cast("list[object]", content if isinstance(content, list) else [])
        if isinstance(part, dict) and "text" in cast("dict[str, Any]", part)
    ]


def _check_merged(messages: list[object]) -> None:
    # inspect_ai's Chat Completions reader merges a turn some scaffolds split,
    # assistant(calls) -> tool -> assistant(content), by moving the second
    # assistant's content into the first and dropping the rest of it, such as
    # a refusal or the model's reasoning, unread. This follows the merge as
    # openai_assistant_message_reducer() in inspect_ai's
    # src/inspect_ai/model/_openai.py makes it.
    reduced: list[dict[str, Any]] = []
    for entry in messages:
        if not isinstance(entry, dict):
            continue
        message = cast("dict[str, Any]", entry)
        if (
            len(reduced) >= 2
            and message.get("role") == "assistant"
            and "tool_calls" not in message
            and "content" in message
            and reduced[-1].get("role") == "tool"
            and reduced[-2].get("role") == "assistant"
            and "tool_calls" in reduced[-2]
            and reduced[-2].get("content") is None
        ):
            dropped = sorted(
                key
                for key, value in message.items()
                # inspect_ai reads no assistant's name, merged or not
                if key not in ("role", "content", "name")
                and value not in (None, "", [], {})
            )
            # the earlier turn's refusal too, which is read as its text only
            # while it has no content
            if reduced[-2].get("refusal"):
                dropped.append("earlier refusal")
            if dropped:
                raise UnreadableError(
                    f"The request holds an assistant's {', '.join(dropped)} that would be merged into an earlier turn and dropped."
                )
            reduced[-2] = reduced[-2] | {"content": message["content"]}
        else:
            reduced.append(message)


def _check_arguments(arguments: object, what: str) -> None:
    # inspect_ai reads a call's arguments as a JSON object. Anything else it
    # maps to the tool's first parameter, but only when it is given the tool's
    # schema, and its converters aren't: without it, the call is read with no
    # arguments and no error, while the client is handed them. See
    # parse_tool_call() in inspect_ai's src/inspect_ai/model/_call_tools.py.
    if not isinstance(arguments, str):
        return
    written = arguments.strip()
    if written and not written.startswith("{"):
        raise UnreadableError(
            f"The {what} holds a tool call whose arguments aren't a JSON object, which would be read as none."
        )


def _check_duplicates(entries: list[object]) -> None:
    # inspect_ai's Responses reader keeps only the last of an assistant's
    # texts that carry the same <content-internal> span, as a scaffold can
    # send one text twice. Two different texts with one span would lose the
    # first unread. It does so within each run of assistant items, which any
    # other item ends. See messages_from_responses_input() and
    # filter_duplicate_assistant_content() in inspect_ai's
    # src/inspect_ai/agent/_bridge/responses_impl.py.
    seen: dict[str, str] = {}
    for entry in entries:
        if not (
            isinstance(entry, dict) and is_assistant_item(cast("dict[str, Any]", entry))
        ):
            seen = {}
            continue
        if cast("dict[str, Any]", entry).get("role") != "assistant":
            continue
        content: object = cast("dict[str, Any]", entry).get("content")
        # each part read as inspect_ai reads it: a refusal is text too
        texts = (
            [content]
            if isinstance(content, str)
            else [
                str(
                    cast("dict[str, Any]", part).get(
                        "text", cast("dict[str, Any]", part).get("refusal", "")
                    )
                )
                for part in cast("list[object]", content)
                if isinstance(part, dict)
            ]
            if isinstance(content, list)
            else []
        )
        for text in texts:
            read = internal_text(text)
            if read is None:
                continue
            value, kept = read
            # compared as inspect_ai compares them: the decoded value, written
            # its way, and the text it reads
            if seen.setdefault(value, kept) != kept:
                raise UnreadableError(
                    "The request holds two different assistant texts that carry the same <content-internal> value, of which only the last would be read."
                )


def _check_choices(entries: list[dict[str, Any]], part: str) -> None:
    # inspect_ai's Chat Completions converter keeps neither a reply's audio nor
    # its transcript, so a reply of audio alone would be judged as empty while
    # the client is handed all of it, and it skips a call in the legacy
    # `function_call` field, which would reach the client unjudged. See
    # chat_choices_from_openai() in inspect_ai's src/inspect_ai/model/_openai.py.
    for entry in entries:
        choices: object = entry.get("choices")
        for choice in cast(
            "list[object]", choices if isinstance(choices, list) else []
        ):
            said: object = (
                cast("dict[str, Any]", choice).get(part)
                if isinstance(choice, dict)
                else None
            )
            if not isinstance(said, dict):
                continue
            if cast("dict[str, Any]", said).get("audio"):
                raise UnreadableError("The reply holds audio, which can't be read.")
            if cast("dict[str, Any]", said).get("function_call"):
                raise UnreadableError(
                    "The reply holds a call in the legacy `function_call` field, which can't be read."
                )


def _check_calls(calls: object) -> None:
    # inspect_ai's Chat Completions converters skip a call that is not a function
    # call, such as a custom tool's, without an error. See
    # chat_tool_calls_from_openai() and messages_from_openai() in inspect_ai's
    # src/inspect_ai/model/_openai.py.
    for call in cast("list[object]", calls or []):
        kind = (
            cast("dict[str, Any]", call).get("type")
            if isinstance(call, dict)
            else getattr(call, "type", None)
        )
        if kind != "function":
            raise UnreadableError(
                f"The body holds a tool call of a kind that can't be read: {kind!r}."
            )


def _model(request: dict[str, Any]) -> str | None:
    model = request.get("model")
    return model if isinstance(model, str) else None


# Each reply is built as the SDK builds one it receives, so that a field value
# this SDK version doesn't know yet, such as a new `finish_reason`, is kept
# rather than an error. inspect_ai's public converters run pydantic's strict
# `model_validate` when they are given a dict, which would make such a reply
# unreadable; they take the built object as it is.


def _completion(reply: dict[str, Any]) -> object:
    from openai._models import construct_type
    from openai.types.chat import ChatCompletion

    return construct_type(type_=ChatCompletion, value=reply)


def _response(reply: dict[str, Any]) -> object:
    from openai._models import construct_type
    from openai.types.responses import Response

    return construct_type(type_=Response, value=reply)


def _chat_stream(events: list[dict[str, Any]]) -> object:
    # the SDK's own accumulator, which inspect_ai's provider relies on
    from openai._models import construct_type
    from openai.lib.streaming.chat import ChatCompletionStreamState
    from openai.types.chat import ChatCompletionChunk

    if any("error" in event for event in events):
        raise UnreadableError("The reply stream ended in an error from the provider.")
    state = ChatCompletionStreamState()
    try:
        for event in events:
            state.handle_chunk(
                cast(Any, construct_type(type_=ChatCompletionChunk, value=event))
            )
        snapshot = state.current_completion_snapshot
    except Exception as ex:
        # the accumulator raises errors of many kinds on chunks out of order
        raise UnreadableError("The reply stream can't be read.") from ex
    # a choice has no finish reason until the chunk that ends it, whatever the
    # SDK's types say
    if not snapshot.choices or any(
        getattr(choice, "finish_reason", None) is None for choice in snapshot.choices
    ):
        raise UnreadableError("The reply stream is not a complete message.")
    return cast(object, snapshot)


def _responses_stream(events: list[dict[str, Any]]) -> object:
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
                return _response(cast("dict[str, Any]", response))
    raise UnreadableError("The reply stream is not a complete response.")
