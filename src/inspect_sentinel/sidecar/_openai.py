from __future__ import annotations

import json
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
    Edits,
    Spliced,
    StreamEvent,
    UnreadableError,
    check_found,
    check_internal,
    check_internal_kept,
    json_object,
    stream_events,
    write_events,
    written_events,
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


def edit_reply(
    content_type: str, body: bytes, chat: bool | None, edits: Edits
) -> bytes:
    """A reply with the calls of `edits` taken out or given new arguments, streamed or not, in either API.

    Raises:
        UnreadableError: If the reply doesn't hold every call of `edits`, a call to change is not a function call, or an edited Responses stream doesn't hold together.
    """
    if content_type.startswith("text/event-stream"):
        events = written_events(body)
        if chat is None:
            chat = any(
                (event.data or {}).get("object") == "chat.completion.chunk"
                for event in events
            )
        if chat:
            return _edit_chunks(events, edits)
        edited = _edit_responses_stream(events, edits)
        _check_responses_stream(edited)
        return write_events(edited)
    reply = json_object(body, "reply")
    if chat is None:
        chat = reply.get("object") == "chat.completion"
    found: list[str] = []
    if chat:
        for choice in cast("list[dict[str, Any]]", reply.get("choices") or []):
            message = cast("dict[str, Any]", choice.get("message") or {})
            calls: object = message.get("tool_calls")
            if isinstance(calls, list):
                message["tool_calls"] = _edit_chat_calls(
                    cast("list[object]", calls), edits, found
                )
    else:
        output: object = reply.get("output")
        if isinstance(output, list):
            reply["output"] = _edit_output(cast("list[object]", output), edits, found)
    check_found(found, edits)
    return json.dumps(reply).encode()


def _edit_chat_calls(
    calls: list[object], edits: Edits, found: list[str]
) -> list[object]:
    kept: list[object] = []
    for call in calls:
        id = _id(call, "id")
        if id is None or id not in edits.targets:
            kept.append(call)
            continue
        found.append(id)
        if id in edits.removed:
            continue
        entry = cast("dict[str, Any]", call)
        function: object = entry.get("function")
        if entry.get("type", "function") != "function" or not isinstance(
            function, dict
        ):
            raise UnreadableError(
                f"The call {id!r} is not a function call, whose arguments this sidecar can change."
            )
        arguments = json.dumps(edits.modified[id].arguments)
        kept.append(
            {
                **entry,
                "function": {
                    **cast("dict[str, Any]", function),
                    "arguments": arguments,
                },
            }
        )
    return kept


def _id(entry: object, key: str) -> str | None:
    value: object = (
        cast("dict[str, Any]", entry).get(key) if isinstance(entry, dict) else None
    )
    return value if isinstance(value, str) else None


def _edit_chunks(events: list[StreamEvent], edits: Edits) -> bytes:
    # A call is told apart by its index, named with its id in its first
    # fragment. A later call's index is moved down for each call taken out
    # before it, and a changed call's arguments go out whole in its first
    # fragment.
    ids: dict[int, str] = {}
    for event in events:
        for call in _fragments(event):
            index = call.get("index")
            if not isinstance(index, int):
                raise UnreadableError("A call in the reply stream has no index.")
            if index not in ids:
                id = call.get("id")
                if not isinstance(id, str):
                    raise UnreadableError(
                        "A call in the reply stream starts without an id."
                    )
                ids[index] = id
    removed = {index for index, id in ids.items() if id in edits.removed}
    for index, id in ids.items():
        if id in edits.modified and _fragments_kind(events, index) != "function":
            raise UnreadableError(
                f"The call {id!r} is not a function call, whose arguments this sidecar can change."
            )
    check_found(ids.values(), edits)
    started: set[int] = set()
    for event in events:
        for choice in _choices(event):
            delta = cast("dict[str, Any]", choice.get("delta") or {})
            calls = cast("list[dict[str, Any]]", delta.get("tool_calls") or [])
            if not calls:
                continue
            kept: list[dict[str, Any]] = []
            for call in calls:
                index = cast(int, call.get("index"))
                if index in removed:
                    continue
                call = dict(call)
                moved = index - sum(1 for taken in removed if taken < index)
                call["index"] = moved
                id = ids[index]
                if id in edits.modified:
                    function = dict(cast("dict[str, Any]", call.get("function") or {}))
                    if index not in started:
                        function["arguments"] = json.dumps(edits.modified[id].arguments)
                    else:
                        function.pop("arguments", None)
                    call["function"] = function
                    if not function:
                        del call["function"]
                    if set(call) == {"index"}:
                        continue
                started.add(index)
                kept.append(call)
            if kept != calls:
                event.changed = True
            if kept:
                delta["tool_calls"] = kept
            else:
                del delta["tool_calls"]
    return write_events(events)


def _choices(event: StreamEvent) -> list[dict[str, Any]]:
    choices: object = (event.data or {}).get("choices")
    if not isinstance(choices, list):
        return []
    return [
        cast("dict[str, Any]", choice)
        for choice in cast("list[object]", choices)
        if isinstance(choice, dict)
    ]


def _fragments(event: StreamEvent) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for choice in _choices(event):
        calls: object = cast("dict[str, Any]", choice.get("delta") or {}).get(
            "tool_calls"
        )
        if isinstance(calls, list):
            fragments += [
                cast("dict[str, Any]", call)
                for call in cast("list[object]", calls)
                if isinstance(call, dict)
            ]
    return fragments


def _fragments_kind(events: list[StreamEvent], index: int) -> object:
    # a call's type, from the first of its fragments that names it
    for event in events:
        for call in _fragments(event):
            if call.get("index") == index and "type" in call:
                return call["type"]
    return "function"


def _call_id(item: object) -> str | None:
    # the id of a call in a Responses output, which inspect_ai reads as the
    # tool call's id; an output of a call has one too, and is left as it is
    if not isinstance(item, dict):
        return None
    entry = cast("dict[str, Any]", item)
    if str(entry.get("type", "")).endswith("_output"):
        return None
    id = entry.get("call_id")
    return id if isinstance(id, str) else None


def _edit_output(output: list[object], edits: Edits, found: list[str]) -> list[object]:
    kept: list[object] = []
    for item in output:
        id = _call_id(item)
        if id is None:
            kept.append(item)
            continue
        if id in edits.removed:
            found.append(id)
            continue
        if id in edits.modified:
            entry = cast("dict[str, Any]", item)
            if entry.get("type") != "function_call":
                raise UnreadableError(
                    f"The call {id!r} is not a function call, whose arguments this sidecar can change."
                )
            found.append(id)
            item = {**entry, "arguments": json.dumps(edits.modified[id].arguments)}
        kept.append(item)
    return kept


def _edit_responses_stream(
    events: list[StreamEvent], edits: Edits
) -> list[StreamEvent]:
    # An item is told apart by its output index, which the later items' events
    # are moved down by for each item taken out before them, so that each
    # event's index is the item's place in the edited output. A changed call's
    # arguments go out whole in its first delta, or in the item it was added
    # with if it has none.
    items: dict[int, str] = {}
    for event in events:
        data = event.data or {}
        index = data.get("output_index")
        if data.get("type") == "response.output_item.added" and isinstance(index, int):
            id = _call_id(data.get("item"))
            if id is not None:
                items[index] = id
    removed = {index for index, id in items.items() if id in edits.removed}
    modified = {index: id for index, id in items.items() if id in edits.modified}
    deltas = {
        index
        for event in events
        if (event.data or {}).get("type") == "response.function_call_arguments.delta"
        and isinstance(index := (event.data or {}).get("output_index"), int)
    }
    found: list[str] = list(items.values())
    written: list[StreamEvent] = []
    started: set[int] = set()
    for event in events:
        data = event.data
        if data is None:
            written.append(event)
            continue
        response = data.get("response")
        if isinstance(response, dict):
            output = cast("dict[str, Any]", response).get("output")
            if isinstance(output, list) and output:
                edited = _edit_output(cast("list[object]", output), edits, found)
                if edited != output:
                    response["output"] = edited
                    event.changed = True
        index = data.get("output_index")
        if not isinstance(index, int):
            written.append(event)
            continue
        if index in removed:
            continue
        moved = index - sum(1 for taken in removed if taken < index)
        if moved != index:
            data["output_index"] = moved
            event.changed = True
        if index in modified:
            arguments = json.dumps(edits.modified[modified[index]].arguments)
            kind = data.get("type")
            item = data.get("item")
            if kind == "response.function_call_arguments.delta":
                if index in started:
                    continue
                started.add(index)
                data["delta"] = arguments
                event.changed = True
            elif kind == "response.function_call_arguments.done":
                data["arguments"] = arguments
                event.changed = True
            elif kind in ("response.output_item.added", "response.output_item.done"):
                item = cast("dict[str, Any]", item) if isinstance(item, dict) else {}
                if item.get("type") != "function_call":
                    raise UnreadableError(
                        f"The call {modified[index]!r} is not a function call, whose arguments this sidecar can change."
                    )
                added = kind == "response.output_item.added" and index in deltas
                data["item"] = {**item, "arguments": "" if added else arguments}
                event.changed = True
        written.append(event)
    check_found(found, edits)
    return written


def _check_responses_stream(events: list[StreamEvent]) -> None:
    # What a client acts on in a Responses stream is not only the response
    # the stream ends with, which is all that is read to judge it: the SDK
    # builds each item from the events, and a client may act on an item when
    # it is done. So an edited stream must hold together: the SDK takes every
    # event, each item is at its place in the output, and each call's
    # arguments are the same in its deltas, its done events and the output.
    from openai import omit
    from openai._models import construct_type
    from openai.lib.streaming.responses._responses import ResponseStreamState
    from openai.types.responses import ResponseStreamEvent

    state: Any = ResponseStreamState(input_tools=omit, text_format=omit)
    try:
        for event in events:
            if event.data is not None:
                state.handle_event(
                    construct_type(
                        type_=cast(Any, ResponseStreamEvent), value=event.data
                    )
                )
    except Exception as ex:
        raise UnreadableError("The edited reply stream doesn't hold together.") from ex
    added: dict[int, dict[str, Any]] = {}
    streamed: dict[int, str] = {}
    said: dict[int, list[str]] = {}
    final: list[object] = []
    for event in events:
        data = event.data or {}
        index = data.get("output_index")
        kind = data.get("type")
        item = data.get("item")
        if kind in ("response.completed", "response.incomplete"):
            response = cast("dict[str, Any]", data.get("response") or {})
            final = cast("list[object]", response.get("output") or [])
        if not isinstance(index, int):
            continue
        if kind == "response.output_item.added" and isinstance(item, dict):
            added[index] = cast("dict[str, Any]", item)
            streamed[index] = str(added[index].get("arguments") or "")
        elif kind == "response.function_call_arguments.delta":
            streamed[index] = streamed.get(index, "") + str(data.get("delta", ""))
        elif kind == "response.function_call_arguments.done":
            said.setdefault(index, []).append(str(data.get("arguments")))
        elif kind == "response.output_item.done" and isinstance(item, dict):
            arguments = cast("dict[str, Any]", item).get("arguments")
            if arguments is not None:
                said.setdefault(index, []).append(str(arguments))
    for index, item in added.items():
        if index >= len(final) or _identity(final[index]) != _identity(item):
            raise UnreadableError(
                "The edited reply stream names an item at a place it doesn't have in the output."
            )
        if item.get("type") != "function_call":
            continue
        out = str(cast("dict[str, Any]", final[index]).get("arguments"))
        heard = said.get(index, []) + [out]
        if streamed.get(index) and streamed[index] != out:
            heard.append(streamed[index])
        if len(set(heard)) != 1:
            raise UnreadableError(
                "The edited reply stream gives a call different arguments in different events."
            )


def _identity(item: object) -> tuple[object, object]:
    entry = cast("dict[str, Any]", item) if isinstance(item, dict) else {}
    return entry.get("type"), entry.get("call_id") or entry.get("id")


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


def chat_turn_at(entries: list[Any], sent: int) -> Spliced | None:
    # the assistant message at `sent`, and the tool messages after it
    turn: object = entries[sent] if sent < len(entries) else None
    if not isinstance(turn, dict):
        return None
    message = cast("dict[str, Any]", turn)
    calls: object = message.get("tool_calls")
    if message.get("role") != "assistant" or not isinstance(calls, list):
        return None
    end = sent + 1
    while (
        end < len(entries)
        and isinstance(entries[end], dict)
        and cast("dict[str, Any]", entries[end]).get("role") == "tool"
    ):
        end += 1

    def rebuild(held: list[Any], told: list[Any]) -> list[Any]:
        return [{**message, "tool_calls": held}, *told]

    return Spliced(
        sent,
        end,
        list(cast("list[Any]", calls)),
        list(entries[sent + 1 : end]),
        rebuild,
    )


def chat_call_ids(entry: object) -> list[str]:
    message = cast("dict[str, Any]", entry) if isinstance(entry, dict) else {}
    calls: object = message.get("tool_calls")
    if message.get("role") != "assistant" or not isinstance(calls, list):
        return []
    return [id for call in cast("list[object]", calls) if (id := _id(call, "id"))]


def chat_call_of(element: object) -> tuple[str | None, object, object, object]:
    id = _id(element, "id")
    if id is None:
        return None, None, None, None
    call = cast("dict[str, Any]", element)
    function: object = call.get("function")
    entry = cast("dict[str, Any]", function) if isinstance(function, dict) else {}
    kind = call.get("type", "function")
    return id, kind, entry.get("name"), _parsed(entry.get("arguments"))


def chat_result_of(element: object) -> tuple[str | None, object]:
    message = cast("dict[str, Any]", element) if isinstance(element, dict) else {}
    if message.get("role") != "tool":
        return None, None
    return _id(message, "tool_call_id"), message


def chat_results_in(entry: object) -> list[object]:
    return [entry] if chat_result_of(entry)[0] is not None else []


def chat_pick(rendered: list[Any], id: str) -> tuple[Any, tuple[Any, ...]]:
    calls = cast("list[Any]", cast("dict[str, Any]", rendered[0])["tool_calls"])
    call = next(call for call in calls if _id(call, "id") == id)
    told = [message for message in rendered[1:] if chat_result_of(message)[0] == id]
    return call, tuple(told)


def responses_turn_at(entries: list[Any], sent: int) -> Spliced | None:
    # the items of the assistant's turn from `sent`, and the outputs after
    # them
    end = sent
    while end < len(entries) and _in_turn(entries[end]):
        end += 1
    if end == sent:
        return None
    last = end
    while last < len(entries) and _is_output(entries[last]):
        last += 1

    def rebuild(held: list[Any], told: list[Any]) -> list[Any]:
        return [*held, *told]

    return Spliced(
        sent, last, list(entries[sent:end]), list(entries[end:last]), rebuild
    )


def _in_turn(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    entry = cast("dict[str, Any]", item)
    return is_assistant_item(entry) or _call_id(entry) is not None


def _is_output(item: object) -> bool:
    return isinstance(item, dict) and str(
        cast("dict[str, Any]", item).get("type", "")
    ).endswith("_output")


def responses_call_ids(entry: object) -> list[str]:
    id = _call_id(entry)
    return [id] if id is not None else []


def responses_call_of(element: object) -> tuple[str | None, object, object, object]:
    id = _call_id(element)
    if id is None:
        return None, None, None, None
    item = cast("dict[str, Any]", element)
    kind = item.get("type")
    if kind == "function_call":
        return id, kind, item.get("name"), _parsed(item.get("arguments"))
    return id, kind, item.get("name"), item.get("input", item.get("action"))


def responses_result_of(element: object) -> tuple[str | None, object]:
    if not _is_output(element):
        return None, None
    return _id(element, "call_id"), element


def responses_results_in(entry: object) -> list[object]:
    return [entry] if responses_result_of(entry)[0] is not None else []


def responses_pick(rendered: list[Any], id: str) -> tuple[Any, tuple[Any, ...]]:
    call = next(item for item in rendered if _call_id(item) == id)
    told = [item for item in rendered if responses_result_of(item)[0] == id]
    # and the message that says why, for a computer call
    told += [
        item
        for item in rendered
        if isinstance(item, dict)
        and cast("dict[str, Any]", item).get("role") == "user"
        and _text(cast("dict[str, Any]", item).get("content")).startswith(f"{id}: ")
    ]
    return call, tuple(told)


def _parsed(arguments: object) -> object:
    # arguments as JSON says them, so that two writings of the same are equal
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _text(content: object) -> str:
    if isinstance(content, list):
        return "".join(
            str(cast("dict[str, Any]", part).get("text", ""))
            for part in cast("list[object]", content)
            if isinstance(part, dict)
        )
    return str(content or "")


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
