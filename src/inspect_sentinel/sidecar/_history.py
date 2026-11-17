"""What a conversation has sent the model, as the core's `step.history` holds it."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any, cast

from inspect_ai.model import ChatMessage

from ._state import Conversation, Sent

# Not sent to the model: the id, source and metadata, and the model an
# assistant message names, which a reader fills from the request it reads, so
# that it changes when a conversation goes on with another model.
_NOT_SENT = {"id": None, "source": None, "metadata": None, "model": None}


RECENT_REQUESTS = 16
"""How many of a conversation's latest requests a request is matched against, to find the one it continues."""


def record_sent(
    conversation: Conversation, input: Sequence[ChatMessage], body: bytes
) -> None:
    """Add what one request of the conversation sent the model to what it has sent.

    A message counts once, as in Inspect's own process, which tells a message it sent before by what it says and where it is, not by its id. A request carries no message ids in any case: each read gives a message a new one. Where Inspect aligns each input against the one before it, a request here is matched by what each message says, in order, against the earlier request it continues. Requests that open alike share a conversation, such as an agent's, a monitor's call that opens as the agent's does, and another agent given the same task, so the request just before may not be that one. Of the conversation's latest requests, it is the one the request begins as, for the most messages, as a request continues the one before it by adding to its end; then the one that shares most with it, as a compaction keeps the opening and the latest turns; then the latest. A message matched there was sent before; any other is new, even one that says what an earlier message said, as a user may say the same thing twice. A compaction that keeps the latest turns behind a summary leaves them matched, and only the summary is new.

    Args:
        conversation: The conversation.
        input: The messages the request sends the model, as read from `body`.
        body: The request body they were read from, which tells an id the request carries from one a reader made up.
    """
    keys = _keys(input, body)
    matched = _matched(conversation, keys)
    for position, message in enumerate(input):
        if position not in matched:
            conversation.sent.append(message)
    conversation.recent.append(Sent(keys, [message.id for message in input]))
    del conversation.recent[:-RECENT_REQUESTS]


def with_sent_ids(
    conversation: Conversation, input: Sequence[ChatMessage], body: bytes
) -> list[ChatMessage]:
    """The request's messages, each that the conversation sent before under the id it had then.

    In Inspect's own process a message keeps its id from one model call to the next, and a monitor can find a message it was shown before by its id, as `NewSinceLastReport` does. A request carries no message ids, so each read gives a message a new one. A message matched to one an earlier request sent, as `record_sent` matches it, is given that message's id, and any other keeps the new one.

    Args:
        conversation: The conversation.
        input: The messages the request sends the model, as read from `body`.
        body: The request body they were read from.
    """
    matched = _matched(conversation, _keys(input, body))
    return [
        message.model_copy(update={"id": matched[position]})
        if matched.get(position) is not None
        else message
        for position, message in enumerate(input)
    ]


def _keys(input: Sequence[ChatMessage], body: bytes) -> list[str]:
    carried = _strings(body)
    return [_digest(_key(message, carried)) for message in input]


def _matched(conversation: Conversation, keys: Sequence[str]) -> dict[int, str | None]:
    # The positions of the request that the earlier request it continues
    # sent, each with the id the message had there.
    begins = [_begun(earlier.keys, keys) for earlier in conversation.recent]
    longest = max(begins, default=0)
    matched: dict[int, str | None] = {}
    for earlier, begun in zip(conversation.recent, begins, strict=True):
        if begun != longest:
            continue
        shared = _shared(earlier.keys, keys, begun)
        # the latest of those alike, as they come latest last
        if len(shared) >= len(matched):
            matched = {now: earlier.ids[was] for now, was in shared.items()}
    return matched


EXACT_PAIRS = 40_000
"""How many pairs of messages, one of the earlier request and one of this, are compared to match what follows the beginning two requests share exactly. Past it, they are matched in order instead, so that a request of many messages alike costs time in proportion to its length, at the cost of matching a repeated message less well (`_in_order`)."""


def _begun(earlier: Sequence[str], keys: Sequence[str]) -> int:
    # for how many messages the request begins as the earlier one did
    return next(
        (
            position
            for position, (was, now) in enumerate(zip(earlier, keys, strict=False))
            if was != now
        ),
        min(len(earlier), len(keys)),
    )


def _shared(earlier: Sequence[str], keys: Sequence[str], begun: int) -> dict[int, int]:
    # The positions of the request that the earlier request also sent, each
    # with its position there: the beginning they share, and what else of the
    # rest matches, in order. A request that adds to the end of the earlier
    # one, as most do, needs no more.
    shared = {position: position for position in range(begun)}
    rest, after = earlier[begun:], keys[begun:]
    if not rest or not after:
        return shared
    if len(rest) * len(after) <= EXACT_PAIRS:
        found = SequenceMatcher(None, rest, after, autojunk=False)
        shared.update(
            (begun + block.b + offset, begun + block.a + offset)
            for block in found.get_matching_blocks()
            for offset in range(block.size)
        )
    else:
        shared.update((begun + now, begun + was) for now, was in _in_order(rest, after))
    return shared


def _in_order(earlier: Sequence[str], keys: Sequence[str]) -> list[tuple[int, int]]:
    # Each message of the request matched to the first like it in the
    # earlier request after the one last matched. For a run of messages
    # alike this matches as the exact matcher does. Where the earlier
    # request holds the same messages scattered before a run of them, a new
    # repetition can be matched with an earlier one and left out, or a
    # message counted again, where the exact matcher prefers the run.
    places: dict[str, list[int]] = {}
    for position, key in enumerate(earlier):
        places.setdefault(key, []).append(position)
    last = -1
    found: list[tuple[int, int]] = []
    for position, key in enumerate(keys):
        candidates = places.get(key)
        if candidates is None:
            continue
        next_place = bisect_right(candidates, last)
        if next_place < len(candidates):
            last = candidates[next_place]
            found.append((position, last))
    return found


def _strings(body: bytes) -> frozenset[str]:
    # every string the request holds, read once, as an id is told carried by
    # being one of them, however the body escapes it
    found: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, dict):
            for item in cast("dict[str, object]", value).values():
                collect(item)
        elif isinstance(value, list):
            for item in cast("list[object]", value):
                collect(item)

    try:
        collect(json.loads(body))
    except ValueError:
        pass
    return frozenset(found)


def _digest(key: str) -> str:
    # what the conversation keeps of each message's key, which is as long as
    # the message
    return hashlib.sha256(key.encode()).hexdigest()


def _key(message: ChatMessage, carried: frozenset[str]) -> str:
    # What the model is sent of the message, with each object's keys in one
    # order, as Inspect compares messages by value and a client may write a
    # call's arguments in another. A reader makes up an id for a provider's
    # tool item or call that came without one, a new one at each read, so an
    # id the request doesn't carry is left out where a reader puts one: a
    # call's, a content item's, and the call a tool message answers. A call's
    # arguments and all else are kept whole.
    dumped: dict[str, Any] = message.model_copy(update=_NOT_SENT).model_dump(
        mode="json"
    )
    if dumped.get("tool_call_id") is not None and not _carried(
        dumped["tool_call_id"], carried
    ):
        dumped["tool_call_id"] = None
    for field in ("tool_calls", "content"):
        items: object = dumped.get(field)
        if isinstance(items, list):
            dumped[field] = [
                _without_made_up_id(item, carried)
                for item in cast("list[object]", items)
            ]
    return json.dumps(dumped, sort_keys=True)


def _without_made_up_id(item: object, carried: frozenset[str]) -> object:
    if not isinstance(item, dict):
        return item
    entry = cast("dict[str, Any]", item)
    if "id" in entry and not _carried(entry["id"], carried):
        return {key: value for key, value in entry.items() if key != "id"}
    return entry


def _carried(id: object, carried: frozenset[str]) -> bool:
    return isinstance(id, str) and id in carried


def not_sent_before(
    input: Sequence[ChatMessage], history: Sequence[ChatMessage] | None
) -> list[tuple[int, ChatMessage]]:
    """The messages of a request that the model has not been sent before, each with its position in the request, counting from 1.

    A message was sent before if `history` holds one that says the same: what the model is sent of it, so all but its id, source, metadata and model. So a message that was rewritten since it was sent is here as its new version, as is an assistant message the client wrote itself, at the end of a request or elsewhere. A message sent before is left out wherever it now sits: one that moved, or that is sent a second time, is not here.

    Args:
        input: The messages the request sends the model.
        history: What the conversation has sent the model before this request, as `step.history` holds it. With None, where that isn't known, every message is returned.
    """
    if history is None:
        return list(enumerate(input, 1))
    sent = {_sent_key(message) for message in history}
    return [
        (position, message)
        for position, message in enumerate(input, 1)
        if _sent_key(message) not in sent
    ]


def _sent_key(message: ChatMessage) -> str:
    said = message.model_dump(mode="json", exclude=set(_NOT_SENT))
    return hashlib.sha256(
        json.dumps(said, sort_keys=True, default=str).encode()
    ).hexdigest()
