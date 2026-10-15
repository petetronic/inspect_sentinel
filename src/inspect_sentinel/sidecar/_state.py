from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from inspect_ai.model import ChatMessage
from inspect_ai.tool import ToolInfo
from inspect_ai.util import Store
from pydantic import JsonValue

from .._context import ProxyContext
from ._exchange import Refuse
from ._host import ListRecorder, Record
from ._inspect_client import Exchange
from ._wire import Offered, Turn

V = TypeVar("V")

MADE_KEPT = 64
"""How many tool calls a conversation keeps what the model was sent for. A call's result is judged without it once it is dropped."""

RUN_EXPIRY = 24 * 60 * 60
"""How long a run's state is kept once no request has named it, in seconds."""


@dataclass
class Rejected:
    """The turns rejected in a row at one point of a client's conversation.

    The client's own conversation never holds them, so they are added to every later request that continues from that point. A model that stopped being told would see itself change course for no reason, and propose the rejected call again.
    """

    sent: int
    """How many entries the client had sent when the turns were rejected."""

    mark: str
    """The mark of those entries."""

    turns: list[Turn]
    """Each rejected turn."""


def occurrence(call: str, index: int) -> str:
    """A tool call told apart from others with its id: a client can use one id for calls in different turns. `index` is how many calls with the id come before it in the client's conversation, which only grows, so it stays the same on every request."""
    return f"{call}#{index}"


@dataclass(frozen=True)
class Made:
    """What the model was sent for the generate that made a tool call."""

    input: list[ChatMessage]
    tools: list[ToolInfo]


@dataclass
class Conversation:
    """One conversation within a run."""

    rejected: list[Rejected] = field(default_factory=list[Rejected])
    """The points at which turns were rejected."""

    noted: JsonValue | None = None
    """What inspect_ai's converters noted while reading the conversation's replies, which reading the next one needs."""

    judged: set[str] = field(default_factory=set[str])
    """Digests of the tool results in the conversation the sentinel has been shown and let go on."""

    issued: dict[str, int] = field(default_factory=dict[str, int])
    """For each call id the client was handed, how many calls with the id its conversation has held, those it held before this sidecar saw it included. A request that holds fewer was compacted, so a call's place among those with its id no longer tells which of them it is. Each is a count, kept for the conversation's life, as forgetting one would hide a compaction."""

    made: dict[str, Made] = field(default_factory=dict[str, Made])
    """For the latest tool calls the client was handed, by `occurrence()`, what the model was sent for the generate that made each. The oldest are dropped past `MADE_KEPT`, as each holds what the model was sent, which grows with the conversation."""


@dataclass
class Run:
    """What the sidecar keeps for one run."""

    store: Store = field(default_factory=Store)
    """The sentinel's store."""

    conversations: dict[str, Conversation] = field(
        default_factory=dict[str, Conversation]
    )
    """The run's conversations, by key."""

    in_a_row: int = 0
    """Rejected turns since a reply last passed."""

    refusal: Refuse | None = None
    """What every later request is refused with, once a sentinel has ended a run an eval names."""

    answered: set[str] = field(default_factory=set[str])
    """For a run no eval names: the marks of the requests whose replies were passed on."""

    ended_at: dict[str, Refuse] = field(default_factory=dict[str, Refuse])
    """For a run no eval names: the marks of the requests the run was ended at, each with what a request that continues from it is refused with."""


@dataclass(frozen=True)
class Held:
    """A request that went on to the provider, kept until its reply is handed over."""

    run: str
    conversation: str
    named: bool
    """Whether an eval named the run."""
    answered: bool
    """Whether the request continues one whose reply was passed on, so that it holds a turn the model wrote."""
    model: str
    recorder: ListRecorder
    input: list[ChatMessage]
    """What the model was sent."""
    calls: Mapping[str, int]
    """How many tool calls with each id the client sent, which the reply's calls follow."""
    offered: Offered
    marks: list[str]
    """The request's marks, for each count of the entries the client sent, before anything was added."""
    proxy: ProxyContext
    """What the proxy said about the request."""
    exchange: Exchange = field(default_factory=Exchange)
    """What was decided so far during the exchange, for an Inspect client's `sentinel` field."""

    @property
    def sent(self) -> int:
        """How many entries the client sent, before anything was added."""
        return len(self.marks) - 1

    @property
    def mark(self) -> str:
        """The mark of those entries."""
        return self.marks[-1]


class State(Protocol):
    """Where the sidecar keeps runs, the requests awaiting a reply, and what was recorded for each request.

    A sidecar that runs as several processes needs them to share this, and a run that outlasts a restart needs it kept. Only the in-memory implementation exists, which hands back the objects it keeps, so the handler changes them in place. A shared implementation would also need a way to write them back.
    """

    async def run(self, key: str) -> Run:
        """The run's state, made new if none is kept."""
        ...

    async def hold(self, id: str, held: Held) -> None:
        """Keep a request until its reply is handed over."""
        ...

    async def release(self, id: str) -> Held | None:
        """The request a reply answers, no longer kept, or None if it isn't."""
        ...

    async def recorder(self, id: str) -> ListRecorder:
        """Where what is recorded for a request and its reply is kept, made new if none is kept."""
        ...

    async def records(self, id: str) -> list[Record] | None:
        """What was recorded for a request and its reply, or None if nothing is kept for it."""
        ...


class MemoryState:
    """State kept in this process, and lost when it stops."""

    def __init__(self, expiry: float = RUN_EXPIRY) -> None:
        """Create the state.

        Args:
            expiry: How long a run is kept once no request has named it, in seconds. A request still awaiting its reply, and what was recorded for a request, are kept as long.
        """
        self._runs = _Expiring[Run](expiry)
        self._held = _Expiring[Held](expiry)
        self._recorders = _Expiring[ListRecorder](expiry)

    async def run(self, key: str) -> Run:
        run = self._runs.get(key)
        if run is None:
            run = Run()
            self._runs.put(key, run)
        return run

    async def hold(self, id: str, held: Held) -> None:
        self._held.put(id, held)

    async def release(self, id: str) -> Held | None:
        return self._held.pop(id)

    async def recorder(self, id: str) -> ListRecorder:
        recorder = self._recorders.get(id)
        if recorder is None:
            recorder = ListRecorder()
            self._recorders.put(id, recorder)
        return recorder

    async def records(self, id: str) -> list[Record] | None:
        recorder = self._recorders.get(id)
        return recorder.records if recorder is not None else None


class _Expiring(Generic[V]):
    # oldest use first, so what has expired is always at the front
    def __init__(self, expiry: float) -> None:
        self._expiry = expiry
        self._kept: OrderedDict[str, tuple[float, V]] = OrderedDict()

    def get(self, key: str) -> V | None:
        self._expire()
        kept = self._kept.get(key)
        if kept is None:
            return None
        self.put(key, kept[1])
        return kept[1]

    def put(self, key: str, value: V) -> None:
        # swept here too: what is only ever put, such as a request whose reply
        # never comes, would otherwise be kept for good
        self._expire()
        self._kept[key] = (time.monotonic(), value)
        self._kept.move_to_end(key)

    def pop(self, key: str) -> V | None:
        self._expire()
        kept = self._kept.pop(key, None)
        return kept[1] if kept is not None else None

    def _expire(self) -> None:
        now = time.monotonic()
        while self._kept:
            key, (used, _) = next(iter(self._kept.items()))
            if now - used < self._expiry:
                return
            del self._kept[key]
