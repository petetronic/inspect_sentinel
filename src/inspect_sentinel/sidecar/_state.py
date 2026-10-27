from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

from inspect_ai.model import ChatMessage
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import Store
from pydantic import JsonValue

from .._context import EvalContext, ProxyContext
from ._exchange import Refuse
from ._host import ListRecorder, Record
from ._inspect_client import Exchange, Withheld
from ._registration import RegisteredTask
from ._wire import Amendment, Edited, Format, Offered, Turn

V = TypeVar("V")

MADE_KEPT = 64
"""How many tool calls a conversation keeps what the model was sent for. A call's result is judged without it once it is dropped."""

RUN_EXPIRY = 24 * 60 * 60
"""How long a run's state is kept once no request has named it, in seconds."""

DROPPED_IDS = 1_000_000
"""How many ids of requests whose records were dropped are kept, so that a request can be told apart from one never seen."""

NotJudged: TypeAlias = Literal[
    "unreadable", "unregistered", "time_limit", "error", "provider_error", "cut_off"
]
"""Why a request or its reply was not judged."""

Outcome: TypeAlias = Literal["proceed", "reject", "terminate"]
"""What an exchange came to."""


@dataclass
class Attempt:
    """One attempt at a request an eval names, from the request to its reply."""

    exchange: str
    """The proxy's id for the exchange, under which its records are kept."""

    state: Literal["judging", "awaiting_reply", "decided", "not_judged"] = "judging"
    """Where the attempt stands."""

    reason: NotJudged | None = None
    """Why the attempt was not judged, when it wasn't."""

    outcome: Outcome | None = None
    """What the attempt came to, once decided."""

    message: str | None = None
    """What the model is told, for a rejection."""

    withheld: Withheld | None = None
    """The model's turn, when the client was refused it because a call in it was rejected, or because a sentinel ended the run at the reply or at a call in it."""

    proposed: Withheld | None = None
    """The model's turn as it proposed it, when the client was handed it with calls taken out."""

    judged: Exchange | None = None
    """What was decided during the attempt, for an Inspect client, from which `request_result` writes the `sentinel` field the client was sent."""

    def decide(self, outcome: Outcome, message: str | None = None) -> None:
        """Mark the attempt decided."""
        self.state, self.outcome, self.message = "decided", outcome, message

    def not_judged(self, reason: NotJudged) -> None:
        """Mark the attempt as one that judging ended without a decision for."""
        self.state, self.reason = "not_judged", reason


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

    written: list[Mapping[str, Amendment]] = field(
        default_factory=list[Mapping[str, Amendment]]
    )
    """For each rejected turn, its calls and their rejections as inspect_ai writes them in the conversation's format, by call id, so that a client that put the turn back itself can be told from one that ran its calls."""

    added: bool = True
    """Whether the turns are added to the requests that continue from the point. A run no eval names that was ended at the point, but isn't remembered as ended there, may next be sent the same request as a fresh start, so they stop being added. They are kept, so that a client that put a turn back itself is still told from one that ran its calls."""


def occurrence(call: str, index: int) -> str:
    """A tool call told apart from others with its id: a client can use one id for calls in different turns. `index` is how many calls with the id come before it in the client's conversation, which only grows, so it stays the same on every request."""
    return f"{call}#{index}"


@dataclass(frozen=True)
class Made:
    """What the model was sent for the generate that made a tool call."""

    input: list[ChatMessage]
    tools: list[ToolInfo]
    tool_choice: ToolChoice
    message_id: str
    """The id the reply holding the call was given, so the call's two steps share it."""


@dataclass
class Conversation:
    """One conversation within a run."""

    rejected: list[Rejected] = field(default_factory=list[Rejected])
    """The points at which turns were rejected."""

    edited: list[Edited] = field(default_factory=list[Edited])
    """The turns the client was handed with calls taken out. Each is kept for as long as the conversation is: it says what the model is told in every later request that holds the turn."""

    noted: JsonValue | None = None
    """What inspect_ai's converters noted while reading the conversation's replies, which reading the next one needs."""

    judged: set[str] = field(default_factory=set[str])
    """Digests of the tool results in the conversation the sentinel has been shown and let go on."""

    issued: dict[str, int] = field(default_factory=dict[str, int])
    """For each call id the client was handed, how many calls with the id its conversation has held, those it held before this sidecar saw it included. A request that holds fewer was compacted, so a call's place among those with its id no longer tells which of them it is. Each is a count, kept for the conversation's life, as forgetting one would hide a compaction."""

    made: dict[str, Made] = field(default_factory=dict[str, Made])
    """For the latest tool calls the client was handed, by `occurrence()`, what the model was sent for the generate that made each. The oldest are dropped past `MADE_KEPT`, as each holds what the model was sent, which grows with the conversation."""

    sent: list[ChatMessage] = field(default_factory=list[ChatMessage])
    """Every message the conversation's requests sent the model, in the order first sent, each once: a step's `history`. Messages a compaction dropped from later requests are kept."""

    recent: list[list[str]] = field(default_factory=list[list[str]])
    """For each of the conversation's latest requests, latest last, a digest of what the model was sent of each of its messages, which a later request is matched against."""


@dataclass
class Run:
    """What the sidecar keeps for one run."""

    store: Store = field(default_factory=Store)
    """The sentinel's store. For a run an eval names, it is the latest attempt's: a new attempt at the sample starts it afresh, as Inspect starts the sample store afresh."""

    attempt: str = ""
    """For a run an eval names, the attempt at the sample that the store and the run's end are of."""

    closed: bool = False
    """Whether the eval ended the sample, once it was scored. No later attempt starts the run again."""

    conversations: dict[str, Conversation] = field(
        default_factory=dict[str, Conversation]
    )
    """The run's conversations, by key."""

    in_a_row: dict[str, int] = field(default_factory=dict[str, int])
    """Rejected turns since a reply last passed, by the attempt an eval names, or `""`. A new attempt at a sample starts its own count, as its model never saw the earlier attempt's turns."""

    refusal: Refuse | None = None
    """What every later request is refused with, once a sentinel has ended a run an eval names, or the eval has ended its sample. A sentinel's end is the attempt's, as the store is."""

    answered: set[str] = field(default_factory=set[str])
    """For a run no eval names: the marks of the requests whose replies were passed on."""

    ended_at: dict[str, Refuse] = field(default_factory=dict[str, Refuse])
    """For a run no eval names: the marks of the requests the run was ended at, each with what a request that continues from it is refused with."""

    eval: EvalContext | None = None
    """The sample the run is, as its task was registered when the run first found it. A registration made later doesn't reach it."""


@dataclass(frozen=True)
class Held:
    """A request that went on to the provider, kept until its reply is handed over."""

    run: str
    conversation: str
    named: bool
    """Whether an eval named the run."""
    sample_attempt: str
    """The attempt at the sample an eval names, or `""`."""
    answered: bool
    """Whether the request continues one whose reply was passed on, so that it holds a turn the model wrote."""
    model: str
    format: Format
    """The API the request is written in, which later requests of the conversation are too."""
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
    attempt: Attempt | None = None
    """The attempt at the request, for an eval that names its requests."""
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
        """The run's state, made new if none is kept. A run its eval ended is not made again: an ended run is handed back, whose `refusal` refuses every request of it."""
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

    async def register(
        self, eval_id: str, task: str, facts: RegisteredTask, expiry: float, limit: int
    ) -> bool:
        """Keep a task an eval registered, in place of any it registered before.

        Args:
            eval_id: The eval's id.
            task: The task's name.
            facts: The task, the samples it runs, and those whose registration is complete.
            expiry: How long to keep it once no request has named it, in seconds.
            limit: How many registrations are kept at once.

        Returns:
            Whether it is kept: False for a new registration when `limit` are kept already.
        """
        ...

    async def registration(self, eval_id: str, task: str) -> RegisteredTask | None:
        """A task an eval registered, or None if none is kept. Asking counts as the task being named, so it is kept for its expiry from now."""
        ...

    async def existing_run(self, key: str) -> Run | None:
        """The run's state, or None if none is kept."""
        ...

    async def end_run(self, key: str) -> bool:
        """Drop a run's state, and remember that it ended. A request or reply of the run still being judged finds it ended too.

        Returns:
            Whether the run was known: kept, or ended before.
        """
        ...

    async def attempt(self, request: str, exchange: str) -> Attempt:
        """Begin a new attempt at a request an eval names.

        Args:
            request: The eval's id for the request. A client that retries the request sends it again under the same id.
            exchange: The proxy's id for this attempt's exchange.
        """
        ...

    async def attempts(self, request: str) -> list[Attempt] | Literal["dropped"] | None:
        """The attempts at a request in order, `dropped` if they were kept and no longer are, or None if the request was never seen."""
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
        # what is left of a run its eval ended: only that it ended
        self._ended = _Expiring[Run](expiry)
        # a request's id outlives its attempts, so the eval can be told they
        # were dropped and not that the request was never seen
        self._dropped = _Expiring[bool](expiry, limit=DROPPED_IDS)
        self._attempts = _Expiring[list[Attempt]](
            expiry, dropped=lambda key: self._dropped.put(key, True)
        )
        # each registration has an expiry of its own, and there are few, so
        # they are swept whole
        self._tasks: dict[tuple[str, str], tuple[float, float, RegisteredTask]] = {}

    async def run(self, key: str) -> Run:
        run = self._runs.get(key)
        if run is not None:
            return run
        ended = self._ended.get(key)
        if ended is not None:
            return ended
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

    async def register(
        self, eval_id: str, task: str, facts: RegisteredTask, expiry: float, limit: int
    ) -> bool:
        self._expire_tasks()
        if (eval_id, task) not in self._tasks and len(self._tasks) >= limit:
            return False
        self._tasks[(eval_id, task)] = (time.monotonic(), expiry, facts)
        return True

    async def registration(self, eval_id: str, task: str) -> RegisteredTask | None:
        self._expire_tasks()
        kept = self._tasks.get((eval_id, task))
        if kept is None:
            return None
        _, expiry, facts = kept
        self._tasks[(eval_id, task)] = (time.monotonic(), expiry, facts)
        return facts

    async def existing_run(self, key: str) -> Run | None:
        return self._runs.get(key)

    async def end_run(self, key: str) -> bool:
        run = self._runs.pop(key)
        if run is None:
            # a run never seen is left as it is, rather than refused from then on
            return self._ended.get(key) is not None
        run.closed = True
        if run.refusal is None:
            # held by any request or reply of the run still being judged
            run.refusal = Refuse(400, "This run has ended.")
        if self._ended.get(key) is None:
            self._ended.put(key, Run(refusal=run.refusal, closed=True))
        return True

    async def attempt(self, request: str, exchange: str) -> Attempt:
        attempts = self._attempts.get(request)
        if attempts is None:
            attempts = list[Attempt]()
            self._attempts.put(request, attempts)
        attempt = Attempt(exchange)
        attempts.append(attempt)
        return attempt

    async def attempts(self, request: str) -> list[Attempt] | Literal["dropped"] | None:
        attempts = self._attempts.get(request)
        if attempts is not None:
            return attempts
        return "dropped" if self._dropped.get(request) is not None else None

    def _expire_tasks(self) -> None:
        now = time.monotonic()
        for key, (used, expiry, _) in list(self._tasks.items()):
            if now - used >= expiry:
                del self._tasks[key]


class _Expiring(Generic[V]):
    # oldest use first, so what has expired is always at the front
    def __init__(
        self,
        expiry: float,
        *,
        limit: int | None = None,
        dropped: Callable[[str], None] | None = None,
    ) -> None:
        self._expiry = expiry
        self._limit = limit
        self._dropped = dropped
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
        if self._limit is not None and len(self._kept) > self._limit:
            self._kept.popitem(last=False)

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
            if self._dropped is not None:
                self._dropped(key)
