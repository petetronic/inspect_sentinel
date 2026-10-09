from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import anyio
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
)
from inspect_ai.tool import ToolCall, ToolCallView
from inspect_ai.util import Store

from .._context import Context, EvalContext, ProxyContext
from .._host import Host
from .._integration import HostContext, resolve_sentinel, run_sentinel
from .._step import AfterToolCall, BeforeToolCall
from .._types import Sentinels
from ._exchange import (
    DECISION_HEADER,
    Answer,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
)
from ._host import ListRecorder, Record, SidecarHost
from ._wire import (
    UnreadableError,
    add_rejections,
    read_reply,
    read_request,
    request_marks,
)

# how many requests awaiting a reply, runs, exchanges' records, and points of
# one run where turns were rejected, a reply was passed on or the run was
# ended are kept before the oldest is dropped
_KEPT = 1024

# consecutive rejections in one run before it is ended, as inspect_ai's bridge
# has it: a model that keeps proposing rejected calls would otherwise never stop
_MAX_CONSECUTIVE_REJECTIONS = 3

# how many request ids an eval may still ask about are remembered. Only the ids
# are kept this long, so an eval can be told a request's records were dropped,
# and not that the request was never seen
_REMEMBERED = 16 * _KEPT

# the header inspect_ai puts an id for each model request in, and sends again
# on each retry of it
REQUEST_ID_HEADER = "x-irid"

_REJECTED = "This call was rejected."
_NOT_RUN = "This call was not run, because another call in the same turn was rejected."

V = TypeVar("V")

State = Literal["judging", "awaiting_reply", "decided", "not_judged"]
"""Where a request stands. `judging`: the sentinel is at work on the request or its reply. `awaiting_reply`: the request went on, and the proxy hasn't handed over a reply, which it never does for a call the provider failed. `decided`: there is an outcome. `not_judged`: judging ended without one."""

Reason = Literal["unreadable", "time_limit", "error", "provider_error"]
"""Why a request was not judged. `unreadable`: it or its reply couldn't be read. `time_limit`: the sentinel passed the handler's time limit. `error`: the sentinel raised. `provider_error`: the provider's reply was an error, so there was nothing to judge."""


@dataclass(frozen=True)
class _Seen:
    run: str
    messages: list[ChatMessage]
    context: Context
    # how many messages the client sent, before anything was added, and their
    # mark
    sent: int
    mark: str
    recorder: ListRecorder
    # whether the proxy named the run, and whether this request continues one
    # whose reply was passed on, so that it holds a turn the model wrote
    named: bool
    answered: bool


@dataclass
class _Run:
    # kept and forgotten together: a run whose state is gone has its results
    # judged again, which is what builds that state
    store: Store
    # digests of the tool results the sentinel has been shown and let go on
    judged: set[str]
    rejected: list[_Rejected]
    # rejected turns since a reply last passed
    in_a_row: int = 0
    # what an eval registered for the run, and a digest of it to tell the same
    # registration sent again from a different one
    eval: EvalContext | None = None
    registered: str | None = None
    # steps put to the sentinel, and whether it ended the run
    steps: int = 0
    ended: bool = False
    # what every later request is refused with, once a sentinel has ended a
    # run the proxy named
    refusal: Refuse | None = None
    # for a run the proxy didn't name, whose key is shared by every
    # conversation that opens alike: the marks of the requests whose replies
    # were passed on, and of the requests the run was ended at, each with what
    # a request that continues from it is refused with
    answered: OrderedDict[str, None] = field(default_factory=OrderedDict[str, None])
    ended_at: OrderedDict[str, Refuse] = field(default_factory=OrderedDict[str, Refuse])


@dataclass
class _Exchange:
    # one request a proxy handed over and its reply: what was recorded while
    # they were judged, where the judging stands, and what was decided
    records: list[Record]
    state: State = "judging"
    reason: Reason | None = None
    outcome: Literal["continue", "reject", "terminate"] | None = None
    message: str | None = None


@dataclass(frozen=True)
class RequestResult:
    """What became of a model request, for the eval that sent it."""

    state: State
    """Where the latest attempt at the request stands."""

    reason: Reason | None
    """Why it was not judged, where `state` is `not_judged`."""

    outcome: Literal["continue", "reject", "terminate"] | None
    """What the sentinel decided for the latest attempt at the request, where `state` is `decided`."""

    message: str | None
    """What the model is told, where the sentinel rejected a call or ended the run."""

    records: list[Record] = field(default_factory=list[Record])
    """What was recorded, in order, over every attempt at the request."""


@dataclass(frozen=True)
class RunResult:
    """What a run came to, for the eval that is about to score it."""

    outcome: Literal["continue", "terminate"]
    """`terminate` if a sentinel ended the run."""

    store: dict[str, Any]
    """The run's store, as the sentinel left it."""


class TimeLimitError(TimeoutError):
    """The sentinel didn't finish judging a request or a reply within the handler's time limit, and was cancelled."""


class RegistrationError(ValueError):
    """A run can't be registered as asked: it is registered already with other details, or it has been judged and was registered as new."""


@dataclass
class _Rejected:
    """The turns rejected in a row at one point of a client's conversation.

    The client's own conversation never holds them, so they are added to every later request that continues from that point. A model that stopped being told would see itself change course for no reason, and propose the rejected call again.
    """

    # how many messages the client had sent when the turns were rejected, and
    # the mark of those messages
    sent: int
    mark: str
    # each rejected turn: the model's message, and what the model is told
    # happened to each call in it, by call id
    turns: list[tuple[ChatMessageAssistant, dict[str, str]]]


class Handler:
    """Judges what a proxy hands over with a sentinel, and answers what the proxy should do.

    It knows nothing of any proxy, nor of how it is reached. The endpoint for proxies turns each message it is sent into a `Request` or a `Reply`, and turns the `Answer` back into what it answers with.
    """

    def __init__(
        self,
        sentinel: Sentinels,
        *,
        host: Host | None = None,
        reject_status: int = 400,
        time_limit: float | None = None,
        request_id_header: str = REQUEST_ID_HEADER,
    ) -> None:
        """Create a handler.

        Args:
            sentinel: One protocol, or a sequence or mapping of monitors and protocols, as a task's `sentinel=` takes.
            host: What the sentinel may do to the outside world. Defaults to a `SidecarHost` with no model roles.
            reject_status: The HTTP status a rejected reply is refused with. The default, 400, is one a client takes as final. A status a client retries by itself, such as 503, brings the client back with the same request, which is then sent on with the rejection added, so the run carries on with a client that knows nothing of sentinels.
            time_limit: The longest the sentinel may take to judge one request or one reply, in seconds, after which it is cancelled. None, the default, is no limit: the sentinel finishes however long it takes, even if the proxy has stopped waiting, so an eval can still ask what was decided.
            request_id_header: The request header that holds the id a client gives a request, which `request_result` is asked by. The default is inspect_ai's own. A client that can't learn that id puts one of its own in another header, and this names that header.
        """
        self._protocol = resolve_sentinel(sentinel)
        self._host: Host = host if host is not None else SidecarHost()
        self._reject_status = reject_status
        self._time_limit = time_limit
        self._request_id_header = request_id_header.lower()
        self._requests: OrderedDict[str, _Seen] = OrderedDict()
        self._runs: OrderedDict[str, _Run] = OrderedDict()
        self._exchanges: OrderedDict[str, _Exchange] = OrderedDict()
        # the proxy's ids for the attempts at each request, by the id the
        # client put on it
        self._attempts: OrderedDict[str, list[str]] = OrderedDict()

    async def request(self, request: Request) -> Answer:
        """Judge a request before the proxy sends it.

        Each tool result in the request that the sentinel hasn't been shown is put to it as an `AfterToolCall` step, in order, before the model is sent any of them. A result is shown once, however often the client sends it. The call has run by then, so the sentinel can let the result go on or end the run, and the first result it ends the run at refuses the request.

        If replies were rejected earlier in the conversation this request continues, the model's rejected messages and the reasons are added where they happened, so the model is still told.

        A request of a run that a sentinel has ended is refused, and the sentinel is not asked again. For a run the proxy didn't name, that is a request which continues the conversation the run was ended in.

        Args:
            request: The request, as the proxy handed it over.

        Raises:
            UnreadableError: If the request can't be read, so neither it nor its reply could be judged.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        # kept before anything is read, so an eval is told of a request that
        # couldn't be
        recorder = ListRecorder()
        exchange = _Exchange(recorder.records)
        _keep(self._exchanges, request.id, exchange)
        asked = request.headers.get(self._request_id_header)
        if asked is not None:
            attempts = self._attempts.get(asked, [])
            _keep(self._attempts, asked, [*attempts, request.id], _REMEMBERED)
        try:
            with anyio.move_on_after(self._time_limit):
                answer = await self._request(request, exchange, recorder)
                exchange.state = (
                    "decided" if exchange.outcome is not None else "awaiting_reply"
                )
                return answer
            raise self._out_of_time("request", request.id)
        except Exception as ex:
            exchange.state, exchange.reason = "not_judged", _why(ex)
            raise

    async def _request(
        self, request: Request, exchange: _Exchange, recorder: ListRecorder
    ) -> Answer:
        messages = await read_request(request.provider, request.body)
        run = request.run if request.run is not None else _run_key(messages)
        marks = request_marks(request.provider, request.body)
        state = self._state(run)
        refusal = state.refusal
        if refusal is None:
            refusal = next(
                (state.ended_at[mark] for mark in marks if mark in state.ended_at),
                None,
            )
        if refusal is not None:
            # the run has ended, and the sentinel is not asked again
            exchange.outcome, exchange.message = "terminate", refusal.message
            return refusal
        # a proxy's requests are not an eval's samples, so the sentinel is
        # briefed with a task and a sample only where an eval registered them
        context = Context(
            path="",
            host=self._host,
            eval=state.eval,
            proxy=ProxyContext(
                provider=request.provider,
                model=request.model,
                user=request.user,
                headers=request.headers,
            ),
        )
        answer: Answer = Pass()
        # a point this request doesn't continue from is kept: a run's key is
        # shared by conversations that open alike, and it may be another's
        told = {
            rejected.sent: rejected.turns
            for rejected in state.rejected
            if rejected.sent < len(marks) and marks[rejected.sent] == rejected.mark
        }
        if told:
            body = await add_rejections(request.provider, request.body, told)
            messages = await read_request(request.provider, body)
            answer = Replace(body)
        seen = _Seen(
            run,
            messages,
            context,
            len(marks) - 1,
            marks[-1],
            recorder,
            named=request.run is not None,
            answered=any(mark in state.answered for mark in marks[:-1]),
        )
        ended = await self._results(
            seen,
            # a result this sidecar wrote for a rejected call is not a tool's
            {id for turns in told.values() for _, results in turns for id in results},
        )
        if ended is not None:
            exchange.outcome, exchange.message = "terminate", ended.message
            return ended
        _keep(self._requests, request.id, seen)
        return answer

    async def _results(self, seen: _Seen, ours: set[str]) -> Refuse | None:
        run, messages = seen.run, seen.messages
        state = self._state(run)
        host_context = HostContext(
            context=seen.context, recorder=seen.recorder, store=state.store
        )
        made: dict[str, tuple[int, ChatMessageAssistant, ToolCall]] = {}
        for index, entry in enumerate(messages):
            if isinstance(entry, ChatMessageAssistant):
                for call in entry.tool_calls or []:
                    made[call.id] = (index, entry, call)
            if not isinstance(entry, ChatMessageTool) or entry.tool_call_id in ours:
                continue
            found = made.get(entry.tool_call_id or "")
            if found is None:
                raise UnreadableError(
                    f"The request holds a result for a call it doesn't hold: {entry.tool_call_id!r}."
                )
            at, assistant, call = found
            digest = _result_digest(call, entry)
            if digest in state.judged:
                continue
            step = AfterToolCall(
                conversation=run,
                message=assistant.text,
                call=call,
                result=entry,
                # left empty: only what the model is sent crosses the wire, and a
                # monitor must not take that for everything the tool returned
                output="",
                view=ToolCallView(),
                input=messages[:at],
                # as it stood when the call ran: up to the model's message
                history=messages[: at + 1],
            )
            state.steps += 1
            decision = await run_sentinel(self._protocol, host_context, step)
            if decision is not None and decision.action == "terminate":
                # not marked as judged, so a client that sends it again is
                # refused again
                return self._end(seen, f"the result of the call to {call.function}")
            state.judged.add(digest)
        return None

    async def reply(self, reply: Reply) -> Answer:
        """Judge a reply before the proxy passes any of it on.

        Each tool call in the reply is put to the sentinel as a `BeforeToolCall` step, in order, and the first that is rejected or terminated refuses the whole reply. A rejection is kept for the later requests of its conversation. The third in a row ends the run.

        The result of a call that passes is judged when the client sends it, in `request`.

        Args:
            reply: The reply, as the proxy handed it over.

        Raises:
            UnreadableError: If the reply can't be read, or answers a request this handler was never shown.
            NotImplementedError: If the sentinel decides `modify`, which a sidecar can't carry out yet.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        # None if the request was never shown, or was dropped since
        exchange = self._exchanges.get(reply.id)
        if exchange is not None:
            exchange.state = "judging"
        try:
            with anyio.move_on_after(self._time_limit):
                answer = await self._reply(reply)
                if exchange is not None:
                    if exchange.outcome is not None:
                        exchange.state = "decided"
                    else:
                        # the one reply that is passed on without an outcome
                        exchange.state, exchange.reason = (
                            "not_judged",
                            "provider_error",
                        )
                return answer
            raise self._out_of_time("reply", reply.id)
        except Exception as ex:
            if exchange is not None:
                exchange.state, exchange.reason = "not_judged", _why(ex)
            raise

    async def _reply(self, reply: Reply) -> Answer:
        if reply.status != 200:
            return Pass()
        seen = self._requests.pop(reply.id, None)
        if seen is None:
            raise UnreadableError(
                f"The reply {reply.id!r} answers a request this sidecar was never shown."
            )
        output = await read_reply(reply.provider, reply.content_type, reply.body)
        # the exchange may have been dropped since the request, if very many
        # requests were handed over in between
        exchange = self._exchanges.get(reply.id) or _Exchange(seen.recorder.records)
        if not output.choices:
            exchange.outcome = "continue"
            return Pass()

        state = self._state(seen.run)
        host_context = HostContext(
            context=seen.context, recorder=seen.recorder, store=state.store
        )

        message = output.message
        calls = message.tool_calls or []
        for call in calls:
            step = BeforeToolCall(
                conversation=seen.run,
                message=message.text,
                call=call,
                view=ToolCallView(),
                input=seen.messages,
                history=seen.messages,
            )
            state.steps += 1
            decision = await run_sentinel(self._protocol, host_context, step)
            # an escalate nobody took proceeds, as it does at the top of an eval
            if decision is None or decision.action in ("continue", "escalate"):
                continue
            if decision.action == "terminate":
                ended = self._end(seen, f"the call to {call.function}")
                exchange.outcome, exchange.message = "terminate", ended.message
                return ended
            if decision.action != "reject":
                raise NotImplementedError(
                    f"A sidecar can't carry out a {decision.action!r} decision yet."
                )
            state.in_a_row += 1
            if state.in_a_row >= _MAX_CONSECUTIVE_REJECTIONS:
                ended = self._end(
                    seen,
                    f"the call to {call.function} after {_MAX_CONSECUTIVE_REJECTIONS} rejected turns in a row",
                )
                exchange.outcome, exchange.message = "terminate", ended.message
                return ended
            # a client that comes back after a rejection sends the same
            # request, so its turns in a row share a point
            rejected = next(
                (
                    known
                    for known in state.rejected
                    if (known.sent, known.mark) == (seen.sent, seen.mark)
                ),
                None,
            )
            if rejected is None:
                rejected = _Rejected(seen.sent, seen.mark, [])
                state.rejected.append(rejected)
                del state.rejected[:-_KEPT]
            # only `message` is for the agent; a decision's explanation is for
            # the log
            why = decision.message or _REJECTED
            rejected.turns.append(
                (message, {c.id: why if c.id == call.id else _NOT_RUN for c in calls})
            )
            exchange.outcome, exchange.message = "reject", why
            return Refuse(
                self._reject_status,
                f"A sentinel rejected the call to {call.function}: {why}",
                {DECISION_HEADER: "reject"},
            )
        state.in_a_row = 0
        if not seen.named:
            _keep(state.answered, seen.mark, None)
        exchange.outcome = "continue"
        return Pass()

    def _out_of_time(self, what: str, id: str) -> TimeLimitError:
        return TimeLimitError(
            f"The sentinel didn't finish judging the {what} {id!r} within {self._time_limit} seconds."
        )

    def _end(self, seen: _Seen, where: str) -> Refuse:
        state = self._state(seen.run)
        state.rejected.clear()
        state.in_a_row = 0
        state.ended = True
        refusal = Refuse(
            400,
            f"A sentinel ended this run at {where}.",
            {DECISION_HEADER: "terminate"},
        )
        if seen.named:
            state.refusal = refusal
        elif seen.answered:
            _keep(state.ended_at, seen.mark, refusal)
        # a run the proxy didn't name, ended before any reply was passed on,
        # is not remembered as ended. Its conversation is still only how it
        # opens, which is all another conversation with the same key has too,
        # and nothing tells the two apart.
        return refusal

    def _state(self, run: str) -> _Run:
        state = self._runs.get(run)
        if state is None:
            state = _Run(Store(), set(), [])
        _keep(self._runs, run, state)
        return state

    def records(self, id: str) -> list[Record] | None:
        """What was recorded while a request and its reply were judged.

        Args:
            id: The request's id.

        Returns:
            The records in order, the request's before the reply's, or None if the request was never seen or its records are no longer kept.
        """
        exchange = self._exchanges.get(id)
        return exchange.records if exchange is not None else None

    def register(
        self, run: str, eval: EvalContext, *, resume: bool = False
    ) -> Literal["new", "resumed"]:
        """Take an eval's account of the sample a run is, so the sentinel is given it as `context.eval`.

        The same registration sent again changes nothing, so an eval that got no answer can send it again.

        Args:
            run: What tells the run from every other: the name its model requests carry.
            eval: The task and the sample. It is the eval's claim, and is not checked.
            resume: Whether the eval is carrying on a run it began earlier.

        Returns:
            `resumed` if the eval is carrying on a run this handler still holds state for, otherwise `new`.

        Raises:
            RegistrationError: If the run is registered already with other details, or has had steps judged and is registered as new.
        """
        state = self._state(run)
        digest = _registration_digest(eval)
        if state.registered is None:
            if state.steps and not resume:
                raise RegistrationError(
                    f"Run {run!r} has already been judged and can't be registered as new."
                )
            state.eval, state.registered = eval, digest
        elif state.registered != digest:
            raise RegistrationError(
                f"Run {run!r} is registered already, with other details."
            )
        return "resumed" if resume and state.steps else "new"

    def request_result(self, id: str) -> RequestResult | Literal["dropped", "unknown"]:
        """What became of a model request, by the id its client put on it.

        A client sends a request again under the same id when it retries, so one id may cover several attempts.

        Args:
            id: The id the client put on the request, in the handler's `request_id_header`. The proxy has to pass that header on for a request to be known by it.

        Returns:
            The latest attempt's outcome with every attempt's records, or `dropped` if the request was seen and its records are no longer kept, or `unknown` if no request with the id was seen.
        """
        attempts = self._attempts.get(id)
        if attempts is None:
            return "unknown"
        kept = [
            exchange
            for exchange in (self._exchanges.get(attempt) for attempt in attempts)
            if exchange is not None
        ]
        if not kept:
            return "dropped"
        return RequestResult(
            kept[-1].state,
            kept[-1].reason,
            kept[-1].outcome,
            kept[-1].message,
            [record for exchange in kept for record in exchange.records],
        )

    def run_result(self, run: str) -> RunResult | None:
        """What a run came to: whether a sentinel ended it, and its store.

        Args:
            run: What tells the run from every other.

        Returns:
            The result, or None if this handler holds nothing for the run.
        """
        state = self._runs.get(run)
        if state is None:
            return None
        return RunResult(
            "terminate" if state.ended else "continue", dict(state.store.items())
        )


def _run_key(messages: list[ChatMessage]) -> str:
    # how the conversation opens: the fallback for a request whose run the
    # proxy doesn't name. Conversations that open alike share a key.
    opening = [m for m in messages if m.role == "system"][:1] + [
        m for m in messages if m.role == "user"
    ][:1]
    digest = hashlib.sha256("\x00".join(m.text for m in opening).encode())
    return digest.hexdigest()[:16]


def _result_digest(call: ToolCall, result: ChatMessageTool) -> str:
    # the call as well as the result: the client wrote both into the request,
    # so an id seen before doesn't vouch for what now stands beside it
    digest = hashlib.sha256(
        json.dumps([call.id, call.function, call.arguments], sort_keys=True).encode()
    )
    digest.update(b"\x00")
    # a message is given a new id each time it is read
    digest.update(result.model_dump_json(exclude={"id"}).encode())
    return digest.hexdigest()


def _why(ex: Exception) -> Reason:
    if isinstance(ex, TimeLimitError):
        return "time_limit"
    if isinstance(ex, UnreadableError):
        return "unreadable"
    return "error"


def _registration_digest(eval: EvalContext) -> str:
    # a message is given a new id each time it is read, so two readings of one
    # registration are compared without them
    sample_input: object = (
        eval.sample_input
        if isinstance(eval.sample_input, str)
        else [m.model_dump(mode="json", exclude={"id"}) for m in eval.sample_input]
    )
    digest = hashlib.sha256(
        json.dumps(
            [
                eval.task,
                eval.task_description,
                eval.sample_id,
                eval.epoch,
                eval.sample_description,
                sample_input,
                eval.metadata,
            ],
            sort_keys=True,
            default=str,
        ).encode()
    )
    return digest.hexdigest()


def _keep(
    kept: OrderedDict[str, V], key: str, value: V, limit: int | None = None
) -> None:
    kept[key] = value
    kept.move_to_end(key)
    while len(kept) > (_KEPT if limit is None else limit):
        kept.popitem(last=False)
