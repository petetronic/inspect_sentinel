from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import anyio
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
)
from inspect_ai.tool import ToolCall
from pydantic import JsonValue

from .._context import ProxyContext
from .._host import HostNative
from .._integration import HostContext, resolve_sentinel, run_sentinel, watched_stages
from .._report import Decision
from .._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from .._types import Sentinels
from ._exchange import Answer, Pass, Refuse, Replace, Reply, Request
from ._host import Record, SidecarHost, stage
from ._inspect_ai import tool_call_view
from ._inspect_client import (
    Exchange,
    Judged,
    Withheld,
    inspect_sample,
    sentinel_field,
    with_sentinel,
)
from ._state import (
    MADE_KEPT,
    Conversation,
    Held,
    Made,
    MemoryState,
    Rejected,
    Run,
    State,
    occurrence,
)
from ._wire import (
    UnreadableError,
    add_rejections,
    call_texts,
    offered,
    read_opening,
    read_reply_noted,
    read_request,
    request_marks,
)

# rejected turns in a row before the run is ended, as inspect_ai's bridge has
# it: a model that keeps proposing rejected calls would otherwise never stop
_MAX_CONSECUTIVE_REJECTIONS = 3

_REJECTED = "This call was rejected."
_NOT_RUN = "This call was not run, because another call in the same turn was rejected."


class TimeLimitError(TimeoutError):
    """The sentinel didn't finish judging a request or a reply within the handler's time limit, and was cancelled."""


class Handler:
    """Judges what a proxy hands over with a sentinel, and answers what the proxy should do.

    It knows nothing of any proxy, nor of how it is reached. The endpoint for proxies turns each message it is sent into a `Request` or a `Reply`, and turns the `Answer` back into what it answers with.

    The four steps are found in the traffic. A request carries the results of the tool calls the client ran, judged as `AfterToolCall`, and is judged as `BeforeGenerate` before it goes to the provider. A reply is judged as `AfterGenerate`, and each tool call in it as `BeforeToolCall`, before the client is handed any of it.

    An Inspect eval names the sample of each request in its `x-inspect-context` header, and the sample is the run. It is told what was decided in the `sentinel` field: on a refusal, and on every reply, which is then answered with `Replace`. Any other client is handed the provider's reply as it was.
    """

    def __init__(
        self,
        sentinel: Sentinels,
        *,
        host: HostNative | None = None,
        state: State | None = None,
        reject_status: int = 400,
        time_limit: float | None = None,
        eval_results: bool = True,
    ) -> None:
        """Create a handler.

        Args:
            sentinel: One protocol, or a sequence or mapping of monitors and protocols, as a task's `sentinel=` takes.
            host: What the sentinel may do to the outside world. Defaults to a `SidecarHost` with no model roles.
            state: Where runs are kept. Defaults to this process's memory.
            reject_status: The HTTP status a rejected turn is refused with. The default, 400, is one a client takes as final. A status a client retries by itself, such as 503, brings the client back with the same request, which then goes on with the rejection added.
            time_limit: The longest the sentinel may take to judge one request or one reply, in seconds, after which it is cancelled. None, the default, is no limit.
            eval_results: Whether an Inspect client's `sentinel` field holds the results: the explanations, references and metadata of the decisions, and what each monitor and protocol recorded. Without them, it holds only what Inspect acts on. A monitor's explanation tells an agent what the monitor looks for, so a client that can't be trusted is not sent them.
        """
        self._protocol = resolve_sentinel(sentinel)
        # a stage nothing watches is never built, as in Inspect's own process
        self._watched = watched_stages(self._protocol)
        self._host: HostNative = host if host is not None else SidecarHost()
        self._state: State = state if state is not None else MemoryState()
        self._reject_status = reject_status
        self._time_limit = time_limit
        self._eval_results = eval_results
        # the tool results being judged now, by run, conversation and digest,
        # so that a request holding one waits for the request judging it
        self._judging: dict[tuple[str, str, str], anyio.Event] = {}

    async def request(self, request: Request) -> Answer:
        """Judge a request before the proxy sends it.

        Each tool result in the request that the sentinel hasn't been shown is put to it as an `AfterToolCall` step, in order. A result is shown once, however often the client sends it. The request is then put to the sentinel as a `BeforeGenerate` step. A stage nothing in the sentinel watches is left out. If either ends the run, the request is refused.

        If turns were rejected earlier in the conversation this request continues, they are added where they happened, so the model is still told.

        A request with `x-inspect-context` belongs to the sample it names, which is its run. A refusal then carries the `sentinel` field.

        A request of a run that a sentinel has ended is refused, and the sentinel is not asked again. For a run no eval names, that is a request which continues the conversation the run was ended in.

        Args:
            request: The request, as the proxy handed it over.

        Raises:
            UnreadableError: If the request, or its `x-inspect-context` header, can't be read, so neither it nor its reply could be judged.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        with anyio.move_on_after(self._time_limit):
            return await self._request(request)
        raise self._out_of_time("request", request.id)

    async def reply(self, reply: Reply) -> Answer:
        """Judge a reply before the proxy passes any of it on.

        The reply is put to the sentinel as an `AfterGenerate` step, then each tool call in it as a `BeforeToolCall` step, in order, leaving out a stage nothing in the sentinel watches. The first call that is rejected refuses the whole turn, and the rejection is kept for the later requests of its conversation. The third rejected turn in a row ends the run. A provider's error is passed on unjudged.

        For an Inspect client, a reply that goes on is answered with `Replace`, the reply with the `sentinel` field added, and a refusal carries the field. The field holds every step judged during the exchange, the request's first.

        Args:
            reply: The reply, as the proxy handed it over.

        Raises:
            UnreadableError: If the reply can't be read, or answers a request this handler was never shown or no longer keeps.
            NotImplementedError: If the sentinel decides `modify`, which a sidecar can't carry out yet.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        with anyio.move_on_after(self._time_limit):
            return await self._reply(reply)
        raise self._out_of_time("reply", reply.id)

    async def records(self, id: str) -> list[Record] | None:
        """What was recorded while a request and its reply were judged.

        Args:
            id: The request's id, as the proxy gave it.

        Returns:
            The records in order, the request's before the reply's, or None if the request was never seen or its records are no longer kept.
        """
        return await self._state.records(id)

    async def _request(self, request: Request) -> Answer:
        sample = inspect_sample(request.headers)
        proxy = ProxyContext(
            provider=request.provider,
            model=request.model,
            user=request.user,
            headers=request.headers,
            inspect=sample,
        )
        opening = _opening(
            await read_opening(request.provider, request.body, request.format)
        )
        # an Inspect eval names the run: its sample, across retries
        named = sample is not None
        key = sample.sample_uuid if sample is not None else opening
        # a conversation is told apart within a named run by how it opens
        conversation_key = f"{key}/{opening}" if named else key
        run = await self._state.run(key)
        conversation = run.conversations.setdefault(conversation_key, Conversation())
        messages = await read_request(
            request.provider, request.body, request.format, start=conversation.noted
        )
        sends = offered(request.provider, request.body, request.format)
        marks = request_marks(request.provider, request.body, request.format)
        refusal = _ended(run, marks)
        if refusal is not None:
            # the run has ended, and the sentinel is not asked again
            return self._with_field(refusal, proxy, Exchange(), run)

        answer: Answer = Pass()
        # only what the client sent holds results to judge: what the sidecar
        # adds below is its own writing
        client = messages
        # A point this request doesn't continue from is kept for the requests
        # that do: a run's key is shared by conversations that open alike. A
        # client that was handed a rejected turn, as an Inspect eval is in the
        # sentinel field, may already have put it back itself.
        holds = {_digest(message) for message in messages}
        told = {
            rejected.sent: turns
            for rejected in conversation.rejected
            if rejected.sent < len(marks) and marks[rejected.sent] == rejected.mark
            if (turns := [t for t in rejected.turns if _digest(t[0]) not in holds])
        }
        if told:
            body = await add_rejections(
                request.provider, request.body, told, request.format
            )
            messages = await read_request(
                request.provider, body, request.format, start=conversation.noted
            )
            answer = Replace(body)

        held = Held(
            run=key,
            conversation=conversation_key,
            named=named,
            answered=any(mark in run.answered for mark in marks[:-1]),
            model=_model(request.provider, request.model),
            recorder=await self._state.recorder(request.id),
            input=messages,
            calls=_calls(client),
            offered=sends,
            marks=marks,
            proxy=proxy,
        )
        said = call_texts(request.provider, request.body, request.format)
        ended = await self._results(held, run, conversation, client, said)
        if ended is not None:
            return self._with_field(ended, proxy, held.exchange, run)
        if BeforeGenerate in self._watched:
            before = BeforeGenerate(
                model=held.model,
                conversation=conversation_key,
                input=messages,
                # the caller's own list is not known behind a proxy
                history=None,
                tools=sends.tools,
                tool_choice=sends.tool_choice,
                config=sends.config,
            )
            if _ends(await self._judge(held, run, before)):
                return self._with_field(
                    self._end(held, run, "before the model was called"),
                    proxy,
                    held.exchange,
                    run,
                )
        # another exchange of the run may have ended it while this one waited
        refusal = _ended(run, marks)
        if refusal is not None:
            return self._with_field(refusal, proxy, held.exchange, run)
        await self._state.hold(request.id, held)
        # an Inspect client is told of the request's steps with the reply
        return answer

    async def _results(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        client: list[ChatMessage],
        said: Mapping[str, Sequence[str]],
    ) -> Refuse | None:
        if AfterToolCall not in self._watched:
            return None
        made: dict[str, tuple[ChatMessageAssistant, ToolCall, str, int]] = {}
        held_calls = _calls(client)
        # the calls the client holds count too, those this sidecar never
        # handed out included
        for call_id, count in held_calls.items():
            issued = max(conversation.issued.get(call_id, 0), count)
            conversation.issued[call_id] = issued
        # how many calls with each id have come before, as a client can use
        # one id for calls in different turns, each with words of its own
        occurrences: dict[str, int] = {}
        for entry in client:
            if isinstance(entry, ChatMessageAssistant):
                for call in entry.tool_calls or []:
                    index = occurrences.get(call.id, 0)
                    occurrences[call.id] = index + 1
                    written = said.get(call.id, ())
                    words = written[index] if index < len(written) else entry.text
                    made[call.id] = (entry, call, words, index)
            if not isinstance(entry, ChatMessageTool):
                continue
            found = made.get(entry.tool_call_id or "")
            if found is None:
                raise UnreadableError(
                    f"The request holds a result for a call it doesn't hold: {entry.tool_call_id!r}."
                )
            assistant, call, words, index = found
            # A call's place among those with its id tells it apart from the
            # others, as a client can use one id for calls in different
            # turns. A conversation that lost calls with its id, to
            # compaction, shifted that place, and a result there can't be
            # told from one judged before, so it is refused rather than
            # judged as a guess.
            if held_calls.get(call.id, 0) < conversation.issued.get(call.id, 0):
                raise UnreadableError(
                    f"The request holds fewer calls with the id {call.id!r} than its conversation did, so its results can't be told apart from those judged before."
                )
            # the call's occurrence too: a client that runs two calls with one
            # id, the same arguments and the same result ran them both
            digest = _result_digest(words, call, entry, index)
            key = (held.run, held.conversation, digest)
            if key in self._judging:
                while (judging := self._judging.get(key)) is not None:
                    await judging.wait()
                # the judgment waited for may have ended the run
                refusal = _ended(run, held.marks)
                if refusal is not None:
                    return refusal
            if digest in conversation.judged:
                continue
            sent = conversation.made.get(occurrence(call.id, index))
            step = AfterToolCall(
                conversation=held.conversation,
                message=assistant.text,
                call=call,
                result=entry,
                # left empty: only what the model is sent crosses the wire, and
                # a monitor must not take that for everything the tool returned
                output="",
                view=tool_call_view(call),
                input=sent.input if sent is not None else None,
                history=None,
                tools=sent.tools if sent is not None else None,
            )
            judging = self._judging[key] = anyio.Event()
            try:
                if _ends(await self._judge(held, run, step)):
                    # not marked as judged, so a client that sends it again
                    # is refused again
                    return self._end(
                        held, run, f"at the result of the call to {call.function}"
                    )
                conversation.judged.add(digest)
            finally:
                del self._judging[key]
                judging.set()
        return None

    async def _reply(self, reply: Reply) -> Answer:
        held = await self._state.release(reply.id)
        answer = await self._judge_reply(reply, held)
        if held is None or reply.status != 200:
            return answer
        if held.proxy.inspect is None:
            return answer
        run = await self._state.run(held.run)
        if isinstance(answer, Refuse):
            return self._with_field(answer, held.proxy, held.exchange, run)
        body = answer.body if isinstance(answer, Replace) else reply.body
        field = self._field(held.exchange, "proceed")
        return Replace(with_sentinel(reply.content_type, body, field))

    async def _judge_reply(self, reply: Reply, held: Held | None) -> Answer:
        if reply.status != 200:
            # nothing came from the model to judge
            return Pass()
        if held is None:
            raise UnreadableError(
                f"The reply {reply.id!r} answers a request this sidecar was never shown, or no longer keeps."
            )
        run = await self._state.run(held.run)
        conversation = run.conversations.setdefault(held.conversation, Conversation())
        output, noted = await read_reply_noted(
            reply.provider,
            reply.content_type,
            reply.body,
            reply.format,
            start=conversation.noted,
        )
        # kept whatever is decided: the provider continues from what it sent
        conversation.noted = noted
        refusal = _ended(run, held.marks)
        if refusal is not None:
            # a sentinel ended the run while the model was answering
            return refusal

        if AfterGenerate in self._watched:
            after = AfterGenerate(
                model=held.model,
                conversation=held.conversation,
                input=held.input,
                history=None,
                tools=held.offered.tools,
                tool_choice=held.offered.tool_choice,
                config=held.offered.config,
                output=output,
            )
            if _ends(await self._judge(held, run, after)):
                return self._end(held, run, "at the model's reply")

        if not output.choices:
            return _ended(run, held.marks) or self._passed(
                held, run, conversation, None
            )
        message = output.message
        # the reply is still read, and kept as passed, when no call is judged
        calls = (message.tool_calls or []) if BeforeToolCall in self._watched else []
        for call in calls:
            step = BeforeToolCall(
                conversation=held.conversation,
                message=message.text,
                call=call,
                view=tool_call_view(call),
                input=held.input,
                history=None,
                tools=held.offered.tools,
            )
            decision = await self._judge(held, run, step)
            if decision is None or decision.action == "proceed":
                continue
            if decision.action == "terminate":
                return self._end(held, run, f"at the call to {call.function}")
            if decision.action != "reject":
                raise NotImplementedError(
                    f"A sidecar can't carry out a {decision.action!r} decision yet."
                )
            # the turn is refused, so the client is never handed the call
            held.exchange.judged[-1].original = call
            # another exchange of the run may have ended it while this one waited
            return _ended(run, held.marks) or self._reject(
                held, run, conversation, message, noted, call, decision
            )
        return _ended(run, held.marks) or self._passed(held, run, conversation, message)

    def _passed(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        message: ChatMessageAssistant | None,
    ) -> Pass:
        run.in_a_row = 0
        if not held.named:
            run.answered.add(held.mark)
        if message is not None:
            seen = dict(held.calls)
            for call in message.tool_calls or []:
                index = seen.get(call.id, 0)
                seen[call.id] = index + 1
                made = Made(held.input, held.offered.tools)
                conversation.made[occurrence(call.id, index)] = made
                # its place, not one more each time: a retry or another
                # reply to the same request hands out a call in the same
                # place, of which the client keeps one
                issued = max(conversation.issued.get(call.id, 0), index + 1)
                conversation.issued[call.id] = issued
            while len(conversation.made) > MADE_KEPT:
                del conversation.made[next(iter(conversation.made))]
        return Pass()

    def _reject(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        message: ChatMessageAssistant,
        noted: JsonValue | None,
        call: ToolCall,
        decision: Decision,
    ) -> Refuse:
        # only `message` is for the agent; a decision's explanation is for the log
        why = decision.message or _REJECTED
        told = {
            other.id: why if other.id == call.id else _NOT_RUN
            for other in message.tool_calls or []
        }
        held.exchange.withheld = Withheld(message, told)
        run.in_a_row += 1
        if run.in_a_row >= _MAX_CONSECUTIVE_REJECTIONS:
            return self._end(
                held,
                run,
                f"at the call to {call.function} after {_MAX_CONSECUTIVE_REJECTIONS} rejected turns in a row",
            )
        # a client that comes back after a rejection sends the same request, so
        # its rejected turns in a row share a point
        rejected = next(
            (
                known
                for known in conversation.rejected
                if (known.sent, known.mark) == (held.sent, held.mark)
            ),
            None,
        )
        if rejected is None:
            rejected = Rejected(held.sent, held.mark, [])
            conversation.rejected.append(rejected)
        rejected.turns.append((message, told, noted))
        return Refuse(
            self._reject_status,
            f"A sentinel rejected the call to {call.function}: {why}",
        )

    async def _judge(self, held: Held, run: Run, step: Step) -> Decision | None:
        recorded = len(held.recorder.records)
        decision = await run_sentinel(
            self._protocol,
            HostContext(
                host=self._host,
                recorder=held.recorder,
                store=run.store,
                proxy=held.proxy,
            ),
            step,
        )
        call = (
            step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else None
        )
        held.exchange.judged.append(
            Judged(stage(step), call, decision, held.recorder.records[recorded:])
        )
        return decision

    def _with_field(
        self, refusal: Refuse, proxy: ProxyContext, exchange: Exchange, run: Run
    ) -> Refuse:
        # an Inspect client is told what was decided, with the refusal
        if proxy.inspect is None:
            return refusal
        # a run an eval names is remembered as ended
        outcome = "terminate" if run.refusal is not None else "reject"
        return replace(refusal, sentinel=self._field(exchange, outcome))

    def _field(
        self, exchange: Exchange, outcome: Literal["proceed", "reject", "terminate"]
    ) -> dict[str, Any]:
        return sentinel_field(exchange, outcome, trimmed=not self._eval_results)

    def _end(self, held: Held, run: Run, where: str) -> Refuse:
        for conversation in run.conversations.values():
            conversation.rejected.clear()
        run.in_a_row = 0
        refusal = Refuse(400, f"A sentinel ended this run {where}.")
        if held.named:
            run.refusal = refusal
        elif held.answered:
            run.ended_at[held.mark] = refusal
        # A run no eval names, ended before any reply in its
        # conversation was passed on, is not remembered as ended. Its
        # conversation is still only how it opens, which is all another
        # conversation with the same key has too, and nothing tells them apart.
        return refusal

    def _out_of_time(self, what: str, id: str) -> TimeLimitError:
        return TimeLimitError(
            f"The sentinel didn't finish judging the {what} {id!r} within {self._time_limit} seconds."
        )


def _ended(run: Run, marks: Sequence[str]) -> Refuse | None:
    # what a request is refused with, if its run has ended: any request of a
    # run an eval names, or one that continues the conversation it was ended
    # in
    return run.refusal or next(
        (run.ended_at[mark] for mark in marks if mark in run.ended_at), None
    )


def _ends(decision: Decision | None) -> bool:
    # an escalate nobody took has already been turned into a terminate
    return decision is not None and decision.action == "terminate"


def _model(provider: str, model: str) -> str:
    # a step names the model as `provider/name`
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


def _opening(messages: Sequence[ChatMessage]) -> str:
    # How the conversation opens: what tells a run apart when the proxy
    # doesn't name it, and a conversation apart within a run. Conversations
    # that open alike share it.
    opening = [m for m in messages if m.role == "system"][:1] + [
        m for m in messages if m.role == "user"
    ][:1]
    digest = hashlib.sha256("\x00".join(m.text for m in opening).encode())
    return digest.hexdigest()[:16]


def _digest(message: ChatMessage) -> str:
    # What makes two messages the same turn. A reply the sidecar read, and the
    # same reply sent back in a later request, are read with different ids,
    # sources and provider details, so only what the turn says is compared.
    calls = [
        [call.id, call.function, call.arguments]
        for call in (
            message.tool_calls if isinstance(message, ChatMessageAssistant) else None
        )
        or []
    ]
    result = message.tool_call_id if isinstance(message, ChatMessageTool) else None
    return hashlib.sha256(
        json.dumps(
            [message.role, message.text, calls, result], sort_keys=True, default=str
        ).encode()
    ).hexdigest()


def _calls(messages: Sequence[ChatMessage]) -> dict[str, int]:
    # how many tool calls with each id the messages hold
    calls: dict[str, int] = {}
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            for call in message.tool_calls or []:
                calls[call.id] = calls.get(call.id, 0) + 1
    return calls


def _result_digest(
    said: str, call: ToolCall, result: ChatMessageTool, index: int
) -> str:
    # everything the step is given from the request, what the model said with
    # the call as well as the call and the result: the client wrote them all
    # into the request, so an id seen before doesn't vouch for what now
    # stands beside it
    digest = hashlib.sha256(
        json.dumps(
            [
                said,
                call.id,
                index,
                call.type,
                call.function,
                call.arguments,
                call.parse_error,
            ],
            sort_keys=True,
        ).encode()
    )
    digest.update(b"\x00")
    digest.update(result.model_dump_json(exclude={"id"}).encode())
    return digest.hexdigest()
