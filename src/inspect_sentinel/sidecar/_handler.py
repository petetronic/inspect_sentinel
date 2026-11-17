from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import anyio
import anyio.abc
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall, ToolCallView
from inspect_ai.util import Store
from pydantic import JsonValue

from .._context import EvalContext, InspectSample, ProxyContext
from .._host import HostNative, HumanAnswer
from .._integration import HostContext, resolve_sentinel, run_sentinel
from .._report import Decision, Escalations
from .._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from .._types import Sentinels
from ..config import watched_stages
from ._errors import TimeLimitError, UnauthorizedError
from ._errors import not_judged as _not_judged
from ._exchange import (
    MAX_BODY_BYTES,
    Answer,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
    succeeded,
)
from ._history import not_sent_before, record_sent, with_sent_ids
from ._host import (
    AskHuman,
    ModelUsed,
    Record,
    RunHost,
    SidecarHost,
    add_usage,
    noting_models,
    stage,
)
from ._inspect_ai import VIEWED_TOOLS, tool_call_view
from ._inspect_client import (
    INSPECT_CONTEXT,
    REQUEST_ID,
    Exchange,
    Judged,
    SignedSample,
    Withheld,
    signed_sample,
)
from ._registration import (
    MAX_TASK_EXPIRY,
    MAX_TASK_REGISTRATIONS,
    TASK_EXPIRY,
    FileTask,
    RegisteredTask,
    RegistrationError,
    RegistrationsFullError,
    SampleFacts,
    Strictness,
    TaskRegistration,
    UnknownTaskError,
    UnregisteredSampleError,
)
from ._reply_stream import (
    ERROR_BODY_BYTES,
    KEEPALIVE_SECONDS,
    PENDING_BYTES,
    ReplyStart,
    ReplyStream,
)
from ._review import Question, RequestAsked, Review, Reviews, escalated
from ._sign import TIMESTAMP_WINDOW, call_values, request_values, unproven
from ._state import (
    MADE_KEPT,
    Attempt,
    Conversation,
    Dropped,
    EndedOn,
    Held,
    Made,
    MemoryState,
    NotJudged,
    Pinned,
    Registered,
    Rejected,
    Run,
    State,
    Stopped,
)
from ._wire import (
    Edited,
    Edits,
    Format,
    UnreadableError,
    amend_request,
    amendments,
    bound_body,
    call_texts,
    edit_reply,
    entry_digests,
    grows,
    held_rejections,
    offered,
    read_opening,
    read_reply_noted,
    read_request,
    request_format,
    request_marks,
    unredacted,
)

logger = logging.getLogger(__name__)

# rejected turns in a row before the run is ended, as inspect_ai's bridge has
# it: a model that keeps proposing rejected calls would otherwise never stop
MAX_RESULT_WAIT = 30.0
"""The longest `request_result` waits for a request still being judged, in seconds."""

# how often a waiting `request_result` looks again for a new attempt
_RESULT_POLL = 0.05

_MAX_CONSECUTIVE_REJECTIONS = 3

_REPLACED = Refuse(400, "A later attempt at this sample replaced this one.")

PENDING = Refuse(
    400,
    "This model call is waiting for a person's review. Nothing is decided yet. An eval gets the call's state with request_result.",
)
"""The refusal for a model call that a sentinel asked a person about, and for every try of it that arrives while the call is paused. The sidecar answers at once, because no connection waits for the person."""

# Refusals for a request or a reply whose judging was stopped with nothing decided, or
# whose model call was closed before it arrived.
_RUN_OVER = Refuse(400, "This run ended while the request waited for a person.")
_CANCELLED = Refuse(
    400, "The eval cancelled this model call while it waited for a person."
)
_CLOSED = Refuse(
    400,
    "This reply's model call was closed while the provider answered. The reply was not judged.",
)
_OTHER_REQUEST = Refuse(
    400,
    f"A different request with this {REQUEST_ID} is, or was, paused for a person's review. This request does not match it, so the sidecar did not judge it.",
)
_OVER = Refuse(
    400,
    "This model call has ended. An eval gets its result with request_result.",
)
FAILED_TRY = "x-sentinel-failed-try"
"""The header the sidecar adds when the judged-reply path returns a provider's error. Its value is the number of the failed try. The sidecar's own refusals never carry it, so an eval can tell a provider's error from a refusal by the sidecar."""

_NO_CALL_NAMED = Refuse(
    400,
    f"The request has no {INSPECT_CONTEXT} header, so the sidecar cannot tell which model call's judged reply to return.",
)
_NO_JUDGED_REPLY = Refuse(
    400,
    f"This sidecar holds no reply for a model call of the eval with this {REQUEST_ID}. Either no person was asked about the call, or the sidecar no longer keeps what it held for it. An eval gets a call's state with request_result.",
)
_NOT_ENDED = Refuse(
    400,
    "This model call has not ended, so the sidecar holds no reply for the eval yet. An eval gets the call's state with request_result.",
)
_NOT_GOING_ON = Refuse(
    400,
    "This model call ended without its turn going ahead, so the sidecar holds no reply for the eval. An eval gets the call's result with request_result.",
)

# A model call a person was asked about is kept after it has ended, with its reply, for
# the state's expiry, unless its sample ends first. The sweep drops it after that.
_SWEEP_EVERY = 60.0
"""How often, in seconds, the handler sweeps for model calls a person was asked about that have been kept for the state's expiry."""

# Headers that describe how the provider's reply was transported. They are left out of
# the judged reply, because they don't apply to the copy an eval fetches.
_NOT_KEPT = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        # the sidecar's own server adds its own to every response
        "date",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "server",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        # The provider's cookies are for the proxy's client. An eval that fetches the
        # judged reply has no use for them, and LiteLLM's guardrail drops them too.
        "set-cookie",
        # the sidecar's own header, left out so that a provider can't set it
        FAILED_TRY,
    }
)

_UNDER_REVIEW = Refuse(
    503,
    "A tool's result in this request is waiting for a person's review, which another model call of the sample asked for. The sidecar did not judge this request. A resend is judged once the review has ended.",
)
"""The refusal for a request that holds a tool's result which a person is reviewing for another model call of the same sample. The status is 503 because a client resends a request by itself after a 503."""

_ANOTHER_EVALS = Refuse(
    400,
    f"The request's {INSPECT_CONTEXT} header names a sample of another eval. The request was not judged.",
)

# what inspect_ai tells the model of a rejected call that has no message
_REJECTED = "Tool call not approved."
_NOT_RUN = "This call was not run, because another call in the same turn was rejected."
_ENDED = "This call was not run, because a sentinel ended the run."


@dataclass(frozen=True)
class RequestResult:
    """What happened to a request an eval names."""

    latest: Attempt
    """The latest attempt at it."""
    records: list[Record]
    """What was recorded, every attempt's in order."""
    reviews: list[tuple[int, Review]]
    """The record of each question a sentinel asked a person, in order. Each comes with the number of the try that asked it, counted from 1."""
    judging_usage: dict[ModelUsed, ModelUsage]
    """What the provider billed for each model the sentinel called while judging every attempt, in the order first billed."""
    failure: Failure | None = None
    """The provider's latest failed exchange, for a model call a person was asked about that is waiting for a resend because of that failure."""


@dataclass(frozen=True)
class JudgedReply:
    """A provider's reply as the proxy passes it on, with the sentinel's edits applied. The sidecar holds it for the eval to fetch, because the eval's model call was paused for a person."""

    status: int
    """The provider's HTTP status."""
    headers: Mapping[str, str]
    """The provider's headers, by lowercase name, without the headers in `_NOT_KEPT`."""
    body: bytes
    """The body: a whole JSON reply, or every event of a streamed reply, from first to last."""
    format: Format
    """The API the request was written in."""
    stream: bool
    """Whether the body is a stream of events."""


@dataclass(frozen=True)
class Failure:
    """A failed exchange with the provider, for a model call a person was asked about."""

    try_number: int
    """The number of the try that was passed on to the provider, counted from 1."""
    reply: JudgedReply | None
    """The provider's error, or None for a reply that was cut off."""
    received: float = 0.0
    """When the error arrived, on the monotonic clock. A wait the provider asked for, such as `retry-after`, is counted from this time."""


@dataclass(eq=False)
class _Operation:
    """A model call that a sentinel asked a person about. It is one operation, however many tries of it arrive.

    The operation's state is the state of `attempt`, the try that first asked a person. A later try reattaches to the operation and adds no attempt.
    """

    eval_id: str
    request_id: str
    """The eval's id for the model call, its `x-irid`."""
    sample: InspectSample
    binding: str
    """A digest of the request that began the operation. A request with the same `x-irid` and a different digest is not a try of this call."""
    attempt: Attempt
    pinned: Pinned
    """What the state keeps for the call past the usual expiry, for as long as the operation lasts."""
    tries: int = 1
    """How many tries of the call have arrived."""
    current_try: int = 1
    """The number of the try whose request or reply the operation holds. That is the try that asked a person, then each try that is passed on to the provider."""
    held: Held | None = None
    """The request as the sidecar judged it, kept while the call waits for a resend."""
    answer_on_resend: Pass | Replace | None = None
    """The answer the sidecar gives the proxy for the resend that is passed on to the provider: the `Pass` or `Replace` that judging the request produced."""
    with_provider: tuple[str, Held] | None = None
    """The request of the try that was passed on to the provider, with the proxy's id for that exchange. It is set from when the try is passed on until the proxy hands over its reply or its failure. The state keeps this request too, but drops it after the usual expiry, and a call a person was asked about can last longer than that. So the operation keeps its own copy, for a reply that arrives after the state has dropped the request."""
    reply: JudgedReply | None = None
    """The provider's reply, once the call is `decided` with the outcome `proceed`. The eval fetches it from the judged-reply path."""
    failure: Failure | None = None
    """The provider's latest failed exchange, set while the call waits for a resend because of that failure."""
    ended: float | None = None
    """When the operation ended, on the monotonic clock. It is kept for the state's expiry after that, so that the eval can still read it. Reading it doesn't extend that time."""
    last_heard: float = 0.0
    """When the sidecar last had contact about the call, on the monotonic clock. It is set when the operation is created, and when the call begins to wait for a resend or for a reply. After that it is updated by each try of the call, by each `request_result` from its eval, and by each part of its reply that a proxy hands over. A waiting call with no contact for the state's expiry is closed."""


@dataclass(eq=False)
class _Judging:
    """The judging in progress of one request or reply, for an eval's attempt at a model call."""

    attempt: Attempt
    sample: InspectSample
    request_id: str
    binding: str
    scope: anyio.CancelScope
    """Cancelled to stop the judging with nothing decided."""
    answerable: anyio.Event
    """Set once the sidecar can answer the exchange that began the judging: the judging has finished, or it has asked a person."""
    limit: anyio.CancelScope | None = None
    """The cancel scope of the time limit. The time a question is open for a person doesn't count against the limit."""
    questions_waiting: int = 0
    """How many questions the judging is waiting on now."""
    time_left: float | None = None
    """How much of the time limit was left when the judging began to wait on a person."""
    answered_pending: bool = False
    """Whether the exchange was answered "pending review". The judging then continues with no connection open."""
    answered_pending_at: float | None = None
    """When the exchange was answered "pending review", on the monotonic clock. Judging after this time isn't time the client spent waiting on its model call."""
    reattach_refusal: Refuse | None = None
    """The refusal for this exchange when another try of the same call asked a person first and holds the operation."""
    answer: Answer | None = None
    """The answer to the exchange that began the judging, set once the judging has finished or has asked a person."""
    error: Exception | None = None
    """The exception the judging raised. The exchange that began the judging raises it."""
    held: Held | None = None
    """For the judging of a reply, the request that the reply answers. If the judging is stopped because nothing is waiting for its exchange, a model call a person was asked about goes back to waiting for a resend, with this request."""


# the judging that the current task belongs to, if the request is an eval's
_JUDGING: ContextVar[_Judging | None] = ContextVar("sidecar_judging", default=None)


@dataclass(eq=False)
class _ResultJudging:
    """The judging in progress of a tool's result that a request holds. Another request that holds the same result waits for this judging and uses its outcome."""

    finished_or_asked: anyio.Event
    """Set once the judging has finished or has asked a person. A request waiting on this judging stops waiting in either case."""
    judging: _Judging | None
    """The judging of the request that holds the result, if that request is an eval's."""
    asked: bool = False
    """Whether the judging asked a person. No connection waits for a person, so from then on a request that holds the same result is answered at once."""


@dataclass(frozen=True)
class SampleResult:
    """What an eval's sample came to."""

    outcome: Literal["proceed", "terminate"]
    """`terminate` if a sentinel ended the sample."""
    store: dict[str, Any]
    """The sentinel's store, by the keys the sentinel wrote."""


class ModifyError(Exception):
    """The sentinel decided to change a tool call that the sidecar can't change in the reply, such as one that is not a function call, so the reply was not judged."""


class Handler:
    """Judges what a proxy hands over with a sentinel, and answers what the proxy should do.

    It knows nothing of any proxy, nor of how it is reached. The endpoint for proxies turns each message it is sent into a `Request` or a `Reply`, and turns the `Answer` back into what it answers with.

    The four steps are found in the traffic. A request carries the results of the tool calls the client ran, judged as `AfterToolCall`, and is judged as `BeforeGenerate` before it goes to the provider. A reply is judged as `AfterGenerate`, and each tool call in it as `BeforeToolCall`, before the client is handed any of it.

    An Inspect eval names the sample of each request in its `x-inspect-context` header, and the sample is the run. It asks what was decided of each request with `request_result`, by the request's `x-irid`. Every client, the eval included, is handed the provider's reply as it was, or as the sentinel edited it.
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
        tasks: Mapping[str, FileTask] | None = None,
        runtime_task_registration: bool = True,
        strictness: Strictness = "open",
        task_expiry: float = MAX_TASK_EXPIRY,
        max_task_registrations: int = MAX_TASK_REGISTRATIONS,
        max_registration_bytes: int = MAX_BODY_BYTES,
        release: Literal["whole", "early"] = "whole",
        keepalive: float | None = KEEPALIVE_SECONDS,
        timestamp_window: float = TIMESTAMP_WINDOW,
        review_timeout: float | None = None,
    ) -> None:
        """Create a handler.

        Args:
            sentinel: One protocol, or a sequence or mapping of monitors and protocols, as a task's `sentinel=` takes.
            host: What the sentinel may do to the outside world. Defaults to a `SidecarHost` with no model roles. A host with its own `ask_human` always answers a sentinel's question itself. Only `SidecarHost`'s `ask_human`, which asks nobody, is replaced by asking a person through an eval, and only while the handler is `serving`.
            state: Where runs are kept. Defaults to this process's memory.
            reject_status: The HTTP status a rejected turn is refused with. The default, 400, is one a client takes as final. A status a client retries by itself, such as 503, brings the client back with the same request, which then goes on with the rejection added.
            time_limit: The longest the sentinel may take to judge one request or one reply, in seconds, after which it is cancelled. None, the default, is no limit.
            eval_results: Whether the judgment `request_result` gives an Inspect eval holds the results: the explanations, references and metadata of the decisions, and what each monitor and protocol recorded. Without them, it holds only what Inspect acts on. A monitor's explanation tells an agent what the monitor looks for, so an eval that can't be trusted is not sent them. A sentinel's question to a person is not sent to such an eval either. Without results, the sidecar answers the question with caution, as it does for a run no eval names. Either way, `request_result` gives how long the judging took and what the judge models' calls cost, by model and role, for the eval's accounting.
            tasks: Tasks registered beforehand, by name, as the tasks folder's files hold them. Nothing an eval registers replaces one, and a request for a sample one doesn't hold is refused.
            runtime_task_registration: Whether an eval may register its tasks itself, at task init.
            strictness: How strict the handler is about a request it can't place. Under every strictness, a request with an `x-inspect-context` header must first pass the check of that header: it is refused, and not judged, if the handler holds no token for the eval it names, if its `timestamp` or `mac` fails the check, or if it names a run another eval started (see `request`). The strictness says what happens to a request that passes, and to one without the header. `open`, the default, judges each of those, without the task's facts where it holds none. `known_tasks` refuses a request whose `x-inspect-context` names a task the handler doesn't hold, and judges one without the header. `registered_only` refuses any request that doesn't name a sample the handler holds, in an epoch the task's registration runs, including one without the header, and refuses a sample's registration that the task's registration doesn't name.
            task_expiry: The longest a task an eval registered is kept once no request has named it, in seconds, whatever the registration asks.
            max_task_registrations: How many tasks registered by evals are kept at once, and how many evals' tokens. A registration past it is refused. It also bounds the evals whose tokens are kept, with runtime registration on or off: a token is kept until no request has named its eval for the run expiry, so an eval that has finished holds a place until then.
            max_registration_bytes: The most a task registered by an eval holds, in bytes of JSON: its registration, the facts of each sample whose registration is complete, and each sample id a sample's registration added. A sample's registration past it fails. It defaults to the largest request body, which is what a task's whole registration could hold when it came in one call.
            release: How a streamed reply is passed on. `whole`, the default, holds every reply until the sentinel has judged it. `early` passes on the text and reasoning that come before the turn's first tool call as they arrive, holds the rest of the turn, and judges it once the turn has ended, as `whole` does. The sentinel is given the whole turn either way.
            keepalive: Under early release, the longest a client is sent nothing while a streamed reply is held, in seconds, after which it is sent a keepalive, an SSE comment that every provider SDK skips. The keepalive sends the status, so a turn refused after it is refused with an error event in the stream. None sends no keepalives.
            timestamp_window: How far, in seconds, the `timestamp` of an Inspect eval's model request or call to the endpoint for evals may be from this process's clock. One further is refused, so a `mac` copied from a log is of no use once the window has passed.
            review_timeout: How long, in seconds, a question waits for a person's answer before it expires, when a sentinel asks a person during a run an Inspect eval names. The eval lists the question with `list_pending_questions` and answers it with `answer_question`. The sidecar answers an expired question with caution: `reject` if that was offered, otherwise `terminate`. The judging then continues. None, the default, means no expiry, because a person may take days. The time a question is open doesn't count against `time_limit`.

        Raises:
            ValueError: If `keepalive`, `timestamp_window` or `review_timeout` isn't a positive number of seconds.
        """
        if keepalive is not None and not keepalive > 0:
            raise ValueError(
                f"A keepalive is sent after some seconds, not {keepalive}."
            )
        if not timestamp_window > 0:
            raise ValueError(
                f"The timestamp window is some seconds, not {timestamp_window}."
            )
        self._timestamp_window = timestamp_window
        self._reviews = Reviews(review_timeout)
        self._judgings_in_progress: set[_Judging] = set()
        # The model calls that a sentinel asked a person about, by eval id and x-irid.
        # Each is kept until its sample ends, or for the state's expiry after the call
        # ended.
        self._operations: dict[tuple[str, str], _Operation] = {}
        # The same operations, keyed by the attempt that asked a person. An attempt
        # isn't hashable by value, so the key is its `id()`. The operation holds the
        # attempt until its entry here is removed, so Python can't reuse the id for
        # another attempt meanwhile.
        self._by_attempt: dict[int, _Operation] = {}
        # the model call each open question was asked for, by question id
        self._operation_by_question: dict[str, _Operation] = {}
        # Provider errors whose bodies are still arriving, by the proxy's id for the
        # exchange. Each entry holds the operation and the failure, which names the
        # failed try.
        self._errors_arriving: dict[str, tuple[_Operation, Failure]] = {}
        # the task group for judging that outlives its exchange, set while the handler
        # is serving
        self._group: anyio.abc.TaskGroup | None = None
        self._release: Literal["whole", "early"] = release
        self._keepalive = keepalive
        self._protocol = resolve_sentinel(sentinel)
        # a stage nothing watches is never built, as in Inspect's own process
        self._watched = watched_stages(self._protocol)
        self._host: HostNative = host if host is not None else SidecarHost()
        self._state: State = state if state is not None else MemoryState()
        self._reject_status = reject_status
        self._time_limit = time_limit
        self._eval_results = eval_results
        # A sentinel's question goes to a person through an eval only if the eval is
        # sent results. An eval that isn't sent results is one the deployment doesn't
        # trust to decide. A host that has its own way to ask a person keeps using it.
        self._asks_eval = (
            eval_results and type(self._host).ask_human is SidecarHost.ask_human
        )
        self._tasks = dict(tasks or {})
        self._runtime_task_registration = runtime_task_registration
        self._strictness: Strictness = strictness
        self._task_expiry = task_expiry
        self._max_task_registrations = max_task_registrations
        self._max_registration_bytes = max_registration_bytes
        # the tool results being judged now, by run, conversation and digest,
        # so that a request holding one waits for the request judging it
        self._judging: dict[tuple[str, str, str], _ResultJudging] = {}
        # The model calls a person was asked about that have a try with the provider
        # now, by the proxy's id for that try's exchange.
        self._operations_with_provider: dict[str, _Operation] = {}

    @property
    def release(self) -> Literal["whole", "early"]:
        """How a streamed reply is passed on."""
        return self._release

    @property
    def eval_results(self) -> bool:
        """Whether an Inspect client is sent the results, or only what it acts on."""
        return self._eval_results

    @property
    def keepalive(self) -> float | None:
        """Under early release, how many seconds a client may be sent nothing before it is sent a keepalive, or None for no keepalives."""
        return self._keepalive

    @property
    def timestamp_window(self) -> float:
        """How far, in seconds, an Inspect eval's `timestamp` may be from this process's clock."""
        return self._timestamp_window

    @property
    def eval_review(self) -> bool:
        """Whether a sentinel's question to a person goes to the Inspect eval that owns the run, for its approval UI. It doesn't when evals aren't sent results, when the host has its own way to ask a person, or when the handler isn't `serving`, because the judging that waits for a person then has no task group to run in."""
        return self._asks_eval and self._group is not None

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """Run the judging that outlives its exchange, for as long as the handler serves.

        A person may take days to answer a sentinel's question, so no connection waits for the answer. The sidecar answers the model call "pending review", and the judging continues here, in a task the handler owns, until it finishes. Leaving this context cancels any judging that is still waiting. A handler that isn't serving asks nobody through an eval.
        """
        async with anyio.create_task_group() as group:
            self._group = group
            group.start_soon(self._sweeping)
            try:
                yield
            finally:
                self._group = None
                group.cancel_scope.cancel()

    @property
    def reviews(self) -> Reviews:
        """The questions a sentinel asked a person, held for Inspect evals to answer."""
        return self._reviews

    async def request(self, request: Request) -> Answer:
        """Judge a request before the proxy sends it.

        Each tool result in the request that the sentinel hasn't been shown is put to it as an `AfterToolCall` step, in order. A result is shown once, however often the client sends it. The request is then put to the sentinel as a `BeforeGenerate` step. A stage nothing in the sentinel watches is left out. If either ends the run, the request is refused.

        If turns were rejected earlier in the conversation this request continues, they are added where they happened, so the model is still told.

        A request with `x-inspect-context` belongs to the sample it names, which is its run. A request whose header can't be read is refused, rather than judged as another client's would be: it says it is an eval's, and a proxy set to let through what can't be judged would let it through unjudged.

        The header's `mac` proves the values that name the run, and the request's `x-irid`, made with the token the eval registered with. A request with the header and no `x-irid` can't be checked, and is refused as one whose header can't be read. Before anything of the run or of the `x-irid` is looked up or changed, a request is refused unjudged, whatever the strictness, if the handler holds no token for the eval, as after a restart; if its `timestamp` is outside the window or its `mac` doesn't match; or if it names a run another eval started. So a client without the eval's token that changes the sample, the attempt or its number, or the `x-irid`, or names a sample of the eval, reaches no run and changes no request's history.

        A request is kept under its `x-irid`, for `request_result`, only once its header has passed the check. One whose header can't be read, one refused unchecked, and one without the header add nothing to the history of any `x-irid`, so a request sent with another request's `x-irid` doesn't change what `request_result` answers for it.

        A conversation belongs to one attempt at the sample. A new attempt starts afresh, as Inspect starts the sample store afresh for it: its conversations, its count of rejected turns in a row, the sentinel's store, and the run's end if a sentinel ended the attempt before. A sample the eval has ended stays ended. Attempts are ordered by the number the header gives each. Once a later attempt has begun, a request of an earlier one is refused unjudged, as nobody waits for it, and so is one of another attempt with the same number: `request_result` says it was not judged, with the reason `replaced_attempt`, so an eval never takes it for a sentinel's terminate. A client that retries a request within the attempt keeps the attempt's conversation.

        A request of a run that a sentinel has ended is refused, and the sentinel is not asked again. For a run no eval names, that is a request which continues the conversation the run was ended in.

        Args:
            request: The request, as the proxy handed it over.

        Raises:
            UnreadableError: If the request can't be read, so neither it nor its reply could be judged.
            UnregisteredSampleError: If the request names a sample that a task registered beforehand doesn't hold, or the handler's strictness refuses a request it can't place.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        try:
            signed = signed_sample(request.headers)
        except UnreadableError as ex:
            # refused rather than raised, but never judged, so not a terminate
            return Refuse(400, str(ex))
        sample = signed.sample if signed is not None else None

        # Proven before anything of its run or its x-irid is looked up or
        # changed, and refused rather than raised, so a proxy that lets
        # through what can't be judged doesn't let through a request that
        # can't prove its run.
        attempt = None
        binding = ""
        if signed is not None:
            refused = await self._unproven_request(signed)
            if refused is not None:
                return Refuse(400, refused)
            binding = _binding(request, signed.sample)
            operation = self._operations.get((signed.sample.eval_id, signed.request_id))
            if operation is not None:
                # This is a try of a model call a person was asked about. The sidecar
                # doesn't judge it again.
                return await self._reattach(operation, request, binding)
            attempt = await self._state.attempt(
                signed.request_id, request.id, signed.sample.eval_id
            )
            # an attempt that is never judged has records too: none
            await self._state.recorder(request.id)

        async def judge() -> Answer:
            try:
                answer = await self._within_limit(
                    self._request(request, sample, attempt, binding),
                    "request",
                    request.id,
                )
            except Exception as ex:
                if attempt is None:
                    raise
                if attempt.settled.is_set():
                    # The call was closed while its request was judged (see below). It
                    # stays closed, whatever the judging raised.
                    logger.warning(
                        "Judging the request %r failed after its model call was closed: %s",
                        request.id,
                        ex,
                    )
                    return _stopped_answer(attempt)
                attempt.not_judged(_not_judged(ex))
                raise
            if attempt is None or signed is None:
                return answer
            if attempt.settled.is_set():
                # The model call was closed while its request was judged, for example
                # because the eval cancelled it. Closing a call cancels the judging only
                # at its next wait, so judging that has nothing left to wait for runs to
                # its end. A closed call stays closed: the sidecar drops the judging's
                # answer, doesn't keep the request for a resend, and reports the call as
                # its closing left it.
                await self._state.release(request.id)
                return _stopped_answer(attempt)
            if answer is _ANOTHER_EVALS:
                # as when the check above refuses it: nothing of the request
                # is kept, so `request_result` reads it as unknown
                await self._state.forget_attempt(
                    signed.request_id, signed.sample.eval_id, attempt
                )
            elif not isinstance(answer, Refuse):
                operation = self._operation_of(attempt)
                if operation is None:
                    attempt.state = "awaiting_reply"
                else:
                    # The exchange was answered "pending review", so the proxy didn't
                    # pass this request on. The sidecar keeps the request until the eval
                    # resends it, and the resend is passed on to the provider.
                    operation.held = await self._state.release(request.id)
                    operation.answer_on_resend = answer
                    attempt.state = "awaiting_resend"
                    # the wait for a resend starts now, however long the person took
                    operation.last_heard = time.monotonic()
            elif answer is _UNDER_REVIEW:
                attempt.not_judged("result_under_review")
            elif attempt.judged is not None and attempt.judged.replaced:
                attempt.not_judged("replaced_attempt")
            else:
                # otherwise a request is refused only when its run has ended
                attempt.decide("terminate")
            return answer

        if attempt is None or signed is None:
            return await judge()
        return await self._exchange(
            judge, attempt, signed.sample, signed.request_id, binding
        )

    async def reply(self, reply: Reply, *, released: bytes = b"") -> Answer:
        """Judge a reply before the proxy passes any of it on.

        The reply is put to the sentinel as an `AfterGenerate` step, then each tool call in it as a `BeforeToolCall` step, in order, leaving out a stage nothing in the sentinel watches. Every call is judged, as in Inspect's own process, unless one ends the run. The rejected calls are taken out of the reply and the rest of the turn goes on, answered with `Replace`, as the other calls of a turn still run in Inspect's own process. Each later request of the conversation that holds the turn has the calls taken out put back, each with its rejection as its result. A turn whose every call is rejected is refused whole, as is one that can't be edited, and its rejections are added to the later requests of its conversation. The third turn refused whole in a row within an attempt at a sample ends the run; a turn in which a call ran starts the count again. A provider's error, a reply in a status outside 2xx, is passed on unjudged; a reply in any 2xx is the model's, and judged, as the provider's SDK reads it so.

        For an Inspect eval, what was decided is kept for `request_result`: every step judged during the exchange, the request's first. A turn handed over with calls taken out or changed, or refused, is kept with the calls a decision acted on redacted, as the eval is handed them.

        Args:
            reply: The reply, as the proxy handed it over.
            released: The start of the reply, as the client has already been sent it, for a reply released early. A reply edited in a way that changes it can't be sent, and is not judged.

        Raises:
            UnreadableError: If the reply can't be read, or answers a request this handler was never shown or no longer keeps.
            ModifyError: If the sentinel decides `modify` of a call the sidecar can't change, and rejects none of the turn's calls.
            TimeLimitError: If the handler has a time limit and the sentinel didn't finish within it.
        """
        held = await self._released(reply.id)
        attempt = held.attempt if held is not None else None
        if attempt is not None:
            if attempt.settled.is_set():
                # The model call was closed while the provider answered, for example
                # because the eval cancelled it. A closed call stays closed: the sidecar
                # doesn't judge the reply, keep it or pass it on.
                return _CLOSED
            attempt.state, attempt.reason = "judging", None

        async def judge() -> Answer:
            try:
                answer = await self._within_limit(
                    self._reply(reply, held, released), "reply", reply.id
                )
            except Exception as ex:
                if attempt is None:
                    raise
                if attempt.settled.is_set():
                    # The call was closed while its reply was judged (see below). It
                    # stays closed, whatever the judging raised.
                    logger.warning(
                        "Judging the reply %r failed after its model call was closed: %s",
                        reply.id,
                        ex,
                    )
                    return _CLOSED
                attempt.not_judged(_not_judged(ex))
                raise
            if attempt is None or held is None:
                return answer
            if attempt.settled.is_set():
                # The model call was closed while its reply was judged, for example
                # because the eval cancelled it. Closing a call cancels the judging only
                # at its next wait, so judging that has nothing left to wait for runs to
                # its end. A closed call stays closed: the sidecar doesn't keep the
                # reply or pass it on, and reports the call as its closing left it.
                return _CLOSED
            operation = self._operation_of(attempt)
            if not succeeded(reply.status):
                if operation is None:
                    attempt.not_judged("provider_error")
                else:
                    self._failed(operation, held, "provider_error", reply)
            elif not isinstance(answer, Refuse):
                if operation is not None:
                    # The sidecar keeps the reply whichever stage the call was paused
                    # at, because the reply it passes on now may never reach the eval.
                    operation.reply = _judged_reply(reply, held.format, answer)
                attempt.decide("proceed")
            elif held.exchange.replaced:
                attempt.not_judged("replaced_attempt")
            elif held.exchange.ended:
                attempt.decide("terminate")
            else:
                attempt.decide("reject", held.exchange.message)
            return answer

        sample = held.proxy.inspect if held is not None else None
        request_id = held.proxy.headers.get(REQUEST_ID) if held is not None else None
        if attempt is None or held is None or sample is None or request_id is None:
            return await judge()
        return await self._exchange(
            judge, attempt, sample, request_id, held.binding, held
        )

    def reply_stream(
        self,
        start: ReplyStart,
        *,
        max_held_bytes: int = MAX_BODY_BYTES,
        max_pending_bytes: int = PENDING_BYTES,
        keepalive: bool = True,
    ) -> ReplyStream:
        """Judge a reply that the proxy hands over as it arrives, rather than whole.

        The proxy runs the stream's `run`, feeds it the provider's bytes, and sends the client what iterating over it gives, all at once. The reply is judged as `reply` judges it, once it has ended.

        Args:
            start: The reply's status and headers, before any of its body.
            max_held_bytes: The most of the reply the stream holds back to judge. A reply larger than it is refused with a 502, and not judged.
            max_pending_bytes: How much of what goes out as it arrives may wait for the proxy to collect it before feeding waits.
            keepalive: Whether the proxy can send its client a keepalive. Where it can't, none is given, so that a refusal keeps its status for as long as nothing else has gone out.
        """
        return ReplyStream(
            self,
            start,
            max_held_bytes=max_held_bytes,
            max_pending_bytes=max_pending_bytes,
            keepalive=self._keepalive if keepalive else None,
        )

    async def reply_not_judged(self, id: str, reason: NotJudged) -> None:
        """Let go of a request whose reply the proxy refused without handing it over, as one larger than it holds.

        The request is no longer kept, and an eval that names it is told its reply was not judged, for the reason given, rather than that it is still awaited.

        Args:
            id: The proxy's id for the request.
            reason: Why the reply was not judged.
        """
        arriving = self._errors_arriving.pop(id, None)
        if arriving is not None:
            # The body of a provider's error didn't all arrive. The sidecar can't give
            # the eval the whole error, and the missing part may be what says whether
            # the call can be retried. So the try fails as a reply that was cut off
            # does: with no reply, which the eval handles as a failed connection.
            operation, failure = arriving
            attempt = operation.attempt
            if (
                operation.ended is None
                and not attempt.settled.is_set()
                and attempt.state == "awaiting_reply"
                and operation.current_try == failure.try_number
            ):
                operation.failure = replace(failure, reply=None)
                attempt.state, attempt.reason = "awaiting_resend", "cut_off"
            return
        held = await self._released(id)
        if held is None or held.attempt is None or held.attempt.settled.is_set():
            return
        operation = self._operation_of(held.attempt)
        if operation is not None and reason == "cut_off":
            # The exchange with the provider failed, so the call waits for a resend.
            self._failed(operation, held, "cut_off", None)
            return
        held.attempt.not_judged(reason)
        self._finish(held.attempt)

    async def error_body(self, id: str, body: bytes) -> None:
        """Store the body of a provider's error, once the proxy has received all of it from the provider.

        The handler doesn't hold a provider's error back. `reply` is shown the error's status and headers as they arrive, with `body_follows` set, and the proxy passes the body on as it comes. For a model call a person was asked about, the eval fetches the error from the sidecar. So the call stays at `awaiting_reply` until the proxy calls this method with the whole body. Only then does `request_result` report that the provider failed, so the eval is never told of an error it can't fetch yet.

        Args:
            id: The proxy's id for the request the error answers.
            body: The error's whole body. It is empty if the error had no body, or if the proxy doesn't copy it, for example because it is too large. A proxy that didn't receive the whole error calls `reply_not_judged` instead.
        """
        arriving = self._errors_arriving.pop(id, None)
        if arriving is None:
            return
        operation, failure = arriving
        attempt = operation.attempt
        if (
            failure.reply is None
            or operation.ended is not None
            or attempt.settled.is_set()
            or attempt.state != "awaiting_reply"
            or operation.current_try != failure.try_number
        ):
            # the call was closed, or moved on, while the error's body arrived
            return
        operation.failure = replace(failure, reply=replace(failure.reply, body=body))
        attempt.state, attempt.reason = "awaiting_resend", "provider_error"

    def answer_question(
        self, eval_id: str, id: str, decision: str, reason: str | None
    ) -> None:
        """Record an eval's answer to a question a sentinel asked a person, as `Reviews.answer` does, and update the model call's state at once.

        Once the answer is recorded, `request_result` reports the call as `judging` until the sentinel has continued, unless another of the call's questions is still open. So an eval that answers and then calls `request_result` never reads `awaiting_review` for a question it has already answered.

        Raises:
            UnknownQuestionError: If the eval has no open question with the id, and no receipt of an answer to it.
            AnsweredError: If the question was already answered with another choice or reason.
            ChoiceError: If `decision` wasn't offered, or is `modify`.
        """
        self._reviews.answer(eval_id, id, decision, reason)
        operation = self._operation_by_question.get(id)
        if operation is None:
            return
        attempt = operation.attempt
        # The call goes back to `judging` once none of its questions is waiting for an
        # answer: each has closed, or has an answer.
        if attempt.state == "awaiting_review" and all(
            review.resolution is not None or review.decision is not None
            for review in attempt.reviews
        ):
            attempt.state = "judging"

    async def cancel_paused_request(self, id: str, eval_id: str) -> bool:
        """Close a model call a sentinel asked a person about, because the eval has given up on it. Nothing is decided for the call.

        The call is closed at whatever state it has reached before its end. The sidecar stops its judging, closes its questions, and passes no further try of it on to the provider. If a request is already with the provider, its reply is not judged, kept or passed on. `request_result` then reports the call as not judged, with the reason `cancelled`. A closed call is never reopened.

        Args:
            id: The eval's id for the model call, its `x-irid`.
            eval_id: The eval that is cancelling. A call of another eval is left unchanged.

        Returns:
            Whether this cancellation closed the model call. It is False for a call that had already ended, a call no person was asked about, and a call the handler doesn't know. So False doesn't mean the call is still paused.
        """
        operation = self._operations.get((eval_id, id))
        if operation is None or operation.attempt.settled.is_set():
            return False
        self._stop("cancelled", lambda _, attempt: attempt is operation.attempt)
        return True

    async def _reattach(
        self, operation: _Operation, request: Request, binding: str
    ) -> Answer:
        # A request that arrives under the `x-irid` of a model call a person was asked
        # about. It is a try of that call only if it is the same request, and the
        # sidecar never judges it again. The sidecar passes it on to the provider only
        # if the call is at `awaiting_resend`, and passes on one try only. Nothing is
        # awaited between finding the call at `awaiting_resend` and marking the try as
        # passed on, so two tries that arrive together can't both reach the provider.
        if binding != operation.binding:
            return _OTHER_REQUEST
        operation.tries += 1
        operation.last_heard = time.monotonic()
        attempt = operation.attempt
        if attempt.settled.is_set():
            return _OVER
        held, answer = operation.held, operation.answer_on_resend
        if attempt.state != "awaiting_resend" or held is None or answer is None:
            return PENDING
        operation.held = None
        operation.current_try = operation.tries
        attempt.state, attempt.reason = "awaiting_reply", None
        self._operations_with_provider[request.id] = operation
        operation.with_provider = (request.id, held)
        await self._state.hold(request.id, held)
        return answer

    async def _released(self, id: str) -> Held | None:
        # Return the request that an exchange's reply answers, and stop keeping it. The
        # state drops a request once nothing has named it for the state's expiry, but a
        # model call a person was asked about lasts for as long as its eval asks about
        # it. So for the try of such a call that is with the provider, the operation's
        # own copy is used if the state has dropped the request. An earlier try's
        # exchange finds nothing.
        held = await self._state.release(id)
        operation = self._operations_with_provider.get(id)
        if operation is None or operation.with_provider is None:
            return held
        exchange, kept = operation.with_provider
        if exchange != id:
            return held
        operation.with_provider = None
        return held if held is not None else kept

    def _failed(
        self,
        operation: _Operation,
        held: Held,
        reason: Literal["provider_error", "cut_off"],
        reply: Reply | None,
    ) -> None:
        # The exchange with the provider failed. For a model call a person was asked
        # about, this is the only case in which the provider is called again. The
        # request is not judged again, and the answers a person gave for it stay.
        # `request_result` reports the failure in `failure`, whatever error the eval's
        # own try received.
        operation.held = held
        # the wait starts now, for a resend or for the rest of an error's body
        operation.last_heard = time.monotonic()
        if reply is not None and len(reply.body) > ERROR_BODY_BYTES:
            # An error body larger than `ERROR_BODY_BYTES` is not kept. This matches an
            # error whose body the proxy passed on as it arrived, which is also kept
            # without a body over that bound. Part of a body might not parse as the
            # error it is, so the sidecar keeps none of it.
            reply = replace(reply, body=b"")
        failure = Failure(
            operation.current_try,
            _judged_reply(reply, held.format, Pass()) if reply is not None else None,
            time.monotonic(),
        )
        if reply is not None and reply.body_follows:
            # The proxy is still passing the error's body on to the client. The call
            # stays at `awaiting_reply` until the proxy reports the whole body with
            # `error_body`, so `request_result` never reports an error that the eval
            # can't fetch yet.
            self._errors_arriving[reply.id] = (operation, failure)
            operation.attempt.state, operation.attempt.reason = "awaiting_reply", None
            return
        operation.failure = failure
        operation.attempt.state, operation.attempt.reason = "awaiting_resend", reason

    def _operation_of(self, attempt: Attempt) -> _Operation | None:
        # the operation the attempt belongs to, if a person was asked about its model
        # call
        return self._by_attempt.get(id(attempt))

    def _finish(self, attempt: Attempt) -> None:
        # A model call a person was asked about has ended, for whatever reason. The
        # sidecar drops what it kept in order to continue the call. It keeps the call's
        # record and its reply, for the eval to read.
        operation = self._operation_of(attempt)
        if (
            operation is None
            or operation.ended is not None
            or not attempt.settled.is_set()
        ):
            return
        operation.ended = time.monotonic()
        operation.held = operation.answer_on_resend = operation.with_provider = None
        for exchange, (with_error, _) in list(self._errors_arriving.items()):
            if with_error is operation:
                del self._errors_arriving[exchange]
        for exchange, with_provider in list(self._operations_with_provider.items()):
            if with_provider is operation:
                del self._operations_with_provider[exchange]
        if self._group is not None:
            self._group.start_soon(self._state.unpin, operation.pinned)

    def _forget(self, key: tuple[str, str]) -> None:
        operation = self._operations.pop(key, None)
        if operation is not None:
            self._by_attempt.pop(id(operation.attempt), None)

    async def _sweeping(self) -> None:
        # runs for as long as the handler serves
        while True:
            await anyio.sleep(min(_SWEEP_EVERY, self._state.expiry))
            self.sweep()

    def sweep(self) -> tuple[int, int]:
        """Close or forget the model calls a person was asked about that have been kept for the state's expiry (`--run-expiry`). A handler that is serving calls this itself, every minute.

        A call that is waiting for a resend or for the provider's reply, and that nothing has named for the expiry, is closed. `request_result` then reports it as not judged, with the reason `expired`, and it is never reopened. The expiry measures only how long the sidecar has had no contact about the call. It says nothing of whether the client has gone: a client that is still there asks about the call or sends a try of it, and either keeps the call. A call with an open question is never closed here, however long the person takes: a question closes when its own time is up (`--review-timeout`), or never.

        A call that ended at least the expiry ago is forgotten. The sidecar kept it since its end for the eval to read, and reading it doesn't extend that time.

        Returns:
            How many calls were closed, and how many were forgotten.
        """
        now, expiry = time.monotonic(), self._state.expiry
        closed = forgotten = 0
        for key, operation in list(self._operations.items()):
            attempt = operation.attempt
            if operation.ended is not None:
                if now - operation.ended >= expiry:
                    self._forget(key)
                    forgotten += 1
            elif (
                attempt.state in ("awaiting_resend", "awaiting_reply")
                and now - operation.last_heard >= expiry
            ):
                logger.info(
                    "The sidecar closed the model call %s of the eval %s. It waited at %s, and no try, lookup or reply arrived for it in %s seconds.",
                    operation.request_id,
                    operation.eval_id,
                    attempt.state,
                    expiry,
                )
                self._expire(operation)
                closed += 1
        return closed, forgotten

    def _expire(self, operation: _Operation) -> None:
        # Close a model call that waited for the state's expiry while no try, lookup or
        # reply arrived for it.
        # It is closed as `cancel_paused_request` closes a call, with the reason
        # `expired`. Such a call has no open question, so there is none to close. Any
        # judging of it that is still running is cancelled.
        attempt = operation.attempt
        if attempt.stop_cause is None:
            attempt.stop_cause = "expired"
        for judging in list(self._judgings_in_progress):
            if judging.attempt is attempt:
                judging.scope.cancel()
        _settle_stopped(attempt)
        self._finish(attempt)

    def reply_progress(self, id: str) -> None:
        """Record that a proxy handed over part of the reply to a request, by the proxy's id for the exchange. If that exchange is the try of a model call a person was asked about that is with the provider, the call's `last_heard` is updated, so the call isn't closed as expired while its reply is still arriving."""
        operation = self._operations_with_provider.get(id)
        if operation is not None:
            operation.last_heard = time.monotonic()

    async def _exchange(
        self,
        judge: Callable[[], Awaitable[Answer]],
        attempt: Attempt,
        sample: InspectSample,
        request_id: str,
        binding: str,
        held: Held | None = None,
    ) -> Answer:
        # Judge an eval's request or reply, and answer the exchange it arrived in. If a
        # person may be asked through the eval, the judging runs in a task the handler
        # owns, so that it can outlive the exchange. At the first question the exchange
        # is answered "pending review", and the judging continues when the answer
        # arrives. `held` is the request that a reply answers.
        judging = _Judging(
            attempt,
            sample,
            request_id,
            binding,
            anyio.CancelScope(),
            anyio.Event(),
            held=held,
        )
        group = self._group
        if group is None or not self._asks_eval:
            # no person is asked through an eval, so the judging never outlives the
            # exchange
            current = _JUDGING.set(judging)
            self._judgings_in_progress.add(judging)
            try:
                return await judge()
            finally:
                self._judgings_in_progress.discard(judging)
                _JUDGING.reset(current)

        async def run() -> None:
            _JUDGING.set(judging)
            try:
                with judging.scope:
                    try:
                        judging.answer = await judge()
                    except Exception as ex:
                        judging.error = ex
                        if judging.answered_pending:
                            # No exchange is left to raise the error in, and
                            # `request_result` reports only that the call was not
                            # judged. So the error is logged here.
                            logger.warning(
                                "Judging the model call %s of the eval %s failed after it waited for a person: %s",
                                request_id,
                                sample.eval_id,
                                ex,
                            )
                if judging.reattach_refusal is None:
                    # Another try of the call may have asked a person while this try was
                    # judged, though this try asked nobody. The call is still one
                    # operation, so the sidecar doesn't return this try's own result.
                    judging.reattach_refusal = self._refusal_if_reattached(judging)
                if judging.reattach_refusal is not None:
                    # Another try of the call holds the operation, or a later try was
                    # decided after this one began. This try adds nothing to what
                    # `request_result` reports for the call, except the usage billed for
                    # its judging.
                    await self._forget_try(judging)
                    judging.answer = judging.reattach_refusal
                    if judging.error is not None:
                        # The sidecar doesn't return this try's own result, and the
                        # error its judging raised is part of that result. The exchange
                        # is answered as the call is, so the error is only logged here.
                        logger.warning(
                            "Judging a try of the model call %s of the eval %s failed. Another try holds the call, so this try is answered as the call is, and its error is only logged: %s",
                            request_id,
                            sample.eval_id,
                            judging.error,
                        )
                        judging.error = None
                elif judging.answer is None and judging.error is None:
                    # stopped, with nothing decided
                    if attempt.stop_cause is None:
                        self._abandoned(judging)
                    else:
                        _settle_stopped(attempt)
                    judging.answer = _stopped_answer(attempt)
                self._finish(attempt)
            finally:
                self._judgings_in_progress.discard(judging)
                judging.answerable.set()

        self._judgings_in_progress.add(judging)
        group.start_soon(run)
        try:
            await judging.answerable.wait()
        except anyio.get_cancelled_exc_class():
            # The exchange's caller went away before anyone was asked, so nothing is
            # waiting for this judging. The judging is cancelled, and its task then
            # records that the try produced no result (`_abandoned`).
            if not judging.answered_pending:
                judging.scope.cancel()
            raise
        if judging.answered_pending:
            return PENDING
        if judging.error is not None:
            raise judging.error
        assert judging.answer is not None
        return judging.answer

    def _abandoned(self, judging: _Judging) -> None:
        # The judging was cancelled because its exchange's caller went away before
        # anyone was asked, and nothing was decided. The try produced no result, so it
        # is recorded as cut off, not as still being judged.
        #
        # For a model call a person was asked about, the judging was of the reply to the
        # try that was passed on to the provider, and that reply is lost with the
        # caller. The call goes back to `awaiting_resend`, as when the proxy reports a
        # reply that was cut off, so that one more try can be passed on. If the call
        # were left at `judging`, every later try would be answered "pending review" for
        # ever.
        attempt = judging.attempt
        if attempt.settled.is_set():
            return
        operation = self._operation_of(attempt)
        if operation is not None and judging.held is not None:
            self._failed(operation, judging.held, "cut_off", None)
        else:
            attempt.not_judged("cut_off")

    async def _superseded(self, judging: _Judging) -> bool:
        # Whether a try of the model call that began after this one has been decided,
        # which means its answer went to the client. A decision of an earlier try
        # doesn't count, because this try may be the client's own retry after that
        # answer was lost.
        tries = await self._state.attempts(judging.request_id, judging.sample.eval_id)
        if not isinstance(tries, list):
            return False
        later = False
        for tried in tries:
            if tried is judging.attempt:
                later = True
            elif later and tried.state == "decided":
                return True
        return False

    async def _forget_try(self, judging: _Judging) -> None:
        # Remove a try that reattached from what `request_result` reports for the call.
        # Its judge models were billed, so their usage is kept: it is added to the try
        # that `request_result` reports, which is the operation's attempt or else the
        # latest other try. The client waited on this try while it was judged. For a
        # call a person was asked about, judging time is added up over the tries, so
        # this try's time is kept too. A call nobody was asked about counts only its
        # latest try's time.
        eval_id, request_id = judging.sample.eval_id, judging.request_id
        operation = self._operations.get((eval_id, request_id))
        into = operation.attempt if operation is not None else None
        if into is None:
            tries = await self._state.attempts(request_id, eval_id)
            if isinstance(tries, list):
                into = next(
                    (
                        tried
                        for tried in reversed(tries)
                        if tried is not judging.attempt
                    ),
                    None,
                )
        if into is not None and into is not judging.attempt:
            for used, billed in judging.attempt.judging_usage.items():
                add_usage(into.judging_usage, used, billed)
            if operation is not None:
                into.judging_time += judging.attempt.judging_time
        await self._state.forget_attempt(request_id, eval_id, judging.attempt)

    def _refusal_if_reattached(self, judging: _Judging) -> Refuse | None:
        # The answer for a try that is being judged when another try of the same model
        # call asked a person first and holds the operation. It is the answer a later
        # try gets: "pending review", or a refusal because the request differs or
        # because the call has ended. None if no other try holds the operation.
        first = self._operations.get((judging.sample.eval_id, judging.request_id))
        if first is None or first.attempt is judging.attempt:
            return None
        if first.binding != judging.binding:
            return _OTHER_REQUEST
        if first.attempt.settled.is_set():
            return _OVER
        return PENDING

    async def eval_token(self, eval_id: str, presented: str) -> str | None:
        """The secret the sidecar keeps for an eval, which a registration that sends a secret must send.

        Args:
            eval_id: The eval's id.
            presented: The secret the caller sent for the eval.

        Returns:
            The secret kept, which the caller sent, as when a registration whose answer was lost is sent again; or None if none is kept, at the eval's first registration or once a restart or the expiry has lost it.

        Raises:
            UnauthorizedError: If the sidecar keeps another secret for the eval.
        """
        kept = await self._state.eval_token(eval_id, renew=False)
        if kept is None:
            return None
        if not hmac.compare_digest(presented.encode(), kept.encode()):
            raise UnauthorizedError(
                f"The eval {eval_id!r} registered with another token."
            )
        return await self._state.eval_token(eval_id)

    def _kept(self, eval_id: str, outcome: Registered) -> None:
        """Raise for a registration, or an eval's secret, that the state didn't keep.

        A token is kept for `--run-expiry` after the eval was last named, apart from the registration that made it, which may expire sooner or, for a task registered beforehand, take no room at all. So the tokens are bounded on their own, by the same number as the tasks registered by evals, and nothing kept is dropped to make room: a kept token guards the records of the eval's runs for as long as they are kept.

        Raises:
            UnauthorizedError: If the handler keeps another secret for the eval.
            RegistrationsFullError: If the handler already keeps as many registrations as it takes, or as many tokens.
        """
        if outcome == "another_token":
            raise UnauthorizedError(
                f"The eval {eval_id!r} registered with another token."
            )
        if outcome == "tokens_full":
            raise RegistrationsFullError(
                f"This sidecar already keeps the {self._max_task_registrations} tasks registered by evals that it takes, or the tokens of as many evals. One is taken again once a kept task or token expires."
            )
        if outcome == "full":
            raise RegistrationsFullError(
                f"This sidecar already keeps the {self._max_task_registrations} tasks registered by evals that it takes. One is taken again once a kept task expires."
            )

    async def keep_eval_token(self, eval_id: str, token: str) -> None:
        """Keep the secret an eval made and sent with a registration that stores no task, unless it is kept already.

        The sidecar never makes a secret, and never sends one back.

        Raises:
            UnauthorizedError: If the handler keeps another secret for the eval.
            RegistrationsFullError: If the handler already keeps as many tokens as it takes registrations.
        """
        self._kept(
            eval_id,
            await self._state.keep_eval_token(
                eval_id, token, self._max_task_registrations
            ),
        )

    async def holds_eval(
        self,
        eval_id: str,
        method: str,
        timestamp: int,
        mac: str,
        signed: Sequence[object] | None = None,
    ) -> bool:
        """Whether the sidecar keeps a secret for the eval, and the caller proved it holds it.

        Args:
            eval_id: The eval's id.
            method: The method called.
            timestamp: When the caller signed, in seconds since the epoch.
            mac: The caller's HMAC of the method, the eval's id and the timestamp, made with the eval's secret.
            signed: The values the `mac` signs, for a method that signs more than those three, as `answer_question` does.

        Returns:
            True if the caller proved it holds the eval's secret; False if the sidecar keeps none for the eval, so it holds nothing of the eval for anyone, as after a restart.

        Raises:
            UnauthorizedError: If the sidecar keeps a secret for the eval, and the `timestamp` is outside the window or the `mac` isn't made with it.
        """
        kept = await self._state.eval_token(eval_id, renew=False)
        if kept is None:
            return False
        problem = unproven(
            kept,
            call_values(method, eval_id, timestamp) if signed is None else signed,
            timestamp,
            mac,
            window=self._timestamp_window,
            what="The call's",
        )
        if problem is not None:
            raise UnauthorizedError(problem)
        # a call that proves it holds the secret keeps it
        await self._state.eval_token(eval_id)
        return True

    async def _unproven_request(self, signed: SignedSample) -> str | None:
        # why a request's header doesn't prove its run, or None if it does
        eval_id = signed.sample.eval_id
        kept = await self._state.eval_token(eval_id, renew=False)
        if kept is None:
            return f"This sidecar holds no registration of the eval {eval_id!r}, as after a restart, so it can't check the request's {INSPECT_CONTEXT} header. The eval registers its task again."
        problem = unproven(
            kept,
            request_values(signed.sample, signed.request_id, signed.timestamp),
            signed.timestamp,
            signed.mac,
            window=self._timestamp_window,
            what=f"The request's {INSPECT_CONTEXT}",
        )
        if problem is not None:
            return f"{problem} The request was not judged."

        # A run belongs to the eval that started it, whether still kept or
        # ended. A client can register an eval of its own and sign with that
        # eval's token, so the token alone doesn't stop it naming another
        # eval's sample.
        held = await self._state.peek_run(_run_key(True, signed.sample.sample_uuid))
        if held is not None and held.eval_id not in (None, eval_id):
            return _ANOTHER_EVALS.message

        # a request that proves its run keeps the eval's token
        await self._state.eval_token(eval_id)
        return None

    async def request_result(
        self, id: str, eval_id: str, wait: float = 0
    ) -> RequestResult | Literal["dropped", "unknown"]:
        """What happened to a request an eval names, by the eval's id for it.

        Args:
            id: The eval's id for the request, as its `x-irid` header gave it. A client that retries a request sends it again under the same id.
            eval_id: The eval asking. A request whose `x-inspect-context` named another eval reads as unknown, as does one that was never kept: one with no `x-inspect-context`, or one whose header couldn't be read or failed the check.
            wait: How long to wait, in seconds, while the latest attempt is still being judged, at most `MAX_RESULT_WAIT`.

        Returns:
            The request's state, with every try's records in order; `dropped` if the records were kept and no longer are; or `unknown` if the request was never seen, or isn't the eval's. For a request no person was asked about, the state is that of its latest try. For a request a person was asked about, the state is that of the one model call, whatever tries have arrived since.
        """
        deadline = time.monotonic() + min(max(wait, 0), MAX_RESULT_WAIT)
        while True:
            attempts = await self._state.attempts(id, eval_id)
            if attempts is None:
                return "unknown"
            if isinstance(attempts, Dropped):
                # told only to the eval the request was of
                return "dropped" if attempts.eval_id == eval_id else "unknown"
            if any(attempt.eval_id != eval_id for attempt in attempts):
                return "unknown"
            operation = self._operations.get((eval_id, id))
            if operation is not None and not any(
                attempt is operation.attempt for attempt in attempts
            ):
                operation = None
            latest = operation.attempt if operation is not None else attempts[-1]
            if operation is not None:
                # the eval asked about the call, which counts as contact
                operation.last_heard = time.monotonic()
            left = deadline - time.monotonic()
            if latest.state != "judging" or left <= 0:
                break
            # woken at once by a decision, and looking again now and then,
            # as a retry of the request may have begun another attempt
            with anyio.move_on_after(min(left, _RESULT_POLL)):
                await latest.settled.wait()
        records: list[Record] = []
        usage: dict[ModelUsed, ModelUsage] = {}
        for attempt in attempts:
            kept = await self._state.records(attempt.exchange)
            if kept is None:
                return "dropped"
            records.extend(kept)
            # every attempt's judging was billed, a retried one's too
            for used, billed in attempt.judging_usage.items():
                add_usage(usage, used, billed)
        reviews = [
            (attempt.review_tries.get(review.id, number), review)
            for number, attempt in enumerate(attempts, 1)
            for review in attempt.reviews
        ]
        if operation is None:
            return RequestResult(latest, records, reviews, usage)
        failed = latest.state == "awaiting_resend" and latest.reason is not None
        return RequestResult(
            latest,
            records,
            reviews,
            usage,
            failure=operation.failure if failed else None,
        )

    async def judged_reply(self, headers: Mapping[str, str]) -> JudgedReply | Refuse:
        """Return the reply that the sidecar holds for a model call a person was asked about, to the eval that fetches it.

        The eval fetches the reply by sending the model call's own request again, under the same `x-irid`, to the judged-reply path. So the request's `x-inspect-context` is checked as any model request's is: the `timestamp` and the `mac` prove the eval, the run and the `x-irid`. The sidecar reads nothing else of the request. The reply is returned once the call is `decided` with the outcome `proceed`, and for as long as the sidecar still keeps it. `request_result` reports that state, and doesn't carry the reply. While the call is at `awaiting_resend` with the reason `provider_error`, the provider's error is returned instead, with the `FAILED_TRY` header set to the number of the failed try. A fetch changes nothing of the call, so the eval can fetch again, and the provider is never called.

        Args:
            headers: The request's headers, by lowercase name.

        Returns:
            The judged reply or the provider's error, or a refusal that says why there is nothing to return: the header is missing, unreadable or fails the check; the sidecar holds no such call for the eval; the call has not ended; or the call ended with an outcome other than `proceed`.
        """
        try:
            signed = signed_sample(headers)
        except UnreadableError as ex:
            return Refuse(400, str(ex))
        if signed is None:
            return _NO_CALL_NAMED
        refused = await self._unproven_request(signed)
        if refused is not None:
            return Refuse(400, refused)

        # Find the call as `request_result` does: it must be the eval's own, by its
        # `x-irid`, and its records must still be kept.
        eval_id = signed.sample.eval_id
        operation = self._operations.get((eval_id, signed.request_id))
        if operation is None:
            return _NO_JUDGED_REPLY
        attempts = await self._state.attempts(signed.request_id, eval_id)
        if (
            attempts is None
            or isinstance(attempts, Dropped)
            or not any(attempt is operation.attempt for attempt in attempts)
        ):
            return _NO_JUDGED_REPLY

        # a fetch counts as contact about the call, as a try does
        operation.last_heard = time.monotonic()
        attempt = operation.attempt
        failure = operation.failure
        if (
            attempt.state == "awaiting_resend"
            and attempt.reason == "provider_error"
            and failure is not None
            and failure.reply is not None
        ):
            # The call waits for a resend because the provider failed it. Return the
            # provider's latest error.
            return _served_error(failure)
        if not attempt.settled.is_set():
            return _NOT_ENDED
        reply = operation.reply
        if attempt.state != "decided" or attempt.outcome != "proceed" or reply is None:
            return _NOT_GOING_ON
        return reply

    async def sample_result(
        self, sample_uuid: str, eval_id: str
    ) -> SampleResult | None:
        """What an eval's sample came to, and the sentinel's store, for the eval to copy into the sample's store before scoring. The handler keeps one run for each sample in each epoch, and its store and end are those of the sample's latest attempt.

        Args:
            sample_uuid: The sample's uuid, as the eval names it.
            eval_id: The eval asking. A sample of another eval reads as unknown.

        Returns:
            The sample's outcome and store, or None if nothing is kept for it, or it isn't the eval's.
        """
        # renewed only once it is found to be the eval's, so that another
        # eval asking keeps nothing alive
        key = _run_key(True, sample_uuid)
        kept = await self._state.peek_run(key)
        # an ended run keeps no store to copy
        if kept is None or kept.eval_id != eval_id or kept.closed:
            return None
        await self._state.existing_run(key)
        return SampleResult(
            "terminate" if kept.refusal is not None else "proceed",
            dict(kept.store.items()),
        )

    async def end_sample(
        self, sample_uuid: str, eval_id: str
    ) -> Literal["ended", "unknown"]:
        """Drop the state kept for an eval's sample, once the eval has scored it. A later request of the sample is refused, as for a sample a sentinel ended.

        Args:
            sample_uuid: The sample's uuid, as the eval names it.
            eval_id: The eval ending it. A sample of another eval is left as it is, and reads as unknown.

        Returns:
            `ended`, or `unknown` for a sample the handler never saw or that isn't the eval's. Ending a sample again has no effect.
        """
        ended = await self._state.end_run(_run_key(True, sample_uuid), eval_id)
        if ended:
            # No one is left to answer the sample's questions, or to resend an answer to
            # one.
            self._stop(
                "sample_ended",
                lambda sample, _: (
                    (sample.eval_id, sample.sample_uuid) == (eval_id, sample_uuid)
                ),
            )
            self._reviews.forget_sample(eval_id, sample_uuid)
            # the replies held for the sample's calls are dropped with the sample
            for key, operation in list(self._operations.items()):
                if (operation.eval_id, operation.sample.sample_uuid) == (
                    eval_id,
                    sample_uuid,
                ):
                    self._forget(key)
        return "ended" if ended else "unknown"

    async def _within_limit(
        self, judging: Awaitable[Answer], what: str, id: str
    ) -> Answer:
        # The time limit covers the sentinel's work on one request or reply. The time a
        # question is open for a person doesn't count: the asker suspends the limit's
        # deadline while it waits.
        with anyio.move_on_after(self._time_limit) as limit:
            current = _JUDGING.get()
            if current is not None:
                current.limit = limit
            return await judging
        raise self._out_of_time(what, id)

    def _stop(
        self, cause: Stopped, stops: Callable[[InspectSample, Attempt], bool]
    ) -> None:
        # Nothing is waiting any more for the attempts that match. Each matching attempt
        # that is being judged is marked as stopped, whether or not it has asked a
        # person yet, so that judging which reaches `human()` later asks nobody. Judging
        # that has asked nobody is left to finish, as before: the checks that follow it
        # keep or drop what it decides.
        #
        # A model call a person was asked about is closed here, at whatever state it has
        # reached, before anything else runs. Its questions close, so from this moment
        # none is listed or can be answered. Its judging is cancelled with nothing
        # decided, no try of it is passed on to the provider, and `request_result`
        # reports it as ended at once. It stays closed: the reply to a request already
        # with the provider is refused when it arrives.
        for judging in list(self._judgings_in_progress):
            attempt = judging.attempt
            if not stops(judging.sample, attempt):
                continue
            if attempt.stop_cause is None:
                attempt.stop_cause = cause
            if self._operation_of(attempt) is not None:
                judging.scope.cancel()
        for operation in list(self._operations.values()):
            attempt = operation.attempt
            if attempt.settled.is_set() or not stops(operation.sample, attempt):
                continue
            if attempt.stop_cause is None:
                attempt.stop_cause = cause
            stop_cause = attempt.stop_cause
            if stop_cause != "expired":
                # a call that expired had no open question to close
                self._reviews.close(
                    [r for r in attempt.reviews if r.resolution is None], stop_cause
                )
            _settle_stopped(attempt)
            self._finish(attempt)

    async def records(self, id: str) -> list[Record] | None:
        """What was recorded while a request and its reply were judged.

        Args:
            id: The request's id, as the proxy gave it.

        Returns:
            The records in order, the request's before the reply's, or None if the request was never seen or its records are no longer kept.
        """
        return await self._state.records(id)

    async def register_task(
        self,
        eval_id: str,
        registration: TaskRegistration,
        expiry: float | None = None,
        token: str | None = None,
    ) -> float | None:
        """Keep a task an eval registered, so that its runs' sentinels are told which sample each is, once each sample's registration is complete.

        A registration replaces the eval's last one for the task, and keeps the samples whose registration was complete that it still names, so that registering again has no effect. A run that has already found its sample keeps what it found.

        Args:
            eval_id: The eval's id, as its requests name it.
            registration: The task, and the samples it runs.
            expiry: How long to keep it once no request has named the task, in seconds. A day if not given.
            token: The secret the registration sent, which is kept if the eval has none. None for a registration that proved it holds the eval's secret.

        A task registered beforehand is already registered, so its registration stores nothing and is not compared with the file, since the answer would tell a caller whether its guess at the file is right. This holds whether or not the handler takes registrations from evals.

        The secret a registration sent is kept with its task in one step, or alone for a task registered beforehand. So a registration that fails leaves no secret behind, and of two first registrations of an eval that send different secrets, only one keeps anything.

        Returns:
            How long it is kept once no request has named the task: `expiry`, or less if that is longer than the handler keeps any. None for a task registered beforehand, which is never dropped.

        Raises:
            RegistrationError: If the handler takes no registrations from evals and the task was not registered beforehand.
            RegistrationsFullError: If the handler already keeps as many registrations as it takes, or as many tokens.
            UnauthorizedError: If the registration sent a secret and the handler keeps another for the eval.
        """
        if registration.task in self._tasks:
            if token is not None:
                await self.keep_eval_token(eval_id, token)
            return None
        if not self._runtime_task_registration:
            raise RegistrationError(
                "This sidecar takes no tasks registered by an eval."
            )
        granted = min(expiry if expiry is not None else TASK_EXPIRY, self._task_expiry)
        kept = RegisteredTask(registration)
        # Read without renewing: the registration's secret is only taken in
        # the step that keeps the task, so one refused there must not have
        # kept the task it would have replaced alive.
        before = await self._state.registration(eval_id, registration.task, renew=False)
        if before is not None:
            kept.keep_samples(
                {
                    id: sample
                    for id, sample in before.samples.items()
                    if id in kept.sample_ids
                }
            )
        if kept.held > self._max_registration_bytes:
            raise self._full(registration.task, kept.held)
        self._kept(
            eval_id,
            await self._state.register(
                eval_id,
                registration.task,
                kept,
                granted,
                self._max_task_registrations,
                token=token,
            ),
        )
        return granted

    async def complete_registration(
        self, eval_id: str, task: str, sample: SampleFacts
    ) -> bool:
        """Keep a sample's facts, before the sample's first epoch, so that its runs' sentinels are told them.

        Completing a sample's registration again replaces its facts, for the runs that start after. A sample the task's registration doesn't name, such as one a task adds while it runs, extends the registration, unless the handler is `registered_only`. A task registered beforehand already holds its samples, so nothing is stored, and the sample is not compared with the file, since the answer would tell a caller whether its guess at the file is right.

        Args:
            eval_id: The eval's id, as its requests name it.
            task: The task's name.
            sample: The sample.

        Returns:
            Whether the task was registered beforehand, so nothing was stored.

        Raises:
            UnknownTaskError: If the eval hasn't registered the task, or its registration is no longer kept.
            RegistrationError: If the handler is `registered_only` and the task's registration doesn't name the sample.
            RegistrationsFullError: If the task's registration would then hold more than the handler takes for a task.
        """
        if task in self._tasks:
            return True
        kept = await self._state.registration(eval_id, task)
        if kept is None:
            raise UnknownTaskError(
                f"This sidecar holds no registration of the task {task!r} by the eval {eval_id!r}. Register the task, then complete the sample's registration."
            )
        if sample.sample_id not in kept.sample_ids:
            if self._strictness == "registered_only":
                raise RegistrationError(
                    f"This sidecar takes only the samples a task's registration names, and the registration of {task!r} doesn't name the sample {sample.sample_id!r}."
                )
        held = kept.holding(sample)
        if held > self._max_registration_bytes:
            raise self._full(task, held)
        kept.complete(sample)
        return False

    def _full(self, task: str, held: int) -> RegistrationsFullError:
        return RegistrationsFullError(
            f"The registration of {task!r} would hold {held} bytes, and this sidecar keeps no more than {self._max_registration_bytes} for a task: its registration, and the facts of each sample whose registration is complete."
        )

    def _view(self, call: ToolCall, run: Run) -> ToolCallView:
        # A request names a tool only by the name the model sees. A task
        # registered beforehand can name the tool that is; otherwise the name
        # is taken for inspect_ai's own tool of that name, if it has a viewer.
        task = self._tasks.get(run.eval.task) if run.eval is not None else None
        tool = task.tools.get(call.function) if task is not None else None
        if tool is None and call.function in VIEWED_TOOLS:
            tool = f"inspect_ai/{call.function}"
        return tool_call_view(call, tool)

    async def _eval(self, sample: InspectSample) -> EvalContext | None:
        # the sample a request names, from the task's registration; asking
        # counts as the task being named, so its registration is kept
        fixed = self._tasks.get(sample.task)
        if fixed is None:
            kept = await self._state.registration(sample.eval_id, sample.task)
            if kept is None:
                if self._strictness != "open":
                    raise UnregisteredSampleError(
                        f"This sidecar judges only the requests of tasks it holds, and holds no task {sample.task!r} of the eval {sample.eval_id!r}."
                    )
                return None
            found = kept.eval_context(sample)
            if self._strictness == "registered_only":
                if found is None:
                    raise UnregisteredSampleError(
                        f"This sidecar judges only the samples it holds, and the registration of {sample.task!r} isn't complete for the sample {sample.sample_id!r}."
                    )
                if not 1 <= sample.epoch <= kept.registration.epochs:
                    raise UnregisteredSampleError(
                        f"The registration of {sample.task!r} runs epochs 1 to {kept.registration.epochs}, not epoch {sample.epoch}."
                    )
            return found
        found = fixed.eval_context(sample)
        if found is None:
            raise UnregisteredSampleError(
                f"The task {sample.task!r} is registered with this sidecar beforehand, without the sample {sample.sample_id!r}."
            )
        return found

    async def _request(
        self,
        request: Request,
        sample: InspectSample | None,
        attempt: Attempt | None,
        binding: str = "",
    ) -> Answer:
        if sample is None and self._strictness == "registered_only":
            raise UnregisteredSampleError(
                f"This sidecar judges only the requests of samples it holds, and the request has no {INSPECT_CONTEXT} header."
            )
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
        sample_attempt = sample.attempt if sample is not None else ""
        # a conversation is told apart within a named run by the attempt it
        # is part of, since a retry opens as the attempt it replaces did, and
        # then by how it opens
        conversation_key = f"{key}/{sample_attempt}/{opening}" if named else key
        run_key = _run_key(named, key)
        if sample is not None:
            # Whose run it is was checked before the request was read, and
            # another eval may have begun the run while it was. So it is
            # checked again here, where nothing is awaited between the check
            # and the run becoming this eval's, and before anything below
            # can reset the run.
            held = await self._state.peek_run(run_key)
            if held is not None and held.eval_id not in (None, sample.eval_id):
                return _ANOTHER_EVALS
        run = await self._state.run(run_key)
        restored = False
        if sample is not None:
            if run.eval_id is None:
                run.eval_id = sample.eval_id
            was = run.attempt
            clash = _begin_attempt(run, sample)
            if run.attempt != was:
                eval_id, sample_uuid, now = (
                    sample.eval_id,
                    sample.sample_uuid,
                    run.attempt,
                )
                self._stop(
                    "replaced_attempt",
                    lambda other, _: (
                        (other.eval_id, other.sample_uuid) == (eval_id, sample_uuid)
                        and other.attempt != now
                    ),
                )
            if clash is not None:
                return self._judged(clash, proxy, Exchange(replaced=True), attempt)
            # the eval holds the calls a decision acted on without their
            # arguments: the model's own go back before anything reads them
            written = request_format(request.provider, request.body, request.format)
            body = unredacted(
                request.provider,
                request.body,
                {
                    id: call
                    for id, (format, call) in run.redacted.items()
                    if format == written
                },
                request.format,
            )
            if body is not None:
                request = replace(request, body=body)
                restored = True
        marks = request_marks(request.provider, request.body, request.format)
        refusal = _ended(run, marks, sample_attempt, request)
        if refusal is not None:
            # the run has ended, and the sentinel is not asked again
            return self._judged(refusal, proxy, Exchange(), attempt)
        # the attempt's store, taken before anything below awaits, since a
        # retry may replace the run's store meanwhile
        store = run.store
        if sample is not None:
            found = await self._eval(sample)
            if run.eval is None:
                # a run keeps the facts it first found
                run.eval = found
        conversation = run.conversations.setdefault(conversation_key, Conversation())
        messages = await read_request(
            request.provider, request.body, request.format, start=conversation.noted
        )
        sends = offered(request.provider, request.body, request.format)

        answer: Answer = Replace(request.body) if restored else Pass()
        # only what the client sent holds results to judge: what the sidecar
        # adds below is its own writing
        client = messages
        # a conversation that holds two calls with one id can't be read, as
        # which of the two a result is of can't be told
        _calls(client)
        # A point this request doesn't continue from is kept for the requests
        # that do: a run's key is shared by conversations that open alike. A
        # client that was handed a rejected turn, as an Inspect eval is in the
        # judgment it asked for, may already have put it back itself.
        holds = {_digest(message) for message in messages}
        continued = [
            rejected
            for rejected in conversation.rejected
            if rejected.sent < len(marks) and marks[rejected.sent] == rejected.mark
        ]
        told = {
            rejected.sent: turns
            for rejected in continued
            if rejected.added
            and (turns := [t for t in rejected.turns if _digest(t[0]) not in holds])
        }
        # the rejections of refused turns the client put back itself
        refused = {
            id: amendment
            for rejected in continued
            for turn, written in zip(rejected.turns, rejected.written, strict=True)
            if _digest(turn[0]) in holds
            for id, amendment in written.items()
        }
        edited = [
            turn
            for turn in conversation.edited
            if turn.sent < len(marks) and marks[turn.sent] == turn.mark
        ]
        # A request that holds a call of an edited turn, which isn't held where
        # the turn was made, as when a client rewrote earlier entries of its
        # conversation, can't be matched with the edit: its results would be
        # judged as of a call that didn't run.
        if _edit_lost(run, edited, client):
            raise UnreadableError(
                "The request holds a call this sidecar changed or took out, after earlier entries of its conversation changed, so which call ran can't be told."
            )
        amended = await amend_request(
            request.provider,
            request.body,
            inserted=told,
            edited=edited,
            format=request.format,
        )
        if amended.body is not None:
            messages = await read_request(
                request.provider,
                amended.body,
                request.format,
                start=conversation.noted,
            )
            answer = Replace(amended.body)
        # what the model is sent, each message the conversation sent before
        # under the id it had then, as in Inspect's own process
        body = amended.body if amended.body is not None else request.body
        messages = with_sent_ids(conversation, messages, body)

        # the last entry in digests, for a run no eval names, so that a run
        # ended at this request remembers it without keeping what it holds
        digests = (
            []
            if named
            else entry_digests(request.provider, request.body, request.format)
        )
        last = digests[-1] if digests else None
        held = Held(
            run=run_key,
            conversation=conversation_key,
            named=named,
            sample_attempt=sample_attempt,
            answered=any(mark in run.answered for mark in marks[:-1]),
            model=_step_model(request.provider, request.model, sample),
            format=request_format(request.provider, request.body, request.format),
            recorder=await self._state.recorder(request.id),
            input=messages,
            offered=sends,
            marks=marks,
            proxy=proxy,
            store=store,
            attempt=attempt,
            last=last,
            request=None if named else request,
            binding=binding,
        )
        said = call_texts(request.provider, request.body, request.format)
        # a changed call's result is of the call that ran, whichever the client holds
        ran = {id: call for turn in edited for id, call in turn.modified.items()}
        restored = amended.restored | held_rejections(
            request.provider, request.body, refused, request.format
        )
        ended = await self._results(
            held, run, conversation, client, said, restored, ran
        )
        if ended is not None:
            return self._judged(ended, proxy, held.exchange, attempt)
        if BeforeGenerate in self._watched:
            before = BeforeGenerate(
                model=held.model,
                conversation=conversation_key,
                input=messages,
                # the agent's own message list is not known behind a proxy
                agent_messages=None,
                # what was sent before this request, which isn't sent yet
                history=list(conversation.sent),
                tools=sends.tools,
                tool_choice=sends.tool_choice,
                config=sends.config,
            )
            if _ends(await self._judge(held, run, before)):
                return self._judged(
                    self._end(held, run, "before the model was called"),
                    proxy,
                    held.exchange,
                    attempt,
                )
        # another exchange of the run may have ended it while this one waited
        refusal = _ended(run, marks, sample_attempt, request)
        if refusal is not None:
            return self._judged(refusal, proxy, held.exchange, attempt)
        # the request goes on to the model, as the client sent it with what
        # this sidecar added
        record_sent(conversation, messages, body)
        await self._state.hold(request.id, held)
        # an Inspect client is told of the request's steps with the reply
        return answer

    async def _results(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        client: list[ChatMessage],
        said: Mapping[str, str],
        restored: frozenset[str] = frozenset(),
        ran: Mapping[str, ToolCall] | None = None,
    ) -> Refuse | None:
        if AfterToolCall not in self._watched:
            return None
        made: dict[str, tuple[ChatMessageAssistant, ToolCall, str]] = {}
        for entry in client:
            if isinstance(entry, ChatMessageAssistant):
                for call in entry.tool_calls or []:
                    made[call.id] = (entry, call, said.get(call.id, entry.text))
            if not isinstance(entry, ChatMessageTool):
                continue
            found = made.get(entry.tool_call_id or "")
            if found is None:
                raise UnreadableError(
                    f"The request holds a result for a call it doesn't hold: {entry.tool_call_id!r}."
                )
            assistant, call, words = found
            call = (ran or {}).get(call.id, call)
            digest = _result_digest(words, call, entry)
            key = (held.run, held.conversation, digest)
            if key in self._judging:
                while (under_way := self._judging.get(key)) is not None:
                    if not under_way.asked:
                        await under_way.finished_or_asked.wait()
                        continue
                    # The judging this request was waiting on asked a person, who may
                    # take days, and no connection waits for a person. So this request
                    # stops waiting. A tool's result is reviewed once, by the model call
                    # that asked.
                    current = _JUDGING.get()
                    reattach_refusal = (
                        self._refusal_if_reattached(current)
                        if current is not None
                        else None
                    )
                    if current is not None and reattach_refusal is not None:
                        # this request is a try of that same model call, so it
                        # reattaches to the call
                        current.reattach_refusal = reattach_refusal
                        current.scope.cancel()
                        await anyio.lowlevel.checkpoint()
                    # This request is another model call of the sample. It is refused
                    # with a 503, so that its client resends it. A resend proceeds once
                    # the review has ended and the result is judged.
                    return _UNDER_REVIEW
                # the judgment waited for may have ended the run
                refusal = _ended(run, held.marks, held.sample_attempt, held.request)
                if refusal is not None:
                    return refusal
            # a rejection the client put back itself is the sidecar's own
            # writing, not the result of a call that ran
            if digest in conversation.judged or entry.tool_call_id in restored:
                continue
            sent = conversation.made.get(call.id)
            # a call this sidecar did not hand out has only the id its
            # reading of the request gave it
            message_id = sent.message_id if sent is not None else assistant.id
            assert message_id is not None
            step = AfterToolCall(
                conversation=held.conversation,
                message=assistant.text,
                message_id=message_id,
                assistant_message=sent.message if sent is not None else None,
                call=call,
                result=entry,
                # not known: only what the model is sent crosses the wire, and
                # a monitor must not take that for everything the tool returned
                output=None,
                view=self._view(call, run),
                input=sent.input if sent is not None else None,
                agent_messages=None,
                # as before the next generate: without what this request sends
                history=list(conversation.sent),
                tools=sent.tools if sent is not None else None,
                tool_choice=sent.tool_choice if sent is not None else None,
            )
            under_way = self._judging[key] = _ResultJudging(
                anyio.Event(), _JUDGING.get()
            )
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
                under_way.finished_or_asked.set()
        return None

    async def _reply(
        self, reply: Reply, held: Held | None, released: bytes = b""
    ) -> Answer:
        answer = await self._judge_reply(reply, held, released)
        if held is None or not succeeded(reply.status):
            return answer
        # an Inspect eval asks what was decided, by `request_result`
        if held.proxy.inspect is not None and held.attempt is not None:
            held.attempt.judged = held.exchange
        return answer

    async def _judge_reply(
        self, reply: Reply, held: Held | None, released: bytes = b""
    ) -> Answer:
        if not succeeded(reply.status):
            # nothing came from the model to judge
            return Pass()
        if held is None:
            raise UnreadableError(
                f"The reply {reply.id!r} answers a request this sidecar was never shown, or no longer keeps."
            )
        run = await self._state.run(held.run)
        conversation = run.conversations.setdefault(held.conversation, Conversation())
        # what the reply is read from, which an edit of it is read back from
        start = conversation.noted
        output, noted = await read_reply_noted(
            reply.provider,
            reply.content_type,
            reply.body,
            reply.format,
            start=conversation.noted,
        )
        # kept whatever is decided: the provider continues from what it sent
        conversation.noted = noted
        # billed whatever is decided, so a client refused the reply still counts it
        held.exchange.usage = output.usage
        refusal = _ended(run, held.marks, held.sample_attempt, held.request)
        if refusal is not None:
            # a sentinel ended the run while the model was answering, or a
            # later attempt replaced this one
            held.exchange.ended = True
            held.exchange.replaced = refusal is _REPLACED
            return refusal

        if AfterGenerate in self._watched:
            after = AfterGenerate(
                model=held.model,
                conversation=held.conversation,
                input=held.input,
                agent_messages=None,
                # as the core gives it: what the conversation has sent by
                # now, another request sent meanwhile included
                history=list(conversation.sent),
                tools=held.offered.tools,
                tool_choice=held.offered.tool_choice,
                config=held.offered.config,
                output=output,
            )
            if _ends(await self._judge(held, run, after)):
                if output.choices:
                    _withhold_ended(held, output.message)
                return self._end(held, run, "at the model's reply")

        if not output.choices:
            return self._ended_meanwhile(held, run) or self._passed(
                held, run, conversation, None
            )
        message = output.message
        # the reply is still read, and kept as passed, when no call is judged
        calls = (message.tool_calls or []) if BeforeToolCall in self._watched else []
        rejected: list[tuple[ToolCall, Decision]] = []
        modified: dict[str, ToolCall] = {}
        for call in calls:
            assert message.id is not None
            step = BeforeToolCall(
                conversation=held.conversation,
                message=message.text,
                message_id=message.id,
                assistant_message=message,
                call=call,
                view=self._view(call, run),
                input=held.input,
                agent_messages=None,
                history=list(conversation.sent),
                tools=held.offered.tools,
                tool_choice=held.offered.tool_choice,
            )
            decision = await self._judge(held, run, step)
            if decision is None or decision.action == "proceed":
                continue
            if decision.action == "terminate":
                _withhold_ended(held, message, call.id, rejected, modified)
                return self._end(held, run, f"at the call to {call.function}")
            if decision.action == "modify" and decision.modified is not None:
                modified[call.id] = decision.modified
            else:
                rejected.append((call, decision))
        refusal = self._ended_meanwhile(held, run)
        if refusal is not None:
            return refusal
        if not rejected and not modified:
            return self._passed(held, run, conversation, message)
        taken = {call.id for call, _ in rejected}
        if all(call.id in taken for call in message.tool_calls or []):
            return await self._reject(held, run, conversation, message, noted, rejected)
        try:
            return await self._edit(
                reply,
                held,
                run,
                conversation,
                output,
                start,
                noted,
                rejected,
                modified,
                released=released,
            )
        except UnreadableError as ex:
            if not rejected:
                raise ModifyError(
                    "The sentinel changed a call this sidecar can't change in the reply."
                ) from ex
            # the reply couldn't be edited as the sentinel decided, so the turn
            # is refused whole, as it was before calls could be taken out
            return await self._reject(held, run, conversation, message, noted, rejected)

    def _passed(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        message: ChatMessageAssistant | None,
        produced: ChatMessageAssistant | None = None,
    ) -> Pass:
        run.in_a_row.pop(held.sample_attempt, None)
        if not held.named:
            run.answered.add(held.mark)
        if message is not None:
            for call in message.tool_calls or []:
                assert message.id is not None
                conversation.made[call.id] = Made(
                    held.input,
                    held.offered.tools,
                    held.offered.tool_choice,
                    message.id,
                    produced if produced is not None else message,
                )
            while len(conversation.made) > MADE_KEPT:
                del conversation.made[next(iter(conversation.made))]
        return Pass()

    async def _edit(
        self,
        reply: Reply,
        held: Held,
        run: Run,
        conversation: Conversation,
        output: ModelOutput,
        start: JsonValue | None,
        noted: JsonValue | None,
        rejected: Sequence[tuple[ToolCall, Decision]],
        modified: Mapping[str, ToolCall],
        *,
        released: bytes = b"",
    ) -> Replace:
        # The rejected calls are taken out and the changed calls go out
        # changed, and the rest of the turn goes on, as in Inspect's own
        # process, where the other calls in a turn still run. The model's
        # message keeps its own calls there, so later requests have them put
        # back.
        message = output.message
        reasons = {
            call.id: decision.message or _REJECTED for call, decision in rejected
        }
        written = await amendments(held.format, message, reasons, modified, noted)
        if held.named:
            for id in reasons:
                run.redacted[id] = (held.format, written[id].call)
        body = await edit_reply(
            reply.provider,
            reply.content_type,
            reply.body,
            reply.format,
            Edits(frozenset(reasons), modified),
            output,
            start=start,
        )
        if not body.startswith(released):
            # what has gone out can't be taken back
            raise UnreadableError(
                "The reply as edited changes what the client was already sent."
            )
        calls = message.tool_calls or []
        conversation.edited.append(
            Edited(
                sent=held.sent,
                mark=held.mark,
                order=tuple(call.id for call in calls),
                taken_out=reasons,
                originals={call.id: call for call in calls if call.id in modified},
                modified=dict(modified),
                amendments=written,
            )
        )
        # an eval is handed neither a call taken out nor a changed call's
        # original arguments
        held.exchange.proposed = Withheld(
            message, reasons, redacted=frozenset(reasons), ran=dict(modified)
        )
        handed = message.model_copy(
            update={
                "tool_calls": [
                    modified.get(call.id, call)
                    for call in calls
                    if call.id not in reasons
                ]
            }
        )
        # some of the turn ran, which is progress, so it counts as a pass
        self._passed(held, run, conversation, handed, produced=message)
        return Replace(body)

    async def _reject(
        self,
        held: Held,
        run: Run,
        conversation: Conversation,
        message: ChatMessageAssistant,
        noted: JsonValue | None,
        rejected: Sequence[tuple[ToolCall, Decision]],
    ) -> Refuse:
        # only `message` is for the agent; a decision's explanation is for the log
        reasons = {
            call.id: decision.message or _REJECTED for call, decision in rejected
        }
        told = {
            other.id: reasons.get(other.id, _NOT_RUN)
            for other in message.tool_calls or []
        }
        # the first rejection is the one the client is refused with
        call, _ = rejected[0]
        why = reasons[call.id]
        count = run.in_a_row.get(held.sample_attempt, 0) + 1
        ends = count >= _MAX_CONSECUTIVE_REJECTIONS
        # made before anything is kept, since making it can fail, and a turn
        # kept without it would leave every later request of the conversation
        # unreadable
        written = (
            None if ends else await amendments(held.format, message, told, (), noted)
        )
        held.exchange.withheld = Withheld(message, told, redacted=frozenset(reasons))
        held.exchange.message = why
        run.in_a_row[held.sample_attempt] = count
        if written is not None and held.named:
            for id in reasons:
                run.redacted[id] = (held.format, written[id].call)
        if written is None:
            return self._end(
                held,
                run,
                f"at the call to {call.function} after {_MAX_CONSECUTIVE_REJECTIONS} rejected turns in a row",
            )
        # a client that comes back after a rejection sends the same request, so
        # its rejected turns in a row share a point
        point = next(
            (
                known
                for known in conversation.rejected
                if (known.sent, known.mark) == (held.sent, held.mark) and known.added
            ),
            None,
        )
        if point is None:
            point = Rejected(held.sent, held.mark, [])
            conversation.rejected.append(point)
        point.turns.append((message, told, noted))
        point.written.append(written)
        return Refuse(
            self._reject_status,
            f"A sentinel rejected the call to {call.function}: {why}",
        )

    async def _judge(self, held: Held, run: Run, step: Step) -> Decision | None:
        recorded = len(held.recorder.records)
        reviewed = len(held.attempt.reviews) if held.attempt is not None else 0
        # which models judged the step, for the eval: its roles don't reach here
        # each judge call's billed usage goes to the attempt as soon as the
        # call is done, so an eval asking while the step is judged sees it
        judging = _JUDGING.get()
        with noting_models(
            held.attempt.judging_usage if held.attempt is not None else None
        ) as models:
            started = time.monotonic()
            try:
                decision = await run_sentinel(
                    self._protocol,
                    HostContext(
                        host=RunHost(self._host, run.eval, self._asker(held)),
                        recorder=held.recorder,
                        store=held.store,
                        proxy=held.proxy,
                    ),
                    step,
                )
            finally:
                # The client waited on its model call while the step was judged, whether
                # or not the judging failed, until the call was answered "pending
                # review". The time a person takes, and any judging after that, is not
                # time the client spent waiting on the call.
                if held.attempt is not None:
                    ended = time.monotonic()
                    if judging is not None and judging.answered_pending_at is not None:
                        ended = min(ended, judging.answered_pending_at)
                    held.attempt.judging_time += max(0.0, ended - started)
        call = (
            step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else None
        )
        held.exchange.judged.append(
            Judged(
                stage(step),
                call,
                decision,
                held.recorder.records[recorded:],
                tuple(models.called),
                tuple(
                    review.id
                    for review in (
                        held.attempt.reviews[reviewed:]
                        if held.attempt is not None
                        else []
                    )
                ),
            )
        )
        return decision

    def _asker(self, held: Held) -> AskHuman | None:
        # A sentinel can ask a person only for a run that an Inspect eval names and
        # whose request passed the check of its header. The eval shows the question in
        # Inspect's approval UI. For any other run there is nobody to ask, and the
        # question is answered with caution. The same holds for every run when evals
        # aren't sent results or the host has its own way to ask a person.
        sample = held.proxy.inspect
        request_id = held.proxy.headers.get(REQUEST_ID)
        attempt = held.attempt
        if (
            not self.eval_review
            or not held.named
            or sample is None
            or request_id is None
            or attempt is None
        ):
            return None

        async def ask(
            step: Step,
            escalations: Escalations,
            choices: Sequence[str],
        ) -> HumanAnswer:
            judging = _JUDGING.get()
            assert judging is not None and judging.attempt is attempt
            # Keep what the try has judged so far, for `request_result`, in case its
            # judging is stopped here. The decisions on the earlier steps stay.
            attempt.judged = held.exchange
            if attempt.stop_cause is not None:
                # Nothing is waiting for this try, so it asks nobody. Its judging is
                # cancelled here, with nothing decided.
                judging.scope.cancel()
                await anyio.lowlevel.checkpoint()
            operation = self._operation_of(attempt)
            if operation is None and await self._superseded(judging):
                # A later try of the call was decided, and the client was given its
                # answer. So this try asks nobody, and is answered as a call that has
                # ended.
                judging.reattach_refusal = _OVER
                judging.scope.cancel()
                await anyio.lowlevel.checkpoint()
            if operation is None:
                # nothing is awaited from here until the operation is stored
                key = (sample.eval_id, request_id)
                reattach_refusal = self._refusal_if_reattached(judging)
                if reattach_refusal is not None:
                    # Another try of the call asked a person while this try was judged.
                    # The call is one operation, and that other try holds it. So this
                    # try is cancelled here and reattaches to the operation.
                    judging.reattach_refusal = reattach_refusal
                    judging.scope.cancel()
                    await anyio.lowlevel.checkpoint()
                # The call becomes one operation, bound to this request. It is stored
                # before anything is awaited, so that no other try can create a second
                # operation.
                operation = _Operation(
                    sample.eval_id,
                    request_id,
                    sample,
                    judging.binding,
                    attempt,
                    Pinned(
                        sample.eval_id,
                        sample.task,
                        held.run,
                        request_id,
                        (attempt.exchange,),
                    ),
                )
                operation.last_heard = time.monotonic()
                self._operations[key] = operation
                self._by_attempt[id(attempt)] = operation
                tries = await self._state.attempts(request_id, sample.eval_id)
                if isinstance(tries, list):
                    operation.tries = max(operation.tries, len(tries))
                    operation.current_try = next(
                        (
                            number
                            for number, tried in enumerate(tries, 1)
                            if tried is attempt
                        ),
                        operation.tries,
                    )
                    # `request_result` returns the records of every try of the request,
                    # so the records of earlier tries are pinned too
                    operation.pinned = replace(
                        operation.pinned,
                        exchanges=tuple(
                            dict.fromkeys(
                                [*(tried.exchange for tried in tries), attempt.exchange]
                            )
                        ),
                    )
                # a person may take days, so the state must not expire what the call
                # needs
                await self._state.pin(operation.pinned)
            asked = time.time()
            timeout = self._reviews.timeout
            call: ToolCall | None = None
            view: ToolCallView | None = None
            result: ChatMessageTool | None = None
            turn: ChatMessageAssistant | None = None
            sent: RequestAsked | None = None
            message = ""
            if isinstance(step, BeforeGenerate):
                # The request the model is about to be sent: the messages it
                # has not been sent before, and the tools it is offered.
                sent = RequestAsked(
                    step.model,
                    len(step.input),
                    tuple(not_sent_before(step.input, step.history)),
                    tuple(tool.name for tool in step.tools),
                    step.tool_choice,
                )
            elif isinstance(step, AfterGenerate):
                # the whole turn, as the model returned it, before any of its
                # calls is judged
                turn = step.output.message if step.output.choices else None
                message = turn.text if turn is not None else ""
            else:
                call, message, view = step.call, step.message, step.view
                if isinstance(step, AfterToolCall):
                    result = step.result
            question = Question(
                eval_id=sample.eval_id,
                sample_uuid=sample.sample_uuid,
                request_id=request_id,
                stage=stage(step),
                call=call,
                message=message,
                view=view,
                result=result,
                turn=turn,
                request=sent,
                escalations=escalated(escalations),
                choices=tuple(choices),
                closes=asked + timeout if timeout is not None else None,
                asked=asked,
            )
            # the review record stays with the try, whether or not the step reaches a
            # decision
            attempt.reviews.append(question.review)
            # The number of the try that asked is stored with the try, not with the
            # operation, because the operation is forgotten when its sample ends. The
            # question's record must name the try that asked for as long as the record
            # is kept.
            attempt.review_tries[question.id] = operation.current_try
            self._operation_by_question[question.id] = operation
            attempt.state = "awaiting_review"
            # A request that is waiting on this judging of a tool's result is answered
            # now, as this exchange is.
            for under_way in self._judging.values():
                if under_way.judging is judging:
                    under_way.asked = True
                    under_way.finished_or_asked.set()
            if not judging.answered_pending:
                # Answer the exchange "pending review" now, so that no connection waits
                # for the person.
                judging.answered_pending = True
                judging.answered_pending_at = time.monotonic()
                judging.answerable.set()
            # the time a question is open doesn't count against the time limit
            limit = judging.limit
            if judging.questions_waiting == 0 and limit is not None:
                if limit.deadline != math.inf:
                    judging.time_left = limit.deadline - anyio.current_time()
                    limit.deadline = math.inf
            judging.questions_waiting += 1
            try:
                answer = await self._reviews.ask(question)
            finally:
                self._operation_by_question.pop(question.id, None)
                judging.questions_waiting -= 1
                if judging.questions_waiting == 0:
                    if limit is not None and judging.time_left is not None:
                        limit.deadline = anyio.current_time() + judging.time_left
                        judging.time_left = None
                    if attempt.state == "awaiting_review":
                        attempt.state = "judging"
            if attempt.stop_cause is not None:
                # The call was closed at the moment its answer arrived, for example by a
                # `cancel_paused_request` in the same turn of the event loop. A closed
                # call decides nothing, so the judging is cancelled here and the answer
                # is not used, as when a call is closed before it asks.
                judging.scope.cancel()
                await anyio.lowlevel.checkpoint()
            return answer

        return ask

    def _judged(
        self,
        refusal: Refuse,
        proxy: ProxyContext,
        exchange: Exchange,
        attempt: Attempt | None,
    ) -> Refuse:
        # an Inspect eval asks what was decided, by `request_result`; a
        # request of a replaced attempt was never judged
        exchange.replaced = exchange.replaced or refusal is _REPLACED
        if proxy.inspect is not None and attempt is not None:
            attempt.judged = exchange
        return refusal

    def _ended_meanwhile(self, held: Held, run: Run) -> Refuse | None:
        # another exchange of the run may have ended it while this one waited
        refusal = _ended(run, held.marks, held.sample_attempt, held.request)
        if refusal is not None:
            held.exchange.ended = True
            held.exchange.replaced = refusal is _REPLACED
        return refusal

    def _end(self, held: Held, run: Run, where: str) -> Refuse:
        held.exchange.ended = True
        refusal = Refuse(400, f"A sentinel ended this run {where}.")
        run.in_a_row.pop(held.sample_attempt, None)
        if held.named:
            # a judgment that outlived its attempt ends only that attempt,
            # which nobody waits for
            if held.sample_attempt != run.attempt:
                return refusal
            run.refusal = refusal
            ended = held.proxy.inspect
            if ended is not None:
                self._stop(
                    "run_ended",
                    lambda other, attempt: (
                        attempt is not held.attempt
                        and (other.eval_id, other.sample_uuid)
                        == (ended.eval_id, ended.sample_uuid)
                    ),
                )
            # nothing of the run goes on, so nothing is put back again
            for conversation in run.conversations.values():
                conversation.rejected.clear()
                conversation.edited.clear()
            return refusal
        if held.answered:
            run.ended_at[held.mark] = refusal
            if held.sent > 0 and held.last is not None:
                run.ended_on.append(
                    EndedOn(
                        held.sent - 1, held.marks[held.sent - 1], held.last, refusal
                    )
                )
        # A run no eval names, ended before any reply in its
        # conversation was passed on, is not remembered as ended. Its
        # conversation is still only how it opens, which is all another
        # conversation with the same key has too, and nothing tells them apart.
        # Other conversations with the same key go on, and still need their
        # turns put back, or told from results. So the rejected turns where
        # this one ended are only no longer added to a request sent there.
        conversation = run.conversations.get(held.conversation)
        for point in conversation.rejected if conversation is not None else []:
            if (point.sent, point.mark) == (held.sent, held.mark):
                point.added = False
        return refusal

    def _out_of_time(self, what: str, id: str) -> TimeLimitError:
        return TimeLimitError(
            f"The sentinel didn't finish judging the {what} {id!r} within {self._time_limit} seconds."
        )


def _withhold_ended(
    held: Held,
    message: ChatMessageAssistant,
    ended_at: str | None = None,
    rejected: Sequence[tuple[ToolCall, Decision]] = (),
    modified: Mapping[str, ToolCall] | None = None,
) -> None:
    # The client records the turn the run ended at, as an in-process sentinel
    # leaves it in the transcript. None of its calls runs, so a changed call
    # isn't shown as it ran. Every call a decision acted on is redacted: the
    # one the run ended at, and those rejected or changed before it. A run
    # ended at the reply had every call acted on. A call judged to proceed, or
    # never judged, keeps its arguments.
    calls = message.tool_calls or []
    acted_on: list[str]
    if ended_at is None:
        acted_on = [call.id for call in calls]
    else:
        acted_on = [ended_at, *(call.id for call, _ in rejected), *(modified or {})]
    redacted = frozenset(acted_on)
    held.exchange.withheld = Withheld(
        message, {call.id: _ENDED for call in calls}, redacted=redacted
    )


def _run_key(named: bool, key: str) -> str:
    # Runs an eval names and runs it doesn't are kept apart. A run no eval
    # names is keyed by how its conversation opens, which anyone who knows the
    # opening can work out, so the eval endpoint, which takes a sample's uuid,
    # must never reach one.
    return f"{'sample' if named else 'opening'}:{key}"


def _stopped_answer(attempt: Attempt) -> Refuse:
    # the refusal for an exchange whose judging was stopped
    if attempt.stop_cause == "replaced_attempt":
        return _REPLACED
    if attempt.stop_cause == "cancelled":
        return _CANCELLED
    return _RUN_OVER


def _settle_stopped(attempt: Attempt) -> None:
    # Settle an attempt that nothing is waiting for any more. It is settled once, with
    # nothing decided for it. If its run ended, it is reported as a request of an ended
    # run is.
    if attempt.stop_cause is None or attempt.settled.is_set():
        return
    if attempt.stop_cause in ("replaced_attempt", "cancelled", "expired"):
        attempt.not_judged(attempt.stop_cause)
    else:
        # Report the attempt as a request of an ended run is reported: the run was
        # ended, by the eval or by a sentinel at another request. The judgment says so,
        # and holds the steps this try had judged before it was stopped. No decision is
        # invented for the step that was waiting.
        if attempt.judged is None:
            attempt.judged = Exchange()
        attempt.judged.ended = True
        attempt.decide("terminate")


_NOT_BOUND = frozenset({INSPECT_CONTEXT, "x-stainless-retry-count"})
"""The headers left out of a request's binding, because each try has its own values for them: Inspect's context, whose `timestamp` and `mac` are made afresh for each try, and the retry count that a provider's SDK adds to its own retries."""


def _binding(request: Request, sample: InspectSample) -> str:
    # A digest of what identifies a request, used to tell a try of a model call from a
    # different request sent under the same `x-irid`. It covers everything the proxy
    # reported about the request that a sentinel can read, or that says how the request
    # is read. A resend is not judged again, so the request that is passed on must be
    # the one the approval was given for. Left out are the header's `timestamp` and
    # `mac`, which are made afresh for each try and checked on each; the proxy's id for
    # the exchange; and a caching hint that changes with the time between tries
    # (`bound_body`).
    try:
        format = request_format(request.provider, request.body, request.format)
    except UnreadableError:
        # an unreadable request is never judged, so it never starts an operation
        format = None
    headers = sorted(
        (name, value)
        for name, value in request.headers.items()
        if name not in _NOT_BOUND
    )
    return hashlib.sha256(
        json.dumps(
            [
                request.provider,
                format,
                request.model,
                request.user,
                headers,
                asdict(sample),
                hashlib.sha256(
                    bound_body(request.provider, request.body, request.format)
                ).hexdigest(),
            ],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _served_error(failure: Failure) -> JudgedReply:
    # A provider's error as the judged-reply path returns it, so that the eval's code for
    # that provider reads it as the provider's own error. The `FAILED_TRY` header marks
    # it with the number of the failed try, which no refusal by the sidecar carries.
    # `x-should-retry: false` tells the eval's SDK not to resend the request by itself,
    # because it would only fetch this error again. Whether the provider is called again
    # is for the eval's own retry rules to decide. A wait the provider asked for is
    # counted from when the error arrived, not from when it is fetched.
    assert failure.reply is not None
    waited = time.monotonic() - failure.received
    headers = {
        name: _time_left(name, value, waited)
        for name, value in failure.reply.headers.items()
    }
    headers[FAILED_TRY] = str(failure.try_number)
    headers["x-should-retry"] = "false"
    return replace(failure.reply, headers=headers)


def _time_left(name: str, value: str, waited: float) -> str:
    # The time left of a wait given in seconds (`retry-after`) or milliseconds
    # (`retry-after-ms`). A date, or any other value, is returned as it was sent.
    scale = {"retry-after": 1.0, "retry-after-ms": 1000.0}.get(name)
    if scale is None:
        return value
    try:
        asked = float(value)
    except ValueError:
        return value
    if not math.isfinite(asked) or asked < 0:
        return value
    return str(math.ceil(max(0.0, asked - waited * scale)))


def _judged_reply(reply: Reply, format: Format, answer: Answer) -> JudgedReply:
    # The reply as the proxy passes it on, kept so that the eval can fetch it and read
    # it as the provider's own reply.
    headers = {
        name: value for name, value in reply.headers.items() if name not in _NOT_KEPT
    }
    headers.setdefault("content-type", reply.content_type)
    return JudgedReply(
        reply.status,
        headers,
        answer.body if isinstance(answer, Replace) else reply.body,
        format,
        "event-stream" in reply.content_type.lower(),
    )


def _begin_attempt(run: Run, sample: InspectSample) -> Refuse | None:
    # A later attempt at the sample starts the run afresh: its model never saw
    # the last attempt's turns, and in Inspect's own process its sample store
    # starts empty. An earlier attempt's request is refused by `_ended`, and
    # another attempt with the same number is refused here, as which of them
    # is the sample's can't be told.
    if run.closed:
        return None
    if sample.attempt_number > run.attempt_number:
        run.attempt = sample.attempt
        run.attempt_number = sample.attempt_number
        run.store = Store()
        run.refusal = None
    elif sample.attempt_number == run.attempt_number and sample.attempt != run.attempt:
        return Refuse(
            400,
            f"Another attempt at this sample has the number {sample.attempt_number}.",
        )
    return None


def _ended(
    run: Run, marks: Sequence[str], attempt: str, request: Request | None = None
) -> Refuse | None:
    # what a request is refused with, if its run has ended: any request of a
    # run an eval names, or one that continues the conversation it was ended
    # in; and any request or reply of an attempt a later one replaced
    if not run.closed and attempt != run.attempt:
        return _REPLACED
    refusal = run.refusal or next(
        (run.ended_at[mark] for mark in marks if mark in run.ended_at), None
    )
    if refusal is not None or request is None:
        return refusal

    # A client may add the user's next message to the entry the ended
    # request ended with, as Claude Code adds it beside the last results,
    # rather than send it as an entry of its own. That request continues the
    # ended conversation too. The request is read once, whatever the run was
    # ended at.
    if not run.ended_on:
        return None
    digests = entry_digests(request.provider, request.body, request.format)
    for ended in run.ended_on:
        if ended.at >= len(marks) or marks[ended.at] != ended.before:
            continue
        if ended.at < len(digests) and (now := digests[ended.at]) is not None:
            if grows(now, ended.digest):
                return ended.refusal
    return None


def _ends(decision: Decision | None) -> bool:
    # an escalate nobody took has already been turned into a terminate
    return decision is not None and decision.action == "terminate"


def _model(provider: str, model: str) -> str:
    # a step names the model as `provider/name`
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


def _step_model(provider: str, model: str, sample: InspectSample | None) -> str:
    # Inspect's own name for the model, as in its process, where an Inspect
    # client gives one that ends in the model the request names: it can add
    # what precedes the name, such as `openai-api/together/`, and not name
    # another model
    claimed = sample.model if sample is not None else None
    if claimed is not None and model and claimed.endswith(f"/{model}"):
        return claimed
    return _model(provider, model)


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


def _edit_lost(
    run: Run, edited: Sequence[Edited], messages: Sequence[ChatMessage]
) -> bool:
    # whether the messages hold a call id of an edited turn of the run that
    # isn't among those the request continues from
    ids = _calls(messages) | {
        message.tool_call_id
        for message in messages
        if isinstance(message, ChatMessageTool) and message.tool_call_id is not None
    }
    return any(
        turn not in edited and ids.intersection(turn.order)
        for conversation in run.conversations.values()
        for turn in conversation.edited
    )


def _calls(messages: Sequence[ChatMessage]) -> set[str]:
    # the ids of the tool calls the messages hold. Separate calls have
    # separate ids, so a conversation that holds two calls with one id can't
    # be read: which of them a result is of can't be told.
    calls: set[str] = set()
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            for call in message.tool_calls or []:
                if call.id in calls:
                    raise UnreadableError(
                        f"The request holds two tool calls with the id {call.id!r}."
                    )
                calls.add(call.id)
    return calls


def _result_digest(said: str, call: ToolCall, result: ChatMessageTool) -> str:
    # everything the step is given from the request, what the model said with
    # the call as well as the call and the result: the client wrote them all
    # into the request, so an id seen before doesn't vouch for what now
    # stands beside it
    digest = hashlib.sha256(
        json.dumps(
            [
                said,
                call.id,
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
