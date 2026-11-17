"""Questions that a sentinel in the sidecar asks a person, held until an Inspect eval answers them.

A sentinel asks a person with `human()`, which calls `Host.ask_human`. For a run that an Inspect eval names, the sidecar holds the question here while the step waits. A question is about one tool call, before it runs or after, about a model's whole turn before any of its calls runs, or about the request a model is about to be sent. The eval lists the question with `list_pending_questions`, shows it in Inspect's approval UI, and sends the person's answer with `answer_question`.

A person may take days to answer, so no connection waits on a question. The handler answers the model call with a "pending review" error and keeps the judging that asked the question. That judging continues when the answer arrives (`design/sidecar.md`, "A review pauses the model call, not a connection"). A question has no expiry unless the operator sets one.

This first way of asking has two temporary exceptions to the sidecar's rules, both in `design/sidecar-plan.md`, step 13:

- The eval signs the answer with its own key, the token it signs every call with. The sidecar accepts an answer only for a question from one of that eval's own runs. Later, reviewers are to answer with credentials of their own, which the sidecar checks.
- The question gives the eval's process the call's arguments, the whole turn, or the messages of the request that were not sent before, even if the person then rejects the call or ends the run. Later, a reviewer surface outside the eval's process is to show the arguments instead.

A question belongs to one model call. A `Review` records what happened to the question. The handler keeps the `Review` with the try of the call that asked, whether or not the step reached a decision.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import anyio
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ContentReasoning,
)
from inspect_ai.tool import ToolCall, ToolCallView, ToolChoice

from .._host import HumanAnswer
from .._report import Escalations
from ._host import Stage, about

logger = logging.getLogger(__name__)

MAX_QUESTIONS_WAIT = 30.0
"""The longest time that `list_pending_questions` waits for a question, in seconds."""

ANSWERABLE = ("approve", "reject", "terminate")
"""The choices that an answer sent through the endpoint for evals may pick. `modify` is left out because it needs a changed call, which Inspect's approval UI can't produce yet."""

_KEPT_RECEIPTS = 10_000
"""How many receipts of answers the sidecar keeps. With a receipt, the sidecar can tell an eval that sends an answer again that the answer was already accepted."""

Cause: TypeAlias = Literal[
    "replaced_attempt",
    "sample_ended",
    "run_ended",
    "cancelled",
    "stopped",
]
"""Why a question's wait was cancelled: a later attempt at its sample began, the eval ended its sample, a sentinel ended its run, the eval cancelled the model call, or the sidecar stopped judging for another reason, as when it shuts down."""


class UnknownQuestionError(Exception):
    """The sidecar holds no open question, and no receipt of an answer, with this id for this eval. The question never existed, belonged to another eval, closed unanswered, or its sample has ended."""


class AnsweredError(Exception):
    """The question was already answered, with another choice or reason."""


class ChoiceError(Exception):
    """The answer picks a choice that the question didn't offer, or one that the endpoint for evals doesn't accept."""


@dataclass(frozen=True)
class Escalated:
    """An escalation that was passed to the protocol that asked, in the form a person is shown."""

    name: str
    """The escalating instance's name."""

    explanation: str | None
    """Why it escalated, if it said."""


@dataclass
class Review:
    """The record of one question: when it was asked, what the eval answered, and how the wait ended. The handler keeps it with the HTTP try that asked.

    It holds the call's id, and none of the call's arguments, view or result, nor any of a turn or a request. The answer is recorded when the sidecar accepts it, separately from how the wait ended. So a try that is stopped just after its answer was accepted keeps the answer on its record, though the answer had no effect.
    """

    id: str
    """The question's id."""

    stage: Stage
    """The stage at which the step was judged."""

    call: str | None
    """The id of the call the question was about. None for a question at a generate stage, which is about a whole turn or a request."""

    asked: float
    """When it was asked, in seconds since the epoch."""

    decision: str | None = None
    """The choice in the eval's answer, if the sidecar accepted one. The eval's signature shows that the eval sent the answer, not that a person chose it."""

    reason: str | None = None
    """The reason the eval's answer gave, if any."""

    answered: float | None = None
    """When the sidecar accepted the answer, in seconds since the epoch."""

    resolution: Literal["answered", "timed_out", "cancelled"] | None = None
    """How the wait ended, or None while the step still waits.

    - `answered`: the sidecar accepted the eval's answer and gave it to the sentinel.
    - `timed_out`: nobody answered in time, so the sidecar gave the step the cautious answer.
    - `cancelled`: the step stopped with no decision. `cause` says why.
    """

    cause: Cause | None = None
    """Why the wait was cancelled, when it was."""

    resolved: float | None = None
    """When the wait ended, in seconds since the epoch."""

    def resolve(
        self, resolution: Literal["answered", "timed_out", "cancelled"]
    ) -> None:
        """Record how the wait ended. Only the first call has an effect."""
        if self.resolution is None:
            self.resolution, self.resolved = resolution, time.time()
            if resolution == "cancelled" and self.cause is None:
                self.cause = "stopped"


@dataclass(frozen=True)
class RequestAsked:
    """What a question before a generate shows of the request the model is about to be sent."""

    model: str
    """The model the request is for."""

    messages: int
    """How many messages the request sends."""

    new: tuple[tuple[int, ChatMessage], ...]
    """The messages the model has not been sent before, each with its position in the request, counting from 1. A message that was rewritten since it was sent is among them."""

    tools: tuple[str, ...]
    """The names of the tools the request offers the model."""

    tool_choice: ToolChoice
    """Which tool the model is to call: `auto`, `any` or `none`, or the one function it must call. A function is kept apart from the three words, since a tool can be named one of them."""


@dataclass
class Question:
    """A question that a sentinel asked a person, for a run that an Inspect eval names. The sidecar holds it only while the step waits."""

    eval_id: str
    """The eval whose run asked the question. Only this eval may see and answer it."""

    sample_uuid: str
    """The run's sample, by the uuid the eval gave it."""

    request_id: str
    """The `x-irid` of the model request in which the step was found. The eval assigns it."""

    stage: Stage
    """The stage at which the step was judged."""

    call: ToolCall | None
    """The call the person decides about, as it will run. None at a generate stage, where the question is about a whole turn or a request."""

    message: str
    """The assistant's text: alongside the call, or the turn's own at `model_output`. Empty at `model_input`."""

    view: ToolCallView | None
    """How the call is shown to the person. None at a generate stage."""

    result: ChatMessageTool | None
    """The call's result, when the stage is `tool_result`."""

    turn: ChatMessageAssistant | None
    """The model's turn, none of which the eval has been given yet. At `model_output` it is what the person decides about: its text, its reasoning and every tool call it makes, each with the call part of its view, before any of them runs. At `tool_call` it is the turn that made the call, without its tool calls. None at `model_input` and `tool_result`, and for a reply that holds no turn. It is a copy for showing (see `shown_turn`)."""

    request: RequestAsked | None
    """The request that the person decides about, at `model_input`, before it is sent to the provider. None at every other stage."""

    escalations: tuple[Escalated, ...]
    """The escalations that were passed to the protocol that asked, in order."""

    choices: tuple[str, ...]
    """What the person may pick."""

    closes: float | None
    """When the question expires, in seconds since the epoch. At that time the sidecar gives the step the cautious answer. None if the operator set no review timeout: the question then waits for as long as its model call does."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """The question's id, which an answer names."""

    asked: float = field(default_factory=time.time)
    """When it was asked, in seconds since the epoch."""

    answer: HumanAnswer | None = None
    """The eval's answer, once the sidecar has accepted it."""

    answer_taken: anyio.Event = field(default_factory=anyio.Event)
    """Set once the question is answered."""

    review: Review = field(init=False)
    """The record of the question. The record is kept after the question itself is dropped."""

    def __post_init__(self) -> None:
        self.review = Review(
            self.id,
            self.stage,
            self.call.id if self.call is not None else None,
            self.asked,
        )

    @property
    def about(self) -> str:
        """What the question is about, in words for a log line."""
        return about(self.stage, self.call.function if self.call is not None else None)


def shown_turn(
    turn: ChatMessageAssistant, calls: Sequence[ToolCall] | None
) -> ChatMessageAssistant:
    """The copy of a model's turn that a question carries, for showing to a person.

    The copy leaves out what only a provider reads. A reasoning block loses its signature and its `internal` value, and one whose text the provider withheld loses the cipher text it holds in place of the text. A person can read none of these, and the eval gets them in the judged reply, which is the turn it sends back to the provider.

    Args:
        turn: The turn, as the model returned it. It is not changed.
        calls: The tool calls the copy has, or None for none.
    """
    content = turn.content
    if not isinstance(content, str):
        content = [
            block.model_copy(
                update={
                    "signature": None,
                    "internal": None,
                    **({"reasoning": ""} if block.redacted else {}),
                }
            )
            if isinstance(block, ContentReasoning)
            else block
            for block in content
        ]
    return turn.model_copy(
        update={"content": content, "tool_calls": list(calls) if calls else None}
    )


@dataclass(frozen=True)
class _Receipt:
    """A record that a question was answered, and with what. It is all that the sidecar keeps of a question once its step stops waiting."""

    eval_id: str
    sample_uuid: str
    decision: str
    reason: str | None


def escalated(escalations: Escalations) -> tuple[Escalated, ...]:
    """The escalations, in the form a person is shown."""
    return tuple(
        Escalated(reported.name, reported.report.explanation)
        for reported in escalations
    )


def cautious(choices: Sequence[str], reason: str) -> HumanAnswer:
    """The answer the sidecar gives when no person answered: `reject` if the question offered it, otherwise `terminate`. Either way the step fails closed."""
    return HumanAnswer(
        decision="reject" if "reject" in choices else "terminate", reason=reason
    )


class Reviews:
    """The questions the sidecar holds for Inspect evals to answer. They are kept in this process, because each step waits on its question here."""

    def __init__(self, timeout: float | None = None) -> None:
        """Create a store that holds no questions yet.

        Args:
            timeout: How long a question waits for an answer, in seconds, before it expires and the sidecar gives its step the cautious answer. None, the default, means no expiry.

        Raises:
            ValueError: If `timeout` isn't a positive, finite number of seconds.
        """
        # None means no expiry. An infinite timeout is refused, because each
        # question tells the eval when it closes, and that time must be finite.
        if timeout is not None and not (timeout > 0 and math.isfinite(timeout)):
            raise ValueError(
                f"The review timeout must be a positive, finite number of seconds, not {timeout}."
            )
        self._timeout = timeout
        self._open: dict[str, Question] = {}
        self._receipts: OrderedDict[str, _Receipt] = OrderedDict()
        # Set and replaced each time a question is asked, so that a waiting
        # `list_pending` wakes at once.
        self._asked = anyio.Event()

    @property
    def timeout(self) -> float | None:
        """How long a question waits for an answer before it expires, in seconds, or None for no expiry."""
        return self._timeout

    async def ask(self, question: Question) -> HumanAnswer:
        """Hold the question until an eval answers it or it expires.

        The handler can cancel the wait by cancelling the judging that the wait is part of. It does so when nobody is left to act on an answer, for example when the sample ends. `close` removes the question at once, and the step then reaches no decision.

        Returns:
            The eval's answer. If the question expired, the cautious answer: `reject` if offered, otherwise `terminate`.
        """
        review = question.review
        self._open[question.id] = question
        self._asked.set()
        self._asked = anyio.Event()
        logger.info(
            "Asked a person about %s at %s, as question %s of the eval %s.",
            question.about,
            question.stage,
            question.id,
            question.eval_id,
        )
        try:
            with anyio.move_on_after(self._timeout):
                await question.answer_taken.wait()
        except anyio.get_cancelled_exc_class():
            # an answer accepted just before the cancellation stays on the
            # record, but has no effect
            review.resolve("cancelled")
            logger.info("Question %s was cancelled (%s).", question.id, review.cause)
            raise
        finally:
            # Remove the question when its step stops waiting, so that the
            # sidecar accepts no later answer for a step that is gone.
            self._open.pop(question.id, None)
        if question.answer is not None:
            review.resolve("answered")
            return question.answer
        review.resolve("timed_out")
        logger.warning(
            "Question %s expired with no answer, so the sidecar gave the %s stage of %s the cautious answer.",
            question.id,
            question.stage,
            question.about,
        )
        return cautious(
            question.choices,
            "Nobody answered before the question expired, so the sidecar gave the cautious answer.",
        )

    async def list_pending(
        self, eval_id: str, sample_uuid: str | None = None, wait: float = 0
    ) -> list[Question]:
        """The eval's unanswered questions, oldest first.

        Args:
            eval_id: The eval that asks. It is never shown another eval's questions.
            sample_uuid: If given, list only this sample's questions.
            wait: How long to wait for a question when there are none, in seconds, at most `MAX_QUESTIONS_WAIT`. The method returns as soon as there is one.
        """
        deadline = time.monotonic() + min(max(wait, 0), MAX_QUESTIONS_WAIT)
        while True:
            found = sorted(
                (
                    question
                    for question in self._open.values()
                    if question.eval_id == eval_id
                    and question.answer is None
                    and sample_uuid in (None, question.sample_uuid)
                ),
                key=lambda question: question.asked,
            )
            left = deadline - time.monotonic()
            if found or left <= 0:
                return found
            asked = self._asked
            with anyio.move_on_after(left):
                await asked.wait()

    def answer(self, eval_id: str, id: str, decision: str, reason: str | None) -> None:
        """Accept an eval's answer to one of its questions. A question takes one answer. The same answer sent again, with the same choice and reason, is accepted and changes nothing.

        The answer goes on the question's record before this returns. So the record holds every answer the sidecar acknowledged, including one whose step was stopped before the answer had an effect.

        Args:
            eval_id: The eval that answers. To it, another eval's question is unknown.
            id: The question's id.
            decision: The choice the answer picks.
            reason: The reason given, if any.

        Raises:
            UnknownQuestionError: If the eval has no open question with the id, and no receipt of an answer to it.
            AnsweredError: If the question was already answered with another choice or reason.
            ChoiceError: If the question didn't offer `decision`, or `decision` is not in `ANSWERABLE`.
        """
        receipt = self._receipts.get(id)
        if receipt is not None and receipt.eval_id == eval_id:
            if (receipt.decision, receipt.reason) == (decision, reason):
                return
            raise AnsweredError(
                f"The question {id!r} was already answered {receipt.decision!r}."
            )
        question = self._open.get(id)
        if receipt is not None or question is None or question.eval_id != eval_id:
            raise UnknownQuestionError(
                f"The sidecar holds no question {id!r} for the eval {eval_id!r}. It may have closed without an answer, or its sample may have ended."
            )
        if question.answer is not None:
            # Already answered, and the receipt has since been dropped. The
            # first answer is kept.
            if (question.answer.decision, question.answer.reason) == (decision, reason):
                return
            raise AnsweredError(
                f"The question {id!r} was already answered {question.answer.decision!r}."
            )
        if decision not in question.choices:
            raise ChoiceError(
                f"The question {id!r} offers {', '.join(question.choices)}, not {decision!r}."
            )
        if decision not in ANSWERABLE:
            raise ChoiceError(
                f"An answer sent through the endpoint for evals must be one of {', '.join(ANSWERABLE)}, not {decision!r}."
            )
        review = question.review
        review.decision, review.reason, review.answered = decision, reason, time.time()
        question.answer = HumanAnswer(decision=decision, reason=reason)
        question.answer_taken.set()
        self._receipts[id] = _Receipt(eval_id, question.sample_uuid, decision, reason)
        while len(self._receipts) > _KEPT_RECEIPTS:
            self._receipts.popitem(last=False)
        logger.info("The eval %s answered question %s: %s.", eval_id, id, decision)

    def close(self, reviews: Sequence[Review], cause: Cause) -> None:
        """Close the questions of these reviews now, and record them as cancelled for `cause`. From then on `list_pending` lists none of them, and an answer to one raises `UnknownQuestionError`, even if the step that waits on it has not stopped waiting yet."""
        for review in reviews:
            self._open.pop(review.id, None)
            if review.resolution is None:
                review.cause = cause
                review.resolve("cancelled")

    def forget_sample(self, eval_id: str, sample_uuid: str) -> None:
        """Drop the receipts of a sample's answers. The handler calls this once the eval has ended the sample. After it, an answer to one of the sample's questions that is sent again raises `UnknownQuestionError`."""
        for id in [
            id
            for id, receipt in self._receipts.items()
            if (receipt.eval_id, receipt.sample_uuid) == (eval_id, sample_uuid)
        ]:
            del self._receipts[id]
