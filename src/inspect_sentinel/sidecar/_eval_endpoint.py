from __future__ import annotations

import base64
import hmac
import json
from pathlib import Path
from typing import Any, Literal

from inspect_ai.tool import ToolCall, ToolFunction
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ._errors import UnauthorizedError
from ._exchange import Refuse
from ._handler import MAX_RESULT_WAIT, Handler, JudgedReply, RequestResult
from ._http import discard_within, read_within
from ._inspect_client import (
    JudgingUsage,
    Judgment,
    judging_usage,
    judgment,
    record_json,
)
from ._json_rpc import (
    ANSWER_REFUSED,
    METHOD_NOT_FOUND,
    QUESTION_ANSWERED,
    REGISTRATION_REFUSED,
    REGISTRATIONS_FULL,
    UNAUTHORIZED,
    UNKNOWN_QUESTION,
    UNKNOWN_SAMPLE,
    UNKNOWN_TASK,
    RpcError,
    answered,
    failed,
    invalid_params,
    read_call,
)
from ._registration import (
    RegistrationError,
    RegistrationsFullError,
    SampleFacts,
    TaskRegistration,
    UnknownTaskError,
)
from ._review import (
    ANSWERABLE,
    MAX_QUESTIONS_WAIT,
    AnsweredError,
    ChoiceError,
    Question,
    UnknownQuestionError,
)
from ._sign import MIN_TOKEN_LENGTH, answer_values
from ._state import NotJudged, Outcome

PATH = "/sentinel/v1"
"""Where the methods are served. The `v1` is the version of the methods."""

JUDGED_REPLY_PATH = f"{PATH}/judged-reply"
"""The path from which an eval fetches the provider's reply that the sidecar holds for a model call paused for a person. It is not a JSON-RPC method. The eval sends the call's own model request here, with its code for that provider, and the sidecar answers as the provider would."""

INFO_PATH = "/sentinel/info"
"""The probe, which has no version, since it is how an eval learns which versions there are."""

VERSIONS = [1]
"""The versions of the methods this sidecar serves."""

TOKEN_VARIABLE = "INSPECT_SENTINEL_EVAL_TOKEN"
"""The environment variable the endpoint's own token is read from, when it has a port of its own."""


_TIMESTAMP = "When the caller signed, in whole seconds since the epoch. A call whose timestamp is further from the sidecar's clock than its window, five minutes by default, is refused with `unauthorized`."

_MAC = 'An HMAC-SHA256, in hex, made with the eval\'s token over `json.dumps([str(v) for v in [method, eval_id, timestamp]], separators=(",", ":"))`, such as `["sample_result","eval_789","1794182400"]`. It proves only that the caller holds the token, not the call\'s other parameters.'


class EvalCall(BaseModel):
    """What every call for an eval presents, but a registration that sends the eval's token: the eval, and proof that the caller holds its token."""

    eval_id: str = Field(description="The eval's id, as its requests name it.")
    timestamp: int = Field(description=_TIMESTAMP)
    mac: str = Field(
        description=f"{_MAC} A call whose `mac` isn't made with the token the sidecar keeps for the eval is refused with `unauthorized`. A sidecar that keeps no token for the eval, as after a restart, holds nothing of it for anyone."
    )


class RegisterTaskParams(TaskRegistration):
    """Parameters of `register_task`: a task an eval runs, and the ids of its samples, sent at task init. It carries either the eval's `token`, or a `timestamp` and a `mac`, and not both."""

    eval_id: str = Field(description="The eval's id, as its requests name it.")
    token: str | None = Field(
        default=None,
        min_length=MIN_TOKEN_LENGTH,
        description=f"The token the eval made for this sidecar, at least {MIN_TOKEN_LENGTH} characters, sent only to a sidecar that holds no token for the eval: at the eval's first registration, when that registration is sent again because its answer was lost, or after the sidecar lost the token in a restart or once it expired. The sidecar keeps it once the registration succeeds, and never sends it back. A sidecar that keeps another token for the eval answers `unauthorized`.",
    )
    timestamp: int | None = Field(default=None, description=_TIMESTAMP)
    mac: str | None = Field(
        default=None,
        description=f"{_MAC} Sent in place of the token once the sidecar holds it. A sidecar that holds no token for the eval answers `unknown_task`, and the eval registers again with its token.",
    )
    expiry: float | None = Field(
        default=None,
        gt=0,
        description="How long to keep the registration once no request has named the task, in seconds. A day if not given.",
    )

    @model_validator(mode="after")
    def _token_or_mac(self) -> RegisterTaskParams:
        partly_signed = (self.timestamp is None) != (self.mac is None)
        if partly_signed or (self.token is None) == (self.mac is None):
            raise ValueError(
                "A registration carries the eval's token, or a timestamp and a mac, and not both."
            )
        return self


class RegisterTaskResult(BaseModel):
    """What the sidecar stored, or that the task was registered beforehand and nothing was stored."""

    eval_id: str
    task: str
    samples: int = Field(description="How many samples the registration named.")
    fixed: bool = Field(
        description="Whether the task was registered beforehand, in the tasks folder. Its sentinel is then given the folder's facts, the registration is not compared with them, and nothing is stored."
    )
    expiry: float | None = Field(
        default=None,
        description="How long the registration is kept once no request has named the task, in seconds: what was asked, or less if the sidecar keeps none that long. Absent when `fixed`, as a task registered beforehand is never dropped.",
    )


class CompleteRegistrationParams(SampleFacts, EvalCall):
    """Parameters of `complete_registration`: one sample of a registered task, sent before the sample's first epoch."""

    task: str = Field(description="The task's name, as `register_task` gave it.")


class CompleteRegistrationResult(BaseModel):
    """That the sample's facts were stored, or that the task was registered beforehand and nothing was stored."""

    eval_id: str
    task: str
    sample_id: str | int
    fixed: bool = Field(
        description="Whether the task was registered beforehand, in the tasks folder. Its sentinel is then given the folder's facts, the sample is not compared with them, and nothing is stored."
    )


class RequestResultParams(EvalCall):
    """Parameters of `request_result`: one model request, by the eval's id for it."""

    id: str = Field(description="The request's `x-irid`.")
    wait: float = Field(
        default=0,
        ge=0,
        description=f"How long to wait, in seconds, while the sentinel is still judging the request, at most {MAX_RESULT_WAIT:g}. The sidecar answers as soon as the state is no longer `judging`. If the state is still `judging`, the eval calls again.",
    )


class ReviewRecord(BaseModel):
    """The record of one question that a sentinel asked a person while it judged the request or its reply."""

    id: str = Field(
        description="The question's id, as `list_pending_questions` returned it."
    )
    try_number: int = Field(
        description="The number of the try at which the sentinel asked the question, counting from 1 in the order the sidecar received the tries. It is the try whose request or reply the sentinel was judging. A client that retries a request sends it again under the same `x-irid`."
    )
    stage: Literal["tool_call", "tool_result", "model_input", "model_output"] = Field(
        description="The stage at which the sentinel asked: `tool_call` before the call runs, `tool_result` after it has run, `model_input` about the request before the model is sent it, `model_output` about the model's whole turn, before any of its calls runs."
    )
    call: str | None = Field(
        default=None,
        description="The id of the tool call the question was about. Left out at a generate stage, where the question was about a request or a whole turn.",
    )
    asked: float = Field(
        description="When the sentinel asked the question, in seconds since the epoch."
    )
    decision: Literal["approve", "reject", "terminate"] | None = Field(
        default=None,
        description="The choice in the eval's answer, if the sidecar accepted an answer. The answer's `mac` proves that the eval sent it, not that a person chose it.",
    )
    reason: str | None = Field(
        default=None, description="The reason in the eval's answer, if it gave one."
    )
    answered: float | None = Field(
        default=None,
        description="When the sidecar accepted the answer, in seconds since the epoch.",
    )
    resolution: Literal["answered", "timed_out", "cancelled"] | None = Field(
        default=None,
        description="How the wait for an answer ended. `answered`: the sidecar gave the sentinel the eval's answer. `timed_out`: the question expired, because the operator set a review timeout, and the sidecar gave the sentinel the cautious answer: `reject` if the question offers it, otherwise `terminate`. `cancelled`: the sidecar stopped the wait and the step reached no decision, so an answer accepted just before had no effect. Left out while the step is still waiting.",
    )
    cause: (
        Literal[
            "replaced_attempt",
            "sample_ended",
            "run_ended",
            "cancelled",
            "stopped",
        ]
        | None
    ) = Field(
        default=None,
        description="Why the sidecar cancelled the wait. `replaced_attempt`: a later attempt at the sample began. `sample_ended`: the eval ended the sample. `run_ended`: a sentinel ended the run at another request. `cancelled`: the eval cancelled the request with `cancel_paused_request`. `stopped`: anything else, such as the sidecar shutting down. Present only when `resolution` is `cancelled`.",
    )
    resolved: float | None = Field(
        default=None, description="When the wait ended, in seconds since the epoch."
    )


class ReplyRecord(BaseModel):
    """A provider's reply as the proxy passes it on. The eval reads it with the same code that reads that provider's replies from the network."""

    status: int = Field(description="The provider's HTTP status.")
    headers: dict[str, str] = Field(
        description="The provider's headers, by lowercase name, without hop-by-hop headers such as `content-length`. It holds only `content-type` if the proxy gave the sidecar no reply headers."
    )
    body: str = Field(
        description="The body, as base64: a whole JSON reply, or a streamed reply's events from first to last. It includes the sentinel's edits. Empty for a provider's error that was too large to keep."
    )
    format: Literal["anthropic-messages", "openai-chat", "openai-responses"] = Field(
        description="The API format of the request, which is also the format of the reply."
    )
    stream: bool = Field(description="Whether the body is a stream of events.")


class FailureRecord(BaseModel):
    """The provider's most recent failure, for a request that is paused for a person. The eval acts on this record, not on the error that its own request returned, because that error may be the sidecar's `pending review`."""

    try_number: int = Field(
        description="The number of the try that the proxy passed on to the provider and that failed, counting from 1. The eval uses it to tell a new failure from one it has already handled."
    )
    reply: ReplyRecord | None = Field(
        default=None,
        description="The provider's error. Left out if the provider's reply was cut off, which the eval treats as a failed connection.",
    )


class RequestResultResult(BaseModel):
    """What happened to the request. With `dropped` or `unknown` records, nothing else is said."""

    id: str
    records: Literal["held", "dropped", "unknown"] = Field(
        description="`held` if the sidecar has the request's records, `dropped` if it had them and no longer keeps them, `unknown` if it never kept the request: one it never saw, one without an `x-inspect-context` it could read and check, or one whose `x-inspect-context` named another eval."
    )
    state: (
        Literal[
            "judging",
            "awaiting_review",
            "awaiting_resend",
            "awaiting_reply",
            "decided",
            "not_judged",
        ]
        | None
    ) = Field(
        default=None,
        description="The state of the request. For a request no person was asked about, it is the state of the latest try. For a request paused for a person, it is the state of the whole model call, however many tries the eval has sent. `awaiting_review`: a question is open, so the eval shows it to a person and sends the answer with `answer_question`. `awaiting_resend`: the provider still has to be called, so the eval resends the same request under the same `x-irid`, and the proxy passes it on to the provider.",
    )
    reason: NotJudged | None = Field(
        default=None,
        description="With `not_judged`: why the sentinel did not judge the request. With `awaiting_resend`: `provider_error` or `cut_off` if the provider failed the last try, which `failure` describes. Left out if the request is only waiting for the eval to resend it.",
    )
    outcome: Outcome | None = Field(
        default=None,
        description="The sentinel's final decision on the request. Present once the state is `decided`.",
    )
    message: str | None = Field(
        default=None, description="What the model is told, for a rejection."
    )
    recorded: list[dict[str, Any]] | None = Field(
        default=None,
        description="What each monitor and protocol recorded, every attempt's in order. Left out where the sidecar doesn't send results.",
    )
    judgment: Judgment | None = Field(
        default=None,
        description="The sentinel's decisions on the request, which the eval acts on and logs. Present once the state is `decided`. Trimmed if the sidecar doesn't send results.",
    )
    reviews: list[ReviewRecord] | None = Field(
        default=None,
        description="The record of each question a sentinel asked a person about the request, in order. It includes a question whose step the sidecar stopped before a decision. A judged step lists the ids of its questions in its own `reviews`. Left out if no person was asked.",
    )
    failure: FailureRecord | None = Field(
        default=None,
        description="The provider's most recent failure. Present with `awaiting_resend` and a `reason`.",
    )
    judging_time: float | None = Field(
        default=None,
        ge=0,
        description="How long the sentinel spent judging the latest attempt's request and reply so far, in seconds. The client waited on its model call meanwhile, so an eval counts this time as waiting, not as the agent's working time. For a request paused for a person, it is the total over the request's tries, and each try counts only while its client was waiting on it. 0 where nothing was judged. Sent with `judging_usage` on every answer about records the sidecar holds, whether or not the judgment is trimmed.",
    )
    judging_usage: list[JudgingUsage] | None = Field(
        default=None,
        description="What the provider billed for each model the sentinel called while judging every attempt at the request, retried attempts included, since each was billed. One entry per role and model that was billed, in the order first billed: a reply from inspect_ai's cache and a call that failed weren't billed. Empty where nothing was billed. Sent with `judging_time` on every answer about records the sidecar holds, whether or not the judgment is trimmed.",
    )


class SampleResultParams(EvalCall):
    """Parameters of `sample_result`: one sample in one epoch, before it is scored."""

    sample_uuid: str = Field(
        description="The sample's uuid, as `x-inspect-context` gives it: one sample in one epoch, the same across the sample's retries."
    )


class SampleResultResult(BaseModel):
    """What the sample came to, and the sentinel's store."""

    sample_uuid: str
    outcome: Literal["proceed", "terminate"] = Field(
        description="`terminate` if a sentinel ended the sample's latest attempt."
    )
    store: dict[str, Any] = Field(
        description="The sentinel's store of the sample's latest attempt, by the keys the sentinel wrote, to copy into the sample's store. Each attempt starts with an empty store, as the sample store does."
    )


class EndSampleParams(EvalCall):
    """Parameters of `end_sample`: one sample in one epoch, once it has been scored."""

    sample_uuid: str = Field(
        description="The sample's uuid, as `x-inspect-context` gives it."
    )


class EndSampleResult(BaseModel):
    """Whether the sample's state was dropped."""

    sample_uuid: str
    state: Literal["ended", "unknown"] = Field(
        description="`ended`, or `unknown` for a sample the sidecar never saw."
    )


class ListPendingQuestionsParams(EvalCall):
    """Parameters of `list_pending_questions`: lists the eval's questions for a person that have no answer yet."""

    sample_uuid: str | None = Field(
        default=None,
        description="If given, the sidecar lists only this sample's questions. It is the sample's uuid, as `x-inspect-context` carries it. Inspect passes the sample whose model call is paused.",
    )
    wait: float = Field(
        default=0,
        ge=0,
        description=f"How long to wait, in seconds, while there are no questions, at most {MAX_QUESTIONS_WAIT:g}. The sidecar answers as soon as there is one.",
    )


class Escalation(BaseModel):
    """An escalation that a monitor or protocol passed to the protocol that asked the person."""

    name: str = Field(description="The name of the monitor or protocol that escalated.")
    explanation: str | None = Field(
        default=None,
        description="Why it escalated, if it gave a reason.",
    )


class MessageShown(BaseModel):
    """One message of a request that the model has not been sent before."""

    position: int = Field(
        description="Where the message is in the request, counting from 1."
    )
    message: dict[str, Any] = Field(
        description="The message, as Inspect writes a `ChatMessage`. It is for showing to the person. Do not add it to the agent's conversation."
    )


class RequestShown(BaseModel):
    """What a question at `model_input` shows of the request the model is about to be sent."""

    model: str = Field(description="The model the request is for.")
    messages: int = Field(description="How many messages the request sends.")
    new: list[MessageShown] = Field(
        description="The messages the model has not been sent before, in the request's order. A message that was rewritten since it was sent is among them, as is an assistant message the client wrote itself. A message sent before is left out wherever it now sits."
    )
    tools: list[str] = Field(
        description="The names of the tools the request offers the model."
    )
    tool_choice: Literal["auto", "any", "none", "function"] = Field(
        description="Which tool the model is to call. `auto`: any of them, or none. `any`: one of them. `none`: none of them. `function`: the one that `tool_function` names."
    )
    tool_function: str | None = Field(
        default=None,
        description="The name of the function the model must call. Present where `tool_choice` is `function`. It is kept apart from `tool_choice`, since a tool can be named `auto`, `any` or `none`.",
    )


class QuestionAsked(BaseModel):
    """A question that a sentinel asked a person about one of the eval's model requests: about a tool call, about the model's whole turn, or about the request itself. The step waits for the answer."""

    id: str = Field(description="The question's id. `answer_question` takes it.")
    sample_uuid: str = Field(
        description="The uuid of the sample whose run the question belongs to."
    )
    request_id: str = Field(
        description="The `x-irid` of the model request that the question paused. The request stays paused until the question has an answer."
    )
    stage: Literal["tool_call", "tool_result", "model_input", "model_output"] = Field(
        description="`tool_call`: the call has not run yet. `tool_result`: the call has run, and its result is about to reach the model. `model_input`: the request has not been sent to the provider. `model_output`: the model's turn is complete, and none of its calls has run."
    )
    call: dict[str, Any] | None = Field(
        default=None,
        description="The tool call with its arguments, as Inspect writes a `ToolCall`. It is for showing to the person. Do not add it to the agent's conversation. Left out at a generate stage.",
    )
    message: str = Field(
        description="The assistant's text: what came with the call, or at `model_output` the turn's own text. Empty at `model_input`."
    )
    view: dict[str, Any] | None = Field(
        default=None,
        description="How to display the call, as Inspect writes a `ToolCallView`. Left out at a generate stage.",
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="The call's result, as Inspect writes a `ChatMessageTool`. Present at `tool_result`.",
    )
    turn: dict[str, Any] | None = Field(
        default=None,
        description="The model's turn, as Inspect writes a `ChatMessageAssistant`. The eval has been given none of it yet, so it is for showing to the person. Do not add it to the agent's conversation: its reasoning has no signature, and none of the cipher text a provider returns in place of reasoning it withheld. At `model_output` it is the whole turn, unless the reply held none: its text, its reasoning and every tool call it makes, each with the call part of its view. A viewer's separate context is only in the call's own question. At `tool_call` it is the turn that made the call, without its tool calls. Left out at `tool_result`, where the eval already has the turn.",
    )
    request: RequestShown | None = Field(
        default=None,
        description="The request the model is about to be sent. Present at `model_input`.",
    )
    escalations: list[Escalation] = Field(
        description="The escalations passed to the protocol that asked the person, in order: who escalated, and why."
    )
    choices: list[str] = Field(
        description=f"The choices the person may pick from. An answer must be one of these, and one of {', '.join(ANSWERABLE)}."
    )
    asked: float = Field(
        description="When the sentinel asked the question, in seconds since the epoch. A client can show how long the question has waited."
    )
    closes: float | None = Field(
        default=None,
        description="When the question expires, in seconds since the epoch, if the operator set a review timeout. At expiry the sidecar gives the sentinel the cautious answer, `reject` if the question offers it and otherwise `terminate`, and the sentinel continues judging. Left out if the question has no expiry.",
    )


class ListPendingQuestionsResult(BaseModel):
    """The eval's unanswered questions, oldest first."""

    questions: list[QuestionAsked]


class AnswerQuestionParams(EvalCall):
    """Parameters of `answer_question`: the person's choice.

    For now the eval sends the answer, signed with the eval's token, and the sidecar accepts it only for a question from one of that eval's own runs. So any client that holds the eval's token can answer that eval's questions. Later, reviewers will answer with credentials of their own, which the sidecar will check.
    """

    id: str = Field(description="The question's id.")
    decision: Literal["approve", "reject", "terminate"] = Field(
        description="The person's choice, one of the question's `choices`."
    )
    reason: str | None = Field(
        default=None,
        description="The person's reason, if they gave one. For `reject`, the sidecar tells the model this reason.",
    )
    mac: str = Field(
        description='An HMAC-SHA256, in hex, made with the eval\'s token over `json.dumps([str(v) for v in [method, eval_id, timestamp, id, decision, reason or ""]], separators=(",", ":"))`. Unlike the `mac` of other calls, it covers the answer, so a `mac` copied from one answer cannot sign a different answer.'
    )


class AnswerQuestionResult(BaseModel):
    """The answer that the sidecar gave the step. Sending the same answer again returns the same result."""

    id: str
    decision: Literal["approve", "reject", "terminate"]


class CancelPausedRequestParams(EvalCall):
    """Parameters of `cancel_paused_request`: cancels a request paused for a person, which the eval has given up on.

    The sidecar closes the request in whatever state it is, with nothing decided. It stops the judging and closes the request's questions. It lets no further try through to the provider. If a try is already with the provider, the sidecar does not judge, keep or pass on its reply. `request_result` then answers `not_judged` with the reason `cancelled`, and the request cannot be resumed.
    """

    id: str = Field(description="The request's `x-irid`.")


class CancelPausedRequestResult(BaseModel):
    """Whether this call cancelled the request."""

    id: str
    cancelled: bool = Field(
        description="Whether this call is what closed the request. False if the request had already ended, if no person was asked about it, or if the sidecar doesn't know it. A second `cancel_paused_request` for the same request also answers false. So false does not mean the request is still paused: call `request_result` for its state."
    )


_METHODS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "register_task": (RegisterTaskParams, RegisterTaskResult),
    "complete_registration": (
        CompleteRegistrationParams,
        CompleteRegistrationResult,
    ),
    "request_result": (RequestResultParams, RequestResultResult),
    "sample_result": (SampleResultParams, SampleResultResult),
    "end_sample": (EndSampleParams, EndSampleResult),
    "list_pending_questions": (ListPendingQuestionsParams, ListPendingQuestionsResult),
    "answer_question": (AnswerQuestionParams, AnswerQuestionResult),
    "cancel_paused_request": (CancelPausedRequestParams, CancelPausedRequestResult),
}

_RESULTS = frozenset({"sample_result"})

# The methods for asking a person through an eval. A sidecar that sends an
# eval no results doesn't serve them, because it doesn't trust that eval to
# answer for a person.
_REVIEW = frozenset(
    {"list_pending_questions", "answer_question", "cancel_paused_request"}
)

_ERRORS: dict[int, tuple[str, str]] = {
    REGISTRATION_REFUSED: (
        "registration_refused",
        "The sidecar takes no tasks registered by an eval, and this task was not registered beforehand; or, for `complete_registration`, the sidecar takes only the samples a task's registration names, and this sample isn't one.",
    ),
    REGISTRATIONS_FULL: (
        "registrations_full",
        "The sidecar keeps as many task registrations as it takes, and takes another once a kept one expires; or, for `complete_registration`, the task's registration holds as much as the sidecar takes for a task. A failure, not a refusal: the deployment has not said that the task may not run.",
    ),
    UNKNOWN_SAMPLE: (
        "unknown_sample",
        "The sidecar holds nothing for the sample, so an eval never mistakes a forgotten sample for a clean one.",
    ),
    UNKNOWN_TASK: (
        "unknown_task",
        "The sidecar holds no registration of the task by the eval: it was never registered, or is no longer kept. The eval registers the task again, then completes the sample's registration. Answered to a `register_task` with a `mac` when the sidecar holds no token for the eval, as after a restart or once the token expired: the eval registers again with its token.",
    ),
    UNKNOWN_QUESTION: (
        "unknown_question",
        "The sidecar holds no such question for the eval. Either the question was never asked, or it belongs to another eval. Or it closed without an answer: it expired and the sidecar gave the sentinel the cautious answer, or its model call was closed first, for example because the eval cancelled the call or ended the sample. Or it was answered, and the sidecar no longer keeps the record of that answer, for example after the eval ended the sample.",
    ),
    QUESTION_ANSWERED: (
        "question_answered",
        "The question already has an answer with a different choice or reason. The sidecar keeps the first answer.",
    ),
    ANSWER_REFUSED: (
        "answer_refused",
        "The answer's choice is not one the question offers.",
    ),
    UNAUTHORIZED: (
        "unauthorized",
        "The call's `mac` isn't made with the token the sidecar keeps for the eval, or its `timestamp` is further from the sidecar's clock than the window allows; or a `register_task` sent another token than the one the sidecar keeps for the eval.",
    ),
}


def eval_routes(
    handler: Handler,
    *,
    max_body_bytes: int,
    token: str | None = None,
    results: bool = True,
) -> list[Route]:
    """The routes of the endpoint for evals: the probe at `/sentinel/info`, the methods at `/sentinel/v1`, and the fetch of a judged reply at `/sentinel/v1/judged-reply`.

    An eval registers its tasks and completes each sample's registration before the sample runs, asks what became of each request and of a sample, and ends a sample once it is scored. The eval makes a token and sends it in a registration to a sidecar that holds none for the eval. Every other call proves the caller holds that token with a `timestamp` and a `mac`, so only the eval reads what was decided of its requests and samples. Every call is a JSON-RPC 2.0 request. No method clears a tally or changes the sentinel. One method can let a tool call go ahead: `answer_question`, which answers a question that the sentinel itself asked a person. For now the eval signs that answer with its own token, and the sidecar accepts it only for the eval's own runs (`design/sidecar-plan.md`, step 13). A call larger than `max_body_bytes` is not read, and is answered with an HTTP 413.

    `/sentinel/v1/judged-reply` is not a method. An eval uses it to fetch what the sidecar holds for a model call that was paused for a person: the provider's reply, once the call is `decided` with the outcome `proceed`, or the provider's error, while the call is `awaiting_resend` because the provider failed it. To fetch it, the eval POSTs the call's own model request to this path, under the same `x-irid` and with an `x-inspect-context` signed as on any model request. The fetch is a model request, and not a method, because the eval reads the answer with its code for that provider, and that code can only send the provider's request. The sidecar reads only the headers that name and prove the call, never the body. It answers with what it holds: the provider's status, headers and body, with a stream's events from first to last. It refuses any other request with a 400 and a JSON `error`, and the eval then gets the call's state from `request_result`. A fetch changes nothing, so the eval can repeat it.

    Args:
        handler: The handler the proxy's requests and replies are put to.
        max_body_bytes: The largest call that is read.
        token: A credential every call to the methods must present as a bearer token, when the endpoint has a port of its own. None where it shares the proxy's port and access control.
        results: Whether the results are sent: `sample_result`, and what each monitor and protocol recorded in `request_result`. A record holds each monitor's explanation, which tells an agent what the monitor looks for, so where an eval can't be trusted they are turned off. `request_result` is still served without them, since it is how an eval learns that a turn was refused or edited. Without results, the sidecar doesn't serve `list_pending_questions`, `answer_question`, `cancel_paused_request` or `/sentinel/v1/judged-reply` either, because an eval that isn't trusted with a monitor's explanation isn't trusted to answer for a person. Build the handler with `eval_results` off as well, so that it asks no question that nobody can answer.
    """

    async def info(http: HttpRequest) -> Response:
        return JSONResponse(
            {
                "service": "inspect_sentinel",
                "versions": VERSIONS,
                "release": handler.release,
                # Whether this sidecar asks a person through the eval. An
                # eval lists questions only at a sidecar that says so here.
                "review": handler.eval_review and results,
            }
        )

    def unauthorized(http: HttpRequest) -> Response | None:
        # the endpoint has a credential of its own only where it has a port
        # of its own
        if token is None:
            return None
        presented = http.headers.get("authorization", "")
        if hmac.compare_digest(presented.encode(), f"Bearer {token}".encode()):
            return None
        return JSONResponse(
            {"error": "A bearer token is required."},
            status_code=401,
            headers={"www-authenticate": "Bearer"},
        )

    async def call(http: HttpRequest) -> Response:
        refused = unauthorized(http)
        if refused is not None:
            return refused
        body = await read_within(http, max_body_bytes)
        if body is None:
            return JSONResponse(
                {"error": f"A call is at most {max_body_bytes} bytes."},
                status_code=413,
            )
        id: int | str | None = None
        try:
            asked = read_call(body)
            id = asked.id
            if not results and asked.method in _RESULTS:
                raise RpcError(
                    METHOD_NOT_FOUND,
                    f"This sidecar doesn't send results, so there is no method {asked.method!r}.",
                )
            if not results and asked.method in _REVIEW:
                raise RpcError(
                    METHOD_NOT_FOUND,
                    f"This sidecar sends an eval no results, so it asks no person through an eval and does not serve the method {asked.method!r}.",
                )
            result = await _answer(handler, asked.method, asked.params, results)
        except RpcError as ex:
            return failed(id, ex)
        return answered(id, result)

    async def judged_reply(http: HttpRequest) -> Response:
        # Every refusal uses a status that a provider's SDK doesn't retry by
        # itself. So a fetch of a reply the sidecar doesn't hold fails once,
        # and the eval then gets the call's state from `request_result`.
        refused = unauthorized(http)
        if refused is not None:
            return refused
        if not results:
            return JSONResponse(
                {
                    "error": "This sidecar doesn't send results, so it asks no person through an eval, and holds no reply for one."
                },
                status_code=404,
            )
        # The headers alone decide the answer, so the sidecar refuses a
        # request before it reads any of the body.
        judged = await handler.judged_reply(
            {name.lower(): value for name, value in http.headers.items()}
        )
        if isinstance(judged, Refuse):
            return JSONResponse({"error": judged.message}, status_code=judged.status)
        # The body is the call's model request. The sidecar discards it
        # without reading what it says.
        if not await discard_within(http, max_body_bytes):
            return JSONResponse(
                {"error": f"A request is at most {max_body_bytes} bytes."},
                status_code=413,
            )
        return _judged_reply_response(judged)

    return [
        Route(INFO_PATH, info, methods=["GET"]),
        Route(PATH, call, methods=["POST"]),
        Route(JUDGED_REPLY_PATH, judged_reply, methods=["POST"]),
    ]


async def _answer(
    handler: Handler, method: str, params: dict[str, Any], results: bool = True
) -> BaseModel:
    if method not in _METHODS:
        raise RpcError(METHOD_NOT_FOUND, f"There is no method {method!r}.")
    try:
        given = _METHODS[method][0].model_validate(params)
    except ValidationError as ex:
        raise invalid_params(ex) from None
    try:
        return await _answered(handler, method, given, results)
    except UnauthorizedError as ex:
        raise RpcError(UNAUTHORIZED, str(ex)) from None


async def _answered(
    handler: Handler, method: str, given: BaseModel, results: bool
) -> BaseModel:
    if isinstance(given, RegisterTaskParams):
        # the token or the proof first: a caller that has neither registers
        # nothing, and reads nothing of what the eval registered before
        if given.token is not None:
            await handler.eval_token(given.eval_id, given.token)
        else:
            assert given.timestamp is not None and given.mac is not None
            if not await handler.holds_eval(
                given.eval_id, method, given.timestamp, given.mac
            ):
                raise RpcError(
                    UNKNOWN_TASK,
                    f"The sidecar holds no token for the eval {given.eval_id!r}, as after a restart or once it expired, so the eval registers the task again with its token.",
                )
        registration = TaskRegistration.model_validate(
            given.model_dump(include=set(TaskRegistration.model_fields))
        )
        try:
            # The token is kept with the registration, in one step. So a
            # registration that fails leaves no token behind, and one that
            # finds another token kept meanwhile leaves no task.
            granted = await handler.register_task(
                given.eval_id, registration, given.expiry, given.token
            )
        except RegistrationsFullError as ex:
            raise RpcError(REGISTRATIONS_FULL, str(ex)) from None
        except RegistrationError as ex:
            raise RpcError(REGISTRATION_REFUSED, str(ex)) from None
        return RegisterTaskResult(
            eval_id=given.eval_id,
            task=given.task,
            samples=len(given.sample_ids),
            fixed=granted is None,
            expiry=granted,
        )
    if isinstance(given, AnswerQuestionParams):
        return await _answer_question(handler, given)
    assert isinstance(given, EvalCall)
    # a sidecar that keeps no token for the eval holds nothing of it
    holds = await handler.holds_eval(given.eval_id, method, given.timestamp, given.mac)
    if isinstance(given, CompleteRegistrationParams):
        if not holds:
            raise RpcError(
                UNKNOWN_TASK,
                f"The sidecar holds nothing of the eval {given.eval_id!r}, which registers the task again.",
            )
        sample = SampleFacts.model_validate(
            given.model_dump(include=set(SampleFacts.model_fields))
        )
        try:
            fixed = await handler.complete_registration(
                given.eval_id, given.task, sample
            )
        except UnknownTaskError as ex:
            raise RpcError(UNKNOWN_TASK, str(ex)) from None
        except RegistrationsFullError as ex:
            raise RpcError(REGISTRATIONS_FULL, str(ex)) from None
        except RegistrationError as ex:
            raise RpcError(REGISTRATION_REFUSED, str(ex)) from None
        return CompleteRegistrationResult(
            eval_id=given.eval_id,
            task=given.task,
            sample_id=given.sample_id,
            fixed=fixed,
        )
    if isinstance(given, RequestResultParams):
        found = (
            await handler.request_result(given.id, given.eval_id, given.wait)
            if holds
            else "unknown"
        )
        if not isinstance(found, RequestResult):
            return RequestResultResult(id=given.id, records=found)
        latest = found.latest
        return RequestResultResult(
            id=given.id,
            records="held",
            state=latest.state,
            reason=latest.reason,
            outcome=latest.outcome,
            message=latest.message,
            recorded=(
                [record_json(record) for record in found.records] if results else None
            ),
            judgment=(
                judgment(
                    latest.judged,
                    latest.outcome,
                    trimmed=not (results and handler.eval_results),
                )
                if latest.state == "decided"
                and latest.judged is not None
                and latest.outcome is not None
                else None
            ),
            judging_time=latest.judging_time,
            judging_usage=judging_usage(found.judging_usage),
            reviews=[
                ReviewRecord(
                    id=review.id,
                    try_number=number,
                    stage=_ASKED_STAGE.validate_python(review.stage),
                    call=review.call,
                    asked=review.asked,
                    decision=_DECISION.validate_python(review.decision),
                    reason=review.reason,
                    answered=review.answered,
                    resolution=review.resolution,
                    cause=review.cause,
                    resolved=review.resolved,
                )
                for number, review in found.reviews
            ]
            or None,
            failure=(
                FailureRecord(
                    try_number=found.failure.try_number,
                    reply=(
                        _reply_record(found.failure.reply)
                        if found.failure.reply is not None
                        else None
                    ),
                )
                if found.failure is not None
                else None
            ),
        )
    if isinstance(given, CancelPausedRequestParams):
        return CancelPausedRequestResult(
            id=given.id,
            cancelled=(
                await handler.cancel_paused_request(given.id, given.eval_id)
                if holds
                else False
            ),
        )
    if isinstance(given, SampleResultParams):
        run = (
            await handler.sample_result(given.sample_uuid, given.eval_id)
            if holds
            else None
        )
        if run is None:
            raise RpcError(
                UNKNOWN_SAMPLE,
                f"The sidecar holds nothing for the sample {given.sample_uuid!r}.",
            )
        return SampleResultResult(
            sample_uuid=given.sample_uuid, outcome=run.outcome, store=run.store
        )
    if isinstance(given, EndSampleParams):
        return EndSampleResult(
            sample_uuid=given.sample_uuid,
            state=(
                await handler.end_sample(given.sample_uuid, given.eval_id)
                if holds
                else "unknown"
            ),
        )
    if isinstance(given, ListPendingQuestionsParams):
        # a sidecar that keeps no token for the eval holds no question for it
        found = (
            await handler.reviews.list_pending(
                given.eval_id, given.sample_uuid, given.wait
            )
            if holds
            else []
        )
        return ListPendingQuestionsResult(
            questions=[_question(question) for question in found]
        )
    raise TypeError(f"No method takes {type(given).__name__}.")


async def _answer_question(
    handler: Handler, given: AnswerQuestionParams
) -> AnswerQuestionResult:
    signed = answer_values(
        given.eval_id, given.timestamp, given.id, given.decision, given.reason
    )
    if not await handler.holds_eval(
        given.eval_id, "answer_question", given.timestamp, given.mac, signed
    ):
        raise RpcError(
            UNKNOWN_QUESTION,
            f"The sidecar holds nothing for the eval {given.eval_id!r}, so it holds no question for it.",
        )
    try:
        handler.answer_question(given.eval_id, given.id, given.decision, given.reason)
    except UnknownQuestionError as ex:
        raise RpcError(UNKNOWN_QUESTION, str(ex)) from None
    except AnsweredError as ex:
        raise RpcError(QUESTION_ANSWERED, str(ex)) from None
    except ChoiceError as ex:
        raise RpcError(ANSWER_REFUSED, str(ex)) from None
    return AnswerQuestionResult(id=given.id, decision=given.decision)


def _judged_reply_response(judged: JudgedReply) -> Response:
    # The reply as the provider sent it and the proxy passed it on, for the
    # eval's code for that provider to read. The sidecar sends the body
    # whole, with a stream's events from first to last. The response gets a
    # length of its own, because the sidecar left out the reply's hop-by-hop
    # headers, such as `content-length`, when it began to hold the reply.
    headers = dict(judged.headers)
    if judged.stream and "event-stream" not in headers.get("content-type", "").lower():
        headers["content-type"] = "text/event-stream"
    return Response(content=judged.body, status_code=judged.status, headers=headers)


def _reply_record(reply: JudgedReply) -> ReplyRecord:
    return ReplyRecord(
        status=reply.status,
        headers=dict(reply.headers),
        body=base64.b64encode(reply.body).decode(),
        format=reply.format,
        stream=reply.stream,
    )


def _question(question: Question) -> QuestionAsked:
    return QuestionAsked(
        id=question.id,
        sample_uuid=question.sample_uuid,
        request_id=question.request_id,
        stage=_ASKED_STAGE.validate_python(question.stage),
        call=(
            _CALL.dump_python(question.call, mode="json", exclude_none=True)
            if question.call is not None
            else None
        ),
        message=question.message,
        view=(
            question.view.model_dump(mode="json", exclude_none=True)
            if question.view is not None
            else None
        ),
        result=(
            question.result.model_dump(mode="json", exclude_none=True)
            if question.result is not None
            else None
        ),
        turn=(
            question.turn.model_dump(mode="json", exclude_none=True)
            if question.turn is not None
            else None
        ),
        request=(
            RequestShown(
                model=question.request.model,
                messages=question.request.messages,
                new=[
                    MessageShown(
                        position=position,
                        message=message.model_dump(mode="json", exclude_none=True),
                    )
                    for position, message in question.request.new
                ],
                tools=list(question.request.tools),
                tool_choice=(
                    "function"
                    if isinstance(question.request.tool_choice, ToolFunction)
                    else question.request.tool_choice
                ),
                tool_function=(
                    question.request.tool_choice.name
                    if isinstance(question.request.tool_choice, ToolFunction)
                    else None
                ),
            )
            if question.request is not None
            else None
        ),
        escalations=[
            Escalation(name=escalation.name, explanation=escalation.explanation)
            for escalation in question.escalations
        ],
        choices=list(question.choices),
        asked=question.asked,
        closes=question.closes,
    )


_CALL = TypeAdapter(ToolCall)
_ASKED_STAGE = TypeAdapter(
    Literal["tool_call", "tool_result", "model_input", "model_output"]
)
_DECISION = TypeAdapter(Literal["approve", "reject", "terminate"] | None)


def schema() -> dict[str, Any]:
    """The JSON Schema of each method's parameters and result, and the errors the methods may answer with.

    A call to method `m` has `params` that fit the definition named in `methods[m].params`, and is answered with a `result` that fits the definition named in `methods[m].result`, or with an error whose code is a key of `errors` or one of JSON-RPC's own.
    """
    models = {model: None for pair in _METHODS.values() for model in pair}
    _, definitions = models_json_schema(
        [(model, "validation") for model in models],
        title="Inspect Sentinel: the endpoint for evals",
        description=f"JSON-RPC 2.0 methods, POSTed to {PATH}, with a probe at GET {INFO_PATH}. A POST of a paused call's model request to {JUDGED_REPLY_PATH} is not a method: it fetches the provider's reply that the sidecar holds for that call, in the provider's format. `methods` names the definitions that a method's `params` and `result` fit. `errors` lists the codes of the method errors of the endpoint's own.",
    )
    return {
        "$schema": GenerateJsonSchema.schema_dialect,
        **definitions,
        "methods": {
            name: {
                "params": {"$ref": f"#/$defs/{params.__name__}"},
                "result": {"$ref": f"#/$defs/{result.__name__}"},
            }
            for name, (params, result) in _METHODS.items()
        },
        "errors": {
            str(code): {"name": name, "description": description}
            for code, (name, description) in _ERRORS.items()
        },
    }


SCHEMA_FILE = Path(__file__).with_name("eval_endpoint.schema.json")
"""Where `schema()` is kept, for an eval's client to be written and tested against without running this package."""


if __name__ == "__main__":
    # rewrites the schema file; run after changing a model above
    SCHEMA_FILE.write_text(json.dumps(schema(), indent=2) + "\n")
