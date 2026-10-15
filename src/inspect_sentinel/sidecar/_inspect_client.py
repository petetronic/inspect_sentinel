from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.tool import ToolCall
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .._context import InspectSample
from .._report import Decision, Failed, Reported
from ._host import Record, Stage
from ._wire import UnreadableError, json_object

INSPECT_CONTEXT = "x-inspect-context"
"""The header an Inspect eval names the sample of each model request in."""

# the events a field can go in, which the provider's SDK reads: the last of
# them in a stream is the one
_LAST_EVENTS = frozenset(
    {"message_delta", "response.completed", "response.incomplete", "response.failed"}
)
_LINE_END = re.compile(r"(\r\n|\r|\n)")
_TOOL_CALL = TypeAdapter(ToolCall)


class _Header(BaseModel):
    model_config = ConfigDict(strict=True)

    version: Literal[1]
    eval_id: str
    task: str
    sample_id: str | int
    epoch: int
    sample_uuid: str
    attempt: str


def inspect_sample(headers: Mapping[str, str]) -> InspectSample | None:
    """The sample an Inspect eval says a request is from, or None if the request has no `x-inspect-context`.

    Args:
        headers: The request's headers, by lowercase name.

    Raises:
        UnreadableError: If the header is there and isn't what Inspect writes, so the request's run can't be told.
    """
    value = headers.get(INSPECT_CONTEXT)
    if value is None:
        return None
    try:
        header = _Header.model_validate_json(value)
    except ValidationError:
        raise UnreadableError(
            f"The request's {INSPECT_CONTEXT} header isn't one this sidecar can read."
        ) from None
    return InspectSample(**header.model_dump(exclude={"version"}))


@dataclass
class Judged:
    """One step judged during an exchange, for the `sentinel` field."""

    stage: Stage
    call: str | None
    """The id of the tool call, at a tool stage."""
    decision: Decision | None
    """The step's final decision, or None if no protocol decided."""
    records: Sequence[Record]
    """What was recorded while the step was judged."""
    original: ToolCall | None = None
    """The call the model proposed, where the client is not handed it as proposed."""


@dataclass(frozen=True)
class Withheld:
    """A turn the client was refused, which the sidecar adds to the later requests of its conversation."""

    message: ChatMessageAssistant
    told: Mapping[str, str]
    """What the model is told happened to each of its calls, by call id."""


@dataclass
class Exchange:
    """What was decided during one exchange, for the `sentinel` field."""

    judged: list[Judged] = field(default_factory=list[Judged])
    """The steps judged, in order."""
    withheld: Withheld | None = None
    """The turn the client was refused, if it was."""
    message: str | None = None
    """What the model is told of the rejection, if the turn was refused for one."""


class SentinelStep(BaseModel):
    """One step judged during an exchange."""

    stage: Stage = Field(
        description="Where the step was judged, as Inspect's sentinel events name it."
    )
    call: str | None = Field(
        default=None, description="The id of the tool call, at a tool stage."
    )
    original: dict[str, Any] | None = Field(
        default=None,
        description="The call the model proposed, as Inspect writes a tool call. There only when the client is not handed the call as the model proposed it.",
    )
    decision: dict[str, Any] = Field(
        description="The step's final decision, as the core package writes a `Decision`."
    )
    recorded: list[dict[str, Any]] | None = Field(
        default=None,
        description="What each monitor and protocol recorded for the step. Left out of a trimmed field.",
    )


class WithheldTurn(BaseModel):
    """The model's turn, which the client was refused."""

    message: dict[str, Any] = Field(
        description="The model's message, every call in it included, as Inspect writes a `ChatMessageAssistant`."
    )
    results: dict[str, str] = Field(
        description="What the model is told happened to each of its calls, by call id. The sidecar adds the turn and these results to each later request of the conversation that doesn't already hold the turn."
    )


class SentinelField(BaseModel):
    """What a sentinel decided during one exchange, for an Inspect client: the `sentinel` field."""

    version: Literal[1] = Field(default=1, description="The version of the field.")
    outcome: Literal["proceed", "reject", "terminate"] = Field(
        description="What the exchange came to: `proceed` when the reply goes on, `reject` when the model's turn was refused and the run goes on, and `terminate` when the run has ended, in this exchange or before it."
    )
    trimmed: bool | None = Field(
        default=None,
        description="True when the sidecar was told not to send results, so each step holds only what Inspect acts on: its stage, its call, `original`, and its decision's action, message and modified call.",
    )
    steps: list[SentinelStep] = Field(
        description="Every step judged during the exchange, in the order they were judged. A step that ended the run is the last. Empty when the run had already ended."
    )
    withheld: WithheldTurn | None = Field(
        default=None,
        description="The model's turn, when the client was refused it because a call in it was rejected.",
    )


def sentinel_field(
    exchange: Exchange,
    outcome: Literal["proceed", "reject", "terminate"],
    *,
    trimmed: bool,
) -> dict[str, Any]:
    """The `sentinel` field for an exchange.

    Args:
        exchange: What was decided during the exchange.
        outcome: What the exchange came to.
        trimmed: Whether to keep only what Inspect acts on, leaving out explanations, references, metadata and records.
    """
    steps: list[SentinelStep] = []
    for step in exchange.judged:
        decision = step.decision if step.decision is not None else Decision.proceed()
        steps.append(
            SentinelStep(
                stage=step.stage,
                call=step.call,
                original=(
                    _TOOL_CALL.dump_python(
                        step.original, mode="json", exclude_none=True
                    )
                    if step.original is not None
                    else None
                ),
                decision=decision.model_dump(
                    mode="json",
                    exclude_defaults=True,
                    include={"action", "message", "modified"} if trimmed else None,
                ),
                recorded=None if trimmed else [record_json(r) for r in step.records],
            )
        )
    withheld = exchange.withheld
    written = SentinelField(
        outcome=outcome,
        trimmed=True if trimmed else None,
        steps=steps,
        withheld=(
            WithheldTurn(
                message=withheld.message.model_dump(mode="json", exclude_none=True),
                results=dict(withheld.told),
            )
            if withheld is not None
            else None
        ),
    )
    return written.model_dump(mode="json", exclude_none=True)


def with_sentinel(content_type: str, body: bytes, field: Mapping[str, Any]) -> bytes:
    """A reply with the `sentinel` field added, beside the provider's own fields, so the provider's SDK still reads it.

    A stream has the field in its last event that the SDK reads: Anthropic's `message_delta`, the Responses API's `response.completed` (or the event that ends a response that didn't complete), or the last chunk of a Chat Completions stream. Every other event is left as it was.

    Raises:
        UnreadableError: If the reply isn't JSON, or is a stream with no event the field can go in.
    """
    if not content_type.startswith("text/event-stream"):
        reply = json_object(body, "reply")
        reply["sentinel"] = field
        return json.dumps(reply).encode()
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as ex:
        raise UnreadableError("The reply stream is not UTF-8.") from ex
    # each line with the line end after it, the last with none
    parts = _LINE_END.split(text)
    lines = list(zip(parts[::2], [*parts[1::2], ""], strict=True))
    target = _last_event(lines)
    if target is None:
        raise UnreadableError("The reply stream has no event the field can go in.")
    first, event, rest = target
    event["sentinel"] = field
    written: list[str] = []
    for index, (line, end) in enumerate(lines):
        if index == first:
            # the event's data, once, on one line
            written.append(f"data: {json.dumps(event)}{end}")
        elif index not in rest:
            written.append(line + end)
    return "".join(written).encode()


def _last_event(
    lines: list[tuple[str, str]],
) -> tuple[int, dict[str, Any], set[int]] | None:
    # the event the field goes in: where its first data line is, the event,
    # and where its other data lines are
    found: tuple[int, dict[str, Any], set[int]] | None = None
    chunk: tuple[int, dict[str, Any], set[int]] | None = None
    data: list[int] = []
    for index, (line, _) in enumerate([*lines, ("", "")]):
        if line.startswith("data:"):
            data.append(index)
            continue
        if line or not data:
            continue
        held, data = data, []
        payload = "\n".join(lines[i][0][5:].lstrip(" ") for i in held)
        try:
            event: object = json.loads(payload)
        except ValueError:
            # OpenAI's chat stream ends with an event that is not JSON
            continue
        if not isinstance(event, dict):
            continue
        event = cast("dict[str, Any]", event)
        where = (held[0], event, set(held[1:]))
        if event.get("type") in _LAST_EVENTS:
            found = where
        elif event.get("object") == "chat.completion.chunk":
            chunk = where
    return found if found is not None else chunk


def record_json(record: Record) -> dict[str, Any]:
    """A record, as the endpoint for evals and the `sentinel` field write it."""
    written: dict[str, Any] = {
        "kind": record.kind,
        "path": record.path,
        "factory": record.factory,
    }
    detail = record.detail
    if isinstance(detail, Reported):
        written["name"] = detail.name
        written["function"] = detail.function
        written["report"] = detail.report.model_dump(mode="json", exclude_defaults=True)
    elif isinstance(detail, Failed):
        written["name"] = detail.name
        written["function"] = detail.function
        written["error"] = f"{type(detail.error).__name__}: {detail.error}"
    else:
        written["name"] = detail
    if record.call is not None:
        written["call"] = record.call
    return written
