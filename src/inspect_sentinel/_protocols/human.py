from __future__ import annotations

from collections.abc import Sequence

from .._context import Context
from .._decorators import protocol
from .._host import HumanAnswer
from .._report import Decision, Escalations
from .._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step
from .._types import Protocol, ProtocolGroup

_CHOICES = ("approve", "reject", "terminate", "modify")
_LEGAL = {
    "tool_call": _CHOICES,
    "tool_result": ("approve", "terminate"),
    "model_input": ("approve", "terminate"),
    "model_output": ("approve", "terminate"),
}
_DEFAULTS = {
    "tool_call": ("approve", "reject", "terminate"),
    "tool_result": ("approve", "terminate"),
    "model_input": ("approve", "terminate"),
    "model_output": ("approve", "terminate"),
}
# why a choice the person may pick before a call is not legal at the stage
_NOT_LEGAL = {
    "tool_result": "after a tool call, since the call has already run",
    "model_input": "before a generate, since reject and modify are not legal at the generate stages yet",
    "model_output": "after a generate, since reject and modify are not legal at the generate stages yet",
}


@protocol
def human(
    stages: Sequence[str],
    choices: Sequence[str] | None = None,
) -> Protocol | ProtocolGroup:
    """Ask a person to decide, every time the step reaches this protocol.

    The person is asked through `Host.ask_human`, which is handed the step and the escalations this protocol was handed and shows them together, so at the end of a `sequential()` the person sees who escalated and why. On its own, `human(stages=["tool_call"])` asks about every call. After a generate, the person sees the whole assistant turn, its text and every tool call it makes, before any of the calls run. Before a generate, the person sees the request the model is about to be sent. The answer maps to a decision: `approve` to `proceed` explained by the person's reason, so the log records why the step went ahead, `reject` to `reject` with the person's reason as both the `message` the agent reads and the `explanation`, `terminate` to `terminate` explained by the reason, and `modify` to a `modify` with the person's replacement call.

    The answer is an ordinary decision: it ends a `sequential()` as any link's decision does, and beside other protocols in a `concurrent()` it is one vote, so another protocol's `reject` outranks the person's `approve`.

    Args:
        stages: The stages to ask at, as `SentinelEvent.stage` records them: `"tool_call"` (before a tool call), `"tool_result"` (after it, with the result the model is about to receive), `"model_input"` (before a generate by the agent's model, with the request about to be sent) and `"model_output"` (after a generate by the agent's model, before its output reaches the agent).
        choices: What the person may pick, from `approve`, `reject`, `terminate` and `modify`. Defaults to `approve`, `reject` and `terminate` before a call, and `approve` and `terminate` after a call and at the generate stages, where `reject` and `modify` are not legal.
    """
    valid = ", ".join(repr(stage) for stage in _LEGAL)
    if isinstance(stages, str) or not stages:
        raise ValueError(f"human needs a list of stages, from {valid}; got {stages!r}.")
    for stage in stages:
        if stage not in _LEGAL:
            raise ValueError(
                f"human cannot ask at stage {stage!r}; the stages it supports are {valid}."
            )
    if len(set(stages)) != len(stages):
        raise ValueError(f"human was given a stage twice: {list(stages)!r}.")
    offered = {stage: _choices(stage, choices) for stage in stages}

    async def ask(
        context: Context,
        step: Step,
        escalations: Escalations,
        stage: str,
    ) -> Decision:
        answer = await context.host.ask_human(step, escalations, offered[stage])
        return _decision(answer, offered[stage])

    async def tool_call(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "tool_call")

    async def tool_result(
        context: Context,
        step: AfterToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "tool_result")

    async def model_input(
        context: Context,
        step: BeforeGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "model_input")

    async def model_output(
        context: Context,
        step: AfterGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "model_output")

    functions: dict[str, Protocol] = {
        "tool_call": tool_call,
        "tool_result": tool_result,
        "model_input": model_input,
        "model_output": model_output,
    }
    if len(stages) == 1:
        return functions[stages[0]]
    return ProtocolGroup(*(functions[stage] for stage in stages))


def _choices(stage: str, choices: Sequence[str] | None) -> tuple[str, ...]:
    if choices is None:
        return _DEFAULTS[stage]
    if isinstance(choices, str) or not choices:
        raise ValueError(
            f"human needs at least one choice, from {', '.join(repr(c) for c in _CHOICES)}; got {choices!r}."
        )
    for choice in choices:
        if choice not in _CHOICES:
            raise ValueError(
                f"human cannot offer {choice!r}; the choices are {', '.join(repr(c) for c in _CHOICES)}."
            )
        if choice not in _LEGAL[stage]:
            raise ValueError(
                f"human cannot offer {choice!r} {_NOT_LEGAL[stage]}; the choices there are {', '.join(repr(c) for c in _LEGAL[stage])}."
            )
    return tuple(choices)


def _decision(answer: HumanAnswer, choices: tuple[str, ...]) -> Decision:
    if answer.decision not in choices and answer.decision != "terminate":
        raise ValueError(
            f"The host answered {answer.decision!r}, which is not one of the choices offered: {list(choices)!r}."
        )
    if answer.modified is not None and answer.decision != "modify":
        raise ValueError(
            f"The host answered {answer.decision!r} with a modified call; only a 'modify' carries one."
        )
    reason = answer.reason or None
    match answer.decision:
        case "approve":
            return Decision.proceed(reason)
        case "reject":
            return Decision.reject(reason, message=reason)
        case "terminate":
            return Decision.terminate(reason)
        case _:
            return Decision(action="modify", modified=answer.modified)
