from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo

from .._context import Context, EvalContext
from .._host import HostNative, HumanAnswer
from .._report import Decision, Failed, Report, Reported
from .._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step

logger = logging.getLogger(__name__)

Stage: TypeAlias = Literal["model_input", "model_output", "tool_call", "tool_result"]
"""Where a step was judged, named as Inspect's transcript names it."""


class SidecarHost:
    """What a sentinel in a sidecar may do to the outside world: the sidecar's `HostNative`.

    A proxy's requests are not an eval's samples, so the sentinel is told of no eval. Models are called by the sidecar itself, never through the proxy it sits beside. A person can't be asked yet: at a tool stage the answer is the most cautious choice offered, so the step fails closed, and at a generate stage asking fails, as it does in Inspect.
    """

    def __init__(self, roles: Mapping[str, str] | None = None) -> None:
        """Create the host.

        Args:
            roles: The model each role generates with, e.g. `{"monitor": "anthropic/claude-haiku-4-5"}`.
        """
        self._roles = dict(roles or {})

    @property
    def eval(self) -> EvalContext | None:
        """None: the sidecar's host serves every run, and `RunHost` tells a run's sentinel which sample it is."""
        return None

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        """Generate with the role's model if the sidecar has one configured, otherwise with `model`.

        A sidecar has no agent's model to fall back on, so with neither it raises.

        Args:
            input: A prompt string or a list of chat messages.
            model: A model name, used when `role` is not configured.
            role: A model role. Defaults to `monitor` when `model` is None.
            tools: Tool definitions to offer the model.
            tool_choice: Which of `tools` the model may or must call.
            config: Generation configuration.
        """
        asked = role if role is not None else ("monitor" if model is None else None)
        resolved = self._roles.get(asked) if asked is not None else None
        if resolved is None and model is None:
            raise ValueError(
                f"The sidecar has no model for role {asked!r}. Configure one, or pass model= to generate()."
            )
        return await get_model(resolved if resolved is not None else model).generate(
            input,
            tools=tools or [],
            tool_choice=tool_choice,
            config=config or GenerateConfig(),
        )

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        """Answer for a person who can't be asked: `reject` if it is offered, otherwise `terminate`.

        An escalation never lets a call through without someone deciding, so the step fails closed, as Inspect's own host does when nobody can answer. A deployment that wants unattended escalations to go on says so in its sentinel, for example with `handle_escalation("proceed")` after `human()`.

        Args:
            step: The step to decide about.
            choices: What a person could have picked.
        """
        if not isinstance(step, BeforeToolCall | AfterToolCall):
            raise NotImplementedError(
                f"A sidecar cannot ask a person at the {stage(step)} stage, as Inspect cannot."
            )
        decision = "reject" if "reject" in choices else "terminate"
        logger.warning(
            "No person can be asked from the sidecar, so the %s stage of the call to %s was answered %s.",
            stage(step),
            step.call.function,
            decision,
        )
        return HumanAnswer(
            decision=decision, reason="No person can be asked from the sidecar yet."
        )


class RunHost:
    """The sidecar's host for one run: the sidecar's own, telling the sentinel which sample the run is."""

    def __init__(self, host: HostNative, eval: EvalContext | None) -> None:
        """Wrap the sidecar's host for a run.

        Args:
            host: The sidecar's host, which calls models and would ask a person.
            eval: The sample the run is, where its task was registered.
        """
        self._host = host
        self._eval = eval

    @property
    def eval(self) -> EvalContext | None:
        """The sample the run is, or None if its task wasn't registered."""
        return self._eval

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        """As the sidecar's host generates."""
        return await self._host.generate(
            input,
            model=model,
            role=role,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
        )

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        """As the sidecar's host asks."""
        return await self._host.ask_human(step, choices)


def stage(step: Step) -> Stage:
    """The stage a step was judged at."""
    if isinstance(step, BeforeGenerate):
        return "model_input"
    if isinstance(step, AfterGenerate):
        return "model_output"
    if isinstance(step, BeforeToolCall):
        return "tool_call"
    return "tool_result"


@dataclass(frozen=True)
class Record:
    """One thing the runner recorded while a sentinel judged a step."""

    kind: Literal["record", "failed", "cancelled", "bypassed", "superseded"]
    """What was recorded, as the runner's `Recorder` methods name it."""

    stage: Stage
    """The stage of the step."""

    path: str
    """The instance path of the monitor or protocol."""

    factory: str
    """Registry name of the instance's factory."""

    detail: Reported[Report] | Failed | str
    """The report or failure with its instance identity, or the instance name for a cancellation or bypass."""

    call: str | None = None
    """The id of the tool call, at a tool stage."""


class ListRecorder:
    """Keeps what the runner records, in order."""

    def __init__(self) -> None:
        self.records: list[Record] = []

    def record(
        self, context: Context, factory: str, step: Step, reported: Reported[Report]
    ) -> None:
        self._add("record", context, factory, step, reported)

    def failed(
        self, context: Context, factory: str, step: Step, failed: Failed
    ) -> None:
        self._add("failed", context, factory, step, failed)

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        self._add("cancelled", context, factory, step, name)

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        self._add("bypassed", context, factory, step, name)

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Reported[Decision]
    ) -> None:
        self._add("superseded", context, factory, step, reported)

    def _add(
        self,
        kind: Literal["record", "failed", "cancelled", "bypassed", "superseded"],
        context: Context,
        factory: str,
        step: Step,
        detail: Reported[Report] | Reported[Decision] | Failed | str,
    ) -> None:
        call = (
            step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else None
        )
        self.records.append(
            Record(kind, stage(step), context.path, factory, detail, call)
        )
