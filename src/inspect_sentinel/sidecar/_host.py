from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    Model,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from .._context import Context, EvalContext
from .._host import HostNative, HumanAnswer
from .._report import Decision, Escalations, Failed, Report, Reported
from .._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step
from ._inspect_ai import Billed, billing, get_model_input_budget

logger = logging.getLogger(__name__)

Stage: TypeAlias = Literal["model_input", "model_output", "tool_call", "tool_result"]
"""Where a step was judged, named as Inspect's transcript names it."""

AskHuman: TypeAlias = Callable[
    [Step, Escalations, Sequence[str]], Awaitable[HumanAnswer]
]
"""A function that asks a person about a step. It takes the step, the escalations and the choices, and waits for the person's answer."""


def asked_about(step: Step) -> str:
    """What a question about the step is about, in words for a log line."""
    if isinstance(step, BeforeGenerate):
        return "the model's request"
    if isinstance(step, AfterGenerate):
        return "the model's turn"
    return f"the call to {step.call.function}"


@dataclass(frozen=True)
class ModelUsed:
    """A model a monitor or protocol called while a step was judged."""

    role: str | None
    """The role it asked for, or None where it named the model without one."""

    model: str
    """The model the call went to, as Inspect names it."""


def add_usage(
    totals: dict[ModelUsed, ModelUsage], used: ModelUsed, usage: ModelUsage
) -> None:
    """Add `usage` to what `totals` holds for `used`, which keeps each model in the order it was first added."""
    totals[used] = totals.get(used, ModelUsage()) + usage


@dataclass
class NotedModels:
    """The models the sidecar's host called within a `noting_models()` block."""

    called: list[ModelUsed] = field(default_factory=list[ModelUsed])
    """Each model called, once each, in the order first called."""

    billed: dict[ModelUsed, ModelUsage] = field(
        default_factory=dict[ModelUsed, ModelUsage]
    )
    """Where what the provider billed for each model's calls is added up as each call is done, in the order first billed: the block's own, or what `noting_models()` was given, such as an attempt's `judging_usage`. A reply from inspect_ai's cache and a call that failed weren't billed, so they add nothing."""


_models_used: ContextVar[NotedModels | None] = ContextVar(
    "sidecar_models_used", default=None
)


@contextmanager
def noting_models(
    billed: dict[ModelUsed, ModelUsage] | None = None,
) -> Generator[NotedModels]:
    """Note each model the sidecar's host calls in the block, and what the provider billed for its calls.

    Args:
        billed: Where to add what each call was billed, as soon as the call is done, such as an attempt's `judging_usage`, so that it counts there while the rest of the block still runs. A dict of the block's own if None.
    """
    noted = NotedModels(billed=billed if billed is not None else {})
    token = _models_used.set(noted)
    try:
        yield noted
    finally:
        _models_used.reset(token)


class SidecarHost:
    """What a sentinel in a sidecar may do to the outside world: the sidecar's `HostNative`.

    A proxy's requests are not an eval's samples, so the sentinel is told of no eval. Models are called by the sidecar itself, never through the proxy it sits beside. This host asks no person itself. A run that an Inspect eval names asks through `RunHost`, and the eval shows the question in its approval UI. For any other run, this host answers a tool stage with the most cautious choice offered, so the step fails closed. At a generate stage it raises an error, as Inspect's own host does.
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
        resolved = self._resolve(model, role)
        noted = _models_used.get()
        used = ModelUsed(_asked(model, role), str(resolved))
        if noted is not None and used not in noted.called:
            noted.called.append(used)
        # What inspect_ai counts as billed, so a reply from its cache counts
        # nothing. `billing()` fills in `billed.usage` as its block is left,
        # so it is read in this outer `finally`, not one inside the `with`,
        # where it is still None; and read even when the call raised, as a
        # call billed before it was cancelled was still billed.
        billed = Billed()
        try:
            with billing(resolved) as billed:
                return await resolved.generate(
                    input,
                    tools=tools or [],
                    tool_choice=tool_choice,
                    config=config or GenerateConfig(),
                )
        finally:
            if noted is not None and billed.usage is not None:
                add_usage(noted.billed, used, billed.usage)

    async def input_budget(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
        config: GenerateConfig | None = None,
    ) -> int:
        """The input budget of the role's model if the sidecar has one configured, otherwise of `model`; resolved as `generate` resolves it."""
        return get_model_input_budget(self._resolve(model, role), config)

    async def count_tokens(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        config: GenerateConfig | None = None,
    ) -> int:
        """The token count of `input` for the role's model if the sidecar has one configured, otherwise for `model`; resolved as `generate` resolves it."""
        return await self._resolve(model, role).count_tokens(input, config)

    def _resolve(self, model: str | None, role: str | None) -> Model:
        # the model a host call names, the way `generate` resolves it: the
        # role's configured model wins, otherwise `model`; with neither it
        # raises, since a sidecar has no agent's model to fall back on
        asked = _asked(model, role)
        resolved = self._roles.get(asked) if asked is not None else None
        if resolved is None and model is None:
            raise ValueError(
                f"The sidecar has no model for role {asked!r}. Configure one, or pass model=."
            )
        return get_model(resolved if resolved is not None else model)

    async def ask_human(
        self, step: Step, escalations: Escalations, choices: Sequence[str]
    ) -> HumanAnswer:
        """Answer for a person who can't be asked: `reject` if it is offered, otherwise `terminate`.

        An escalation never lets a call through without someone deciding, so the step fails closed, as Inspect's own host does when nobody can answer. A deployment that wants unattended escalations to go on says so in its sentinel, for example with `handle_escalation("proceed")` after `human()`.

        Args:
            step: The step to decide about.
            escalations: The escalations the asking protocol was handed; unused, since no person is asked.
            choices: What a person could have picked.
        """
        decision = "reject" if "reject" in choices else "terminate"
        logger.warning(
            "No person can be asked from the sidecar, so the %s stage of %s was answered %s.",
            stage(step),
            asked_about(step),
            decision,
        )
        return HumanAnswer(
            decision=decision, reason="No person can be asked from the sidecar yet."
        )


def _asked(model: str | None, role: str | None) -> str | None:
    # the role a host call asks for: `monitor` when it names neither
    return role if role is not None else ("monitor" if model is None else None)


class RunHost:
    """The sidecar's host for one run. It wraps the sidecar's own host, tells the sentinel which sample the run is, and asks a person when the run has someone to ask."""

    def __init__(
        self,
        host: HostNative,
        eval: EvalContext | None,
        ask: AskHuman | None = None,
    ) -> None:
        """Wrap the sidecar's host for a run.

        Args:
            host: The sidecar's host, which calls models. When no person can be asked, it gives the cautious answer.
            eval: The sample the run is, where its task was registered.
            ask: Asks a person about a tool step. It is given for a run that an Inspect eval names, because that eval's approval UI shows the question. None when nobody can be asked.
        """
        self._host = host
        self._eval = eval
        self._ask = ask

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

    async def input_budget(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
        config: GenerateConfig | None = None,
    ) -> int:
        """As the sidecar's host reads the input budget."""
        return await self._host.input_budget(model=model, role=role, config=config)

    async def count_tokens(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        config: GenerateConfig | None = None,
    ) -> int:
        """As the sidecar's host counts tokens."""
        return await self._host.count_tokens(
            input, model=model, role=role, config=config
        )

    async def ask_human(
        self, step: Step, escalations: Escalations, choices: Sequence[str]
    ) -> HumanAnswer:
        """Ask a person about the step, if the run has someone to ask. Otherwise return the answer of the sidecar's own host."""
        if self._ask is not None:
            return await self._ask(step, escalations, choices)
        return await self._host.ask_human(step, escalations, choices)


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
