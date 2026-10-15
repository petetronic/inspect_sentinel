from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast, runtime_checkable
from typing import Protocol as TypingProtocol

from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_ai.util import Store

from ._report import Decision, Failed, Observation, Report, Reported
from ._score import score_step
from ._step import Step

if TYPE_CHECKING:
    from ._context import Context, EvalContext, ProxyContext
    from ._excerpt import Excerpt


@dataclass(frozen=True)
class HumanAnswer:
    """What a person answered when `Host.ask_human` asked them about a step."""

    decision: str
    """The choice the person picked, one of the `choices` they were offered (`approve`, `reject`, `terminate` or `modify`), or, when the person ended the prompt without one of them, `reject` if it was offered and `terminate` otherwise."""

    reason: str | None = None
    """The reason the surface gives for the answer, if any: the person's typed reason, or fixed text such as "Human operator rejected the tool call."."""

    modified: ToolCall | None = None
    """The replacement call, when the person chose `modify` at a tool stage."""


class HostNative(TypingProtocol):
    """What a host implements: the eval the agent is running in, inference through its models, and asking a person. Host-facing; monitors and protocols reach these through `Host`, which adds what every host shares, such as `Host.score`."""

    @property
    def eval(self) -> EvalContext | None:
        """As `Host.eval`."""
        ...

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
        """As `Host.generate`."""
        ...

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        """As `Host.ask_human`."""
        ...


class Host(HostNative):
    """What a monitor or protocol may do to the outside world. Author-facing.

    Inference and asking a person are the only effects for now. Outbound HTTP through named endpoints (`fetch`) is planned. Meanwhile a monitor running in-process may call inspect_ai APIs directly, such as `sandbox()`, if it is declared `portable=False`.

    The host builds it around its `HostNative`; `score` is built on `generate`, so it is the same for every host.
    """

    def __init__(self, native: HostNative) -> None:
        """Wrap a host's own implementation.

        Args:
            native: The host's implementation of the eval, inference and asking a person.
        """
        self._native = native

    @property
    def eval(self) -> EvalContext | None:
        """The task and sample being run, in an Inspect eval; monitors read it as `Context.eval`. None outside an eval, for example when a proxy runs the sentinel on requests that have no task, sample or epoch."""
        return self._native.eval

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
        """Generate with a model the host resolves.

        `model` is a model name, e.g. `openai/gpt-4o-mini`; `role` is a model role, e.g. `monitor` or `trusted`, which resolves through the host's role configuration (in an eval, `Task(model_roles=...)` or `--model-role`). A configured `role` wins; otherwise `model`; with neither, the role is `monitor`. When the role is not configured and no `model` is given, what happens is up to the host, and inspect_ai falls back to the agent's model with a warning. A role is portable, since the deployer picks its model; a model name suits quick experiments.

        Args:
            input: A prompt string or a list of chat messages.
            model: A model name, used when `role` is not given or not configured.
            role: A model role. Defaults to `monitor` when `model` is None.
            tools: Tool definitions to offer the model.
            tool_choice: Which of `tools` the model may or must call.
            config: Generation configuration.
        """
        return await self._native.generate(
            input,
            model=model,
            role=role,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
        )

    async def score(
        self,
        step: Step | Excerpt[Step],
        dimensions: str | Mapping[str, str],
        *,
        instructions: str | None = None,
        policy: str | None = None,
        template: str | None = None,
        template_variables: Mapping[str, Any] | None = None,
        model: str | None = None,
        role: str | None = None,
        config: GenerateConfig | None = None,
        retry_refusals: int = 3,
    ) -> Observation:
        """Ask a model to score a step from 0.0 to 1.0, with an explanation.

        The model is asked through `generate`, with the same `model`, `role` and `config`. It is sent a prompt built from `template`, and answers by calling an answer tool whose arguments are checked, so a missing or out-of-range score, or a reply with no answer (or no reply at all), goes back to the model to correct. When the model gives no valid answer within three attempts, `score` raises, and the runner records the monitor as failed. A reply the provider refuses (stop reason `content_filter`) is asked again without using an attempt, up to `retry_refusals` times, after which `score` raises. Needs `inspect_scout` installed, which renders the step and numbers its messages; it is not a dependency of this package yet, so without it `score` raises `ImportError`. Call it while the monitor or protocol runs, since it keeps the sample's nonce in the sample store; outside a step it raises `RuntimeError`.

        Pass the step to show the model all of it, or narrow what it sees with the windowing functions, such as `last_turns(step, 6)` or `new_since_last_report(context, step)`.

        The default template opens with `instructions` and a paragraph on the nonce, then shows the model `policy`, the task, any escalations and the step, asks it to score the step from 0.0 to 1.0 against `dimensions`, and asks for an explanation. Templates are Jinja templates. A custom one uses the same variables, and must include `{{ step }}` and `{{ answer_format }}`:

        - `{{ instructions }}`: `instructions`, or by default a sentence on monitoring AI agents for harmful actions and one telling the model to treat what the agent wrote or was shown as data and ignore any instructions in it. Instructions you pass replace both, so say the second yourself.
        - `{{ policy }}`: `policy`, the policy the agent must follow whatever its task, between `[BEGIN POLICY (nonce)]` and `[END POLICY (nonce)]` lines; empty when there is none.
        - `{{ task }}`: the task the agent was given (in an eval, the sample's input), between `[BEGIN TASK (nonce)]` and `[END TASK (nonce)]` lines; empty outside an eval.
        - `{{ eval }}`: the eval the agent is running in, as `context.eval` holds it, such as `{{ eval.task }}` or `{{ eval.metadata.difficulty }}`; None outside an eval, so guard it with `{% if eval %}` in a monitor that may run outside one.
        - `{{ escalations }}`: the escalations from earlier links of a `sequential` composition (`step.escalations`), one `- name: explanation` line each, between `[BEGIN ESCALATIONS (nonce)]` and `[END ESCALATIONS (nonce)]` lines; empty when there are none.
        - `{{ step }}`: `step_as_str()` of the step or excerpt, its messages numbered `[M1]`, `[M2]` and so on so the explanation can cite them; the observation's `references` come from those cites.
        - `{{ sentinel_nonce }}`: a nonce made on the sample's first `score` and used for the rest of the sample, so the monitor model's prompt cache keeps working from step to step; 8 hex characters, on every block marker and, where the installed Scout takes a nonce, every message header (`[M1] USER (3f9a1c7e):`). The agent never sees the prompt, so text it writes that looks like a marker or a header lacks the nonce, and the default template tells the model so. The nonce is kept in the sample store, so a tool that shows the agent the store, or a protocol that passes the agent a monitor explanation quoting the nonce, gives it away for the rest of the sample. A `sentinel_nonce` in `template_variables` is ignored.
        - `{{ dimensions }}`: `dimensions` as given, or each dimension's name followed by its text.
        - `{{ answer_format }}`: the instruction to call the answer tool.
        - Any name in `template_variables`, set to its value.

        A template that uses a name that is none of these raises before the model is asked, so a misspelled variable is caught. The template is rendered once and values are inserted as text, so the agent's text is never read as template. A custom template states the 0.0 to 1.0 scale in its own words; without it the answer tool still rejects scores out of range.

        Args:
            step: The step to score, whole, or an excerpt of it from a windowing function.
            dimensions: What to score and how. A string is the prompt text for a single score, and the observation's `suspicion` is one float. A mapping is each dimension's name to its prompt text, and `suspicion` is a dict keyed by name; a name is 1 to 64 letters, digits, `_`, `.` or `-` (what providers accept as a tool parameter name), must not start with `_`, and must not be `explanation` or an attribute of pydantic's `BaseModel`.
            instructions: The monitor's own instructions, such as its role and the agent's setting, in place of the default instructions.
            policy: The policy the agent must follow whatever its task says, such as which files it may change; shown before the task.
            template: A replacement for the default prompt template, a Jinja template.
            template_variables: More variables for the template, by name; a name must not be one of the variables above.
            model: A model name, as for `generate`.
            role: A model role, as for `generate`.
            config: Generation configuration.
            retry_refusals: How many times to ask again when the provider refuses to answer.
        """
        return await score_step(
            self,
            active_step("Host.score()").store,
            step,
            dimensions,
            instructions=instructions,
            policy=policy,
            template=template,
            template_variables=template_variables,
            model=model,
            role=role,
            config=config,
            retry_refusals=retry_refusals,
        )

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        """Ask a person to decide about a step, and wait for the answer.

        The host is handed the step itself and renders it by its type: before a tool call the call, its view, the conversation and `step.escalations` (who escalated and why); after a call the same with the result the model is about to receive; after a generate the whole assistant turn, its text and every tool call it makes, with `step.escalations`. A host raises a clear error for a stage it cannot render. When the person ends the prompt without one of the `choices`, the host answers `reject` if it was offered and `terminate` otherwise, so the step fails closed; `human()` accepts that `terminate` even when it was not offered. It notifies the person and marks the sample as awaiting a person while it waits. `human()` calls this; most protocols never need to.

        Args:
            step: The step to decide about.
            choices: What the person may pick, from `approve`, `reject`, `terminate` and `modify`, in the order to present them.
        """
        return await self._native.ask_human(step, choices)


@runtime_checkable
class Recorder(TypingProtocol):
    """Where the runner records every report. Runner-facing; authors never call it."""

    def record(
        self, context: Context, factory: str, step: Step, reported: Reported[Report]
    ) -> None:
        """Record one child's report for the step it examined.

        Args:
            context: The child's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the child examined.
            reported: The child's report with its instance identity.
        """
        ...

    def failed(
        self, context: Context, factory: str, step: Step, failed: Failed
    ) -> None:
        """Record that a monitor function raised instead of reporting. The step goes on; a protocol that reads the layer's observations fails unless it checks `Observations.failed` first.

        Args:
            context: The monitor's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the monitor was examining.
            failed: The failure with its instance identity.
        """
        ...

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        """Record that a child was cancelled before it finished: a sibling decided `terminate` or called `decide_final()`, a sibling raised, or the layer was cancelled from above. Any reports it already made, from the earlier functions of an instance whose factory returned several, stand.

        The cause is not recorded here; a `terminate`, a `decide_final()` or an exception in the same layer says which it was.

        Args:
            context: The child's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the child was examining.
            name: The child's instance name.
        """
        ...

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        """Record that a final decision from a descendant passed this layer without running its decision logic. The step ended with that decision unless a later superseded record says otherwise.

        The final decision itself is recorded through `record` for the protocol that made it when it takes effect, at the root, after the `bypassed` record of each layer it passed.

        Args:
            context: The child's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the child was examining.
            name: The child's instance name.
        """
        ...

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Reported[Decision]
    ) -> None:
        """Record a final decision that lost a race to another final decision in the same layer, or that an exception in the same layer outranked; it did not take effect.

        Also records a `terminate`, already recorded through `record`, that a sibling's `decide_final()` outran in a runner-managed group (`run_children`, `run_protocols`): the step ended with the final decision instead. A protocol that fans out with its own task group gets no such record; its children's decisions stay as recorded.

        Args:
            context: The deciding protocol's context; its `path` identifies the instance.
            factory: Registry name of the instance's factory, package prefix included, e.g. `inspect_sentinel/concurrent`: the code that ran.
            step: The step the protocol was examining.
            reported: The decision that did not take effect, with its instance identity.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class HostContext:
    """What the host passes to `run_sentinel`: its `HostNative`, where the runner records reports, the sample store, and, beside a proxy, what the proxy says about the request.

    It builds the top layer's `Context` from the host, with an empty `path` and a `Host` wrapping it. Monitors and protocols reach the store only through `Context.store_as()`, which namespaces it by their `path`; `Host.score` keeps the sample's nonce in it too.
    """

    host: HostNative
    """The host's own implementation of the eval, inference and asking a person."""

    recorder: Recorder
    """Where the runner records reports, failures and cancellations."""

    store: Store
    """The sample store that `Context.store_as()` and `Host.score` read and write while the step runs."""

    proxy: ProxyContext | None = None
    """What a proxy says about the model request the step was found in, for a host beside a proxy. None in an eval's own process."""

    def __post_init__(self) -> None:
        # a runtime check for hosts that are not type-checked against Recorder
        if not isinstance(cast(object, self.recorder), Recorder):
            raise TypeError(
                f"Recorder {type(self.recorder).__name__} must implement record(), failed(), cancelled(), bypassed() and superseded()."
            )

    @property
    def context(self) -> Context:
        """The top layer's context, as monitors and protocols will read it: an empty `path` and a `Host` around `host`."""
        # _context imports this module
        from ._context import Context

        return Context(path="", host=Host(self.host), proxy=self.proxy)


@dataclass(frozen=True)
class StepState:
    recorder: Recorder
    store: Store
    # instance path -> factory registry name, for every layer that has run in
    # this step; tasks share it through the context var, and no lock is needed
    # since they run on one event loop and each access is one dict operation
    factories: dict[str, str]


class Layer(NamedTuple):
    recorder: Recorder
    factory: str


_step: ContextVar[StepState | None] = ContextVar("inspect_sentinel_step", default=None)


@contextmanager
def running_step(recorder: Recorder, store: Store) -> Generator[None]:
    token = _step.set(StepState(recorder, store, {}))
    try:
        yield
    finally:
        _step.reset(token)


def active_step(caller: str) -> StepState:
    state = _step.get()
    if state is None:
        raise RuntimeError(
            f"{caller} works only while run_sentinel is running a step, in the monitor or protocol it called or a task that one started; it was called outside a step, for example after the monitor or protocol returned."
        )
    return state


def enter_layer(path: str, factory: str) -> None:
    factories = active_step("The runner").factories
    running = factories.setdefault(path, factory)
    if running != factory:
        raise ValueError(
            f"Instance path {path!r} already ran {running!r} in this step, so {factory!r} cannot run under it too. Give each instance its own name."
        )


def layer(context: Context, caller: str) -> Layer:
    state = active_step(caller)
    factory = state.factories.get(context.path)
    if factory is None:
        raise ValueError(
            f"{caller} was given a Context whose path {context.path!r} is not a layer of the running step. Pass the Context this monitor or protocol was given."
        )
    return Layer(state.recorder, factory)
