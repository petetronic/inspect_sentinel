from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallView, ToolChoice, ToolInfo, ToolResult

from ._report import Decision, Reported


@dataclass(frozen=True)
class HumanDecision:
    """A decision a person made through `human()` at an earlier step of the agent's conversation. `human_decisions` on a step holds them, and `human_decision_for()` finds the one covering a call.

    Use it to annotate the past call it covers when rendering the conversation into a prompt, such as "reviewed by a person: approved", so a monitor does not flag already-reviewed history again. It is not a standing permission: never lower suspicion of a new call because it resembles one a person approved. The same command can mean something else once the state has changed (an `rm -rf build/` approved in one working directory deletes something else after a `cd`), and an agent can get a benign call approved and then issue a similar harmful one.
    """

    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]
    """The stage of the step it was made at, as `SentinelEvent.stage` records it: one of the four stages, at each of which `human()` can ask."""

    step_id: str
    """The id of the step it was made at, as `SentinelEvent.step_id` records it: the tool call id at a tool stage, the generate's id (the `uuid` of its first `ModelEvent`) at a generate stage."""

    call_ids: tuple[str, ...]
    """The tool calls the decision covered: the step's call at a tool stage, and every call the turn makes at `model_output`. How a prompt says which calls a person approved. Empty for a review of a turn with no tool calls, which has no key to annotate a message by: `step_id` is then the generate's id, which no message in the conversation carries."""

    decided: Reported[Decision]
    """The `human()` instance that asked (`name`, `path`, `function`) and a detached copy of its decision, which carries nothing of the step it was made at beyond its own fields: `approve` is a `proceed`, and the person's reason is the `explanation` (and, for a `reject`, the `message`). The reason is untrusted text, which can quote the agent, so quote it when rendering it into a prompt.

    It is what the person decided, not what happened to the step: beside other protocols in a `concurrent()`, a peer's `reject` outranks the person's `approve`, so a call a person approved may not have run. Mark only calls that ran, which have a result without an error in the history (a call rejected before it ran also gets a tool message, with an `approval` error, so a tool message alone does not prove the call ran).

    A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, a `reject` or `terminate`, and its `explanation` says so (for example "ACP client cancelled the permission request."). Render the explanation, and do not present such a decision as the person's judgment.
    """


@dataclass(frozen=True)
class BeforeGenerate:
    """A generate by the agent's model, before the request is sent and before the cache lookup.

    `input`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. `input`, `agent_messages`, `tools` and `config` are the host's own objects, borrowed for the step, not copies, and `history` holds the logged copies of what was sent: monitors and protocols must not mutate any of them (copy first to edit), since a change could alter the agent state, the real request or the log.
    """

    model: str
    """The model the request is for, as Inspect names it, such as `anthropic/claude-haiku-4-5`. Behind a proxy it is Inspect's name where an Inspect client gives it, otherwise the provider and the model the request names; `Context.proxy.model` is the proxy's own name for it."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: Sequence[ChatMessage]
    """Exactly what the model is sent: the caller's messages after the model's own preparation (the configured system message, reasoning history, merged consecutive messages). In-process these are the host's own messages; monitors must not mutate them."""

    agent_messages: Sequence[ChatMessage] | None
    """The outermost running `@agent` agent function's own message list: the `messages` of the `AgentState` object it was called with, read live at every stage. A scaffold may curate this list, for example to keep hidden side-task instructions out of a control monitor's view, and it may contain retries. For the actual model traffic, including instructions sent but kept out of this list, use `input` and `history`. To omit the request lists and show the agent's own list, use `StepPreprocessor(input_window=None, history_window=None, agent_messages_window=AllMessages())`. Other stage content remains: every stage renders the request's tools and tool choice while `input` is known (a tool stage's are the producing request's); `AfterGenerate` renders the reply, including reasoning by default before the scaffold has curated it; both tool stages render the assistant text and call; and `AfterToolCall` renders the result. Strict blindness also needs a stage-specific `transform` or a custom prompt that curates these surfaces; a transform can clear `tools` and `tool_choice`. The list is never approximated from `input` or `history`. A compaction does not drop turns from the list and adds its summary message (overflow recovery also drops the reply that overflowed); `input` carries the compacted request. With `truncation` set, `react`'s overflow filter trims this list when forced compaction is not available or does not help (no compaction, a model agent as `model=`, a failed forced compaction, or a second overflow right after one), while `history` keeps the turns.

    The same source is read by reference at all four stages. An agent that continues with another state object shows the object it was called with; `react` keeps its own object when a model agent returns a new one. Sub-agents run through handoff or `as_tool()` have their own conversations and lists. Hidden input stays out only when the scaffold that owns the conversation is the outermost `@agent` agent function.

    `None` when no `@agent` agent function is running (for example a solver calling `generate()`); a callable-object or `partial` agent does not register this list. Also `None` for bridged agents, in replay, and under a sidecar or proxy host. Monitors must treat a present list as read-only and must not mutate it.
    """

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent: the inputs of all its model calls taken together, each message once, where it was first sent. The opening prompts as the model saw them come first, a message the scaffold rewrote before sending shows as it was sent, and turns a compaction later dropped are still here. What was already sent is decided by content and position, not by message id: each call's input is aligned against the previous call's. So a message the host rebuilds with a fresh id for every call, such as a configured system message, is not repeated where it was, while a message sent again changed appears again as its new version, where that version was sent, as do a message moved behind a later one, a message one call dropped and a later call restored, and an equal message sent again at a later turn. inspect_ai sends an earlier assistant turn without its reasoning under `reasoning_history="last"`, so such a turn appears once with its reasoning and once without. Each message shows as it was sent: a message the scaffold edits or elides in place after sending it (as `react`'s `on_continue` can elide an old tool result) keeps its sent content here, and appears again as its new version if it is sent again. The model's outputs appear once a later call sends them back. It does not yet include `input`, which is about to be sent. Only this agent instance's model calls count: a sub-agent or a parallel agent has its own conversation, and a compaction summary, a scorer or a monitor is not an agent turn. A model call the scaffold makes itself inside its agent loop does count, unless the scaffold marks it with `inspect_ai.model.model_origin("helper")`, which also leaves it unchecked, or runs it as an agent of its own, through `inspect_ai.agent.run()`. Sub-agents inside one bridged CLI share its conversation. `None` when the host cannot produce it: in a replay from a log, until replay is built. In-process Inspect always provides it. Beside a proxy, the sidecar builds it from the requests it sees, which carry neither message ids nor the agent instance, so it differs: messages are matched by what they say, a conversation is told apart by how it opens, so agents whose conversations open alike share one history, every model call of the conversation counts as an agent turn, and a sidecar that joins a conversation midway starts the history there. Unlike `agent_messages`, it is what was sent: a rebuilt or compacted input, and instructions a scaffold sends but keeps out of its own list, are here; outputs and results the model was never sent back are not. In-process the messages are the copies of what was sent that the transcript's `ModelEvent`s hold, not the host's own objects; monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent's conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: list[ToolInfo]
    """The tools the model is offered, after `tool_choice` filtering. In-process these are the host's own objects; monitors must not mutate them."""

    tool_choice: ToolChoice
    """The tool choice the model is sent."""

    config: GenerateConfig
    """The generation configuration the model is sent, with the task's and the model's settings merged in. In-process this is the host's own object; monitors must not mutate it. Behind a proxy it is read from the request, so it holds what the model was sent, the provider's defaults included, and settings that govern only Inspect's own client, such as `max_retries`, `timeout` and `max_connections`, are None."""


@dataclass(frozen=True)
class AfterGenerate:
    """A generate by the agent's model that has completed, before its output reaches the agent.

    `input`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. `input`, `agent_messages`, `tools`, `config` and `output` are the host's own objects, borrowed for the step, not copies, and `history` holds the logged copies of what was sent: monitors and protocols must not mutate any of them (copy first to edit), since a change could alter the agent state, the real request, the response the agent receives or the log.
    """

    model: str
    """The model the request was for, as `model` is at `BeforeGenerate`; `output.model` names the model that answered."""

    conversation: str
    """Id linking this agent's steps across compactions."""

    input: Sequence[ChatMessage]
    """Exactly what the model was sent, as for `BeforeGenerate`; monitors must not mutate it."""

    agent_messages: Sequence[ChatMessage] | None
    """The outermost running agent function's own message list, read live by reference, or `None`, as for `BeforeGenerate.agent_messages`. The agent may not have added `output` yet. Monitors must not mutate it."""

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent, or `None` when the host cannot produce it, as for `BeforeGenerate.history`. It includes `input`; `output` is not in it yet. Monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent's conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: list[ToolInfo]
    """The tools the model was offered; monitors must not mutate them."""

    tool_choice: ToolChoice
    """The tool choice the model was sent."""

    config: GenerateConfig
    """The generation configuration the model was sent, as `config` is at `BeforeGenerate`; monitors must not mutate it."""

    output: ModelOutput
    """What the model returned; the first choice's message is what the agent receives. Some providers return an output with no choices, so check `output.choices` before reading `output.message`. In-process this is the host's own output, which the agent receives; monitors must not mutate it."""


@dataclass(frozen=True)
class Truncation:
    """A cut the host made to a tool's output before the model saw it: `AfterToolCall.truncated` when there was one."""

    original_bytes: int
    """The size of the tool's output, in UTF-8 bytes of its text."""

    limit_bytes: int
    """The limit the output was cut to, in bytes."""


@dataclass(frozen=True)
class BeforeToolCall:
    """A tool call about to execute.

    `input`, `tools`, `tool_choice` and `assistant_message` all come from one model call: the one that produced the call (in-process, the transcript's `ModelEvent`). So they are the same live and in a replay from a log.

    They are set or `None` together. They are `None` when no model call produced the call (a pre-filled message, a human agent, or a call the scaffold made itself), when the transcript event that recorded the model call is no longer available, or when the host does not have it (a replay from messages). A scaffold that rebuilds the model's message under a new id before running its calls gets `None` live, while a replay from a log finds the model call.

    `input`, `tools`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps for the conversation rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. Monitors and protocols must not mutate them or the messages and tool definitions in them.
    """

    conversation: str
    """Id linking this agent's steps across compactions."""

    message: str
    """Assistant text alongside the call."""

    message_id: str
    """Id of the assistant message that made the call, so a monitor model can cite it."""

    assistant_message: ChatMessageAssistant | None = field(default=None, kw_only=True)
    """The assistant message that made the call, as the model returned it: its text, its reasoning and all of its tool calls, this one among them (find it by `call.id`). It is set for a solver that calls `generate()` in its own loop.

    `None` when the model call that returned it is not available (see `BeforeToolCall`).

    It shows the calls as the model proposed them: an approver's or a sentinel's `modify` changes `call`, not this message. Through a sentinel sidecar, the calls the sidecar took out are redacted, as in the `ModelEvent`. A turn's steps share one object, the host's own: monitors must not mutate it."""

    call: ToolCall
    """The call as it will execute."""

    view: ToolCallView
    """Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise."""

    input: Sequence[ChatMessage] | None
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when that model call is not available (see `BeforeToolCall`); it is never approximated from `agent_messages`. `context.eval.sample_input` is the sample's input, the assignment."""

    agent_messages: Sequence[ChatMessage] | None
    """The outermost running agent function's own message list, read live by reference, or `None`, as for `BeforeGenerate.agent_messages`. For `react` and other scaffolds that append to the state they were called with, it includes the assistant message holding the call. Monitors must not mutate it."""

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent: the inputs of all its model calls taken together, each message once, where it was first sent. The opening prompts as the model saw them come first, a message the scaffold rewrote before sending shows as it was sent, and turns a compaction later dropped are still here. What was already sent is decided by content and position, not by message id: each call's input is aligned against the previous call's. So a message the host rebuilds with a fresh id for every call, such as a configured system message, is not repeated where it was, while a message sent again changed appears again as its new version, where that version was sent, as do a message moved behind a later one, a message one call dropped and a later call restored, and an equal message sent again at a later turn. inspect_ai sends an earlier assistant turn without its reasoning under `reasoning_history="last"`, so such a turn appears once with its reasoning and once without. Each message shows as it was sent: a message the scaffold edits or elides in place after sending it (as `react`'s `on_continue` can elide an old tool result) keeps its sent content here, and appears again as its new version if it is sent again. The model's outputs appear once a later call sends them back. It includes the input of the call that produced this tool call. Only this agent instance's model calls count: a sub-agent or a parallel agent has its own conversation, and a compaction summary, a scorer or a monitor is not an agent turn. A model call the scaffold makes itself inside its agent loop does count, unless the scaffold marks it with `inspect_ai.model.model_origin("helper")`, which also leaves it unchecked, or runs it as an agent of its own, through `inspect_ai.agent.run()`. Sub-agents inside one bridged CLI share its conversation. `None` when the host cannot produce it: in a replay from a log, until replay is built. In-process Inspect always provides it. Beside a proxy, the sidecar builds it from the requests it sees, which carry neither message ids nor the agent instance, so it differs: messages are matched by what they say, a conversation is told apart by how it opens, so agents whose conversations open alike share one history, every model call of the conversation counts as an agent turn, and a sidecar that joins a conversation midway starts the history there. Unlike `agent_messages`, it is what was sent: a rebuilt or compacted input, and instructions a scaffold sends but keeps out of its own list, are here; outputs and results the model was never sent back are not. In-process the messages are the copies of what was sent that the transcript's `ModelEvent`s hold, not the host's own objects; monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent's conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: Sequence[ToolInfo] | None = None
    """The tool definitions offered to the model call that produced the call, as the transcript's `ModelEvent` recorded them, after any model hooks (so they can differ from that call's `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` when that model call is not available (see `BeforeToolCall`); it is never approximated from the tools available now. Monitors must not mutate them."""

    tool_choice: ToolChoice | None = None
    """The tool choice of the model call that produced the call, as the transcript's `ModelEvent` recorded it. `None` when that model call is not available (see `BeforeToolCall`)."""


@dataclass(frozen=True)
class AfterToolCall:
    """A tool call that has executed, before the model sees its result.

    `input`, `tools`, `tool_choice` and `assistant_message` all come from one model call: the one that produced the call (in-process, the transcript's `ModelEvent`). So they are the same live and in a replay from a log.

    They are set or `None` together. They are `None` when no model call produced the call (a pre-filled message, a human agent, or a call the scaffold made itself), when the transcript event that recorded the model call is no longer available, or when the host does not have it (a replay from messages). A scaffold that rebuilds the model's message under a new id before running its calls gets `None` live, while a replay from a log finds the model call.

    `input`, `tools`, `history`, `agent_messages` and `human_decisions` are read-only sequences that can share the lists the host keeps for the conversation rather than copy them; a monitor that wants a list calls `list()` on one, at its own cost. Monitors and protocols must not mutate them or the messages and tool definitions in them.
    """

    conversation: str
    """Id linking this agent's steps across compactions."""

    message: str
    """Assistant text alongside the call."""

    message_id: str
    """Id of the assistant message that made the call, so a monitor model can cite it."""

    assistant_message: ChatMessageAssistant | None = field(default=None, kw_only=True)
    """The assistant message that made the call, as the model returned it: its text, its reasoning and all of its tool calls, this one among them (find it by `call.id`). It is set for a solver that calls `generate()` in its own loop.

    `None` when the model call that returned it is not available (see `AfterToolCall`).

    It shows the calls as the model proposed them: an approver's or a sentinel's `modify` changes `call`, not this message. Through a sentinel sidecar, the calls the sidecar took out are redacted, as in the `ModelEvent`. A turn's steps share one object, the host's own: monitors must not mutate it."""

    call: ToolCall
    """The call as executed, after any `modify`."""

    result: ChatMessageTool
    """What the model will see, after truncation; `truncated` says whether it was cut."""

    output: ToolResult | None
    """The tool's untruncated return value, before any cut `truncated` reports. `None` when the host cannot see it: beside a proxy, which sees only the `result` the model is sent, and in a replay from a log, which records only that `result`. In-process Inspect always provides it. Judge what the model saw by `result`, which every host provides."""

    view: ToolCallView
    """Rendering of the tool call and its context: custom when the tool registers a viewer, a default Python-call rendering otherwise."""

    input: Sequence[ChatMessage] | None
    """Exactly what the model was sent for the generate that produced the call: what a monitor should judge the call against. `None` when that model call is not available (see `AfterToolCall`); it is never approximated from `agent_messages`. `context.eval.sample_input` is the sample's input, the assignment."""

    agent_messages: Sequence[ChatMessage] | None
    """The outermost running agent function's own message list, read live by reference, or `None`, as for `BeforeToolCall.agent_messages`. The agent may not have added `result` yet. Monitors must not mutate it."""

    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    """Every message this agent instance has sent to the model so far, in the order sent, or `None` when the host cannot produce it, as for `BeforeToolCall.history`. It includes the input of the call that produced this tool call; `result` is not in it yet. Monitors must not mutate them."""

    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    """Decisions a person made through `human()` at earlier steps of this agent's conversation (`conversation`), oldest first, for annotating the past calls a person already reviewed when rendering the conversation, so a monitor does not flag them again; `human_decision_for()` finds the one covering a call. Decisions from a parent or sibling agent's conversation are not included. Not a standing permission: mark the exact call, and never lower suspicion of a new call that resembles an approved one (see `HumanDecision`). A step is its stage and its step id, so a decision before a call is here after that call, and a decision about a turn at `model_output` is here at each of its calls' tool stages. It holds only decisions made at steps that had finished before this step began, so never one from this step or from a parallel step still running, and only those that took effect (one a sibling's `decide_final()` superseded is dropped); it never holds a monitor's observation or another protocol's decision. A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, with an explanation that says so; render it, never as the person's judgment. Empty when there are none; `None` when the host cannot tell, as under a proxy that keeps no record of them."""

    tools: Sequence[ToolInfo] | None = None
    """The tool definitions offered to the model call that produced the call, as the transcript's `ModelEvent` recorded them, after any model hooks (so they can differ from that call's `AfterGenerate.tools` when a hook changed them): what each tool does bears on how suspicious a call to it is. `None` when that model call is not available (see `AfterToolCall`); it is never approximated from the tools available now. Monitors must not mutate them."""

    tool_choice: ToolChoice | None = None
    """The tool choice of the model call that produced the call, as the transcript's `ModelEvent` recorded it. `None` when that model call is not available (see `AfterToolCall`)."""

    truncated: Truncation | Literal[False] | None = None
    """Whether the host cut the tool's output before the model saw it in `result`. A `Truncation` when it did; `False` when it did not, which is always so for structured content; `None` when this host cannot tell (a proxy, a replay from messages alone, or a log older than the record of cuts). `if step.truncated:` is true only for a cut. It is built from the value the transcript's `ToolEvent.truncated` records (`Truncation(*event.truncated)` when that is set), so a monitor reads the same answer live and from a log. A cut made before the host's limit applies, by a sandbox output limit or a timeout, is not reported here: `result.error` reports it, with type `"limit"` or `"timeout"`. Use this field, not the "output was too long" notice in `result`, to tell whether the host cut the result: that notice is text a tool's output can contain, so it can be forged."""


Step: TypeAlias = BeforeGenerate | AfterGenerate | BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage: a protocol, or a host."""


def human_decision_for(step: Step, call_id: str) -> HumanDecision | None:
    """The latest decision a person made through `human()` that covers the tool call `call_id`, from `step.human_decisions`.

    A call can be decided about more than once: its turn at `model_output` covers each of its calls, then the call itself before it runs and its result after, so the latest is the most specific. Look decisions up by call id, which the model's provider assigns and which stays the same however a scaffold copies or rebuilds its messages; message ids do not, so there is no lookup by message, and a review of a turn with no tool calls, which stays in `step.human_decisions`, is found by none. None when no decision covers the call, and also when `step.human_decisions` is None because the host cannot tell; check that field to tell the two apart. Use the result to annotate that exact call, not to judge a new call that resembles it (see `HumanDecision`).

    Args:
        step: The step being examined.
        call_id: The id of a tool call earlier in the conversation, as `ToolCall.id`.
    """
    for decision in reversed(step.human_decisions or ()):
        if call_id in decision.call_ids:
            return decision
    return None
