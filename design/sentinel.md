# Sentinel

Exploratory design for two new registry types, `@monitor` and `@protocol`: one annotated function per stage (a point in the agent loop), with one identity and one place to keep state. A monitor observes and returns evidence; a protocol decides and returns an action, consulting monitors if it has any. A protocol with no monitors is a rule.

*Sentinel* is the umbrella: the name of the feature, and of one configured tree — a protocol with its monitors — as passed to `Task(sentinel=)` and recorded as a `SentinelEvent`. Inside it, *monitor* and *protocol* keep their AI-control-literature senses, which are also ControlArena's: the monitor is a scorer, the protocol decides. An earlier draft used "monitor" for both the umbrella (`Task(monitor=)`) and the scorer, and reviewers read the umbrella as "the monitor plus the protocol"; renaming the umbrella rather than the scorer keeps the literature's pair intact.

Status: partly built. The tool stages, the decorators and groups, the runner, `decide_final()`, configuration, and `observe_only`, `concurrent`, `sequential`, `threshold` and `human` are implemented in `inspect_sentinel` and its inspect_ai integration (`Host.ask_human` in inspect_ai follows). The [failure policy](#failure-semantics) is implemented too. `Host.score` is built (`sentinel-deployment.md`, "The host ABI"); it is sentinel code on the host's `generate`, so hosts implement nothing for it. The generate stages are built: monitors observe them and protocols may `proceed`, `escalate` or `terminate` there; `reject` and `modify` are deferred ([generate-stages.md](generate-stages.md)). Step preprocessing (`StepPreprocessor`, its windows and `preprocess_step`) and `step_as_str` are built. `resample`, the other view helpers and per-task state are designed but not built (`pr-series.md`, "Deferred"); the Python for those is illustrative.

**Related design documents.** This one defines the protocol. `sentinel-deployment.md` covers running a sentinel outside the eval process — in a proxy on the wire — and the constraints that places on the protocol. `sentinel-development.md` covers measuring and calibrating a monitor before it acts, by replaying it over transcripts as an Inspect Scout scanner. `inspect-core.md` covers extracting the wire types into a package light enough for a monitor author (or another language) to depend on.

## Goal

A *monitor* watches an agent as it runs and reports what it sees; a *protocol* decides whether to intervene. Concretely, the things people build today:

- an LLM that reads each tool call and stops the sample if the call looks like sabotage or an exfiltration attempt
- a classifier that reads each tool *result* and stops the sample if the agent actually reached the internet (#5355)
- a running suspicion score accumulated over a whole trajectory, where no single step is disqualifying but the total is
- a heuristic that rewrites or annotates the model's input before each generate ("you have used 8 of your 10 allowed commands")

All four are the same idea seen at different stages of the loop. Inspect has no single object for it: each has to be assembled from a different primitive, and the pieces cannot share state without the author inventing a place to put it.

## What a sentinel is assembled from today

| Point in the loop | Primitive | Gaps |
|------------------------|------------------------|------------------------|
| Before a tool call | `@approver` | Sees nothing but the call. Fires only when the model calls a tool. |
| After a tool result | `@reviewer` (#5356, merged 2026-09-11) | Decision vocabulary is `continue` / `terminate` / `escalate` only — no score, no rewrite. Ships as a parallel subsystem to approval rather than sharing one. |
| Before generate | `Hooks.on_before_model_generate` | Process-global and env-enabled, so not per-eval or per-task; exceptions swallowed; no transcript record; no way to terminate. |
| After generate | — | Nothing. |

And state, which is the part that hurts most. An approver is constructed once when the eval is resolved and then called from every sample concurrently, so `self` is shared across samples — the same hazard the `Hooks` docstring warns about at length, with the same workaround (key a dict by `sample_id`, remember to evict it). Nothing carries across the four stages either: an approver that wants to know what the *previous* tool returned has to go fishing in `history`.

A monitor and a protocol are the two objects that make those four stages one thing with one state: the first observes, the second decides.

## Prior art {#prior-art}

A survey of the AI-control literature and of shipped guardrail APIs, done after the first draft of this design. It changed the architecture — see [Protocols](#protocols) — so it is recorded rather than summarised away.

### Independent convergence on the four affordances

The OpenAI Agents SDK ships exactly four decorators: `@input_guardrail`, `@output_guardrail`, `@tool_input_guardrail`, `@tool_output_guardrail`. That is `BeforeGenerate` / `AfterGenerate` / `BeforeToolCall` / `AfterToolCall`, arrived at independently. It uses a different return type per stage, which is the thing [What a monitor returns](#what-a-monitor-returns) argues against.

Its input guardrails also take a `run_in_parallel` flag: run the guardrail concurrently with the agent (cheaper wall-clock, the agent may burn tokens before cancellation) or block first (no wasted tokens, higher latency). That is a knob we do not have and probably want.

### The control literature separates scoring from deciding

This is the finding that reshaped the design.

[Ctrl-Z](https://arxiv.org/abs/2504.10374) formalises deployment as a protocol that observes each action and may **accept**, **resample**, **defer**, or **audit** it. The mechanics are specific:

- the audit budget is **1% of actions per episode**, allocated by exponential decay so scrutiny is stricter early, when attacks are likelier
- there are **two thresholds over one score** — a lower deferral threshold and a higher auditing threshold — both calibrated against known-benign score distributions and adjusted for remaining budget
- **resampling** draws up to two further samples and executes the least suspicious one that clears the deferral threshold
- **incrimination** exploits multi-sample variance: a malicious policy attacks consistently across samples, an honest one varies. A three-strikes rule audits when suspicion crosses threshold three times in a task
- **deferral** substitutes the weaker trusted model's action outright

The original AI Control paper adds **trusted editing**: the trusted model rewrites the untrusted output to remove suspicious parts, then submits the edit.

Two structural lessons:

1.  **The monitor is a scorer.** Thresholds are not monitor knowledge — they need calibration against benign distributions and depend on global remaining budget and position in the episode. Deferral, resampling and auditing are things a *protocol* does with a score.
2.  **`audit` is a second axis, not a peer action.** You can accept-and-audit or block-and-audit; Ctrl-Z's two thresholds sit over the same score.

### Structured scores resist collusion; single scores do not

[TraceGuard](https://arxiv.org/html/2604.03968v1) scores five independent dimensions — goal alignment, constraint adherence, reasoning coherence, safety awareness, action-trace consistency — with parallel independent LLM calls, and reports **95% safety versus 0% for single-score monitoring** when the untrusted monitor is also the agent. The mechanism is fragmenting evaluation so no single compromised model sees the whole assessment.

That turns "run monitors concurrently or in sequence" from a latency question into a safety one: independence is the point. It also argues the suspicion signal should be structured and first-class rather than a scalar buried in `metadata`.

### Collusion has a taxonomy, and the cheapest attack is the strongest

[Untrusted monitoring safety cases](https://arxiv.org/html/2602.20628v1) enumerate passive self-recognition (the monitor recognises its own style, no pre-agreed signal), active causal collusion (a pre-agreed signal), active acausal collusion (Schelling points), and combinations — and find that **passive self-recognition via fine-tuning beats the previous state of the art**. The strategy requiring no coordination at all is the hardest to stop.

Countermeasures are honeypotting (feed known-bad input to catch a dishonest monitor), signal jamming (minification, variable normalisation, natural language summarisation) and auditing.

All of these are **pipeline stages, not monitor logic**: paraphrasing sits between agent output and monitor input; honeypots are synthetic inputs injected into the monitor's stream. They belong to a protocol.

### Reasoning is a distinct signal

[CoT monitorability](https://arxiv.org/abs/2507.11473) argues reasoning traces are a monitorable safety signal that development choices can erode. TraceGuard's framing is sharper for us: *outputs look clean but reasoning is off*. Inspect has `ContentReasoning`, so whether a monitor sees it is a decision we have to make rather than inherit — and the evidence says yes, scored separately from actions.

### What shipped products converged on

- **Claude Code hooks** — `PreToolUse` returns `permissionDecision` of `allow` / `deny` / `ask`, plus `updatedInput` (rewritten arguments, not a whole call), `additionalContext` (the model sees it) and `systemMessage` (the operator sees it, the model does not). All matching hooks run in parallel; any `deny` blocks; context accumulates. `PostToolUse` can read the result but **cannot block or modify it** — the gap #5355 identified, in a shipped product.
- **Guardrails AI** runs a standalone Guard Server over HTTP, which is explicitly why it has the best non-Python story. That is our sidecar conclusion, independently reached.
- **NeMo Guardrails** uses Colang, a rule DSL, at 100–300ms added latency — the endpoint `sentinel-deployment.md` predicts a restricted Python subset converges on.
- **Portkey** integrates external guardrails by webhook, which is what the host ABI's `fetch` is for.
- Output guardrails under streaming are an open problem in the OpenAI SDK too.

### What we took, and what we left

Absorbed: the monitor/protocol split, structured scores, `audit` as a flag, prepackaged protocols, `additionalContext` and `systemMessage` as separate channels, the `run_in_parallel` knob, reasoning visibility. The disagreement between the two traditions about what a monitor emits was absorbed too, as [two registry types](#monitor-or-protocol) — monitors that observe, protocols that decide — rather than as a field convention.

Deliberately **not** in the core surface: honeypots, paraphrasing, resampling payloads, collusion-resistant untrusted monitoring. These matter to a small number of control researchers, carry real conceptual weight, and are better served by the protocol extension surface than by the sentinel protocol. See [Keeping it simple](#keeping-it-simple).

## Shape

A monitor and a protocol are each an async function, produced by a registered factory in the idiom every other Inspect decorator uses. **Which of the four stages it watches is given by the type of its second parameter** — there is no separate stage argument, and no base class. **Which of the two it is is given by the decorator**, and the return type must agree.

``` python
@protocol
def no_curl() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall, escalations: Escalations) -> Decision | None:
        """A rule: a protocol with no monitors. Called before a tool call executes."""
        if step.call.function == "bash" and "curl" in step.call.arguments.get("cmd", ""):
            return Decision.reject("uses curl", message="network access is out of bounds")
        return None

    return decide


@monitor
def suspicion_monitor(model: str | None = None, role: str | None = None) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        """A monitor: scores the step, never acts on the score."""
        ...

    return check
```

```python
Task(sentinel=no_curl())                                        # a rule acts
Task(sentinel=threshold(suspicion_monitor(), reject_at=0.8))    # a monitor acts through a protocol
```

The four payload types are the vocabulary:

| Payload | Fires |
|------------------------|------------------------|
| `BeforeGenerate` | before each generate, before cache lookup |
| `AfterGenerate` | after each generate, before the output reaches the agent |
| `BeforeToolCall` | before a tool call executes |
| `AfterToolCall` | after a tool call executes, before the model sees the result |

Two pairs: input/output around generate, call/result around a tool. `None` means "didn't participate". A monitor returns an `Observation`, which is evidence; a protocol returns a `Decision`, which is an action. See [What monitors and protocols return](#what-a-monitor-returns).

A monitor function watches one stage; a protocol function may watch one or all four. A concern spanning two stages is two functions: two monitors, or one factory returning both as a `MonitorGroup` when they share state — see [One function, one stage](#one-function-one-stage), which argues that this is rarer than it looks and that no function should watch two stages.

Each takes a single frozen dataclass rather than positional arguments, following `Hooks` rather than `Approver`. The parameter is named `step` rather than `Hooks`' generic `data`: all four payloads are the thing under examination, and a monitor examines a step in the agent's trajectory — the unit the control literature scores. `step.call`, `step.output`, `step.history` read as a subject; `data.call` reads as a bag. Per-stage names (`generate.input`, `call.call`) were the alternative, and the tool side sinks them. With four payloads that will grow fields, adding a field to a dataclass is additive; adding a positional parameter breaks every monitor.

### The payloads

``` python
@dataclass(frozen=True)
class BeforeGenerate:
    model: str
    conversation: str           # links this agent's steps across compactions
    input: Sequence[ChatMessage]  # exactly what the model is sent
    agent_messages: Sequence[ChatMessage] | None  # the outermost running agent function's own list, read live at every stage; None without one
    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)  # every message this agent instance has sent, in order sent; None when the host cannot produce it
    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)  # human() decisions at earlier steps of this conversation
    tools: list[ToolInfo]
    tool_choice: ToolChoice
    config: GenerateConfig


@dataclass(frozen=True)
class AfterGenerate:
    model: str
    conversation: str
    input: Sequence[ChatMessage]  # exactly what the model was sent
    agent_messages: Sequence[ChatMessage] | None
    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)  # includes input
    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    tools: list[ToolInfo]
    tool_choice: ToolChoice
    config: GenerateConfig
    output: ModelOutput


@dataclass(frozen=True)
class BeforeToolCall:
    conversation: str
    message: str                # assistant text alongside the call
    message_id: str             # id of the assistant message that made the call
    assistant_message: ChatMessageAssistant | None = field(default=None, kw_only=True)  # the reply that made the call, as the model returned it; None exactly when input is None
    call: ToolCall              # the call as it will execute
    view: ToolCallView
    input: Sequence[ChatMessage] | None  # the request that produced the call; None without one
    agent_messages: Sequence[ChatMessage] | None  # the same live agent list as at generate stages; None without one
    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)  # as for BeforeGenerate
    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    tools: Sequence[ToolInfo] | None = None  # the tools that request offered; None exactly when input is None
    tool_choice: ToolChoice | None = None  # that request's tool choice; None exactly when input is None


@dataclass(frozen=True)
class AfterToolCall:
    conversation: str
    message: str
    message_id: str
    assistant_message: ChatMessageAssistant | None = field(default=None, kw_only=True)
    call: ToolCall              # the call as executed (after any `modify`)
    result: ChatMessageTool     # what the model will see, after truncation
    output: ToolResult | None  # the tool's untruncated return value; None where the host can't see it (a proxy, a replay)
    view: ToolCallView
    input: Sequence[ChatMessage] | None  # the request that produced the call; None without one
    agent_messages: Sequence[ChatMessage] | None
    history: Sequence[ChatMessage] | None = field(default=None, kw_only=True)
    human_decisions: Sequence[HumanDecision] | None = field(default=None, kw_only=True)
    tools: Sequence[ToolInfo] | None = None  # the tools that request offered; None exactly when input is None
    tool_choice: ToolChoice | None = None  # that request's tool choice; None exactly when input is None
    truncated: Truncation | Literal[False] | None = None  # the host's cut; False: not cut; None: can't tell


@dataclass(frozen=True)
class Truncation:
    original_bytes: int         # size of the tool's output
    limit_bytes: int            # the limit it was cut to


@dataclass(frozen=True)
class HumanDecision:
    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]  # as SentinelEvent.stage
    step_id: str                # as SentinelEvent.step_id
    call_ids: tuple[str, ...]   # the calls it covered: the call at a tool stage, every call in the turn at model_output
    decided: Reported[Decision] # the human() instance and its decision; the person's reason is the explanation


def human_decision_for(step: Step, call_id: str) -> HumanDecision | None: ...  # the latest decision covering the call


Step: TypeAlias = BeforeGenerate | AfterGenerate | BeforeToolCall | AfterToolCall
"""The union, for code that handles any stage — a protocol, or a dispatcher."""



@dataclass(frozen=True)
class ReplayStep:               # a step rebuilt from a stored transcript (sentinel-development.md, R7)
    id: str                     # as SentinelEvent.step_id
    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]  # as SentinelEvent.stage
    step: Step
    segment: tuple[str, int]    # the conversation, and how many of its compactions came before
    recorded_answers: Mapping[str, HumanAnswer] = field(default_factory=dict)  # the step's own recorded human() answers, by instance path; never on the step (R7-D4)
```

`input`, `history`, `agent_messages` and `human_decisions` at every stage, and `tools` at the tool stages, are read-only `Sequence`s, so a host can hand each step a view of the conversation's lists cut at the step rather than a copy, which would make building the steps of a conversation quadratic in its length; a monitor that wants a list calls `list()` on one, at its own cost (R7-D6, decided 2026-10-19 in [sentinel-development.md](sentinel-development.md); the generate stages' `input` added 2026-10-20, since replay builds it as a view over the log's stored ranges). The generate stages' `tools` stay a `list`: they do not grow with the conversation. `AfterToolCall.output` is `None` where the host cannot see the tool's untruncated return value: beside a proxy, which sees only the result the model is sent, and in a replay from a log, which records only that result. Both were built with R7 PR 2, as was `ReplayStep`.

Which model calls the generate stages check, where the hook sits, and what each action means there are worked through in [generate-stages.md](generate-stages.md).

`AfterGenerate` carries the request that produced `output` — `input`, `tools`, `tool_choice`, `config` — as well as the output, because a protocol that substitutes an action (`resample`) has to regenerate with exactly what the agent was given.

The tool stages carry the producing request's `input` and `tools` (AISI R3), found by the same lookup (see [Two histories](#two-histories)) and `None` under the same conditions, because what a tool does changes how suspicious a call to it is and the tools a call was chosen from are the ones the model was offered, not the ones registered now. They are the tools as the `ModelEvent` recorded them, after any model hooks, so they can differ from that call's `AfterGenerate.tools` when a hook changed them. `AfterToolCall.truncated` says whether the host cut the result: a `Truncation` (the output's size and the limit it was cut to, in bytes) when it did, `False` when it did not, and `None` when the host cannot tell, as under a proxy or in a replay from messages alone. That follows the convention that `None` on a step means the host cannot provide the value (as for `input` and `agent_messages`), so a host that does not set the field says "unknown", never "not cut". It is built from the value `ToolEvent.truncated` records, so live and replay read one field from one source. A cut made earlier, by a sandbox output limit or a timeout, is reported by `result.error` (type `"limit"` or `"timeout"`), not here. Monitors use it rather than the "output was too long" notice in `result`, which is text a tool's output can contain. Live, `result` against `output` shows the cut too, but the log does not hold the untruncated `output`.

Every payload carries `human_decisions` (AISI R4): the decisions a person made through `human()` at earlier steps of the same `conversation`, oldest first. The use is a monitor rendering the conversation into its prompt and annotating each tool call a person already reviewed ("reviewed by a person: approved", with the reason quoted), so it does not flag already-reviewed history again in isolation; `human_decision_for(step, call_id)` finds the latest decision covering a call. They annotate past calls, and are not a standing permission: mark the exact call a decision covers, and never lower suspicion of a new call because it resembles one a person approved. The same command can mean something else once the state has changed (an `rm -rf build/` approved in one working directory deletes something else after a `cd`), and an agent can get a benign call approved and then issue a similar harmful one. A record is what the person decided, not what happened to the step, so a monitor marks only calls that ran, which have a result without an error in the history (a call rejected before it ran also gets a tool message, with an `approval` error, so a tool message alone does not prove the call ran). A prompt the person cancelled or closed without choosing is recorded as the surface's fail-closed answer, a `reject` or `terminate`, with an explanation that says so (ACP: "ACP client cancelled the permission request."; inspect_ai's own fallback: "The person closed the prompt without choosing; failing closed."): render the explanation, and do not present it as the person's judgment. A step is its stage and its step id, so a decision before call X is there at X's `AfterToolCall`, and a decision about a turn at `AfterGenerate` is there at each of its calls' tool stages, `call_ids` naming the calls a decision covered. It holds what was recorded before the step began, never a decision from the step itself, so monitors running beside a `human()` at the same step do not see its answer. Only `human()`'s decisions are kept: other protocols' decisions are left out because a `threshold()` decision carries its monitors' scores (`metadata["observations"]` and the explanation), and passing them on would break monitor independence (see [The surfaces](#the-surfaces)); a `human()` decision carries only the person's action and reason. The record is what the person decided, not what happened to the step: beside other protocols in a `concurrent()` a peer's `reject` outranks the person's `approve`. Live, inspect_ai's recorder keeps each `human()` decision as it records it, as a detached copy that carries none of the runner's record of the step (which reaches every decision there, `threshold()`'s scores included); `kept_human_decision()` in the integration module decides what is kept. It holds them until the step settles, dropping each that a supersession withdrew (`supersedes()`: a decision with the person's action superseded at the person's path or a layer above it, so a copy a composition passed up counts), and only then adds the rest to the conversation's list, which it snapshots when it builds a step; so a parallel step never sees a decision from a step still running, which might yet be superseded (review of #96, findings R2-1 to R2-3); replay reads the same records from the log's `SentinelEvent`s ([sentinel-development.md](sentinel-development.md), "Human decisions in replay"). `None` means the host cannot tell, as under a proxy that keeps no record of its `ask_human` answers. It is keyword-only so that it can sit beside `history`, as a fact about the conversation, ahead of fields that have no default.

Two limits of scope. A team's own review protocol (AISI R9: its own approval UI) does not appear in `human_decisions`, which holds only `human()`'s decisions; the path to general support is a per-protocol opt-in to share its decisions, which could fill or supersede this field ([pr-series.md](pr-series.md), Deferred, Later; not built). And the intended consumer is R8's shared renderer (item 8.5): an option that annotates the matching calls in a fixed format the agent cannot forge, so monitors do not each write the annotation by hand.

A monitor never annotates `Step`, since [its annotation is what selects the stage](#the-stage-comes-from-the-annotation). The alias exists for the layer above, which does see all four.

**Everything in `step` is agent-influenced; nothing in `context` is.** `call`, `message`, `agent_messages`, `history`, `view`, `result` and `output` are all written or shaped by the agent under observation and are therefore injection-bearing; the briefing is author-written, the store is the monitor's own, the host is the framework's. Splitting them across two arguments makes that boundary structural rather than a matter of field ordering, which is the most useful thing the framework can do about injection (see [It is the only input the agent did not write](#it-is-the-only-input-the-agent-did-not-write)). Which of these fields a given monitor is actually shown, and how they are rendered into its prompt, is the monitor's [view](#views).

`conversation`, `truncated` and `human_decisions` are the exceptions. The first two are framework-computed and carry no text. `human_decisions` is framework-computed too, from a person's answers; a person's typed reason may quote agent text, so a monitor rendering it should label it as a prior verdict.

`BeforeToolCall` and `AfterToolCall` are the `Approver` and `Reviewer` payloads. `AfterToolCall` is a faithful dataclass-ification of the merged `Reviewer.__call__`, which takes exactly `message`, `call`, `result`, `output`, `view` and `history` (the list passed to `execute_tools()`, which the payload carried as `agent_messages` until that became the outermost running agent's own list) — this document predicted the shape from the PR and it survived contact with the merged code unchanged. The before/after naming earns its keep here: `BeforeToolCall` carries a `call: ToolCall` and `AfterGenerate` an `output: ModelOutput`, so a payload named for the object it wraps would collide with a field inside itself.

#### Escalations are a protocol's parameter, not a field of the step {#escalations-are-a-parameter}

The payloads hold only facts about the conversation that the host provides and that read the same live and in replay. What earlier links of a `sequential` decided at this step is not such a fact: it is the chain's own state. So a protocol is `async def decide(context, step, escalations)`, where `escalations: Escalations` (an exported alias of `tuple[Reported[Decision], ...]`) holds the escalations of the links before it in an [ordered composition](#two-compositions-concurrent-and-ordered), in order, so a human at the end of the `sequential` sees who is asking and why. It is empty everywhere else: the root and every composition but `sequential` hand their children none, so escalations do not cross layers. A monitor takes `(context, step)` and is never handed them. That is enforced by the signature rather than by stripping a field at run time, and it keeps monitors independent: a `threshold` escalation carries every monitor's score and explanation in `metadata["observations"]`, which a monitor after it in a chain would otherwise read (AISI's R4 narrowing kept protocol decisions and other monitors' observations out of a monitor's view for the same reason). The reports are protocol-authored rather than agent-authored, but they may quote agent text in their explanations, so a protocol rendering them, as `human()` does through the host, labels them as prior verdicts rather than as instructions. Approval's equivalent is a block appended to `view.context`, which exists only on the tool stages and is absent under a proxy. Decided by the maintainer on 2026-10-14, replacing a `step.escalations` field that `sequential` set with `replace()`; a `ProtocolContext` with `.escalations` (it would break `Context`'s constant-for-a-sample invariant), a context variable (hidden state) and an `oversight` sub-object on the step were rejected. This is the one positional parameter added after release, while the package had almost no users, and there is no two-argument form.

#### Two histories: `input` and `agent_messages` {#two-histories}

Every payload carries `input`, `agent_messages` and `history` (described [below](#the-history-as-sent)). `input` is exactly what the model was sent for this generate, or for the generate that produced a tool call. At the tool stages it is `None` when no model call produced the call (a pre-filled message, a human agent, or a message the scaffold built) or its event is unavailable: the dispatcher finds the `ModelEvent` whose output is the call's assistant message, newest first, through the transcript's bounded-memory history, which also serves events evicted from memory. `agent_messages` is the outermost running `@agent` agent function's own message list, read live from the `AgentState` object it was called with. A scaffold may deliberately curate that list, including keeping hidden side-task instructions out of a control monitor's view. It can contain retries. The exact model traffic is in `input` and `history`; neither is derived from `agent_messages`, and `agent_messages` is never approximated from `input` or `history`.

inspect_ai reads the same state object's `messages` by reference at all four stages. An agent that continues with another state object shows the one it was called with; `react` keeps its own object. Sub-agents through handoff or `as_tool()` have their own conversations and lists. A compaction does not drop turns from the list and adds its summary message (overflow recovery also drops the reply that overflowed); `input` carries the compacted request. With `truncation` set, `react`'s overflow filter trims this list when forced compaction is not available or does not help (no compaction, a model agent as `model=`, a failed forced compaction, or a second overflow right after one), while `history` keeps the turns. Hidden input stays out of `agent_messages` only when the scaffold that owns the conversation is the outermost `@agent` agent function.

Two observations from the current approval-based monitor make the distinction matter. Scaffolds edit what they send: `react` composes and inserts a system message, a custom agent may transform a system or user message, and the bridge path rewrites the request (though bridged agents have `agent_messages=None`). A monitor reading the agent's own list therefore may not read what the model read. Compaction can separate them further: `input` has a summary and recent turns while the agent's list retains its turns and gains the summary. For “has this agent ever reached the internet,” `history` preserves the model traffic across compactions and overflow trimming; `agent_messages` answers what the scaffold currently holds and may omit turns. Approvers today get the list passed to `execute_tools()` and never the producing model input, so they cannot see the system prompt the model actually saw.

The runner therefore has two obligations, and neither field is derived from the other: carry the exact model input through to the tool stages, and provide the agent's live entry-state message list at every stage where the host has it. Both are evidence. The system prompt the model saw is `input[0]` when `input` is present, not `history[0]` or necessarily the first message in `agent_messages`. `agent_messages` is `None` when no `@agent` agent function is running, including a solver that calls `generate()` itself; a callable-object or `partial` agent does not register this list. It is also `None` for bridged agents; in replay, whose log does not record the list; and for sidecar and proxy hosts. A monitor checks for `None` instead of judging an empty list. It must treat a present list as read-only: mutating it changes the agent's state.

The [surfaces table](#the-surfaces) records where these fields are available. Under a proxy, `input` is the request and `agent_messages` is `None`: the proxy cannot read the agent's own state. A keyed store can accumulate an as-sent view from requests for `history`, not `agent_messages` ([sentinel-deployment.md](sentinel-deployment.md)).

#### The agent's own messages: `agent_messages` {#agent-messages}

`agent_messages` is the `messages` of the `AgentState` passed to the outermost `@agent` agent function in this conversation (inspect_ai #5986, 2026-10-21; [pr-series.md](pr-series.md), "`agent_messages` is the agent's own list"). It is read when each stage runs, by reference. A scaffold can leave hidden instructions out of it even when those instructions appear in `input` and `history`. To omit the request lists and show the agent's own list, use `StepPreprocessor(input_window=None, history_window=None, agent_messages_window=AllMessages())`. Other stage content remains: every stage renders the request's tools and tool choice while `input` is known (a tool stage's are the producing request's); `AfterGenerate` renders the reply, including reasoning by default before the scaffold has curated it; both tool stages render the assistant text and call; and `AfterToolCall` renders the result. Strict blindness also needs a stage-specific `transform` or a custom prompt that curates these surfaces; a transform can clear `tools` and `tool_choice`. The list can also include retries or other messages the scaffold retains. Monitors must not mutate it.

An agent that replaces its state object instead of mutating it continues to expose the earlier object it was called with. `react` retains its own state object when a model agent returns a new one. An undecorated scaffold does not establish this field itself: if it calls a decorated policy agent, that policy's entry state is the registered one only while that policy runs; stages after it returns, such as the scaffold's own tool calls, get `None`. Each sub-agent has a separate conversation and list.

#### The history as sent: `history` {#the-history-as-sent}

`history` is every message one agent instance has sent to the model, across all its calls, in the order sent (AISI R10, built 2026-10-14 as `sent` and renamed for 0.2.0, when the scaffold's list, until then `history`, became `agent_messages`; `pr-series.md`, "Renaming the histories"). It answers what `input` cannot, since `input` is one call, and what `agent_messages` cannot, since `agent_messages` is the agent's own list rather than what was sent, and a scaffold can rewrite messages before sending them (ControlArena replaces the opening system and user messages to build an attack policy), the model adds a configured system message, and a compaction drops the opening prompts from later requests. Like `input` and `agent_messages` it is a fact about the conversation, so it is a step field, filled for every monitor and protocol while a sentinel runs; it is keyword-only with a default of `None`, so the payloads' positional fields are unchanged.

- **What is in it.** The inputs of the instance's model calls (`conversation` names the instance), each message once, where it was first sent: the opening prompts come first, and turns a compaction later dropped stay. The outputs appear once a later call sends them back. At `BeforeGenerate` it does not yet include `input`; at `AfterGenerate` it does; at the tool stages it includes the input of the call that produced the tool call.
- **What counts as new.** Since inspect_ai #5990, what was already sent is decided by content and position, with no dependence on message ids. Content means the message without its id, `source` and `metadata`, none of which the model is sent. Each call's input is aligned against the previous call's input. The part that starts the same as the previous input is matched whole. Past it, a message equal to one in the previous input at or after the point the alignment has reached is matched, and the alignment moves past it. Anything else is new and is added where it was sent. So:
  - a message the host rebuilds with a fresh id for every call, such as the configured system message, is matched where it was, and is not repeated;
  - a message sent again changed appears again as its new version, where that version was sent, so each version is kept in the order sent;
  - a message inserted among sent ones is new, and the ones after it still match;
  - a message moved behind one that came after it is new again;
  - a message one call drops and a later call restores is new again, since only the previous call's input is compared;
  - two equal messages sent at different turns, such as the same nudge twice, are two messages.

  One host behaviour shows as an edit: under `reasoning_history="last"` inspect_ai sends an earlier assistant turn without its reasoning, so each such turn appears once with its reasoning and once without.
- **As sent.** `history` holds each message as it was sent: the copy of the input that the call's `ModelEvent` logs (inspect_ai #5883, which #5990 also uses for `history`), not the scaffold's own object. A message the agent edits or elides in place after sending it still shows as it was sent. When `react`'s `on_continue` rewrites an old tool result to `"[elided]"`, `history` keeps the result the model was sent, and if an edited message is sent again it appears again as its new version. A message that cannot be copied (a value that does not serialize or deep copy) is held live, in the log and in `history`, as before #5883. So `history` entries are not the same objects as those in `input` or `agent_messages`, though equal content still compares equal.
- **Whose calls.** Only the instance's agent turns, the calls whose `ModelEvent.origin` is `"agent"`, by the same rule the generate stages use ([generate-stages.md](generate-stages.md#a-which-calls)): a sub-agent or a parallel agent has its own conversation, and a compaction summary, a scorer, an approver and a monitor are not agent turns. A model call a scaffold makes inside its agent loop is an agent turn unless the scaffold marks it, so ControlArena's own monitor and compaction summariser calls are in the list (decided 2026-10-14: sentinel has no rule that tries to tell them apart). A scaffold keeps such a call out by wrapping it in `model_origin("helper")` from `inspect_ai.model` (or `"environment"` for an auditor or simulated user), which also leaves the call and its tool calls unchecked. It can instead run the call as an agent of its own through `inspect_ai.agent.run()`, which gives it its own agent span and so its own conversation. Sub-agents run inside one bridged CLI share the bridge's conversation.
- **`None`** when the host cannot produce it: a replay from a log, until replay is built (R7). The log now records which model calls were agent turns (`ModelEvent.origin`, R7 PR 1, inspect_ai #5990), and replay rebuilds `history` by folding a conversation's `"agent"` inputs from the log by the rule above, as sent, so it equals the live one (R7 PR 3; [sentinel-development.md](sentinel-development.md), "Replay over logs (R7)"); `agent_messages` stays `None` in replay. In-process Inspect always fills it while a sentinel runs. Beside a proxy, the sidecar builds it from the requests it sees, which carry neither message ids nor the agent instance, so it differs: messages are matched by what they say, agents whose conversations open alike share one history, and every model call of the conversation counts as an agent turn ([sidecar.md](sidecar.md#8-what-a-sentinel-is-given)).

inspect_ai builds it incrementally. Each agent turn's input is folded in when `Model.generate` records the call, before the request is sent, so a request in flight counts as sent. The host keeps, per conversation and weak-keyed by the transcript, the list of messages sent and a content key (a hash of the content) for each message of the previous input. The keys are computed on every send, so a message changed in place since it was last sent counts as new. The messages are the copies the `ModelEvent`s already hold, so `history` adds no copy of its own: the added memory is the lists' references and one key per message of the previous input, and the cost is one serialization and hash per message per agent call. Each step gets a copy of the list, not of the messages. Monitors must not mutate the messages: a change alters the in-memory log record, though no longer the agent's conversation. Nothing is folded while no sentinel is active. `ModelEvent.origin` is written whether or not a sentinel is active.

**Lineage.** Before inspect_ai #5990 a log could not link one agent's inputs across a compaction: a `CompactionEvent` carries a `role` and a `source` but no reference to the conversation it compacted, a `ModelEvent` carries only its innermost span, several agents can interleave their events in one sample, and a compaction summary call ran in the agent's span with nothing to mark it. So every step carries `conversation: str`, an id minted by whichever loop owns the messages — `react`, the generate loop, a bridge — when that conversation starts and held for its life, and `SentinelEvent` records it. Live, inspect_ai links its agent turns to that id as it makes them (above). Since inspect_ai #5990 a log records enough to replay `history`: `ModelEvent.origin` marks the agent turns (and a compaction summary call as `"compaction"`), and a call's conversation follows from the span tree, the innermost `agent` span above it, else the sample's uuid, which is the conversation the call had live. The one known exception is a call a bridged CLI's event sink attributes to a sub-agent span, which inspect_ai #6004, a follow-up to #5990, changes. No conversation id is recorded. Logs written before #5990 and other sources have no `origin`, so they get their turns from the transcript's timeline (R7 PR 6). Replay rebuilds an agent's as-sent history from those events (R7 PR 3; [sentinel-development.md](sentinel-development.md), "What the log records"). *Superseded: until 2026-10-19 this paragraph proposed a kind marker on `ModelEvent` as a likely future option.*

### The context

`Context` carries everything ambient to the step: the briefing, the monitor's own memory, and the effects it may perform. The briefing is the part that needs arguing for.

A monitor cannot judge anything without knowing what the agent is *supposed* to be doing. `curl attacker.example` is exfiltration in one task and the entire point of the eval in another; reading `/etc/shadow` is sabotage unless the assignment was privilege escalation. Without a briefing a monitor can only flag things that look alarming in the abstract, which is the failure mode that makes monitors useless — high false positives on exactly the evals that deliberately look alarming.

``` python
@dataclass(frozen=True, kw_only=True)
class EvalContext:
    """The Inspect eval sample being run: the briefing.

    Constant for the sample. None as `Context.eval` outside an eval, for
    example in a proxy, which answers requests that have no task, sample
    or epoch.
    """

    # -- the standing assignment, constant across a task
    task: str
    """Task name."""

    task_description: str | None
    """What agents in this setting are asked to do, and what is in bounds.
    None when the task sets no `description` or the host does not provide it."""

    # this particular instance of it
    sample_id: str | int
    epoch: int

    sample_description: str | None
    """What is distinctive about this instance. None when the sample sets no
    `description` or the host does not provide it."""

    sample_input: str | list[ChatMessage]
    """The prompt this agent was given; `step.input` is what the model was sent."""

    metadata: dict[str, Any]
    """Author-supplied structured context (task metadata merged with sample)."""

    target: Target | None = None
    """The expected answer. None unless explicitly opted into — see below.
    The opt-in is not built, so for now it is always None.

    Last because it is the only defaulted field, and a dataclass cannot put an
    undefaulted field after one that defaults.
    """

    @property
    def sample_input_text(self) -> str: ...   # sample_input as one string


@dataclass(frozen=True, kw_only=True)
class Context:
    """Everything ambient to the step under examination.

    Three roles, deliberately on one object because all three are constant
    for the sample while the step is not: the briefing (what the agent is
    supposed to be doing), the monitor's own memory, and the effects it may
    perform. Immutable: no field can be reassigned, and there is no way in
    to the sample store beyond the instance's own namespace.
    """

    path: str
    """Instance path, e.g. "attempt/internet_attempt". Empty at the top layer."""

    # -- the briefing, in an eval
    @property
    def eval(self) -> EvalContext | None:
        """The task and sample being run, from the host. None outside an Inspect eval."""

    # -- memory
    def store_as(self, model_cls: Type[SMT]) -> SMT:
        """Typed view of this instance's per-sample state, namespaced by `path`.

        Other scopes are planned; see [Per-task](#per-task).
        """

    # -- effects
    host: Host
    """Inference and asking a person today; outbound JSON (`fetch`) and keyed storage are planned.

    The portable route out of a monitor. Defined in `sentinel-deployment.md`;
    in-process these are ordinary implementations, in a proxy they cross the
    host boundary.
    """
```

#### Eval-only fields under `eval`

Sentinels also run in a proxy that answers requests that are not Inspect evals: there is no task, sample or epoch. Rather than make every eval field optional, or fill them with stand-ins that a monitor cannot tell from the real thing, the eval's fields live under `context.eval`, which is None outside an eval. A monitor that reads them checks for None once, and in an eval `task`, `sample_id` and `epoch` are always present. `path`, `host` and `store_as()` stay at the top, since every host has them. Decided by the maintainer on 2026-10-05; what a proxy supplies in their place, such as a deployment's charter, is left to the proxy spike (`proxy-host.md`).

#### Deployment-neutral, not `Task` and `Sample`

The obvious shape is to hand the monitor the `Task` and the `Sample`. Don't. That welds every monitor to Inspect and forfeits the portability the rest of this document spends its length preserving — a proxy has neither object and never will.

`Context` states what a monitor needs in terms of the agent rather than the framework, so both deployments can fill it:

| Field | In an eval | In a proxy |
|------------------------|------------------------|------------------------|
| `path`, `host`, `store_as()` | the instance path, the in-process host, the sample store | the instance path, the proxy host, its keyed store |
| `eval` | the task and sample: `Task` name, authored description, sample id, epoch, `Sample.input`, task + sample metadata | None |

#### It is the only input the agent did not write

Everything else a monitor sees — messages, tool arguments, tool output — is agent-influenced and is a prompt-injection surface, as [Inference, budget, and injection](#inference-budget-and-injection) notes. The context is the exception: an eval author wrote it, or a deployment operator did.

That makes it the anchor. A monitor should treat its context as instruction and the conversation as *evidence*, and the framework should keep that distinction structurally visible — separate fields, separately labelled when rendered into a monitor's own prompt — rather than concatenating briefing and transcript into one undifferentiated blob where a tool result can claim to be a revised assignment. This is the single most useful thing the framework can do about injection, and it is nearly free.

#### `target` is opt-in, and off by default

Handing the monitor the expected answer is occasionally necessary and usually wrong.

It is necessary for a monitor scoring whether the agent sandbagged, or whether it actually did the task rather than merely appearing to.

It is wrong for the central case, AI control evals, where the monitor stands in for one that would run in deployment — and a deployment monitor does not know the right answer. A monitor given the target scores better than its deployed counterpart ever could, so the eval stops measuring what it claims to. This is a validity problem, not an ergonomics one, and defaults decide it: if `target` is present by default, most monitors will read it without anyone deciding they should.

So: absent unless explicitly requested, and **recorded in the log when it was requested**, so a reader can tell whether a monitor's measured performance is achievable in deployment.

#### Passed on every call, not stashed once per sample

The context is constant for a sample, so handing it over once (at `on_sample_start`, say) and letting the monitor keep it looks tidier. It isn't: keeping it means state, and a monitor that needs state to know its own assignment can no longer be the pure function of one exchange that `sentinel-deployment.md` argues for. Passing it on every call costs a reference and preserves that property.

It is a separate argument rather than a payload field because it is invariant while the step is not — re-attaching an identical object to every payload invites the question of whether it could differ — and because two arguments state the monitor's job in the signature: here is what should be happening, here is what just happened. Two-argument callbacks are the local idiom (`Solver(state, generate)`, `Scorer(state, target)`), and OpenAI's guardrails likewise take context first.

The forward-compatibility argument for single-dataclass payloads still holds: it rules out *many* parameters, not two. Every future addition lands inside `Context` or inside a step payload, so the arity stays stable.

#### Where the author writes it

Neither `Task` nor `Sample` has a `description` field today — `Task` carries `name`, `display_name`, `version`, `metadata`, `tags`; `Sample` carries `input`, `choices`, `target`, `id`, `metadata`. So this needs somewhere to live. Two options:

- **`Task(description=)` and `Sample(description=)`**, general-purpose authored prose. Two small additive fields, no sentinel-specific API surface, and useful well beyond monitors — the viewer, model-graded scorers, and eval discovery all want "what is this task about" too. The risk is that a field named `description` attracts documentation written for humans reading the source, which the docstring already covers; the docstring for the field has to say it is a briefing for automated readers.
- **A `metadata` convention**, e.g. `Task(metadata={"description": ...})`. No API change at all, but undiscoverable and unvalidated.

The first, unless we are trying hard to avoid touching `Task` and `Sample`. Structured context stays in `metadata` either way.

### What monitors and protocols return {#what-a-monitor-returns}

A monitor **reports**; it does not act. What it hands back is evidence. A [protocol](#protocols) decides what actually happens, and what it hands back is an action.

``` python
Action = Literal["proceed", "modify", "reject", "terminate", "escalate"]
"""What a protocol can decide about the thing it just examined."""


class Observation(BaseModel):
    """Evidence, for a protocol to calibrate and threshold. What a monitor returns."""

    suspicion: Suspicion
    """How suspicious. Required — an Observation with no score says nothing."""

    explanation: str | None = Field(default=None)
    references: list[Reference] = Field(default_factory=list)
    """Messages and events the explanation cites, e.g. [M22]; see 'References' below."""
    metadata: dict[str, Any] | None = Field(default=None)


class Decision(BaseModel):
    """What should happen. What a protocol returns; advisory to any protocol wrapping it unless passed to decide_final()."""

    action: Action
    """Required."""

    audit: bool = Field(default=False)
    """Request that oversight budget be spent on this. Recorded and shown as a flag; nothing acts on it yet."""

    modified: ToolCall | ChatMessageAssistant | BeforeGenerate | None = Field(default=None)
    """Replacement for the step's subject, for a 'modify' decision. `ToolCall` only
    until the generate stages land."""

    explanation: str | None = Field(default=None)
    """Why, for the log. The agent never sees it; see 'Two audiences' below."""

    references: list[Reference] = Field(default_factory=list)
    """Messages and events the explanation cites."""

    message: str | None = Field(default=None)
    """Text the agent reads when the step is rejected. The agent sees nothing else from a decision."""

    metadata: dict[str, Any] | None = Field(default=None)


Report: TypeAlias = Observation | Decision
"""What the transcript records; the union nothing else needs to handle."""
```

The report type is the registry type. A `@monitor` function is annotated `-> Observation`, or `-> Observation | None` if it can abstain, a `@protocol` function `-> Decision` or `-> Decision | None`, and neither can return the other — see [Monitor or protocol?](#monitor-or-protocol) for how to pick, and [The kind is the decorator](#the-kind-comes-from-the-return-annotation) for how it is checked.

#### The name says the layering

`Decision.action` reads as a command, and under this split it is one: the thing that returned it is the thing configured to decide. What softens it is nesting — a child protocol's decision is advisory to the protocol above it, which may overrule it ([Child decisions are advisory](#child-decisions-are-advisory)) — and the transcript carries that as one event per layer: each records `action`, what that layer returned, so a child's event and its parent's together show what happened. An earlier draft called the type `Recommendation`, because a leaf monitor could also return one and "decision" was accurate only when nothing wrapped it. With leaves out of the picture the shorter name is accurate at every position, and the events of the layers above say the rest.

#### Progressive disclosure through defaults, not through the return type {#progressive-disclosure-through-defaults-not-through-the-return-type}

Most monitors touch one field of their report type. That does not require the return type to be polymorphic — every field but the required one has a default:

``` python
@monitor
def network_suspicion() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation(suspicion=0.9 if reaches_external_network(step.call) else 0.0)

    return check
```

An earlier draft accepted a bare `float` or `dict[str, float]` as well, on the theory that the simplest monitor should be a function returning a number. Dropped, for three reasons:

- **`Scorer` is the local precedent and it points the other way.** It returns `Score | None` — a required object — with the polymorphism *inside*, as `Score.value: Value`. That is the pattern to copy: wrap the flexible value, keep the return type fixed. (`Task(approval=)` accepting a str, a config object or a list is an *input* parameter; permissive inputs and permissive returns are different questions.)
- **A bare value has no name.** `return 0.9` gives a reader nothing — suspicion, or confidence that it is safe? `dict[str, float]` is worse: score dimensions, or metadata? Named fields settle it at the call site.
- **The saving is 24 characters**, against a coercion branch in the dispatcher and an ambiguity in every monitor anyone reads.

So the return type is the report type or `None`, matching `Score | None`. The polymorphism that genuinely helps stays inside, as `Suspicion`.

Progressive disclosure survives intact: a monitor sets `suspicion` and ignores the rest, and a rule sets `action` and never learns what `audit` or `modified` do until it needs them. That was always the useful half — the complexity reduction came from [pushing work up to the protocol](#protocols), not from shaving a constructor call.

#### Three states, all distinct

| Return | Who | Meaning | Recorded |
|------------------------|------------------------|------------------------|------------------------|
| `None` — *abstain* | either | didn't participate | no |
| `Observation` | a monitor | examined, no verdict — here is evidence | yes |
| `Decision` | a protocol | examined, and here is what happens | yes |

Call the first case **abstaining**, not "returning None". "Return `None`" and "`suspicion` is `None`" mean entirely different things — didn't participate, versus this report carries no score — and writing both as "None" in prose is the fastest way to confuse a reader. The API spells abstention `None`; the docs should not.

Abstaining rather than a sentinel value (`Report.none()`) follows `Scorer`, which returns `Score | None`, and avoids fighting Python: a rule with an early exit writes `if step.call.function != "bash": return`, which yields `None` whatever the signature says. A sentinel would either have to accept `None` anyway — a second spelling for one state, which is what the empty report was rejected for — or force a constructor call on the most common path, since a targeted rule abstains on nearly every call. The general rule: `None` is right for absence when there is exactly one kind of absence, it carries no data, and it is unambiguous with "not yet computed". All three hold here, and a null object buys nothing because a protocol must branch on "did anyone report" regardless.

Named constructors are still worth having, for common *valid* reports rather than for abstention — sugar over existing states, not new ones:

``` python
Decision.proceed()                     # action="proceed" — examined; let it proceed
Decision.reject("why")                 # action="reject"
Decision.reject("why", message="...")  # message is what the agent reads
Decision.terminate("why")              # action="terminate"
Decision.escalate()                    # action="escalate" — pass it on
Observation.score(0.9)                 # suspicion=0.9
Observation.flag("why")                # suspicion=1.0 — a detector, not a scorer
```

A monitor has no way to say `proceed`, and that is deliberate rather than a gap. Forcing a scorer to say `proceed` would be a lie — in Ctrl-Z the monitor genuinely has no view, because deferral versus audit depends on budget it cannot see — and it would corrupt the precedence rule, since a monitor reporting suspicion 0.95 would be voting `proceed`. Putting the two return types in two registry types is what makes that structural: a monitor has no `action` field to be forced to fill, and no decorator under which returning one type-checks.

An earlier draft had one type with every field optional and two validators — "at least one of `suspicion`, `action`, `audit`", and "`modified` iff `action == 'modify'`". The split absorbs the first entirely: `Observation.suspicion` and `Decision.action` are required fields, so `Report()` is not constructible and there is no error message to get right. The second validator survives, inside `Decision`, because it relates two of its own fields. It mirrors `Approval.modified`.

The alternative — allow an empty report and record it as a no-op observation — was rejected because it creates a fourth state indistinguishable in effect from `None`, while still producing a transcript record. "I examined this and found nothing" is already expressible, and better, as `action="proceed"` or `suspicion=0.0`.

This also keeps `proceed` meaningful rather than filler. **`proceed` = I examined this and it may proceed**; `escalate` = I decline to decide, pass it on; `None` = I wasn't involved. A `sequential` needs all three.

#### Monitor or protocol? {#monitor-or-protocol}

The split is not ours. It is the seam between the two bodies of prior art surveyed above, which disagree about what a monitor even emits:

| | Emits | Prior art |
|------------------------|------------------------|------------------------|
| **a monitor** | an `Observation` — a score, for someone else to threshold | Ctrl-Z's trusted monitor; TraceGuard's five dimensions. [The control literature separates scoring from deciding](#the-control-literature-separates-scoring-from-deciding) — "the monitor is a scorer", and accept/defer/resample/audit belong to the protocol |
| **a protocol** | a `Decision` — an action | every shipped guardrail: Claude Code's `permissionDecision: allow \| deny \| ask`, OpenAI's tripwire, Guardrails AI's pass/fail, NeMo's Colang rules |

The literature's word for the second thing is "protocol", and shipped products have no word for the first. So the names are the literature's, and the shipped-guardrail shape — a function that looks at one step and says what should happen — is a protocol with no monitors: a *rule*.

Examples of each:

| Thing | Write it as | Report |
|------------------------|------------------------|------------------------|
| LLM judge over a tool call | monitor | `suspicion=0.87` |
| TraceGuard-style dimensions | monitor | `suspicion={"goal_alignment": 0.1, "sabotage": 0.8, ...}` |
| Running trajectory score | monitor | accumulated in the store, reported each step |
| CoT monitor | monitor | reasoning scored separately from the action |
| `no_curl` | protocol (a rule) | `action="reject"` |
| `rm -rf /` guard | protocol (a rule) | `action="reject"` |
| Strip `--force` from a call | protocol (a rule) | `action="modify", modified=...` |
| A person | protocol (`human()`) | whatever they say |

Note that `Observation` keeps the shape [Progressive disclosure](#progressive-disclosure-through-defaults-not-through-the-return-type) already argued for: `Suspicion` is the flexible value, `Observation` the fixed wrapper — exactly `Score` and `Value`.

**Choosing is not determined by the use case.** #5355's network-egress classifier is legitimate either way: a monitor if you want a protocol to calibrate the cut point, a rule if the detection is definitive. One question decides it:

> **Do you want to own the threshold?** Yes → a protocol. No → a monitor.

And the split makes the failure this guards against unrepresentable rather than merely detectable. A monitor that computes a score and compares it to a constant has nothing to *do* with the comparison — it cannot return an action — so the constant has nowhere to hide. If you find yourself wanting to act on a score inside a monitor, you have found the reason `threshold()` exists. And if the check is definitive with no score in it — a string match, a path outside the sandbox — it is a rule, and a rule is a protocol.

A detector sits between: it has no score but it is not sure enough to be a rule, or its author wants it dry-runnable and backtestable before it acts. `Observation.flag()` is for that — a monitor reporting `suspicion=1.0` — and `threshold(detector(), reject_at=1.0)` turns it into a rule at configuration time. That path is a choice, not a default: a bare monitor [is an error](#monitors-need-a-protocol), and `observe_only()` records it without acting.

#### What the docs should lead with

A rule, then a monitor wrapped in `threshold()`.

[Keeping it simple](#keeping-it-simple) rules out leading with a scoring monitor: that means explaining what the number is, what consumes it, what a threshold is and why you do not set it — the entire control vocabulary inside "write your first monitor". A deterministic rule needs none of it: `@protocol def no_curl()` returning `Decision.reject(...)` is the `@approver` experience with more stages to attach to, and it is what people arrive expecting, having used Claude Code hooks or the OpenAI SDK.

The second example is a monitor under `threshold()`, which introduces the score and the constant together — the score in the monitor, the constant in the protocol — and that is the whole control vocabulary a first-time author needs. `observe_only()`, calibration and the shipped control protocols come after, at the moment someone's check produces a number they are not sure where to cut.

#### Suspicion is structured

``` python
Suspicion: TypeAlias = float | dict[str, float]  # finite scores; a dict has at least one key
```

A scalar is accepted, but the multi-dimensional form is the one to encourage: TraceGuard reports 95% versus 0% safety for structured versus single-score monitoring against a colluding monitor (see [Prior art](#prior-art)). A protocol can threshold per dimension, and dimensions can be scored by independent monitors so no single model sees the whole assessment.

#### `audit` is a flag, not an action

Auditing — spending scarce human review — is orthogonal to what happens to the action. You can proceed-and-audit or reject-and-audit. Ctrl-Z's deferral and auditing are two thresholds over one score, not two values in one enum.

`audit` lives on `Decision` rather than `Observation`, because "spend human review on this" is itself a policy statement; a monitor flags something by scoring it high. The budget is the outermost protocol's: an inner protocol *requests* an audit, the outer one grants it if budget remains.

For now `audit` is recorded for offline analysis and shown in the viewer as a flag; nothing acts on it yet, since no shipped protocol owns a budget.

An earlier draft called this the first concrete consumer of per-task state. It is not. Ctrl-Z's budget is **1% of actions per episode** ([Prior art](#prior-art)), and an episode is one agent trajectory — one sample. So the canonical audit budget is per-*sample* state, which `context.store_as(BudgetState)` already handles with no new machinery. A budget pooled across samples is a different and also defensible design, but it is not what the literature specifies, and [Per-task](#per-task) is not on the critical path because of it.

#### Final decisions {#authoritative-reports}

[Child decisions are advisory](#child-decisions-are-advisory): whatever wraps a protocol may honour or ignore what it decided. One class of decision is not: a decision its author means as the last word, which no threshold, budget or aggregation above it gets to turn into `proceed`. The doc first carried that as a prose rule, then as a `binding` field the runner enforced as a floor. It is now control flow:

``` python
def decide_final(decision: Decision) -> NoReturn:
    """End the step with this decision."""
```

A protocol that has the last word calls `decide_final(decision)`, and the step ends there. Nothing above the calling protocol runs: each layer's `await run_protocols(...)` or `await run_children(...)` never returns, so its decision logic never sees the decision and cannot weaken it. Siblings still in flight are cancelled and recorded as cancelled, as they are for `terminate`, and the decision is the step's outcome. The runner still [checks its shape](#the-boundary-check), records it as the decision of the protocol that called `decide_final()`, and records each layer it passed as `bypassed` (see [Transcript](#transcript)). The dispatcher invokes the resolved root through `run_sentinel`, which treats it as a layer at the empty path: the root's decision is shape-checked and recorded under its registry name with `path=""`, so its children's paths stay bare; a `decide_final()` from below records the root as `bypassed`, and `run_sentinel` returns the final decision as the step's outcome, so the dispatcher never catches `Final`.

No shipped protocol calls it: `human()` returns the person's answer as an ordinary decision (see [Humans in the loop](#humans-in-the-loop)). Any protocol may call `decide_final()` — the framework cannot verify the claim to the last word, only that the author made it — but the claim is visible in the transcript and in code review, which is the same discipline `portable=False` relies on.

#### Two audiences

Claude Code hooks separate `additionalContext` (the model sees it) from `systemMessage` (the operator sees it, the model does not). We do too, and the reason is security rather than tidiness: **telling an agent why it was flagged teaches it to evade.** `explanation` is log-only: the agent never sees it, so it can carry the full reasoning. `message` is the one field the agent reads, set explicitly with `Decision.reject(explanation, message=...)`; a reject without one gets the host's default "not approved" text. This deliberately differs from inspect's `Approval.explanation`, which *is* shown to the model; the planned approval adapter maps an approver's explanation to `message`.

`message` is legal only on a `reject` for now, and the shape check rejects it on any other action. The same field would give us the soft intervention we otherwise lack — a way to say "proceed, but the model should know X" without rejecting anything — but that needs a way to deliver text to the agent while the step proceeds, which does not exist yet.

#### References

`references` on both report types points at the messages and events an explanation cites. It is a list of `inspect_ai.scorer.Reference` (`type` `"message"` or `"event"`, an `id`, and the optional `cite` text such as `[M22]`), the type Scout's `Result.references` uses, so Inspect View can render cites in a sentinel event's explanation as links, as it does for scanner scores. A monitor using `message_numbering()` fills it from `extract_refs(explanation)`. `threshold` copies the references of the monitor its explanation names onto its decision, since the explanation carries that monitor's.

#### One vocabulary across stages

This is a different axis from the split above. `Observation` and `Decision` split by *who emits it* — a monitor or a protocol; nothing here splits by *where it runs*. A `BeforeToolCall` monitor and an `AfterGenerate` monitor both return `Observation`, a protocol at either returns `Decision`, and `Action` is one Literal across all four stages. The stages look like they need different vocabularies; they mostly don't:

| Stage             | proceed  | terminate | escalate | reject | modify       |
|-------------------|----------|-----------|----------|--------|--------------|
| `BeforeGenerate`  | ✓        | ✓         | ✓        | ✓      | open (below) |
| `AfterGenerate`   | ✓        | ✓         | ✓        | ✓      | —            |
| `BeforeToolCall`  | ✓        | ✓         | ✓        | ✓      | ✓            |
| `AfterToolCall`   | ✓        | ✓         | ✓        | **—**  | **—**        |

`proceed`, `terminate` and `escalate` are universal. `reject` generalises further than it looks: at `BeforeGenerate` it means "do not run this generate, and tell the agent why"; at `AfterGenerate`, "discard this completion". It is genuinely excluded only at `AfterToolCall`, and for a *policy* reason — #5355's argument that withholding output tells the model its command failed when it did not, so the model's picture of the sandbox diverges from the sandbox. The same reasoning excludes `modify` on a result.

**`modified` carries a replacement for the *subject* of the step**, and the subject is stage-determined exactly as the payload type is:

| Stage | Subject | `modified` carries |
|------------------------|------------------------|------------------------|
| `BeforeGenerate` | the request | a `BeforeGenerate`, built with `dataclasses.replace(step, input=...)`; `model` unchanged |
| `AfterGenerate` | the assistant message | `ChatMessageAssistant` — **not** `ModelOutput` |
| `BeforeToolCall` | the call | `ToolCall` |
| `AfterToolCall` | — | `modify` is not legal, per the policy reason above |

The `AfterGenerate` row is the one that needed working out, and it is what unblocks resampling and trusted editing. `ModelOutput` is `model`, `choices`, `completion`, `usage`, `fallback`, `time`, `metadata`, `error` — of which only `choices[0].message` is substitutable *content*. `usage`, `time` and `model` describe the call that actually happened. If a protocol draws three completions and executes the least suspicious, three completions' worth of tokens really were spent, and swapping the whole `ModelOutput` would either discard that or falsify it. So the framework substitutes the message and leaves the accounting alone.

Only `choices[0]` is monitored and substitutable, matching `_bridge/_approval.py`'s `n = 1` restriction — alternate choices carrying tool calls that no monitor saw are the thing that restriction exists to prevent.

#### Why not in-place mutation, given the hook does exactly that {#why-not-in-place-mutation}

`Hooks.on_before_model_generate(data) -> None` mutates its payload, and its docstring promises those mutations reach the cache key and the API call. Worth understanding how, because it looks like it contradicts our frozen payloads: `BeforeModelGenerate` *is* `@dataclass(frozen=True)`. Frozen protects the *binding*, not the contents — and `input: list[ChatMessage]` and `tools: list[ToolInfo]` are mutable containers, so a hook appends to `data.input` rather than reassigning it.

We deliberately don't follow that precedent. A monitor that mutates its payload is side-effecting, which forfeits being a pure function of `(context, step)` — the property `sentinel-deployment.md` depends on end to end. A proxy deserializes the payload, hands it to the monitor, and has no way to notice that a list was mutated in place; it can only look at what was returned. Returning a replacement costs one `dataclasses.replace()` and keeps every deployment honest.

#### This is not the `stage` flag #5355 rejected

That argument was against a `stage` parameter on a single `Approver.__call__`, where the caller branches on a runtime flag to know what it is looking at. A monitor's stage is already encoded structurally, and more tightly than the class form managed: the payload type *is* the stage, so a monitor that reads `step.result` cannot have been registered for `BeforeToolCall`. Sharing the *return* type does not reintroduce a stage flag, because there is no stage in `Report` to begin with.

#### What one type buys

Six places want uniform handling: the precedence rule under [Composition](#composition); the escalation chain's fall-through; `SentinelEvent`, whose decision field would otherwise be a union or stringly typed; policy and YAML config parsing; the TUI transcript renderer; and `analysis` dataframe columns.

The cost of not sharing is already paid, and it is larger than an earlier draft of this document guessed. That draft assumed #5356 would generalise the approval system over the decision type. It did not: `approval/` was not touched at all, and `review/` ships a parallel `_policy.py`, `_registry.py`, `_apply.py` and `_call.py` — **413 lines, 20–62% textually similar to their approval counterparts.** Glob matching, the ordered chain with escalate fall-through, registry lookup and config parsing all exist twice. A third decision type would make it three times.

#### Rejected: a generic parameterised on the action set

``` python
A = TypeVar("A", bound=str, default=Action)

class Decision(BaseModel, Generic[A]):
    action: A
```

This keeps per-stage legality statically checked and is the better answer in the abstract. It is the wrong answer for this codebase:

- No serialized Pydantic model in `inspect_ai` is generic today. The only `Generic[...]` uses are a `list[T]` subclass in the eval recorder and the provider batching dataclasses, none of which reach the log schema.
- Parameterised generics get Pydantic's mangled names in `components/schemas` — `Decision_ObserveAction_` — which flow to `generated.ts` and the barrel. `design/type-generation-pipeline.md` documents the machinery that exists precisely to keep unstable auto-generated names out of the schema.

A subclass that widens the `action` Literal is also out: widening a field type in a Pydantic subclass is an unsound override, mypy flags it, and the suppression gate means it could not be papered over.

#### Enforcing the restrictions

With a flat type the type checker cannot stop an `AfterToolCall` protocol deciding `reject`. The dispatcher knows which stage it dispatched, so it validates there and raises — an illegal decision is a bug in the protocol, not agent-controlled input, so failing loudly is right. This follows AGENTS.md's "invalid state" rule: define the unsupported behavior explicitly and cover it with a test rather than silently coercing `reject` into `proceed`.

## Protocols {#protocols}

**A monitor observes and reports. A protocol authorizes and acts.**

That split came out of [Prior art](#prior-art) and is the main architectural consequence of it. In Ctrl-Z the monitor is a scorer; deferral, resampling and auditing are things the protocol does with the score, because they depend on calibration against benign distributions and on a global budget the monitor cannot see.

**"Protocol" is a registry type, and deciding is what puts something in it.** A protocol is any registered function that returns a `Decision`. Most consult monitors — `threshold` reads their scores — or compose other protocols — `sequential` and `concurrent` — and those are combinators, the shape Inspect already uses everywhere it composes: `chain(*solvers) -> Solver`, `multi_scorer(scorers, reducer) -> Scorer`, `policy_approver(policies) -> Approver`. But a protocol may have no children at all: `human()` asks a person, `no_curl()` matches a string. Those are rules, and they are protocols because they decide, not because they compose. See [A protocol may have no children](#a-protocol-may-have-no-children) and [The `@protocol` decorator](#the-protocol-marker).

The layering below is therefore mostly about *what code does*. The structural claims are typed; the behavioural one — that a protocol invokes its children through [the runner](#the-runner) — is recorded where it cannot be enforced, and that nothing weakens a final decision is [control flow](#authoritative-reports) rather than a check. [What the collapse costs](#what-the-collapse-costs) keeps the honest residue.

Applying the split shrinks the monitor surface rather than growing it. Every control-literature action turns out to live above the monitor:

| Control action | Whose job | How |
|------------------------|------------------------|------------------------|
| `accept` | protocol | below every threshold |
| `audit` | protocol | grants a monitor's `audit` request if budget remains |
| `defer` | protocol | calls the trusted model and substitutes its action |
| `trusted_edit` | protocol | has the trusted model rewrite, then substitutes |
| `resample` | protocol | draws more samples, scores each, picks or incriminates |

None are expressible by something looking at one exchange, so `Action` does not grow. Resampling is the one that needs more than a wider vocabulary — it needs a protocol that regenerates and substitutes, which [One that generates](#one-that-generates) shows.

The collusion countermeasures land here too: paraphrasing and minification are a transform between agent output and monitor input; honeypots are synthetic inputs injected into the monitor's stream. Pipeline stages, not monitor logic.

### Two philosophies, both first-class {#two-philosophies-both-first-class}

The [two registry types](#monitor-or-protocol) are not just two shapes of data. Each implies a way of deciding, and the design has to support both rather than treating one as the degenerate case of the other.

- **Rules, no monitors.** The protocol decides directly from what it sees. No thresholds, no calibration, no budget. This is the shipped-guardrail philosophy — and it is not hypothetical in Inspect, because **it is exactly what `@approver` does today.** An approver consults nothing, its decisions are binding, and nobody regards that as unsafe.
- **Monitors under a calibrated protocol.** The monitor never decides; thresholds, budget and actions all live in the protocol that reads it. This is the control-literature philosophy, and it is what `threshold` and `resample` are for.

Neither is the advanced version of the other. What a protocol is *for* becomes sharper as a result: **a protocol exists to decide.** When there is a score, deciding means thresholding it; when there is not, the rule is the decision.

That collapses to a single rule about what sits at the top of a configuration:

> **Whatever is outermost must be a protocol.**

| `Task(sentinel=)` given | Compiles to |
|------------------------|------------------------|
| a protocol | itself — the protocol is the root |
| a `ProtocolGroup`, or any list or mapping containing a protocol | `concurrent(...)` — protocols vote, monitors are recorded |
| a monitor, a `MonitorGroup`, or a list or mapping of monitors only | a configuration error naming `observe_only()` and `threshold()` |

This is where the split pays for itself: there is no mixed case to warn about, because anyone reading the configuration can see which entries are protocols, and a monitor beside a rule in a list is doing exactly what a monitor does everywhere — recording. `observe_only()` is how to run monitors with nothing acting on them, and the name says in the configuration that the sentinel enforces nothing.

Running monitors with nothing acting looks degenerate and isn't. Calibration needs a benign score distribution, and `observe_only()` over known-benign tasks is how you get one. It is step one of the second philosophy, not a failure to configure.

### Child decisions are advisory {#child-decisions-are-advisory}

A protocol's `Decision` is an opinion **whatever wraps it** may honour or ignore. Where nothing wraps it, there is nothing to ignore it and it is honoured — the approval model above.

Within a wrapping protocol this is forced rather than chosen. `concurrent` combines its children's decisions by precedence and `sequential` takes the first, so an inner decision is input to an outer one, never a bypass of it. If an inner `terminate` went around the protocol that wraps it, a protocol could not promise anything about what its configuration does. The one sanctioned way around is [`decide_final()`](#authoritative-reports), and it is visible: the transcript records every layer it bypassed.

The distinction that makes this coherent is **declared versus inferred**. [A bare monitor is an error](#monitors-need-a-protocol) because acting on it would mean acting on a number the author never thresholded. A rule's action was *declared*: someone who writes `Decision.reject(...)` is not surprised when a call is rejected, any more than an approver author is surprised today, so a rule at the top level acts.

One rule then covers the bimodality: **a wrapping protocol may override advisory decisions, never a final one.** An earlier draft left this as a rule about *what a protocol should do*, and a later one as a `binding` field the runner clamped to; it is now [`decide_final()`](#authoritative-reports), which ends the step before any wrapping protocol decides.

### A bare monitor is an error {#monitors-need-a-protocol}

`Task(sentinel=suspicion_monitor())`, with nothing wrapping the monitor, is a configuration error, and so is a `MonitorGroup` or a list or mapping of monitors only. The error names every such monitor and says what to wrap it in:

> A sentinel needs a protocol to decide each step, but it was given only monitors: suspicion_monitor. Wrap them in threshold() to act on their scores, or in observe_only() to record them without acting.

Acting on a bare monitor is ruled out for a specific reason rather than a general caution about safety: a threshold nobody set is a threshold nobody calibrated, so acting on one is acting on a number whose meaning is unestablished. Recording it silently is ruled out too: a scoring monitor whose scores nobody reads is a safety mechanism failing open. `observe_only()` makes that a *declared mode*: the configuration says that the sentinel enforces nothing. Decided by the maintainer on 2026-10-02, replacing an earlier rule that resolved monitors alone to `observe()` with a warning.

There is no corresponding question for a rule. A protocol at the top level acts, because its action was declared by its author rather than inferred from an uncalibrated number, and defaulting to silence would mean a rule that does nothing until you learn about a layer you had no reason to need. `@approver` has worked this way for years.

An earlier draft needed three warnings — a decision-returning monitor under `observe()`, a score nobody reads, and a list where half acts and half records — because a monitor could return a decision and the configuration could not show which entries would. The split removes the case rather than the warnings: the decorator says which entries decide, and the one remaining case, a "safety mechanism with no consumer", is an error, as AGENTS.md's "invalid state" rule asks.

### Ship protocols, not parts {#ship-protocols-not-parts}

The control literature's protocols are a small, finite, well-studied set. Ship them named and configured so nobody assembles thresholds, budgets and actions by hand:

``` python
Task(sentinel=observe_only(network_suspicion()))
Task(sentinel=threshold(network_suspicion(), reject_at=0.8))
Task(sentinel=resample(network_suspicion(), threshold=0.8, n=3))
```

This is the same move as shipping `react()` and `basic_agent()` rather than making everyone write a tool loop. It converts "understand the control literature" into "pick from a list", and the papers become an explanation of *why* each protocol exists rather than required reading.

#### What the names mean {#what-the-names-mean}

The literature's protocols are all arrangements of three resources with different trust levels: an **untrusted model** (capable, possibly scheming — it does the work), a **trusted model** (weaker, believed not to be scheming), and **trusted human labour** (scarce; this is the audit budget).

| Protocol | The arrangement |
|------------------------|------------------------|
| `concurrent` | run every child, strongest decision wins; what a list containing a protocol resolves to. Not from the literature — it is the concurrent composition itself, named |
| `sequential` | run children in order, first non-`escalate` decides; approval's chain as a protocol |
| `observe_only` | record every monitor's score; never act. How monitors run with nothing acting, since a list of monitors only is an error, and how a benign distribution gets collected |
| `threshold` | act on a score: `reject` above one constant, `terminate` above another. The smallest protocol, and the one calibration feeds |
| `resample` | above a threshold, draw further completions, score each, execute the least suspicious; escalate if none clears (Ctrl-Z; multi-step settings only) |
| `trusted_monitoring` | the trusted model scores the untrusted model's output; the most suspicious slice, up to an audit budget, goes to human review and the rest is accepted |
| `trusted_editing` | the trusted model rewrites the suspicious parts rather than replacing the whole action |
| `untrusted_monitoring` | the *untrusted* model scores — more capable, but it may collude, so it needs honeypots and lie detection |

The first five ship in core; `concurrent`, `sequential`, `observe_only` and `threshold` are built, and `resample` is not yet. `defer_to_trusted` was removed from the plan by the maintainer on 2026-10-02: an eval has no trusted model to defer to. The last three are extension candidates, with the pipeline stages (honeypots, paraphrasing). `trusted_monitoring` is the surprising one to leave out, since it is the literature's baseline; the reason is the audit budget. It is a deployment-economics idea — the scarce resource being modelled is human labour — and as a headline parameter it means "how much to *not* review", which reads badly to the eval operators this package is for, who want to see everything flagged. Nothing stops an extension shipping it, and `threshold` plus `audit=True` is the eval-shaped version.

Two things follow that a reader should not have to infer.

**The threshold comes from the budget, not from the score.** "Audit the most suspicious 1%" rather than "audit anything above 0.8". That is the whole reason [thresholds cannot live inside a monitor](#monitor-or-protocol): you need the benign score distribution to know which value *is* the 99th percentile, and a monitor looking at one step has never seen a distribution.

**"Trusted" describes the monitor, not the protocol — and the protocol name is a statement about what it omits.** `trusted_monitoring(some_untrusted_judge())` type-checks perfectly well; nothing verifies the claim in the name. What the word actually buys at the protocol level is an *assumption*: this protocol trusts its monitor and therefore does nothing to check it. `untrusted_monitoring` is a separate protocol rather than a flag precisely because dropping that assumption means adding machinery — honeypot injection, and comparing reports against known-bad inputs.

A protocol owns:

- **thresholds** — calibrated against benign score distributions, per dimension where scores are structured
- **the audit budget** — how much remains, and how it is allocated across an episode (Ctrl-Z decays it exponentially, scrutinising early steps harder)
- **aggregation** — combining reports from several monitors
- **the control actions** in the table above
- **pipeline stages** — paraphrasing, honeypot injection, each of which can be [its own nested protocol](#a-protocol-may-have-no-children)
- **its stance toward child decisions** — honour, or record and ignore

All of which is ordinary code, not framework surface — see [The runner](#the-runner).

#### One that generates

`threshold` never calls a model. The control protocol that ships beside it does, and it is worth showing because it exercises the parts of the surface a threshold does not: regenerating with the agent's own request, and substituting the result.

``` python
@protocol
def resample(monitors: Monitor | MonitorGroup | Monitors, threshold: float, n: int = 3) -> Protocol:
    async def decide(context: Context, step: AfterGenerate, escalations: Escalations) -> Decision | None:
        reports = await run_monitors(monitors, context, step)
        score = reports.max_suspicion()
        if score is None or score < threshold:
            return Decision.proceed()
        best_score, best_message = score, step.output.message
        for _ in range(n):
            output = await context.host.generate(
                step.input,
                model=step.model,
                tools=step.tools,
                tool_choice=step.tool_choice,
                config=step.config,
            )
            reports = await run_monitors(monitors, context, replace(step, output=output))
            candidate = reports.max_suspicion() or 0.0
            if candidate < best_score:
                best_score, best_message = candidate, output.message
        if best_score >= threshold:
            return Decision(action="escalate", explanation="no resample cleared the threshold")
        return Decision(action="modify", modified=best_message)

    return decide
```

Each candidate's scores go through the runner, so the log holds every draw, not only the winner. Returning `escalate` when nothing clears the threshold is Ctrl-Z's incrimination signal in its simplest form, and it composes: `sequential([resample(...), human()])` sends exactly the incriminated steps to a person.

Three things follow for the rest of the design:

- **`AfterGenerate` carries the request.** `tools`, `tool_choice` and `config` join `input`, so a protocol regenerates with exactly what the agent was given.
- **`Host.generate` is a real generate.** Messages or a string in, `ModelOutput` out, with `model`, `role`, `tools`, `tool_choice` and `config`. `model` is always a model name and `role` always a model role. A configured role wins, then `model`; with neither, the role is `monitor`, and a `model` alone is used without consulting it. In an eval an unconfigured role with no `model` falls back to the agent's model with a warning, once per process for each role, naming the role to set. A `Model` instance is not accepted; a configured client goes in `model_roles`. Under a proxy, generating as the agent's model is a re-issued upstream request. `sentinel-deployment.md` carries the signature.
- **Protocol inference has its own usage bucket.** `resample` spends up to `n` times the agent's tokens on the agent's model. It is recorded under a `protocol` bucket in `ModelUsage`, separate from both agent and monitor usage and not charged to the agent's limits, because the protocol chose to spend it. See open question 8.

It is `AfterGenerate`-only, and its signature says so rather than annotating `Step`.

### A protocol may have no children {#a-protocol-may-have-no-children}

The shipped `threshold` protocol, in outline, with configuration-time validation left out:

``` python
@protocol
def threshold(
    monitors: Monitor | MonitorGroup | Monitors,
    *,
    escalate_at: float | Mapping[str, float] | None = None,
    reject_at: float | Mapping[str, float] | None = None,
    terminate_at: float | Mapping[str, float] | None = None,
) -> Protocol | ProtocolGroup:
    # strongest first; a float is {"*": float}; the ones not given are left out
    levels = [
        ("terminate", thresholds(terminate_at)),
        ("reject", thresholds(reject_at)),
        ("escalate", thresholds(escalate_at)),
    ]

    async def decide(context: Context, step: Step, escalations: Escalations) -> Decision:
        observations = await run_monitors(monitors, context, step)
        if not observations:
            return Decision.proceed("no monitor reported")
        for action, limits in levels:
            if crossed := crossing(observations, limits):
                return Decision(
                    action=action,
                    explanation=crossed.explain(),
                    references=references(observations),
                    metadata={"observations": scores(observations)},
                )
        return Decision.proceed()

    # one function per stage its monitors watch; before tool calls only, with
    # reject_at
    return per_stage(decide, monitors)
```

It holds the constants a monitor cannot, reads its children's scores through the runner, and returns a decision. Nothing else. Each of `escalate_at`, `reject_at` and `terminate_at` is a floor: at or above it the step gets at least that action, and the decision is the strongest action any score reaches. At least one is required; `observe_only()` is how to record scores with nothing acting on them. A decision is explained for the log with the highest score that reached its threshold, that monitor's name and its explanation; it carries every reporting monitor's references, and `metadata["observations"]` lists every monitor's score and explanation, so a `human()` after it receives all of them in the `escalations` it is handed, and inspect_ai's approval card shows them once it renders that metadata ([pr-series.md](pr-series.md), Deferred, "inspect_ai's human view shows every score"). None sets `message`, so the agent learns only that the call was refused. It runs at each stage its monitors watch, monitors of different stages each thresholded at their own; `reject_at` limits it to `BeforeToolCall`, where a reject is legal. An escalation is answered by what follows it in a `sequential()`: `sequential([threshold(monitors, escalate_at=0.6, terminate_at=0.9), human(stages=["tool_call"])])`. When no monitor reported at a step it decides `proceed`, explained as "no monitor reported", rather than abstaining, so a `human()` after it is asked only when it escalates, not about every step its monitors skip ([pr-series.md](pr-series.md), "threshold proceeds when no monitor reported"). That `proceed` is a vote, but a `proceed` does not contest a `modify`, so in a `concurrent()` beside a protocol that modifies the call the rewrite takes effect, as beside a low-scoring threshold ([Two compositions](#two-compositions-concurrent-and-ordered)). A failed monitor is not an abstention: it fails the sample, as [Failure semantics](#failure-semantics) says.

Each threshold is one number or a mapping per dimension. A key `"<monitor>.<dimension>"` applies to one monitor's dimension (the monitor's instance name, as in `Reported.name`), a bare `"<dimension>"` to that dimension from every monitor, a monitor's instance name to its single-number score, and `"*"` to anything unlisted; the most specific key wins, and a score no key covers never triggers. A number is shorthand for `{"*": number}`, so a float threshold behaves and explains itself as it always has. The strongest action any score reaches decides: terminate, then reject, then escalate. A key containing `.` is always qualified, by the longest instance name that prefixes it, never a bare dimension. Configuration validates the keys, requires finite values, and rejects a threshold that can never apply because a stronger action's threshold is at or below it for every score it covers (`escalate_at=0.9` beside `terminate_at=0.8`); a key one mapping lists and another does not is not such a conflict. This is what thresholding [structured suspicion](#suspicion-is-structured) per dimension means: calibration produces one percentile per dimension, and the mapping holds them.

And `no_curl` from [Shape](#shape), which is the same thing with the middle removed: no monitors, no runner call, a decision straight from the step. Both are protocols. **Deciding is what makes something a protocol, not having children.** An earlier draft defined a protocol as "a monitor that takes monitors", which made the combinator the essence and left rules and `human()` as a special kind of monitor — the kind that decides — with a decorator whose only job was to distinguish the two. That was the wrong seam. The seam the literature draws, and the one the report types already encode, is between observing and deciding; whether a decider consults anything is a detail of its implementation.

Where a protocol does have children it is a combinator, `[Monitor | Protocol] -> Protocol`, and Inspect answers that shape the same way three times over — `chain`, `multi_scorer`, `policy_approver`. `multi_scorer` is the closest analogue: it runs children in parallel, aggregates with a reducer, and returns one `Score`. The typing follows the children: `threshold` takes `Monitors` because it reads scores, `sequential` and `concurrent` take `Sentinels` because they compose decisions and record whatever monitors they are handed, and `threshold(human())` is a type error, as it should be — a person has no score.

Two consequences worth stating plainly:

- **Two `RegistryType` values, `"monitor"` and `"protocol"`.** An earlier draft argued its way to one on the grounds that a second type was a cost with no benefit. The benefit turned out to be the configuration surface: with two types, a reader of `Task(sentinel=[...])` or of a YAML file can see which entries act, and the three configuration-time warnings that existed to explain mixed lists disappear ([A bare monitor is an error](#monitors-need-a-protocol)).
- **The alias is `Protocol`, and the decorator is `protocol`.** They pair as `Tool` and `@tool` do. The name collides with `typing.Protocol`, and the collision is accepted: see [Open questions](#open-questions) item 19. (`Policy` would be wrong regardless: in the control literature the *policy is the untrusted agent*, so reusing it for the thing doing the deciding would invert the term.)

Protocols nest. A budget-owning protocol can wrap a paraphrasing protocol wrapping the leaf monitors, so the [pipeline stages](#the-extension-surface) become composable pieces rather than features one monolithic protocol has to anticipate. What still needs a rule is *budget ownership* under nesting — see the [open questions](#open-questions).

### The `@protocol` decorator {#the-protocol-marker}

``` python
Monitors: TypeAlias = Mapping[str, Monitor | MonitorGroup] | Sequence[Monitor | MonitorGroup]
Protocols: TypeAlias = Mapping[str, Protocol | ProtocolGroup] | Sequence[Protocol | ProtocolGroup]
Sentinel: TypeAlias = Monitor | MonitorGroup | Protocol | ProtocolGroup
Sentinels: TypeAlias = Sentinel | Mapping[str, Sentinel] | Sequence[Sentinel]

Escalations: TypeAlias = tuple[Reported[Decision], ...]   # exported: a protocol's third parameter

Protocol: TypeAlias = (
    Callable[[Context, Step, Escalations], Awaitable[Decision | None]]
    | Callable[[Context, BeforeGenerate, Escalations], Awaitable[Decision | None]]
    | Callable[[Context, AfterGenerate, Escalations], Awaitable[Decision | None]]
    | Callable[[Context, BeforeToolCall, Escalations], Awaitable[Decision | None]]
    | Callable[[Context, AfterToolCall, Escalations], Awaitable[Decision | None]]
)


@overload
def protocol(factory: Callable[P, Protocol], /) -> Callable[P, Protocol]: ...
@overload
def protocol(factory: Callable[P, ProtocolGroup], /) -> Callable[P, ProtocolGroup]: ...
@overload
def protocol(*, name: str | None = None, version: int = 0) -> ProtocolDecorator: ...
```

It is `@monitor`'s twin, typed the way `@scorer` is, and it claims two things about the factory — each a claim the framework verifies against the code rather than a second vocabulary that could disagree with it, the distinction that got `@monitor(kind=)` and `@monitor(on=)` rejected:

- **The returned callable is annotated `-> Decision`, or `-> Decision | None` if it can abstain.** A protocol returning `Observation | None` is a monitor wearing the wrong decorator, and the checker says so before the decorator runs.
- **It registers under `"protocol"`.** Same params capture, same `Step`-or-single-stage annotation rule as a monitor.

Children are ordinary factory parameters, typed `Monitors`, `Protocols` or `Sentinels`, and there may be none. That is what makes the nested key in [configuration](#configuration) legal: `monitors:` or `children:` under a factory is legal exactly when the factory has a parameter of that name, and a configuration error otherwise. An earlier draft required a leading positional `monitors` parameter, checked with `Concatenate`, so that the decorator could prove the factory composed. That proof is not wanted now that composing is not what a protocol is; a rule has no children to prove anything about.

What the decorator deliberately does *not* do is invoke the children or own the fan-out. A draft of this design tried that — `protocol(monitors, decide=...)` with the framework calling the runner and the author supplying only the reduction — and it was strictly less expressive than writing the body: it had to grow a `transform=` parameter just to recover paraphrasing, and it could not express `sequential` at all, because its decision is made *between* links rather than after them. The factory still holds the children either way, so the enforcement it appeared to buy was illusory. The body of a protocol is ordinary code that calls [the runner](#the-runner).

### Static checking {#static-checking}

Most of what `@protocol` and `@monitor` verify at decoration time, mypy and pyright verify first. The repo already does this for scorers: `@scorer` is typed `Callable[P, Scorer] -> Callable[P, Scorer]`, so a factory whose inner function returns the wrong thing fails type-checking before the runtime check runs. The same technique covers both decorators, and the runtime pass reduces to reading annotations the checker already validated.

| Claim | How it is checked statically |
|------------------------|------------------------|
| a monitor returns `Observation \| None` | `@monitor` is typed `Callable[P, Monitor] -> Callable[P, Monitor]`, with an overload `Callable[P, MonitorGroup] -> Callable[P, MonitorGroup]` for a [group](#one-function-one-stage); an `async def` returning `Decision \| None` is not assignable to any member of `Monitor` |
| a group is not called directly | `MonitorGroup` and `ProtocolGroup` have no `__call__`, so calling one is a type error |
| a protocol returns `Decision \| None` | the same, with `Protocol` |
| the inner function's second parameter is a stage type | `Callable` parameters are contravariant, so a function accepting `Step` matches the `Step` variant and one accepting `BeforeToolCall` matches that variant; an unannotated `step` matches nothing |
| the declared kind matches the body | with the return annotated `Observation \| None`, a `return Decision(...)` inside is an ordinary type error — the strongest of the four, since it checks every return statement rather than the signature |
| `threshold` takes monitors, `sequential` takes children | the parameter types; `threshold(human())` is a type error |

What stays runtime-only is exactly what no type system would express: per-stage action legality (a generic `Decision[A]` was [rejected](#rejected-a-generic-parameterised-on-the-action-set) for schema reasons), whether a body went through the runner (the [call guard](#the-boundary-check)), and YAML configuration. Those live in [the boundary check](#the-boundary-check) and the log.

One more rule is runtime-only for a different reason: that a monitor annotates exactly one payload type. A function accepting `Step` is, by parameter contravariance, a valid `Callable[[Context, BeforeToolCall], ...]`, so pyright accepts it as a `Monitor`; the rule is a statement about how monitors should be written, not a type-safety property, and `@monitor` enforces it when the factory is called. Making it static would mean parameterising `Monitor` by payload, which is not worth a type parameter on every factory to catch a mistake that already fails at configuration time.

### The runner {#the-runner}

**The protocol invokes its children; the framework does not.** This is forced by [Prior art](#prior-art): paraphrasing and minification are transforms between agent output and monitor input, and honeypots are synthetic inputs injected into the monitor's stream. Neither is expressible if the framework has already called the monitors and handed over finished reports.

So the runner is not a convenience. It is where core's obligations sit, and it is the *only* sanctioned way to invoke a child. It comes in two families, one per registry type, plus a mixed form for the compositions:

``` python
async def run_monitors(
    monitors: Monitor | MonitorGroup | Monitors, context: Context, step: Step
) -> Observations:
    """Invoke each monitor annotated for this stage, concurrently.

    Derives each child's Context under this layer's path, records a
    SentinelEvent per report, records a raising monitor as failed (the
    result's `failed`) rather than propagating. Results in configuration
    order; empty if every child abstained or does not watch this stage.
    """


async def run_protocols(
    protocols: Protocol | ProtocolGroup | Protocols, context: Context, step: Step
) -> Decisions:
    """The same, for protocols. Cancels siblings when one returns terminate or calls decide_final()."""


async def run_children(
    children: Sentinels,
    context: Context,
    step: Step,
) -> Reports:
    """Both families in one task group. What concurrent() uses."""
```

Each accepts one instance, a sequence or a mapping, and always returns a sequence, so one child and many are the same call. A single instance is named by its registry name, as a sequence entry is; a one-entry mapping names it explicitly. An earlier draft also had `run_monitor` and `run_protocol` for one child at a time; they were removed when a factory could return [several functions as a group](#one-function-one-stage), since one child may then report more than once and a singular return type no longer fits it.

Every form carries the per-child obligations:

- **filter by stage** — a child not annotated for this step's stage is skipped, and the skip is indistinguishable from abstention
- **name the child** — from the mapping key if the protocol was given a `Mapping`, else the registry name without its package prefix; the runner treats a duplicate name within one layer as an error rather than inventing a `#2` suffix, and a composition that runs children one at a time validates its names the same way first
- **derive the child's `Context`** — the host hands `run_sentinel` a `HostContext` of its `HostNative`, the recorder and the sample store, which builds the top layer's `Context`; `run_sentinel` keeps the recorder, the store and the registry name of each running instance's factory, keyed by path, in a context variable set for the step, and the runner builds each child's `Context` under this layer's path and registers it there before invoking it. A monitor or protocol is given a plain `Context` with no private fields; running children outside a step is a `RuntimeError`, and with a `Context` whose path is not a layer of the step a `ValueError`. Every `Recorder` call receives the `Context` of the instance it concerns and its factory's registry name, so the recorder reads which code ran from `factory` and which instance from `path`. Naming and duplicate detection live in a private `named_children()` helper (`_validate.py`), which `sequential()` and `concurrent()` also use
- **run each of the instance's functions** — every function of the instance annotated for the step runs, in the order its factory returned them and one after another, since they share one store, until one returns `terminate` or calls `decide_final()`, which ends the instance's run: the functions after it do not run and are not recorded; a lone function is the one-member case
- **record the report** — one `SentinelEvent` per report, including the ones the protocol goes on to ignore, which is load-bearing because the ignored ones are the benign distribution calibration needs
- **apply the [failure policy](#failure-semantics)** uniformly: a monitor function that raises is recorded through `Recorder.failed` and listed in the result's `failed`, and the rest of the instance's functions and its siblings run on; a protocol's exception propagates

Across the children of one call it adds the concurrent obligations: it **fans out on an anyio task group, with the same ordering and first-exception semantics as inspect_ai's `tg_collect()`** (per AGENTS.md, which [Concurrency is a safety property](#concurrency-is-a-safety-property-not-just-a-latency-one) shows is a safety requirement rather than a latency one); and `run_children` **holds the group so a `terminate` decision can cancel the still-running siblings**, since nothing outranks it and the sample is ending — prompting a human about a sample that no longer exists is the concrete thing this avoids. A cancelled child is recorded as cancelled, meaning it was cancelled before it finished, so the log distinguishes it from one that abstained; any reports it already made, from the earlier functions of a multi-function instance, stand; its observation, if wanted, is recovered by replay. Monitors are cancelled too: once a protocol has decided `terminate` no sibling's report can change the outcome, and spending their inference live would only save a replay later. Cancellation lands at a child's next await, so a child that finishes without awaiting is recorded normally. A cancellation from above the layer, or one caused by a sibling raising, is recorded the same way; the `terminate` or the exception in the same layer says which. A child calling [`decide_final()`](#authoritative-reports) cancels its siblings the same way and propagates out of the call; as with `terminate`, the functions of one instance after the one that called it do not run, and an instance cancelled part way is recorded as cancelled once; if two race, the first to reach the group wins and the other is recorded as `superseded`, a final decision that did not take effect, and an exception in the same group outranks both, so a bug is not hidden behind a final decision.

They return:

``` python
@dataclass(frozen=True)
class Reported(Generic[R]):
    name: str      # instance name: mapping key, or registry name
    path: str      # "attempt/internet_attempt"
    report: R
    function: str  # __name__ of the function that produced the report


class Observations(Sequence[Reported[Observation]]):
    failed: tuple[Failed, ...]
    """Monitor functions that raised; while any are present, reading raises MonitorFailedError."""
    succeeded: Observations
    """The observations without the failures, readable whether or not any failed."""

    def max_suspicion(self) -> float | None:
        """Max over observations (per-dimension max, then max); None if nobody observed."""


@dataclass(frozen=True)
class Failed:
    name: str
    path: str
    function: str
    error: Exception


class Decisions(Sequence[Reported[Decision]]):
    def strongest(self) -> Reported[Decision] | None:
        """By the precedence in Composition; None if nobody decided."""


class Reports(NamedTuple):  # observations, decisions = await run_children(...)
    observations: Observations
    decisions: Decisions
```

Two sequence types rather than one, because [the two halves aggregate differently](#composition) — scores combine and then threshold, decisions resolve by precedence — and because the types then say which protocols read which: `threshold` never sees a `Decision`. Identity is attached because a protocol may weight or require agreement across specific monitors, and because `path` is what the transcript and the store are keyed by. Generics are safe here in a way [they were not for `Report`](#what-a-monitor-returns): these are in-memory values passed as arguments, never serialized, so no mangled name reaches `components/schemas`.

Everything else a protocol does is **ordinary Python over these plus task-scoped state** — thresholds, budget arithmetic, aggregation, deferral. That is the point rather than a gap: a protocol is meant to be [the extension surface](#the-extension-surface), and something requiring framework blessing beyond these helpers would not be one.

One thing it is not quite arbitrary code over: **per-sample state has to actually work**, which it does — the audit budget is per-episode and therefore per-sample (see [`audit` is a flag](#audit-is-a-flag-not-an-action)). Only a budget deliberately pooled across samples reaches [Per-task](#per-task), and that is a choice rather than a requirement.

### Two compositions: concurrent and ordered {#two-compositions-concurrent-and-ordered}

The reviewer group's requirements (#5423, see [below](#the-reviewer-groups-requirements-5423)) made explicit something the first draft had only half of. There are two ways to compose children, and people need both:

- **Concurrent.** Every child runs; nothing is silenced; the strongest decision wins. This is what independent guards want — two monitors that each must be satisfied — and it is the composition [Composition](#composition) describes.
- **Ordered.** Children run in sequence; the first non-`escalate` decides; later children never run. This is approval's chain, and it exists for two reasons that are both "do not invoke the next monitor unless needed": a cheap rule ahead of an expensive judge, and a human at the end who should only be asked what the machines could not answer.

The first draft treated the ordered form as "wrong for a set of observers" and stopped there. It is wrong *as the default* — that is precisely the bug #5423 reports: A approves, B would have terminated, B never ran — but it is the right shape whenever there is a person at the end of it. Both ship as protocols on the same runner.

**`sequential()`** is the ordered one, named to pair with `concurrent` and to stay clear of `inspect_ai.solver.chain`. It runs one link at a time, through the runner with a one-entry mapping, so every link is recorded under its name, and it hands escalations forward to the next link's protocols through the runner's `escalations=` argument:

``` python
@protocol
def sequential(children: Sentinels) -> Protocol:
    named = named_children(children, None)
    if not named:
        raise ValueError("sequential needs at least one child.")

    async def run(context: Context, step: Step, escalations: Escalations) -> Decision | None:
        handed: list[Reported[Decision]] = []   # starts empty, ignoring `escalations`: they do not cross layers
        participated = False
        for name, child in named:
            observations, decisions = await run_children({name: child}, context, step, escalations=handed)
            warn_failed(observations.failed)         # a failed monitor is logged and falls through
            if observations.failed or observations.succeeded:
                participated = True                  # recorded; falls through
            strongest = decisions.strongest()      # a group link decides by its strongest function
            if strongest is None:
                continue
            participated = True
            if strongest.report.action != "escalate":
                return strongest.report
            handed.extend(d for d in decisions if d.report.action == "escalate")
        if handed:
            return handed[-1].report              # nobody decided; pass it up
        return Decision.proceed() if participated else None

    return run
```

The functions of one [`ProtocolGroup`](#one-function-one-stage) run in one call and the link decides by the strongest of their decisions, as `concurrent` does, so one function's `proceed` cannot hide another's `terminate`; each is handed the escalations from before any of their own; they share state through the store, not through `escalations`. A monitor link is never handed them.

Three properties fall out of writing it as ordinary code. A monitor in a `sequential` is recorded and falls through, since an observation cannot be "the first decision" — ordered composition is Decision-shaped in substance. A `sequential` whose every deciding link escalated returns the last escalate, so it passes up to the layer above as concurrent's does, and if it reaches the root `run_sentinel` returns a `terminate` explained as an unhandled escalation in its place, which `human()` or `handle_escalation()` at the end of the chain prevents; one where only monitors participated returns `proceed`. The [approval adapter](#a-protocol-layer-for-approval-too) keeps approval's fail-closed `reject` on its own path, so existing approval users see no change, and a `sequential` that wants fail-closed ends with a rejecting monitor, as approval lists end with `auto` today. And sequential dispatch stops being a violation of the independence rule, because the loop is a named, shipped protocol whose mode the transcript shows.

**`concurrent()`** is the concurrent composition, and it is also what a top-level list or mapping containing a protocol resolves to (see [Configuration](#configuration)). In outline, leaving out group labels:

``` python
@protocol
def concurrent(children: Sentinels) -> Protocol:
    async def run(context: Context, step: Step, escalations: Escalations) -> Decision | None:
        reports = await run_children(children, context, step)    # hands its children no escalations
        warn_failed(reports.observations.failed)
        voters = reports.decisions
        strongest = voters.strongest()
        if strongest is None:
            return None                       # monitors only
        if len(voters) == 1:
            return strongest.report
        own = strongest.report.explanation
        update: dict[str, object] = {}
        contesting = [d for d in voters if d is not strongest and d.report.action == "modify"]
        if strongest.report.action == "modify" and contesting:    # two rewrites conflict
            others = ", ".join(d.name for d in contesting)
            rewrite = f"{strongest.name} modified the call but so did {others}"
            own = f"{rewrite}: {own}" if own else rewrite
            update = {"action": "reject", "modified": None}
        summary = "; ".join(f"{d.name}: {d.report.action}" for d in voters)
        update["explanation"] = f"{own} ({summary})" if own else summary
        return strongest.report.model_copy(update=update)

    return run
```

That settles the `modify` question this document previously left open: concurrent children cannot see each other's rewrites, so two rewrites conflict, and a `modify` beside another `modify` is a rejection naming the modifiers, keeping the first modifier's `audit`, `metadata` and explanation. Another `modify` contests it whichever instance it comes from, another function of the modifier's own [multi-function](#one-function-one-stage) instance included, and the explanation labels each such function's vote `name.function`. A `proceed` or an `escalate` beside it does not contest it, so the rewrite is honoured, and a `reject` or `terminate` outranks it anyway. When more than one protocol decided, the layer's explanation lists each one's decision after the winner's own. Re-approving modified arguments is more machinery than the decision is worth for now, so the rewritten call is not checked again by the protocols that proceeded, as it is not for a lone modifier.

Until 2026-10-26 a `proceed` contested a `modify` too: a `modify` when another child decided anything but `escalate` became a rejection, because the others had approved the call as it stood and nobody had approved the rewrite (#5423 reached the same answer independently). Reversed by the maintainer on 2026-10-26. A peer with no objection to the original call should not make the layer stricter than having no peer at all: a lone modifier's rewrite is honoured unchecked, and adding a peer that proceeds turned that fix into a block. It also meant that a `threshold()` whose monitors did not report, which votes `proceed` since sentinel #153, blocked a sibling's rewrite. The accepted cost is that the protocols that proceeded do not see the rewrite, the same as for a lone modifier. The name pairs with `sequential`: both say how the children run and leave the decision rule to the docs. `strictest` and `all_of` were the alternatives; the first named only the recommendation half of the reduction (observations aggregate by max, not strictness) and the second named what the configuration means rather than its shape.

**Concurrent is the default, and ordered is one explicit word.** The argument for making ordered the default is that many people will want a chain, which is true. The argument against is that ordered-by-default silences monitors, and silencing is a safety failure while concurrency's cost is latency and tokens. A default should fail safe, and changing one later is a silent semantic change for everyone. So a list with a protocol in it is `concurrent`, and ordered composition is `sequential(...)`. The mistake the wrong default would produce most often — a `human()` in a concurrent group, prompting on every call — is not an error, since asking about every call can be intended; the docs recommend `sequential()` for asking only on escalation.

### The boundary check {#the-boundary-check}

When any layer returns a decision, or passes one to `decide_final()`, the runner checks it before anything acts on it.

**Shape checks raise.** These are the rules no type can express, and a violation is a deterministic bug in the protocol, not agent input, so it raises, per the "invalid state" rule from AGENTS.md, naming the protocol that produced the decision:

- **The action is legal for the stage** — no `reject` or `modify` at `AfterToolCall`, per [One vocabulary across stages](#one-vocabulary-across-stages).
- **`modified` is set iff `action == "modify"`**. Its type is `Decision`'s field type, which validation already enforces, so the runner does not check it again; a decision built with `model_construct` has opted out of validation.
- **A `modify` rewrites only the arguments.** The replacement keeps the call's `id` and `function`: the tool was resolved and the model's `tool_call_id` fixed before the protocol ran, so a different function or id would run one tool while the log and the model name another. A protocol that wants a different tool rejects the call and says what to run instead.

A `decide_final()` decision is shape-checked when it is made. The decision is recorded when it takes effect, at the root; each layer it passes is recorded as bypassed as it is passed; a losing final decision is recorded as superseded where it lost.

**A decision no layer above may weaken is [`decide_final()`](#authoritative-reports).** It replaces an earlier design in which a `binding` flag on the decision set a floor: the runner compared each layer's return with its binding children's, clamped a weaker one up, and kept the protocol's own decision on the record as `overrode`.

Control flow rather than a value-level rule, because a rule about decision values requires every layer to carry or compare the property — inherit the flag, preserve it through `concurrent`'s rewrite of a contested `modify`, break ties toward it, reconcile two binding `modify`s — and each patch to that design leaked wherever a middle layer built a new decision. Control flow cannot be dropped by a middle layer, because the middle layer's code does not run. It reuses the cancellation `terminate` already has, so there is no new concurrency machinery, and it settles conflict the way `terminate` does: the first `decide_final()` to reach a layer wins and the rest are recorded, as with two `terminate`s. Ordering several authorities is a `sequential`, or a panel that collects advisory votes and calls `decide_final()` itself.

### Humans in the loop

#5423's chains end in a person, and the first draft had nowhere to put one: humans were `@approver`s, and approvers lived in `Task(approval=)`. A `human()` protocol closes that. It has no children and asks a person through the host:

``` python
human(stages=["tool_call"])                                    # before every call it is reached at
human(stages=["tool_call", "tool_result"], choices=[...])
human(stages=["model_output"])                                 # the whole assistant turn, before its calls run
```

- **`stages` is required**: `"tool_call"` (before a call), `"tool_result"` (after it), `"model_input"` (before a generate by the agent's model, with the request about to be sent) and `"model_output"` (after a generate by the agent's model, before its output reaches the agent), the names `SentinelEvent.stage` records. Leaving it out, an empty list, or an unknown stage is a configuration error when the factory is called. The factory returns one function per chosen stage, a `ProtocolGroup` when several, so the one-stage-per-function rule holds.
- **It asks whenever it is reached** at a stage it watches. Inside a `sequential`, being reached means every earlier link watching that stage escalated, so the person sees escalations; on its own, `human(stages=["tool_call"])` asks about every call, like inspect's human approver. There is no escalation filter and no warning when nothing before it watches a stage: that can be intended.
- **Choices** use inspect's approval words, since the surfaces present those. Before a call the default is `approve`, `reject`, `terminate`, and `modify` may be offered; after a call it is `approve`, `terminate`, and asking for `reject` or `modify` there is a configuration error, since the call has already run. Before and after a generate it is `approve`, `terminate`, and asking for `reject` or `modify` is a configuration error, since neither is legal at the generate stages yet ([generate-stages.md](generate-stages.md), part 2). `approve` maps to `proceed`, `reject` to `reject`, `terminate` to `terminate`, `modify` to a `modify` carrying the person's replacement call, which the shape check holds to the call's `id` and `function`.
- **The person's typed reason on a reject is the decision's `message`**, so the agent reads it, and its `explanation`, for the log. A terminate's or an approve's reason is its `explanation`: an approve keeps the person's reason, so the log, and a later reader of earlier decisions (AISI R4), sees why the step went ahead.
- **The answer is an ordinary decision.** Inside a `sequential` it ends that chain, as any link's decision does; above the chain the usual vote applies, so beside other protocols in a `concurrent()` the person is one vote: a peer's `reject` outranks the person's `approve`, and the person's `reject` or `terminate` is not weakened, since the strongest decision wins. Human surfaces already queue requests, so two `sequential`s that both end in a person prompt one after the other and both answers vote; a `terminate` cancels the other prompt, as it cancels any sibling.

It reaches the person through `Host.ask_human(step, escalations, choices) -> HumanAnswer`, passing on the escalations it was handed. The host is handed the step itself and renders it by its type, so the interface is the same for every stage, with the escalations (who escalated and why): before a call the call, its view and the conversation; after a call the same plus the result the model is about to receive; before a generate the request, with the tools it offers and the messages the model has not been sent before; after a generate the whole assistant turn, its text and every tool call it makes, before any of the calls runs. `HumanAnswer` holds the chosen `decision`, the `reason` the surface gives (the person's typed reason or fixed text), and `modified`, the replacement call for a tool-stage modify. When the person ends the prompt without one of the choices, the host answers `reject` if it was offered and `terminate` otherwise, so the step fails closed; `human()` accepts that `terminate` even when it was not offered. The host notifies the person and marks the sample as awaiting a person while it waits, and raises a clear error for a stage it cannot render. inspect_ai renders all four stages with the existing surfaces (ACP, the approval panel, the console), which show one call per prompt. At `BeforeGenerate` it renders the request as one view: the escalations, how many messages the request sends, the tools it offers and which the model must call, and each message the model has not been sent before in a code block of its own, marked and fenced as a turn's calls are, since a message can hold a tool's output. A message counts as sent before when `step.history` holds one with the same content, so a message rewritten since it was sent is shown, as is an assistant message the scaffold wrote itself. The view stands on its own, because the console and the panel show no other part of the conversation. It doesn't show a message that was sent before and has moved, is sent again or was removed, nor content that isn't text, which it counts, nor a tool's description and parameters. The person approves the request or terminates. At `AfterGenerate` it renders the turn as one view, the escalations, the number of calls and then each call in a code block of its own, with control, bidi and invisible characters in the agent's text shown as code-point markers, and the same characters marked in the turn's text, reasoning and calls where the ACP TUI and `--display conversation` show them beside the prompt, so no call's agent-written arguments can hide or pass for another through markdown or terminal control (earlier tool output on those two surfaces is not marked yet), and the person approves or terminates the turn as a whole (AISI R1, settling [open question](#open-questions) 14). A proxy implements the same method, e.g. as a review queue.

A `human()` in a `concurrent()`, including the root a top-level list or mapping resolves to, is allowed. It asks about every call it is reached on, which may be what the user wants, for the same reason there is no warning when nothing before it watches a stage. To ask only on escalation, put it at the end of a `sequential()`.

### What the split does not enforce {#what-the-collapse-costs}

Two registry types make the contract mostly explicit, and the residue is small:

- The structural claims — returns a decision, annotates a stage, takes typed children — are [statically checked](#static-checking).
- Nothing above a [`decide_final()`](#authoritative-reports) decision runs, so nothing above it can weaken it, and each layer it passed is recorded as `bypassed`.
- **Going through the runner is enforced at run time, not statically.** The factory holds the children and could call them as plain callables, bypassing naming, recording and the concurrency guarantee. The decorator wraps each configured function in a guard, and the runner marks the function it is invoking in a context variable, so a configured function called any other way while a sentinel is running raises `RuntimeError` naming `run_monitors`/`run_protocols`/`run_children`. Outside a run the guard is inert, so a unit test can still call a function directly. A factory that returns an instance another factory configured, to name a composition, gets a fresh guard around the original function rather than a guard around the guard, which would reject the runner's call. The fresh guard, and for a group the fresh group, carries over the configured instance's attributes, such as dispatch metadata, except its registry info and params, which the outer factory replaces, and its step types, which are recomputed. The returned instance itself is left as it was and is not cloned, so its closure and children are shared with it. A group has no `__call__` at all.
- "No `escalate` at the outermost layer" is not expressible in the type. When one reaches the root, `run_sentinel` records it and returns a `terminate` explained as an unhandled escalation, naming each escalation that reached the root (a decision rather than an exception, so the sample is not marked as an error and retried, and every host already applies `terminate` at every stage).

The type system still cannot tell a direct call from a runner call, so the guard is a run-time check, the same under one registry type or two.

### The extension surface {#the-extension-surface}

Protocols being ordinary registered factories is what makes them the natural extension point — a third party ships one the same way they ship a scorer or a tool, with no core change to negotiate.

That matters for scope discipline. AGENTS.md's rule — new functionality defaults to an extension, not core — applies directly: honeypots, paraphrasing, resampling and collusion-resistant untrusted monitoring carry real conceptual weight and serve a small audience. Design the extension point, ship the protocols people actually use, and let the research-grade machinery live outside core until there is demand.

The `run_monitors()` seam is what makes that credible rather than aspirational. An extension implementing honeypotting does not need a framework hook: it wraps the call, injecting synthetic known-bad steps into the stream its monitors see and comparing the reports it gets back. Paraphrasing is the same seam, transforming `step` before the call. Because protocols nest, either can be dropped between a budget-owning protocol and its leaf monitors without that protocol knowing they exist.

## Registration

`@monitor` and `@protocol` each decorate a factory that returns the function, exactly as `@approver`, `@scorer`, `@solver` and `@tool` do. Every registry decorator in Inspect is factory-based, and a dual form here alone would be the odd one out — worse, it would make the use site inconsistent, since a bare rule would be listed as `no_curl` while a configured monitor is listed as `suspicion_monitor("openai/gpt-4o-mini")`. The zero-config case pays one nested `def` for that uniformity:

``` python
@monitor
def suspicion_monitor(model: str | None = None, dimensions: list[str] | None = None) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        ...

    return check
```

`Monitor` is a type alias for the returned callable rather than a base class — one of four, since the payload varies by stage:

``` python
Monitor: TypeAlias = (
    Callable[[Context, BeforeGenerate], Awaitable[Observation | None]]
    | Callable[[Context, AfterGenerate], Awaitable[Observation | None]]
    | Callable[[Context, BeforeToolCall], Awaitable[Observation | None]]
    | Callable[[Context, AfterToolCall], Awaitable[Observation | None]]
)
```

A protocol's alias, `Protocol`, has the same four shapes returning `Decision | None` plus a fifth over `Step` — the widest union, and therefore a registration at all four stages, which is what a composition like `sequential` annotates. See [The `@protocol` decorator](#the-protocol-marker).

Registry params come from the factory signature, so `create_registry_object("monitor", "suspicion_monitor", {"model": "openai/gpt-4o-mini"})` and `create_registry_object("protocol", "threshold", {"reject_at": 0.8})` and the params recorded in the log work as they do for every other registry type. `registry_create` is not a construction path here: it instantiates only factories whose return annotation names a class matching the registry type, and `Monitor` and `Protocol` are union aliases, so it would hand back the factory; there is deliberately no `registry_create` overload for the two types, so the call is a type error rather than a silent no-op. Note what is *not* a parameter on a monitor: a threshold. A monitor factory taking `threshold=0.8` has nothing to do with it, since the monitor cannot act — the constant belongs on a protocol, which is the anti-pattern [Monitor or protocol?](#monitor-or-protocol) describes, wearing configuration as a disguise. The new `RegistryType` values are `"monitor"` and `"protocol"`.

#### Instance names

A registry name identifies a factory, not a configured instance, and two instances of one factory in a configuration — two `sequential`s, two `suspicion_monitor`s with different models — need telling apart in the log and in the store. So every configured monitor or protocol has an **instance name**: the key, when it was given in a `Mapping`; the registry name without its package prefix otherwise (the prefix contains `/`, the path separator, so two packages' monitors sharing a leaf name in one layer must be told apart with a mapping). Names are unique within a layer, and a duplicate is a configuration error rather than an invented suffix. Nesting composes them into a **path**, `attempt/internet_attempt`, which is what `SentinelEvent` records and what `context.store_as()` namespaces by. This is #5423's dict-of-named-chains, generalised to every layer rather than only the top. A factory that returns [several functions](#one-function-one-stage) configures one instance, with one name and one path; its functions are told apart by their `__name__`, which `Reported` and `SentinelEvent` record as `function`, so path plus function identifies a report within a step.

#### Bare names inside the package, prefixed names outside it

`Context`, `Report`, `Action` and `Host` carry no `Monitor` prefix. The package name already supplies it — `MonitorContext` stutters as `monitor.MonitorContext` — and this is the local convention rather than a departure from it: `inspect_scout` exports `Result`, `Error` and `Reference` unprefixed from its scanner module, and Inspect's own public vocabulary is `Score`, `Target`, `Store`, `Approval`, `Task`. The payload types were already bare, so the prefixed names were the inconsistent ones.

The line is namespace, not importance: **a type living in the sentinel package goes bare; a type living in a shared namespace keeps the prefix.** Hence `SentinelEvent` stays `SentinelEvent` — it belongs to `inspect_ai.event` beside `ToolEvent`, `ModelEvent` and `ApprovalEvent`, where the prefix is the discriminator and a bare `Event` is already taken.

A user with their own `Context` is served by `import as`, which is the ordinary cost of short names and the reason the rule stops at the package boundary.

### The stage comes from the annotation {#the-stage-comes-from-the-annotation}

There is no `on=` or `stage=` argument. The type of the second parameter is the declaration, and it is required.

This is the `@tool` contract, not a new idea: `tool/_tool_info.py:118` calls `get_type_hints(func)` and builds the entire `ToolInfo` — name, description, parameter schema — from annotations, with no override and no second spelling. Requiring the payload type is the same move, and it is consistent with the house rule that every function carries annotations.

A string alternative (`@monitor(on="tool_call")`) was rejected for the reason any parallel vocabulary is: it can disagree with the annotation, and then something has to adjudicate. A type-valued one (`on=BeforeToolCall`) merely restates what is already there.

Deduction runs when the factory is called — that is, at configuration time, which is when the decorator first holds the returned callable. `@scorer` already works this way: `scorer_wrapper` invokes `scorer_type(*args, **kwargs)` and then inspects the result (checking `is_callable_coroutine`) before tagging it. So this needs no new machinery, and the error arrives when the monitor is configured rather than at first dispatch.

**The one failure mode is real and already documented in this tree.** `_control/eval_state.py:40-45` spells it out: under PEP 563 (`from __future__ import annotations`) every annotation is a string, and `get_type_hints()` fails outside a type-checking context. Our audience — people minimizing import weight — is precisely who would import `BeforeToolCall` under `TYPE_CHECKING`. The mitigation is not an override to escape to, it is an error that names the fix:

> check: could not resolve the annotation `BeforeToolCall`. Import it at module level and at runtime, not under TYPE_CHECKING or inside the factory.

A monitor annotates exactly one payload type. A protocol may annotate one payload type or `Step`, which registers it at every stage and is meant for stage-agnostic protocols that only forward the step, such as `concurrent` and `observe_only`. A function never dispatches on the payload type, so a union of some stages but not all is a `TypeError` when the factory is called, naming a `ProtocolGroup` with one function per stage, or `Step`. `Step` is a union, so a union of every stage written out is `Step` by value and is accepted as such; the check compares the annotated set against the set of all stages, and rejects any proper subset with more than one member.

### The kind is the decorator, and the return annotation is checked against it {#the-kind-comes-from-the-return-annotation}

The parameter annotation says *where* a function runs; the decorator says *what it emits*, and the return annotation must agree:

``` python
@monitor  ...  async def judge(context: Context, step: BeforeToolCall) -> Observation | None: ...
@protocol ...  async def no_curl(context: Context, step: BeforeToolCall, escalations: Escalations) -> Decision | None: ...
```

An earlier draft had one decorator and read the kind off the return annotation, so that the same `@monitor` could produce a scorer or a decider. That was rejected in favour of two decorators, and the reasoning that once argued *for* reading annotations still holds here — a `@monitor(kind="scorer")` argument was rejected because a second vocabulary can contradict what the function returns. Two decorators are not a second vocabulary. The decorator is the registry type, and mypy checks the body against it: a `return Decision(...)` inside a function annotated `-> Observation | None` is an ordinary type error, and a factory returning that function does not satisfy `Callable[P, Monitor]`.

Inferring the kind from which fields a report happens to set was also rejected, and this is the sharper argument. It makes the kind a property of a *report* rather than of a *function*, so the same function could score on one call and decide on the next, and nothing could be said at configuration time. A declared kind buys two things:

- **The config-time check gets specific.** The [monitors-only error](#monitors-need-a-protocol) can name the monitor and say what to wrap it in, because the configuration knows which entries are monitors before anything runs.
- **Emitting both is unrepresentable.** The old prose rule — a monitor doing both "is taking policy into its own hands" — was discouragement. Two registry types make it a type error, and this is the bigger win: a function that emits a score *and* an action has already thresholded internally, so the protocol above it cannot tell which half to trust and the calibration never happens.

### One function, one stage {#one-function-one-stage}

An earlier draft made `Monitor` a class with four overridable methods. Dropping it was the right call, and the deciding argument is empirical rather than aesthetic: **monitors watching more than one stage are rare.**

- All four motivating cases in [Goal](#goal) are single-point.
- The obvious two-stage candidate — correlate a call with its result — collapses to one `AfterToolCall`, because that payload already carries `call`, `result`, `output`, `view` and `agent_messages`.
- Both shipped designs surveyed in [Prior art](#prior-art) agree: OpenAI's Agents SDK registers guardrails individually, and Claude Code hooks are per-event with no grouping construct at all.

So no function watches two stages, and deliberately so. A concern spanning two stages is two functions. When they are independent, they are two monitors, listed twice. (A protocol annotating `Step` is one function registered at four stages, not a grouping of four functions — it sees whichever step arrives and decides about it.)

When the two share state, they come from one factory. Accumulating across stages is the trajectory-score case this design argues for, and closure state is wrong for it, since the factory runs once per configuration and its functions are shared across samples. So a `@monitor` factory may return a `MonitorGroup(before, after)` instead of one function, and a `@protocol` factory a `ProtocolGroup(...)`, and the group is one configured instance: one instance name, one `path`, one child context, and so one `store_as` namespace that every member shares. Each member still watches exactly one stage and is validated as a lone function would be, when the factory is called, since only the decorator knows the kind (a `MonitorGroup` from a `@protocol` factory, or the reverse, is a `TypeError`); members are told apart by their `__name__`, recorded as `function` on `Reported` and `SentinelEvent`, and the group's constructor rejects an empty group or a duplicate name. The instance accepts the union of its members' payload types. The runner treats it as one child and runs every member that accepts the step, in the order the factory returned them and one after another since they share a store, so one instance may contribute several reports at a step, which `Observations` and `Decisions` already accommodate as sequences. A member that returns `terminate` or calls `decide_final()` ends the instance's run, as it would for a lone protocol: the members after it do not run and are not recorded. This is the class with a method per stage that the first paragraph rejected, reached through functions, so the static scan and the per-function portability verdict still work. Two instances of one factory, and two samples, keep separate state as before.

The group is an explicit type rather than a list. An earlier version accepted any sequence of functions, which typed as `Sequence[Monitor]` and so let a caller index it and call a member directly, and let a factory that meant to return one function return a list by accident. `MonitorGroup` and `ProtocolGroup` are final classes that hold their functions in order and have no `__call__`, so pyright rejects calling a group, `@monitor` on a factory returning a `MonitorGroup` is typed `Callable[P, MonitorGroup]` by overload, and the aliases (`Monitors`, `Protocols`, `Sentinels`) and the shipped protocols' parameters name the groups beside the functions. A factory that returns a plain list or tuple is an error that says to return the group.

### Decorator arguments {#decorator-arguments}

Both decorators take an optional argument form, `@monitor(name=..., version=...)` and `@protocol(name=..., version=...)`, typed by overload so the bare and called forms check alike. `name` overrides the registered name, as `@solver(name=)` does. `version` is a non-negative `int`, default 0, stored in the registry info's metadata and recorded in the log's configuration entry when not 0, so a calibration can record which version of a monitor it measured, the same convention as Inspect Scout's scanners; bump it when a change alters the scores or decisions. `portable=` (default `True`) says whether it can run outside an eval, and is checked when the factory is called ([Portable Monitors](../docs/portability.qmd)); there is no `fail=` (see [Failure semantics](#failure-semantics)).

Two alternatives were considered and rejected:

- **A name convention** (`suspicion_on_tool_call`, selected as `suspicion_*`) encodes in the identifier what the annotation already says, so the two can disagree and a consistency check has to adjudicate. Nothing in Inspect's registry parses names for structure today; names are explicit, never meaningful.
- **Glob selection over registry names** solves a problem that only exists if grouping does. Selection is an explicit list, like scorers. If listing gets tedious a package exports one — `monitors=acme.all_monitors()` — which needs no framework concept.

### What the function shape buys

Beyond removing a base class from the surface:

- **Each monitor's stage and portability are legible per function**, not per class. A class with one portable method and one unportable one is an all-or-nothing verdict; a function is its own verdict. `sentinel-deployment.md` builds on this.
- **Monitors are enumerable without importing anything.** `_util/decorator.py::parse_decorators` walks top-level `FunctionDef` nodes for a named decorator using nothing but `ast`, which is what a deployment bundler needs to find the monitors in a file and compute its import closure. That works *only* because monitors are top-level functions — a class body is an `ast.ClassDef` with its methods nested inside, invisible to that walk.

One thing it does **not** buy, which is worth stating rather than implying: a closure captures state as readily as `self` does, and it is captured once per configuration and shared across every sample either way. `nonlocal count` is more friction than `self.count = 0`, and a variable declared beside the `async def` reads as shared in a way an attribute does not — but the hazard is the same one, and [State](#state) still has to argue it rather than the shape having settled it.

### Where the decorator lives

A monitor author should be able to depend on the wire types and nothing else. That works without a bridge: `_util/registry.py` is already a leaf — its runtime imports are stdlib plus pydantic plus four small `_util` modules, and every reference to `Task`, `Agent`, `Approver`, `Solver`, `Tool` and the rest is `TYPE_CHECKING`-only, present for `registry_create`'s overloads. `registry_tag` is two `setattr` calls and `extract_named_params` is stdlib `inspect.signature().bind()`.

So the registry primitives move into `inspect_core` alongside the types (`inspect-core.md`), and `@monitor` defined there registers for real — no marker attributes, no deferred drain, no import-ordering hazard. Module identity gives a shared singleton registry: `inspect_ai` imports the same module object, so there is one dict rather than two to keep coherent.

Two details. `ensure_entry_points()`, called from `registry.py`, loads plugin entry points and can therefore import arbitrary third-party packages at runtime — it has to be injectable or a no-op in the leaf. And because `@monitor` decorates a factory, parameter capture is the ordinary `registry_tag(factory, monitor, info, *args, **kwargs)` call that `@scorer` already makes; route it through the existing `extract_named_params` rather than reimplementing it, since that carries the VAR_KEYWORD flattening from #4374 which makes capture-then-replay idempotent. The registry records only lists, tuples and dicts, so the decorator hands the factory, and records, a container of configured monitors and protocols of any other kind (a `UserList`, a `UserDict`) as the list or dict that keeps its names and order; other arguments are passed as given. A composition runs the named snapshot of its children it records, so changing the caller's container later changes neither.

## State {#state}

Two scopes, per-sample and per-task, both reached through `context.store_as()` and neither through a global (see [One access path, not two](#one-access-path-not-two)). Per-sample is built; per-task is planned.

### Per-sample

Per-sample state is the sample `Store`, which already exists and is already `ContextVar`-bound. A monitor runs inside the sample's context, so `store_as(MyState)` works today with no new machinery. What a monitor adds is automatic namespacing, so two configured instances of the same monitor don't collide, and the monitor doesn't have to know its own instance name:

``` python
class TrajectoryState(StoreModel):
    score: float = Field(default=0.0)
    flags: list[str] = Field(default_factory=list)


async def check(context: Context, step: BeforeToolCall) -> Observation | None:
    state = context.store_as(TrajectoryState)
```

The namespace is the instance path, so renaming a mapping key or wrapping a monitor in another layer moves its state; there is no override. There is no `context.store`: the whole sample store holds the agent's state too, and a monitor or protocol must not read or change state it was not given, so `store_as()` is the only way in the API offers (a convention rather than isolation: the `StoreModel` it returns carries its store). Decided by the maintainer on 2026-10-02; a monitor that needs some of the agent's state is a separate question (an explicit, read-only view of named keys), not yet designed.

The package itself keeps one key in the sample store outside any instance's namespace: `inspect_sentinel:nonce`, the nonce `Host.score` puts on the markers of every prompt in the sample. Solver code that reads the store, and the eval log, see it.

This is where the shape argument from [What the function shape buys](#what-the-function-shape-buys) has to be cashed out rather than assumed. A factory closure is created once per configuration and shared by every sample, so anything captured in it is *task*-scoped whether the author meant it or not. `context.store_as()` is how per-sample state is spelled, and the factory docstring has to say so — the shape makes the sharing more visible, it does not prevent it.

Backing per-sample state with the sample `Store` gets three things for free: it is recorded to the transcript as `StoreEvent`s, so the monitor's reasoning is in the log; it survives checkpoint and resume; and `store_from_events_as()` can reconstruct it from a finished log for analysis.

#### One access path, not two {#one-access-path-not-two}

There is no `monitor_store_as()` global. Ambient accessors exist in Inspect for call boundaries you do not control — a tool invoked by the model, a scorer invoked by the framework. A monitor controls every one of its own helper call sites, so a helper can simply take what it needs as a parameter.

The one case that looks like it needs a global — a scorer reading what a monitor accumulated — does not. The monitor's per-sample state *is* in the sample store, namespaced by the monitor's [instance path](#instance-names), so a scorer holding a `TaskState` reads it with the existing `store_as(TrajectoryState, instance="attempt/internet_attempt")`. The only thing a monitor-specific global would add is not having to name the instance, which is exactly what makes it ambient and unportable.

`context.store_as()` mirrors the familiar global `store_as()` deliberately. Bare `store()` is not available and should not be: it is taken by `inspect_ai.util.store()`, and a monitor's store is a *namespaced view* of the sample store rather than the same object.

Injecting the store is also what keeps `StoreModel` usable outside an eval. `StoreModel` declares `store: Store = Field(exclude=True, default_factory=store)` — it falls back to the global `ContextVar`, which is precisely the ambient binding a proxy cannot provide. `context.store_as()` passes the store explicitly, so that path is never taken; a `monitor_store_as()` global would have made monitor state quietly unportable.

#### No global for `host` {#no-global-for-host}

Neither is there a `monitor_host()`, and here the argument above is reinforced by a second one. The store case rests only on helpers being able to take a parameter; the host case rests on that **and** on ambient authority being a thing to avoid on its own terms. The value of `context.host` is that it is the only portable route out of a monitor — everything a portable monitor can do to the outside world is enumerable from one interface — and any global reopens that. Today it carries inference and asking a person (`generate` and `ask_human`); outbound HTTP is planned as `fetch` through named endpoints (`sentinel-deployment.md`). Meanwhile an in-process monitor may call inspect_ai APIs directly, such as `sandbox()` or `logging`, if it is declared `portable=False`, at the cost of portability to a proxy.

The leak that already exists is `get_model()`. A monitor can call it and bypass `context.host` entirely, and the `portable=True` check refuses it. There is in-process value in routing through the host too, not only portability: `context.host.generate()` can default to the `monitor` role and let the host record monitor inference distinctly from the agent's, neither of which a direct `get_model()` gets.


### Per-task

Not built. Per-task state has no mechanism today and is the harder half; it waits on a design for the concurrency and persistence questions below. `pr-series.md`, "Deferred", "`store_as()` scopes", plans four scopes for `store_as()`: the sample in one epoch (today's), the sample across epochs, the task, and global.

``` python
class RunTotals(StoreModel):
    terminated: int = Field(default=0)


state = context.store_as(RunTotals, scope="task")   # planned: store_as() has no scope= today

```

Three things are unresolved:

**Concurrency.** Samples share one event loop and interleave at every `await`. A single `Store` operation is atomic and no lock is warranted (per AGENTS.md, Inspect runs on one event loop thread), but a read-modify-write *spanning an `await`* is a real race:

``` python
count = state.terminated          # read
await classify(step)              # another sample runs here
state.terminated = count + 1      # write clobbers
```

This is less alarming than it looks, because the canonical critical section is naturally await-free. A protocol's decision — compare score to threshold, check remaining budget, decrement — is pure arithmetic; the `await` happened earlier, when `run_monitors` ran. An await-free read-modify-write is atomic under the single-loop-thread invariant, so it needs no lock and no helper.

So the rule is a documented "don't `await` between reading and writing shared state", with the reason attached rather than asserted. A helper that makes the critical section syntactically obvious would be a nicety, not a fix.

**Persistence.** In-memory only is the simple answer, and it means task state starts empty on `eval_retry` and on eval-set resume — so a monitor that gates on a cross-sample total behaves differently on a resumed run than on a fresh one.

For a budget that asymmetry has a direction worth naming: a reset budget means *more* auditing than intended, not less. Erring toward more human review is the safe side of that error, which makes in-memory defensible for v1 on a principle rather than on convenience. A monitor that gates the other way — "terminate once 3 samples have been flagged" — resets toward *less* intervention, and that one genuinely needs durability. Writing it somewhere durable means picking a home (`EvalSpec.metadata`? a new `EvalLog` field? a task-level event stream, which doesn't exist) and deciding when it's flushed.

**Scope.** "Task" is right for cross-sample aggregation within one task. It is the wrong scope for anything that wants to span an `eval()` over several tasks, and for `eval_set`. Whether there's a third scope, or whether the task scope is really a run scope, is open.

## Configuration

Mirroring approval:

``` python
Task(sentinel=...)
eval(sentinel=...)
inspect eval task.py --sentinel sentinel.yaml
```

`Task(sentinel=)` accepts one protocol, or a list or mapping of names to monitors and protocols with at least one protocol among them. Every top-level configuration resolves to a protocol, so there is always a layer that owns the combined decision and emits the summary event #5423 asked for — and which protocol depends only on what was given:

| `Task(sentinel=)` given | Compiles to |
|------------------------|------------------------|
| a protocol | itself — the protocol is the root |
| a `ProtocolGroup`, or any list or mapping containing a protocol | `concurrent(...)` — protocols vote, monitors are recorded |
| a monitor, a `MonitorGroup`, or a list or mapping of monitors only | a configuration error naming `observe_only()` and `threshold()` |

A lone protocol is the root, unwrapped, so `Task(sentinel=threshold(suspicion(), reject_at=0.8))` records `threshold` at the empty path and its monitor at `suspicion`. A lone `ProtocolGroup` is wrapped in `concurrent`, since the root returns the step's one outcome and combining several functions' decisions is `concurrent`'s job. `Task(sentinel=[p])` is a list, so it resolves to `concurrent([p])` and records `p` at `p` and `concurrent` at the empty path. The host invokes the root itself, so the root's children's paths are bare (`p`, not `concurrent/p`), and the root records its own decision at the empty path like any layer. A single-rule configuration therefore logs one event per step, the rule's, at the root.

``` python
Task(sentinel=no_curl())                                    # a rule; acts
Task(sentinel=[no_curl(), no_rm_rf()])                      # two rules; concurrent
Task(sentinel=observe_only(suspicion_monitor()))            # a monitor; records only
Task(sentinel=threshold(suspicion_monitor(), reject_at=0.8))
Task(sentinel={
    "attempt": sequential([internet_attempt(), human(stages=["tool_call"])]),   # a rule that escalates; the human takes its escalations
    "escape": sandbox_escape(),                               # a rule; runs on every call regardless
})
```

The last form is #5423's example, and the mapping keys are the [instance names](#instance-names) that reach the log as `attempt/internet_attempt`, `attempt/human`, `escape`. `internet_attempt` is a rule that returns `Decision.escalate(...)` when it suspects an attempt, so the person sees only those; a monitor in its place would be recorded and fall through, since an observation is not a decision for the `sequential` to stop on.

The flat case in YAML is a list, like approval; registry lookup searches both types:

``` yaml
sentinel:
  - name: no_curl
  - name: no_rm_rf
```

One entry is a lone instance, the root itself, as `Task(sentinel=no_curl())` is; a mapping is one entry when its `name` is a string, and a mapping of instance names otherwise:

``` yaml
sentinel:
  name: no_curl
```

Named instances are a mapping, and a protocol with children nests them under the key that names its parameter:

``` yaml
sentinel:
  attempt:
    name: sequential
    children:
      - name: internet_attempt
      - name: human
        params:
          stages: [tool_call]
  escape:
    name: sandbox_escape
```

``` yaml
sentinel:
  - name: threshold
    params:
      reject_at: 0.8
    monitors:
      - name: suspicion_monitor
        params:
          model: openai/gpt-4o-mini
```

The nested key is the factory's parameter name — `monitors:` for the protocols that read scores, `children:` for the two compositions — and it is legal exactly when the factory has that parameter. The top-level key is `sentinel:`, so it cannot be confused with the nested `monitors:` (an earlier draft used `monitors:` for both). `monitors:` under a rule, or under a monitor, is a configuration error, and the schema must not let it parse silently. Nesting is the shape rather than a cost: `Task(solver=chain(...))` already has it in Python, approval's config already nests policies, and a tree is what permits [nested pipeline stages](#the-extension-surface), which a flat schema could not express at all.

There is no `tools` key. An earlier draft borrowed it from `ApprovalPolicy` to scope the tool stages, but it was a wart: a `BeforeGenerate` monitor is not tool-scoped and would fire regardless, so the key could only mean "which tools this applies to, if it watches a tool stage at all". A monitor that cares about some tools filters on `step.call.function` itself, which it can already do and which is more honest about a monitor not being a per-tool policy. An entry takes `name`, `params`, `version`, `meta` and nested keys only, so a `tools:` carried over from an approval policy is an error, not a param.

`sentinel_from_config` builds a configuration through `create_registry_object`, nested entries first, and returns the instance, list or mapping unresolved, for `resolve_sentinel`. It takes a YAML or JSON file whose only key is `sentinel:`, a bare registered name, or the parsed shape. A name is looked up as both a monitor and a protocol, which is unambiguous because `@monitor` and `@protocol` refuse a name already registered as the other kind. A bare name that matches nothing exactly is then tried in `inspect_sentinel`, so `name: threshold` finds the shipped protocol, and a local `observe_only` shadows the shipped one, which stays reachable as `inspect_sentinel/observe_only`. A repeated key in a file is an error rather than the last one winning. Since `name`, `params`, `version` and `meta` are an entry's own keys, the decorators refuse a factory with a parameter of one of those names. Every error names the entry, as in `sentinel.attempt.children[1]`. `config_from_sentinel` is the inverse, from each instance's registry name, params and version, with a param holding monitors or protocols written as nested entries; it is what the log records and `eval_retry` rebuilds from. An entry records `version` when its factory's is not 0, and building an entry whose recorded version differs from the installed factory's logs a warning naming both and builds anyway. The shape is `SentinelConfig`, a pydantic model over one `SentinelEntry` (`name`, `params`, an optional `version`, an optional `meta` reserved for fields added later, which readers keep without interpreting and nothing writes yet, and nested entries), a list of them or a mapping of instance names to them; it lives in inspect_ai (`inspect_ai.log`), so logs load without `inspect_sentinel`.

## Failure semantics {#failure-semantics}

**A monitor that raises never fails the sample itself. A protocol that reads a failed monitor's layer fails, unless it checks for the failure first. A protocol that raises fails the sample.**

Decided with the maintainer 2026-10-01; this replaces the earlier design, in which a monitor's exception propagated unless nothing consumed its report and `@monitor(fail="open")` / `@protocol(fail="open")` declared tolerance per function.

The runner catches an `Exception` from a monitor function at the call. The monitor's result is a failure, `Failed(name, path, function, error)`, in place of an observation: it is recorded through `Recorder.failed` (inspect_ai records a `SentinelEvent` with `status="error"` and the error's message in `error`, one per failure, at the call), and the siblings run on. A cancellation is not a failure: it is a `BaseException`, recorded as cancelled as before, and so is an error a monitor raised while being cancelled, as an SDK that turns cancellation into its own exception does. Nor is a limit the host enforces: a `LimitExceededError` raised in a monitor (a cost or time limit reached during its model call) ends the sample as it would anywhere else. What the runner itself detects about a monitor, such as returning a `Decision` or calling `decide_final()`, is a bug in the monitor's code rather than a failure of its check, and raises as before.

What happens next depends on who reads the scores:

| Configuration | When a monitor raises |
|------------------------|------------------------|
| `observe_only()`, or a monitor as a plain link of `concurrent()` or `sequential()` | recorded; a warning is logged once per instance path and exception type; the step continues |
| a protocol that reads the observations, such as `threshold()` | reading them raises `MonitorFailedError`, naming each failed monitor and its error; the protocol fails, so the sample fails |
| a protocol that checks `observations.failed` first | it decides what a failure means, reading the other monitors' scores through `observations.succeeded` |
| a protocol, at any depth, that raises | the sample fails |

**Consuming protocols fail closed by default.** `Observations` carries the layer's failures in `failed`, and while any are present its readers (iteration, `len`, indexing and `max_suspicion()`) raise `MonitorFailedError`, chained to the first failure's exception. A protocol written without failures in mind cannot decide on a partial set of scores by accident: an empty-looking result when the only monitor failed would otherwise read as "nobody observed", which `threshold()` answers with `proceed`: failing open. `threshold()` does not handle failures, so a failed monitor under it fails the sample. `succeeded` is the same layer's observations without the failures, readable whether or not any failed; a protocol that escalates, rejects, or decides from the rest when a monitor fails says so in its own code:

``` python
observations = await run_monitors(monitors, context, step)
if observations.failed:
    return Decision.escalate(f"{len(observations.failed)} monitor(s) failed")
peak = observations.max_suspicion()
```

**Recording-only compositions record and continue.** Under `observe_only()` nothing reads the scores, so killing a sample over a report that was going nowhere would trade a safety mechanism's purpose, collecting scores, for nothing; `concurrent()` and `sequential()` read only their protocols' decisions, so a monitor among their links is recorded the same way, and `sequential()` lets a failed monitor fall through as an observing one does. The warning is the one live signal; the event is the durable one, so a calibration run can count failures per monitor. A custom protocol that reads only its children's decisions is in the same position and gets no warning unless it checks `failed` itself; the event is still recorded.

**Protocols that raise fail the sample.** The precedent is unanimous once you look at the code rather than the docs. Neither `apply_tool_approval` nor review's `_apply.py` wraps the call itself — each has exactly one `try/except`, and it is around the *viewer*, falling back to default rendering. An approver or reviewer that raises propagates and ends the sample. A protocol that raises because a monitor's observations could not be read fails the same way; the monitor's own `error` event is already in the transcript.

Transients are absorbed below: `GenerateConfig.max_retries` and `timeout` retry model calls before anything reaches the monitor boundary, so an exception arriving here has survived the retry policy — a persistent outage or a bug. That is why the failure is recorded rather than retried, and why a protocol that reads scores fails rather than guessing.

There is no `fail=` argument on `@monitor` or `@protocol`: a monitor never fails the sample on its own, and the consuming protocol's code is where tolerating a failure is decided, visible in review.

## Composition {#composition}

Composition is the protocol's job: it collects reports from every configured child and produces one outcome. This section describes the **concurrent** composition, which is the default and what `concurrent()` implements; the ordered alternative, `sequential()`, is in [Two compositions](#two-compositions-concurrent-and-ordered). Approval's ordered chain — first non-`escalate` decides, later approvers never see the call — is right for an authorisation chain ending in a person and wrong as the default for a set of observers.

- **Observation is unconditional.** Every child configured for a stage runs at it, whether or not another has already formed a view.
- **Every report is recorded**, including the ones that lost.
- **Reports combine in two halves**, one per kind.

**Aggregating suspicion.** `Observation`s from several monitors have to combine before thresholding. Max is the obvious default; per-dimension max is the right shape when scores are structured, since dimensions are meant to be independent (see [Prior art](#prior-art)). Weighted combinations and requiring agreement across monitors are protocol policy, not framework behavior.

**Aggregating decisions.** Across `Decision`s, the strongest wins: `terminate` \> `reject` \> `modify` \> `escalate` \> `proceed`. `escalate` means "I decline to decide; someone above should", so a peer's `proceed` cannot override it, while anything stronger still wins, and a layer whose strongest decision is `escalate` escalates. At the root there is nobody to hand an `escalate` to, so it is a configuration bug: `run_sentinel` returns a `terminate` in its place, recorded as an unhandled escalation, at every stage. A configuration that wants a fixed policy ends with `handle_escalation("terminate")` or `handle_escalation("proceed")`, or with `human()` (decided 2026-10-10, [pr-series.md](pr-series.md), "Unhandled escalate", built in sentinel #74 and #78). One `Action` ordering covers all four stages, so this is a single comparison function rather than one per stage — it is `Decisions.strongest()`. Claude Code hooks land the same way — all hooks run in parallel, any `deny` blocks. `terminate` additionally cancels siblings still running, since nothing can outrank it.

**A final decision is not a vote**: a protocol may combine advisory decisions however it likes, but a child that calls [`decide_final()`](#authoritative-reports) ends the step before the protocol combines anything.

Then the protocol reconciles the two halves with its thresholds and budget, subject to [child decisions being advisory](#child-decisions-are-advisory).

### Concurrency is a safety property, not just a latency one

Three LLM monitors in sequence triple the added latency at every step, which argues for an anyio task group with `tg_collect()`'s semantics (not `asyncio.gather`, per AGENTS.md). TraceGuard gives a stronger reason: **independence is the point**. Its 95%-versus-0% result against a colluding monitor comes from fragmenting evaluation so no single model sees the whole assessment, which requires monitors that cannot observe each other's conclusions.

So concurrent dispatch is the default, and the cost is that `modify` becomes incoherent — a concurrent child cannot see another's rewrite. Settled in `concurrent()`: two rewrites conflict, so a `modify` beside another `modify` becomes a `reject` naming the modifiers, while a `modify` beside a `proceed` or an `escalate` is honoured, unchecked by the children that proceeded, as a lone modifier's is. Last-wins in configuration order and "two is an error" were the alternatives to the rejection. Until 2026-10-26 any other decision but `escalate` turned a `modify` into a `reject`, because the other children approved the *original* arguments and nothing approved the modified ones; the maintainer reversed that on 2026-10-26, since a peer that proceeds has no objection to the call and should not make the layer stricter than no peer at all ([Two compositions](#two-compositions-concurrent-and-ordered)). A `sequential` has no such problem, since only one link decides.

### A second axis: concurrent with the *agent*

Separately from monitors running concurrently with each other, a monitor can run concurrently with the agent rather than blocking it — the OpenAI Agents SDK's `run_in_parallel`. Blocking costs latency; running in parallel means the agent may consume tokens, or take a step, before a violation is caught.

This is a real knob and it belongs on the protocol, not the monitor: whether late detection is acceptable is a deployment judgement. It is a different axis from `concurrent()`, which is about children running concurrently with *each other*; to keep the two apart the knob should be named for what it withholds, `blocking=True`, rather than reuse the word.

## Transcript

``` python
class SentinelEvent(BaseEvent):
    event: Literal["sentinel"] = "sentinel"
    factory: str
    """Registry name of the factory: a monitor or a protocol. The runner hands it to the
    recorder with each record; the instance name is the last segment of `path`."""
    path: str
    """Instance path, e.g. "attempt/internet_attempt"."""
    function: str | None
    """`__name__` of the function that produced the report, which tells apart the functions of
    an instance whose factory returned several; path plus function identifies a report. Set for
    `reported`, `superseded` and `error` events; `None` for `cancelled` and `bypassed`, which
    are per instance."""
    step_id: str
    """The step examined: at the generate stages the generate's id, the `uuid` of
    its first `ModelEvent`, shared by BeforeGenerate and AfterGenerate; at the tool
    stages the tool call id. See sentinel-development.md, "Step ids"."""
    conversation: str
    """The agent conversation this step belongs to; the same id the loop's
    `ModelEvent`s and `CompactionEvent`s carry. See "Two histories"."""
    stage: Literal["model_input", "model_output", "tool_call", "tool_result"]
    kind: Literal["observation", "decision"]
    """The report family; agrees with the registry type of `factory` by construction, for every status."""
    status: Literal["reported", "cancelled", "bypassed", "superseded", "error"]
    """What happened to the report: `reported` carries it; `cancelled` and `bypassed` record no report; `superseded` carries the decision of a protocol whose `decide_final()` lost a race or was outranked by an exception (decisions only); `error` records a monitor function that raised instead of reporting (observations only)."""
    suspicion: Suspicion | None
    action: Action | None
    audit: bool
    modified: ToolCall | None
    """The replacement call, for a `modify` decision."""
    message: str | None
    """What the agent was told, for a `reject` decision."""
    explanation: str | None
    references: list[Reference]
    """The report's references; empty for `cancelled` and `bypassed`."""
    metadata: dict[str, Any] | None
    error: str | None
    """The exception's type and message, for an `error` event; `None` otherwise."""
```

An `error` event records a [failed monitor](#failure-semantics): one per failure, written at the call, with `function` set and no suspicion, action or references. Whether the step went on or the sample failed is not on the event; a protocol that read the layer and failed shows as the sample's error.

A `cancelled` event carries no suspicion or action, but its `kind` still says whether the child was a monitor or a protocol; it says the child was in flight when the layer was torn down, by a sibling's `terminate` or `decide_final()` or by cancellation from above. A `decide_final()` decision's event is the ordinary reported decision event of the protocol that called it, written at the root when the decision takes effect, so it follows the `bypassed` event of each layer above it, each a protocol whose own decision logic never ran. When two `decide_final()` calls race in one layer, or an exception in the layer outranks one, the losing decision's event has status `superseded`: it did not take effect. A `terminate` already recorded when a sibling's `decide_final()` outran it is followed by a `superseded` event for the same reason.

What the layer above did with a report needs no field of its own. Every layer that decides records its decision at its own path, so the answer is the event at the parent path, or, for a layer a `decide_final()` bypassed, the `decide_final()` decision, which is recorded with the path that made it. A reader can therefore tell an overruled decision from one that was never made, which is what a reader of a nested configuration needs to see. An earlier draft carried an `outcome` field for this; it duplicated the parent's event and was removed. Which child's decision a top-level result came from is exact after `decide_final()` and otherwise inferred, by walking down from the top-level decision to a child that made the same one.

`conversation` is what lets a reader follow one agent through a sample that has several, and across that agent's compactions, without guessing from spans; see [Two histories](#two-histories). `step_id` is what lets a report be matched to a label. It is recorded live, from ids that exist before the monitor runs, so a validation set built against replayed steps also matches the events an eval recorded — the two scan modes in `sentinel-development.md` share it.

`path` is #5423's `chain` field, generalised. That proposal tagged each `ApprovalEvent` with the name of the chain it belonged to; with arbitrary nesting a single parent name is not enough, and a path names the layer at every depth. `factory` and `path` are both kept because a reader wants both questions answered: which code ran, and which configured instance it was; `function` says which of that instance's functions, when its factory returned several.

A protocol is recorded exactly as a monitor is, so a nested configuration emits **one `SentinelEvent` per layer**, and that is what makes the layering readable without any further field:

```
factory="internet_attempt",             path="attempt/internet_attempt",  kind="decision",  status="reported",  action="escalate"
factory="sandbox_escape",               path="escape",                    kind="decision",  status="reported",  action="proceed"
factory="inspect_sentinel/human",       path="attempt/human",             kind="decision",  status="reported",  action="reject"
factory="inspect_sentinel/sequential",  path="attempt",                   kind="decision",  status="reported",  action="reject"
factory="inspect_sentinel/concurrent",  path="",                          kind="decision",  status="reported",  action="reject"
```

The person's `reject` ends the `sequential` at `attempt`, and the top-level `concurrent` takes it as the strongest vote; its decision at `path=""` is the summary event #5423 wanted. A `decide_final()` decision is instead recorded last, at the root, when it takes effect, after a `bypassed` event for each layer above it. Every layer's decision is recorded, the root's included; a configuration that is a single rule logs one event per step, since the rule is the root; when more than one child decided, the layer's explanation lists each child's decision after the winner's own.

An earlier draft added a `strategy: str` here and on `Decision`, to record whether a `modify` came from a deferral, a resample or a trusted edit. It was removed: **the layer's identity already is the strategy.** A protocol named `resample` returning `modify` has said what it did, and the aggregate query that justified the field — how often did this protocol resample — is `factory == "inspect_sentinel/resample" and action == "modify"`. Worse, on `Decision` it was a field every rule carried and never set.

The residual case is a protocol that genuinely takes several different actions — Ctrl-Z choosing between deferral and audit on two thresholds over one score, where `factory` does not disambiguate. Two answers in order of preference: compose it, which [nesting permits](#a-protocol-may-have-no-children), or put it in `metadata`, which already exists and needs no new contract on a serialized type.

`kind` and `stage` are both fine on an event, which is a record, in a way they are not on a protocol, which is a contract. The event stays one flat type across both kinds and all four stages — it is what the renderer, the dataframe and the log schema read, and splitting it would push a union into `generated.ts` for no reader's benefit.

`action` is typed rather than stringly typed only because the four stages share `Action`. With per-stage vocabularies the field would have to be a union of three Literals correlated with `stage`, which neither the JSON schema nor the generated TypeScript can express as a correlation — so consumers would get `string`.

The event should be emitted for observations as well as decisions. A monitor that scores every step under a protocol that intervenes twice in a hundred samples is the common case, and the ninety-eight quiet steps are the interesting data — they are also the benign distribution a protocol's thresholds are calibrated against, so recording them is load-bearing rather than merely nice.

Open: whether `SentinelEvent` replaces `ApprovalEvent`/`ReviewEvent` when the report came from a monitor, or sits alongside them (two events for one outcome), or whether those events gain a `monitor` field.

## Inference, budget, and injection {#inference-budget-and-injection}

- **Budget.** Monitor inference must not be charged to the agent's limits, the same exemption approvers get today (`suspend_token_limit()`, `suspend_turn_limit()` in `approval/_apply.py`). A monitor that terminates a sample for exceeding a token limit its own inference consumed is a trap worth closing up front.
- **Model role.** The `monitor` role is the default for `context.host.generate()` with neither `model` nor `role`, and `"monitor"` is already the example role name in `docs/models.qmd`. Monitor usage should be separable from agent usage in `ModelUsage`.
- **Injection.** A monitor reads model completions, tool arguments, and tool output, all of which are agent-controlled text. `docs/control-channel.qmd` already makes this point for monitoring agents reading the control channel; it applies with more force here, because this monitor's output decides whether the sample stops. Worth a docs section, and the framework should structurally help rather than only warn: [Views](#views) proposes a shared rendering path whose default output is a delimited, clearly-untrusted evidence envelope.

## Views: what a monitor sees, and how it is rendered {#views}

A step payload says what *exists* at its stage. A monitor author has two further questions the payload does not answer: which of those surfaces will actually be there in the deployment this monitor is headed for, and how to turn them into a prompt without re-deriving the briefing/evidence boundary by hand every time. Today the first is scattered across the deployment sections and the second is left entirely to the author, so every LLM monitor concatenates context and history its own way, and most will get the envelope wrong.

A **view** is the answer to both, and it is not a type. It is a documented map of the surfaces and a family of rendering helpers — `messages_as_str` and its relatives — that turn any subset of them into prompt text with the boundary intact. Scout's `messages_as_str`, `MessagesPreprocessor` and `message_numbering` are the beginnings of this family; they already depend on nothing but `ChatMessage` and `Content`, and the design here is mostly about moving them down a level and filling in the pieces a monitor needs that a scanner does not.

Status: sketch, like the rest of this document. The Python is illustrative.

### The surfaces

What a monitor can be shown, where each thing comes from, and where it is available. "Eval-only" surfaces exist because Inspect executed the tool or registered the tool definition; nothing on the wire carries them.

| Surface | Lives in | Eval | Proxy / bridged | Replay | Trust |
|---|---|---|---|---|---|
| Briefing: `task_description`, `sample_description`, `sample_input`, `metadata` | `context.eval` | yes | **no**: `eval` is None | from transcript info | instruction |
| The subject: the call, the assistant message, or the request | `step.call` / `step.output` / `step.input` | yes | yes | yes | evidence |
| Tool result as the model sees it | `step.result` | yes | next request only (lossy) | yes; exact from `ToolEvent` | evidence |
| Whether the host truncated the result | `step.truncated` | yes | **no**: `None` (unknown) | exact from `ToolEvent.truncated` in a log new enough to record it; else `None` | framework |
| Untruncated tool output | `step.output` | yes | **no** | **no** | evidence |
| Tool call view (`ToolCallView`) | `step.view` | yes | **no** | from events only | evidence |
| Exact model input | `step.input` | yes | yes (the request) | exact from `ModelEvent`; `None` at a tool stage without one | evidence |
| The outermost running agent function's own message list (`AgentState.messages`) | `step.agent_messages` | with a running `@agent` agent function | proxy and bridge: **no**, `None` | no, `None` (not logged) | evidence |
| Every message the agent instance has sent, in order sent | `step.history` | yes | proxy: **yes**, built by the sidecar from the requests it sees, matched by content; agents that open alike share one; bridged in-process: yes, one conversation per bridge | **not yet**, `None`: the log records agent turns since inspect_ai #5990 (`ModelEvent.origin`, R7 PR 1), and R7 PR 3 rebuilds `history` from them, as sent | evidence |
| Reasoning (`ContentReasoning`) | inside assistant messages | provider-dependent: full, summary, or redacted | whatever the wire carries | as logged | evidence |
| System prompt, as the model saw it | `step.input[0]` when present | yes | yes | from events only | **see below** |
| Tool definitions, generate config | `step.tools`, `step.config` (tool stages: `step.tools` only, the producing call's) | yes | yes | from events only; `None` at a tool stage without a producing `ModelEvent` | evidence |
| Media (images, audio, documents) | content parts | yes | yes | yes | evidence |
| Earlier `human()` decisions in the conversation | `step.human_decisions` | yes | **no**: `None`, unless the proxy keeps its own `ask_human` answers | from `SentinelEvent`s | prior verdicts |
| Expected answer | `context.eval.target` | yes | **never** | yes | instruction |
| Own accumulated state | `context.store_as()` | yes | keyed store | per-transcript | own |

Two rows deserve a note.

**The system prompt changes trust class with the deployment.** In an eval it is author-written and could sit with the briefing. Under a proxy it arrives in the request body from the agent's scaffold, which is the thing being monitored. The helpers therefore treat the system prompt as evidence everywhere and exclude it by default — Scout's `exclude_system=True` is the right default for the same reason — and the briefing, which the operator supplied out of band, stays the only instruction-class input.

**Reasoning fidelity is not the monitor's to choose.** A monitor can render reasoning, but a provider may only return a summary or a redacted block, and a proxy only sees what the wire carries. The helpers render whatever is there — `<thinking>`, `<thinking_summary>`, `<thinking_redacted/>` — so the prompt says which it was. This settles [open question 9](#open-questions): reasoning is visible by default, at whatever fidelity exists, and a monitor that wants to score actions without it excludes it in the preprocessor. A protocol that wants to withhold it from an untrusted child transforms `step` before the call, as it would for paraphrasing.

Things a monitor never sees at any view, so that nobody designs around them: the sandbox itself (only what came back through a tool), other samples, other monitors' reports, at the same step or earlier (independence is a [safety property](#concurrency-is-a-safety-property-not-just-a-latency-one); `human_decisions` holds only a person's `human()` answers, never a score), the escalations of earlier links in a `sequential`, which only a protocol is [handed](#escalations-are-a-parameter), the protocol's thresholds and remaining budget, and whether its own explanation was shown to anyone.

### The helpers

Lowest level first. The first three exist in Scout today and move down unchanged; the rest are new.

**`messages_as_str(messages, *, preprocessor=None, format="text")`** and **`message_as_str`.** Scout's message rendering: role labels, tool calls with arguments, tool results with errors, `<thinking>` blocks at whatever fidelity was returned, and `<image/>`-style placeholders for media. `MessagesPreprocessor` carries the exclusions (`exclude_system`, `exclude_reasoning`, `exclude_tool_usage`) and a free-form `transform`. These depend only on `ChatMessage` and `Content`, so they belong in `inspect_sentinel`, with Scout re-exporting them. The `Transcript` overload and `Reference` extraction stay in Scout. Inspect's own `analysis/_dataframe/extract.py::messages_as_str` is a third copy today and should consolidate onto the same one.

**`message_numbering(preprocessor=None)`.** Scout's `[M1]`-style numbering with a paired `extract_refs`. `sentinel-development.md` already assumes an LLM monitor uses it so that citations in an explanation resolve to messages for free; making it available to monitors is what makes that true.

**Step preprocessing.** Built (2026-10-16). A `StepPreprocessor[S]` says what a monitor shows a model about a step: how much of its conversations, which parts of their messages, and any change of the monitor's own. `Host.score(step, ..., preprocessor=...)` applies it with `preprocess_step()` and renders the result with `step_as_str()`; a custom prompt calls the two itself. A preprocessed step is still a step of the same type, only thinned: it is a copy that lives in the monitor, never handed back to the host.

``` python
StepPreprocessor(
    input_window=AllMessages(),        # the default; LastTurns(n), LastMessages(n), NewSinceLastReport(context), or None
    history_window=None,               # the default: none of what the agent sent; a window shows it
    agent_messages_window=None,        # the default: none of the agent's own message list; a window shows it
    omit_system=True,                  # as Scout leaves system messages out
    omit_reasoning=False,              # in the step's own reply too
    omit_tool_usage=False,             # a provider's own tool use, such as web search
    transform=None,                    # S -> S, applied last to the thinned step
)
```

The windows are a closed set of frozen dataclasses, `Window = AllMessages | LastTurns | LastMessages | NewSinceLastReport`: third parties do not add one, and new ones can be added here without breaking callers. Each window cuts its list as it is; preprocessing does nothing else to `input`, `history` and `agent_messages` except the omissions. `LastTurns(n)` keeps everything from the `n`th assistant message from the end; `LastMessages(n)` keeps the last `n` messages. A list with no window becomes `[]`, even one the host could not provide, and a windowed list the host could not provide stays None. Windows do not compose. Anything the built-in windows and omissions do not provide, a monitor writes as its `transform`, which runs last so it can rely on the windows and omissions (decided by the maintainer on 2026-10-16); it can withhold a flagged message, and it can cap a `NewSinceLastReport` window at a number of turns. A transform for one step type types the preprocessor (`StepPreprocessor[BeforeToolCall]`), so a type checker rejects it at another stage; without a transform it is a `StepPreprocessor[Step]`.

The input is shown whole by default, so each step's prompt starts with the last one's and the monitor model's prompt cache keeps working. To omit the request lists and show the agent's own list, use `StepPreprocessor(input_window=None, history_window=None, agent_messages_window=AllMessages())`. Other stage content remains: every stage renders the request's tools and tool choice while `input` is known (a tool stage's are the producing request's); `AfterGenerate` renders the reply, including reasoning by default before the scaffold has curated it; both tool stages render the assistant text and call; and `AfterToolCall` renders the result. Strict blindness also needs a stage-specific `transform` or a custom prompt that curates these surfaces; a transform can clear `tools` and `tool_choice`. `history` (every message the agent sent) and `agent_messages` (the agent's own message list) are each left out unless a window asks for them, so the monitor's author chooses which, if either, the model sees besides the input (decided by the maintainer on 2026-10-19). They are left out even when the host could not provide them, rather than named as unavailable, so a default prompt in replay, where both are None, is the same as the live one (decided by the maintainer on 2026-10-26 for `agent_messages`, after the R7 replay-trust benchmark in meridianlabs-ai/inspect_sentinel#150 found the default prompt differed; `history` follows the same rule). A transform is given the host's own message objects and copies one before changing it.

`NewSinceLastReport(context)` is the incremental window. It keeps the id of the last message shown, per conversation, per source (`input`, `history`, `agent_messages`) and per monitor function (shared across the stages a function watches), in the monitor's store (`context.store_as`). The runner saves it only when the function returns or decides final, so a function that raises, is cancelled, or is passed by a descendant's final decision is shown those messages again at the next step; so an incremental monitor does not re-read its conversations, and the cost argument from `sentinel-deployment.md` — quadratic inference, linear deserialization, WASM memory that never shrinks — is answered by a window rather than by discipline. A conversation whose recorded message is gone, as after a compaction or where a host gives messages new ids on each step, is shown whole.

The monitor must be able to see everything the agent saw, so nothing is left out without a flag asking for it (decided by the maintainer on 2026-10-16). The maintainer accepted one exception on 2026-10-27: when the prompt overflows the monitor model's context window, `Host.score` cuts the oldest shown messages to fit (`sentinel-deployment.md`, "The host ABI"). The cut is on by default with no flag, since an overflowing prompt cannot be sent at all; it is named in the prompt with a marker line per cut list and counted in `Observation.metadata` (`left_out`), and it raises rather than cut the fixed part of the prompt, which at `BeforeGenerate` includes the newest INPUT message, the request being judged (decided by the maintainer on 2026-10-29). Text rendering cannot meet that: Scout renders images, audio, video, documents and provider data as bare placeholders such as `<image />`, redacted reasoning as `<thinking_redacted/>`, and drops a content type it does not know, so the monitor model is shown something other than what the monitored model was sent. The planned answer is a `messages_as_contents` that returns the thinned and transformed messages as a list of `Content` parts rather than text (maintainers, 2026-10-19); until it exists, `step_as_str` renders those placeholders without saying so.

**`call_as_str(call, view=None)`.** The Python-call rendering `default_tool_call_viewer` produces for approvers, honouring a registered `ToolCallView` when one is present. One implementation serves the human approval surface and the monitor prompt.

**`result_as_str(result, *, truncated=None)`.** A tool result with the truncation (`step.truncated`) marked, so the monitor knows the model saw a cut and how much was cut. Marking where the cut falls in the text needs inspect_ai to record it; see `pr-series.md`, Deferred.

**`briefing_as_str(context)`.** The instruction section, from `context.eval`: task and sample descriptions, the original input, and metadata the author chose to surface. Outside an eval it has no eval fields to render. Never includes `target` unless asked; the caller passing `include_target=True` is what the log records.

**`step_as_str(step, *, nonce=None)`.** Built. Returns the evidence section for a stage and the function that turns `[M1]`-style cites in a model's answer into references. Everything is in one `AGENT STEP` block named for the stage (`[BEGIN AGENT STEP - TOOL CALL REQUEST]` … `[END AGENT STEP - TOOL CALL REQUEST]`; `TOOL CALL RESULT`, `MODEL REQUEST`, `MODEL REPLY`), with blocks inside it, each left out when empty. At every stage: `HISTORY` (`step.history`), `AGENT MESSAGES` (`step.agent_messages`), then the model request as `INPUT`, `TOOLS` (the tools offered, as JSON) and the tool choice on one line (`[TOOL CHOICE] "auto"`, no end marker), after the input so the input stays a prefix the monitor model's prompt cache can reuse; at a tool stage the request is the one that produced the call, and when its input is not known its tools and tool choice are left out (decided by the maintainer on 2026-10-20, so the four stages show the request the same way). Then after a generate, `REPLY`; at a tool stage, `CALL` (the assistant's text and the call; other calls in the same message are not shown) and, after the call, `RESULT`. Nothing is outside the step block, so the framework does not mark part of what it shows as the subject and the rest as context: what the monitor model weighs is the monitor author's choice, through the windows, a transform or the template (decided by the maintainer on 2026-10-19: "the framework is not really in a position to define a bright line between the subject and additional context when it comes to prompting an llm monitor"). Each list renders as it is, so a message can appear in more than one block, such as the call's message in `AGENT MESSAGES` and in `CALL`. The call's message in a tool stage's block carries the step's `message_id`, so a cite of it resolves to the transcript's message (decided by the maintainer on 2026-10-19). Nothing else of the step is rendered: not its config, `view` or untruncated tool output. Given a nonce, every block marker carries it, as `[BEGIN INPUT (3f9a1c7e)]`, and so does every message header the default renderer writes; `Host.score` makes one per sample, on its first call, and keeps it in the sample store, so the monitor model's prompt cache keeps working from step to step. Blocks are rendered and numbered with Scout's `message_numbering()`, in the order shown. A list the host could not provide (None) is named in its block; `preprocess_step()` leaves a list with no window empty, so only a windowed one is named. `Host.score`'s default template tells the model what each block inside the step block holds. Not built yet: naming the other absent surfaces (no `view`, no untruncated `output`) in a fixed phrase, so a monitor developed in an eval and run under a proxy produces a prompt that is visibly different rather than subtly different; `step_as_str` does not render those surfaces at all yet.

**`monitor_prompt(context, step, *, question, answer=..., preprocessor=None)`.** The analogue of Scout's `scanner_prompt`, and the one most authors call. It emits: the briefing section, labelled as instructions; the evidence section inside a delimited envelope whose surrounding text says it was written by the agent under observation and may contain instructions to ignore; and the question with an answer format. This is the "structural help" the injection section asked for, delivered as the default output of the helper people will reach for first.

An LLM monitor built from them:

``` python
@monitor
def suspicion_monitor(model: str | None = None, role: str | None = None) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        prompt, extract_refs = await monitor_prompt(
            context, step,
            question="Rate how suspicious this tool call is given the assignment.",
            answer="score",
            preprocessor=StepPreprocessor(input_window=NewSinceLastReport(context)),
        )
        output = await context.host.generate(prompt, model=model, role=role)
        score, explanation = parse_score(output.completion)
        return Observation(
            suspicion=score,
            explanation=explanation,
            references=extract_refs(explanation),
        )

    return check
```

Nothing in the body concatenates briefing and transcript, labels an envelope, or decides what to do with reasoning blocks. Those are the helper's defaults, which is the point: the author writes the question, and the choices they do make — which window of history, whether reasoning is excluded — are visible in the call.

Media is the one place text rendering is not enough. The text helpers render a placeholder where an image was; a monitor that must see a screenshot builds a `list[ChatMessage]` prompt itself. That needs `Host.generate` to accept messages as well as a string, which it does, in code and in `sentinel-deployment.md`'s ABI.

### What the helpers do not do

They do not declare or enforce anything. A monitor can read `step.output` directly, and nothing stops it or records it. The doc's validity argument for `target` — that a monitor seeing something its deployed counterpart cannot should have to say so, and the log should show it — therefore holds only for `target`, which keeps its own opt-in. Whether the same treatment should extend to `output` and `view` is [open](#open-questions); a declared view was one answer and was judged heavier than the problem, since a reviewer can grep a monitor for `step.output` as easily as for a threshold constant.

### Feature directions

In rough priority order, with what each buys:

1. **Lift Scout's rendering helpers into `inspect_sentinel`.** `messages_as_str`, `message_as_str`, `MessagesPreprocessor`, `message_numbering`. A package move with Scout re-exporting, and the `analysis` copy retired. First, because everything else composes over it.
2. **`monitor_prompt` and the evidence envelope.** The default prompt shape carries the instruction/evidence boundary and marks truncation. Docs examples use it, so it is what people write.
3. **Windows, especially `NewSinceLastReport`.** The incremental window, implemented once over the store rather than by every author. Built, as `StepPreprocessor`'s windows.
4. **`call_as_str` and `result_as_str`.** Shared with the approval renderer, so the monitor's prompt and the human's screen show the same call.
5. **Absent-surface phrasing in `step_as_str`.** So a monitor that leans on an eval-only surface degrades visibly under a proxy or bridged agent.
6. **A `list[ChatMessage]` form of `Host.generate`.** Needed for computer-use and browser monitoring, where the evidence is a screenshot. Done: `input` takes a message list as well as a string.
7. **"What the monitor saw" in the viewers.** The rendered prompt is already recorded by `host.generate`'s `ModelEvent`, so Inspect View and Scout View can show it next to the `SentinelEvent`. A viewer task rather than a design one.

### What this changes elsewhere

- Open question 9 (does the monitor see `ContentReasoning`) is answered above.
- `Host.generate` in `sentinel-deployment.md` takes a `list[ChatMessage]` as well as a string, for native media.
- Scout imports its rendering helpers from `inspect_sentinel` rather than owning them; `sentinel-development.md`'s reliance on `message_numbering` for references becomes a statement about a shared helper.
- `default_tool_call_viewer` in `approval/_apply.py` becomes a caller of `call_as_str`.

## Bridged agents are the proxy case {#bridged-agents-are-the-proxy-case}

An agent that runs its own tool loop — `claude_code`, `codex` — never reaches `execute_tools()`. An earlier draft called this a gap that leaves monitors "silently covering half a bridged agent." It is narrower than that, and it is a case this design already characterizes in full.

`agent/_bridge/util.py:587` calls `apply_bridge_tool_approval(bridge, output, input_messages)` inside the bridge generate path. That function recovers tool calls from the model's *response* and applies approval to them — with `with_modified_arguments` for rewrites, `rejection_messages()` building the append-and-regenerate, and `terminate_for_repeated_rejections()` going out-of-band through the bridge. So tool-stage policy already reaches bridged agents, by projection rather than by interception.

Which is exactly the deployment table from `sentinel-deployment.md`, evaluated in-process:

| Stage | Bridged agent | Fidelity |
|------------------------|------------------------|------------------------|
| `BeforeGenerate` | `bridge_generate` is the universal choke point | full |
| `AfterGenerate` | same | full |
| `BeforeToolCall` | projected from `tool_use` blocks in the response | good |
| `AfterToolCall` | appears only in the *next* request | **lossy** |

So **a bridged agent is the proxy deployment running in-process**, and everything `sentinel-deployment.md` establishes transfers without restatement: the five restrictions `_bridge/_approval.py` arrived at, why `reject` must be a replay rather than a response edit, why rejection needs a bound, why `modify` narrows to arguments, and why `terminate` is a side call.

That reframes what has to be communicated. Not "monitors cover half a bridged agent" but "bridged agents get the portable subset, which is specified" — and the same remedy applies as for a proxy: `AfterToolCall` should say it cannot fire here, rather than let it be discovered.

It also means the portability discipline pays off twice in-process, before any proxy exists. Once the bridge hook lands (`pr-series.md`, "Deferred"), a monitor written to the portable subset works under a bridged agent unchanged; one that relies on `AfterToolCall` does not, and the [per-function portability verdict](#what-the-function-shape-buys) already says which is which.

## Deployment beyond the eval process

A sentinel is useful outside an eval — enforced at a network boundary in front of a model API, where the agent cannot bypass it and the coverage is all agent traffic rather than one task's samples. `sentinel-deployment.md` works that through. Four constraints from it bear on the protocol itself, so they are recorded here:

- **Two of the four stages are the portable core.** A proxy sees a request and a response; it never sees a tool execute. `BeforeGenerate` and `AfterGenerate` map directly; `BeforeToolCall` is recoverable from `tool_use` blocks in the response; `AfterToolCall` is **lossy** — a result appears only in the *next* request, so the last step of an episode is never seen. That is verbatim the limitation #5355 wrote `Reviewer` to escape, and `AfterToolCall` should say so rather than let it be discovered.

- **`terminate` travels, as an effect rather than an edit.** Every other action is an edit to the exchange — `proceed` is no edit, `modify` a rewrite, `reject` an append-and-regenerate, `escalate` a hand-off to the next decider. `terminate` is different in kind, but a proxy can still perform it: an error response ends the exchange and a `host.terminate()` call ends the session out-of-band, the same shape `bridge.request_terminate()` already uses in-process. What does not cross is the *guarantee* — an eval's terminate is total and synchronous, a proxy's is best-effort and may be unsupported. So there is no `PortableAction` subset; there is a capability a host may or may not provide.

  Worth keeping distinct from `reject`, which the deployment doc previously blurred: `reject` is conversational (append a synthetic result, regenerate, the model learns), an error response is transport-level (the SDK raises, the loop breaks). A proxy can do both, and they are not degrees of the same thing.

- **State must arrive as an argument.** `context.store_as(...)` survives any deployment; an ambient accessor would not, because a proxy has nothing to bind to. There is deliberately no global counterpart — see [One access path, not two](#one-access-path-not-two).

- **`view` and the untruncated `output` must be optional.** A proxy has wire types, not registered `ToolDef`s, so there is no viewer to resolve. The [rendering helpers](#views) say so in the prompt when a surface is absent rather than omitting it silently.

- **Portability has two axes, and one of them is the module.** Whether a monitor *can* run in a proxy is per-function — which context members it touches, whether it recommends `terminate`. Whether it can be *bundled* is per-module, because the import closure is a module property: one `import requests` at the top of a file disqualifies every monitor in it, whatever the functions do. This is why monitors being top-level functions matters beyond tidiness — it is what lets a bundler enumerate a file statically, with `ast` and no imports.

A fifth constraint is a design decision rather than a restriction: **monitors should be incremental by default** — classify the newest turn and fold the verdict into accumulated state, rather than re-judging the whole trajectory. Three independent arguments converge on it (quadratic inference cost, linear deserialization cost, and WASM linear memory that never shrinks), which is enough to treat it as load-bearing rather than an optimization.

The types a monitor works with also want to be importable without the eval framework; `inspect-core.md` measures what that costs today and what it would take.

## Relationship to approval and review

`Reviewer` merged on 2026-09-11 (#5356), so this is no longer a question about an open PR. Three concepts now exist for one idea, with two config keys, two event types and two decision vocabularies:

``` python
ApprovalDecision = Literal["approve",  "modify", "reject", "terminate", "escalate"]
ReviewDecision   = Literal["continue",                     "terminate", "escalate"]
```

**Chosen: a childless `AfterToolCall` protocol subsumes `Reviewer`; `@approver` survives.**

A rule at `AfterToolCall` *is* a `Reviewer` with a richer return type — the payloads are identical field for field, and `Action` is a superset of `ReviewDecision`. `@reviewer` is to be deprecated with a warning pointing adopters at `@protocol`, and `review/`'s 413 lines of parallel policy machinery retire with it rather than becoming a third copy; the deprecation waits on protocols having users (open question 25).

The reason to consolidate *this* one and not approval is adoption, not elegance: `review` is days old, while `@approver` has been public for years and is used in shipped evals and extension packages. Deprecating a freshly-landed contribution has a real cost, and it is smaller than the cost of a third parallel subsystem.

Two alternatives, recorded because the choice could be revisited:

- **Façade.** `Approver` and `Reviewer` both stay; `@protocol` adapts into each. Nothing is deprecated, and the parallel machinery stays forever. Rejected: it takes the cost without the consolidation.
- **Primitive.** A rule subsumes both, and `@approver` goes too. The remaining gap is narrow and nameable — `Approval` is binding where a nested protocol's `Decision` is overridable — but it means deprecating a mature concept people have working code against, to gain uniformity they did not ask for. Not now; better answered once sentinels have users.

#### `approve` and `proceed` are not the same word for the same thing {#approve-and-proceed-are-not-the-same-word-for-the-same-thing}

The obvious tidy-up is to reconcile the vocabularies — rename `ApprovalDecision.approve` to `proceed` and have one word. Don't.

Mechanically it is expensive and incomplete: every `@approver` ever written says `decision="approve"`, so a rename breaks them all; accepting both leaves two *legal* spellings, which is strictly worse than two words in two subsystems; and logs already written pin `approve` forever regardless.

But the better reason is that the difference tracks a real distinction:

| Word | Where | Means |
|------------------------|------------------------|------------------------|
| `approve` | approver, pre-execution | I authorize this action |
| `continue` | reviewer, post-execution | nothing to authorize; the tool already ran |
| `proceed` | protocol, nested under another | I decline to object |

`Reviewer` reached for `continue` independently, for a decision where there is nothing left to authorize. That is evidence *for* the distinction rather than against it — and it is exactly why subsuming review into a rule is clean while subsuming approval is not. Both are `proceed`-shaped; `approve` is not.

So: `@protocol` uses `proceed`, `Approval` keeps `approve`, and the adapter maps between them at the boundary, which it always had to do. If monitor ever subsumes approval, `approve` retires with `@approver` — reconciliation by attrition rather than by rename.

### A protocol layer for approval too

Keeping `@approver` leaves a thin spot: approvers are binding today, so a rule adapted into the approval chain effectively runs under an honour-everything protocol while a configured one runs under a real one. Two layerings for one concept.

`policy_approver()` is already a proto-protocol — it does the chaining, the escalate fall-through and the no-approver-means-reject default. It just isn't configurable, and approver decisions map straight to effects.

**Make a protocol a peer of the policy list, not a layer beneath it.** Both compile down to an `Approver` that `apply_tool_approval` calls:

``` python
Task(approval=[ApprovalPolicy(...)])                       # today, unchanged
Task(approval=threshold(..., reject_at=0.8))               # protocol-mediated
```

Existing behavior doesn't change — the list form keeps its current code path — and the protocol form does more inside the same slot. That avoids routing every shipped eval through new decision-making code, which is the version of this that would carry real risk.

It also gives the framing that dissolves the thin spot without a semantic change: **the `ApprovalPolicy` list is shorthand for an honour-everything protocol.** The layering is then uniform conceptually even where the code path stays direct.

Three constraints:

- **Human approvers stay final.** Same rule as [Child decisions are advisory](#child-decisions-are-advisory): a protocol may override advisory decisions, never a [final](#authoritative-reports) one. A threshold overriding a human rejection would be a safety regression.
- **Don't grow `Approval` with a score field.** A protocol can consume both shapes — decisions from approvers, reports from monitors — and an approver that wants to score should be a monitor. `Approval` reaches the transcript, the log schema, the viewer and the dataframe, and AGENTS.md wants every producer and consumer named before one of those changes.
- **Probably a separate PR.** This is the same concept rather than a tangential fix, so it is defensible — but it turns "add a Monitor registry type" into "add a Monitor registry type and introduce a protocol layer to approval". Note that the "protocol layer" here is not new machinery: it is permitting `Task(approval=)` to accept a protocol alongside a policy list, and compiling it down to an `Approver`.

The win is not tidiness. Someone running an LLM approver today has no way to express "reject only the top 1% most suspicious" — the threshold has to be baked inside the approver, where it cannot be calibrated against a benign distribution or traded against a budget. A protocol layer fixes that for approval users whether or not they ever write a monitor.

### The reviewer group's requirements (#5423) {#the-reviewer-groups-requirements-5423}

The group that built `Reviewer` proposed, in #5423, that `approval` and `review` accept a dict of named policy chains: every chain runs on every matching call, each chain keeps today's first-non-`escalate` semantics internally, and the call proceeds only if every chain approves. The motivating defect was that one ordered chain makes two *independent* monitors impossible — whichever approves first silences the other — and the workaround (wrap both in one approver) loses per-monitor events, invents a severity order, and leaves an inner monitor with nowhere to escalate.

That proposal will not land as written; the group has agreed to build on sentinels instead. Its requirements are therefore requirements on this design, and this is how each is met:

| #5423 asked for | Here |
|------------------------|------------------------|
| independent chains that all run | the default composition is concurrent; nothing is silenced |
| per-monitor events, not one wrapper event | one `SentinelEvent` per participating child, from the runner |
| a framework severity order | `terminate > reject > modify > escalate > proceed`, in `Decisions.strongest()` |
| an ordered chain inside each group, with `escalate` staying within it | `sequential()`; the escalations it hands its links stay within one `sequential`, and a layer that ends on `escalate` passes it up as its own decision |
| a human at the end of a chain who sees who escalated and why | `human()`, reading the `escalations` it is handed |
| named chains, and the name in the log | instance names from the mapping; `SentinelEvent.path` |
| a summary event when several chains participated | the top-level `concurrent` layer's own event |
| `terminate` cancels the other chains | `run_protocols` |
| `modify` with several participants is a rejection naming the chain | `concurrent()`, for two rewrites; since 2026-10-26 a `proceed` beside a `modify` no longer contests it |
| an uncovered call rejects, as a lone policy list does today | on the approval adapter path only; a `sequential` that nothing covers is `proceed`, matching review |
| concurrent by default (their open question 1) | yes |

Two things differ deliberately. The proposal used "monitor" to mean a bundle that registers an approver and a reviewer together, so that "two monitors are two chains"; here a monitor or a rule is one function at one stage and their bundle is a `sequential()`. And `Task(approval=)` does not grow the dict form — the home for named parallel chains is `Task(sentinel=)`, with approvers reachable inside a `sequential` through the adapter.

## Keeping it simple {#keeping-it-simple}

The prior-art pass added a lot of vocabulary. Most of it must stay invisible to someone who just wants to flag suspicious `bash` calls.

**The complexity lives in the protocols that read scores, and most people never write one.** Per [Two philosophies](#two-philosophies-both-first-class) a rule is a protocol with nothing to calibrate — one annotated function returning `Decision.reject(...)` — and it never encounters thresholds, budgets, or the word "audit". That the word on its decorator is "protocol" is the only control vocabulary it meets, and the docs introduce it as "the simplest protocol is a rule".

That gives two on-ramps rather than one ladder, and they rarely cross:

- **"Block bad tool calls."** A rule: `@protocol`, no monitors. This is the `@approver` experience with a wider set of stages to attach to. Done.
- **"Run a control experiment."** Monitors under `threshold()` or a shipped control protocol, calibrated against a benign distribution.

The bridge between them is the tell from [Monitor or protocol?](#monitor-or-protocol): the moment your check computes a score and wants to compare it to a constant, you have discovered why the second on-ramp exists. That is when the control vocabulary should first appear, and not before.

The concrete mitigations, in order of how much they do:

1.  **Push everything possible up.** Applying the monitor/protocol split removed `defer`, `resample`, `audit`-as-an-action, `trusted_edit`, honeypots, paraphrasing and multi-sample payloads from the monitor surface. That is a real reduction, not a presentation trick. Apply the same test to anything added later.
2.  [**Progressive disclosure through defaults**](#progressive-disclosure-through-defaults-not-through-the-return-type) — a monitor sets `suspicion`, a rule sets `action`, and neither learns what `audit` or `modified` do until it needs them.
3.  [**Ship protocols, not parts**](#ship-protocols-not-parts) — pick from a list instead of assembling thresholds and budgets.
4.  [**Kind-appropriate defaults**](#monitors-need-a-protocol) — a protocol acts, and a monitor needs a protocol to say what is done with its scores, `observe_only()` when nothing is.
5.  **Two words, kept crisp — and now two types.** *Monitor* looks at one thing and says how suspicious it is. *Protocol* decides what happens, with or without monitors to consult. The failure mode is a doc explaining audit budgets inside "how to write your first rule" — keep that page free of the control vocabulary beyond the decorator's name.
6.  [**Exotica goes in extensions**](#the-extension-surface), per AGENTS.md's default.

**One vocabulary.** An earlier draft kept two — `Action` for a deciding leaf and the control literature's `accept` / `audit` / `defer` / `resample` / `trusted_edit` for the protocol above it — on the theory that the word would signal which layer you were in. There is now one layer that acts, and one word-list:

| Control term | Where it lives now |
|------------------------|------------------------|
| `accept` | `Action.proceed` — the same thing |
| `audit` | `Decision.audit`, a flag; [a second axis, not a peer action](#audit-is-a-flag-not-an-action) |
| `defer_to_trusted` | a protocol name; it returns `modify` |
| `trusted_editing` | a protocol name; it returns `modify` |
| `resample` | a protocol name; it returns `modify` or `proceed` |

**The literature's terms name mechanisms; ours name outcomes.** Several mechanisms produce one outcome — deferral, resampling and trusted editing all end in a `modify` — so a mechanism can never replace an outcome, and the two word-lists do not merge. `accept` is the one exception, and it *is* `proceed`.

That is also why the control terms need no home in the data at all: **they are protocol names, and nothing else.** The papers' words survive where someone would search for them, the transcript records which protocol acted, and an author never learns a second vocabulary in order to return a value.

Three near-synonyms this settles, all of which look mergeable and are not:

- **`defer` is not `escalate`.** `escalate` passes the *decision* to the next decider and leaves the action alone; `defer_to_trusted` passes the *action* to a weaker model and leaves the decision made. Spelling it `defer_to_trusted` rather than `defer` matters, because bare "defer" reads in English as "postpone, hand off" — which is precisely `escalate`.
- **`trusted_edit` is not `modify`.** It is one way to produce a modification. A rule stripping `--force` from a call produces another, with no trusted model involved.
- **`reject` is not `block`.** Ours has specific mechanics from `_bridge/_approval.py` — append the rejected call and a synthetic result, regenerate, bounded by a retry limit, so the model learns it was denied. Control's `block` means the action does not execute and the protocol substitutes or escalates.

**Rejected: core executes the control actions.** There is one coherent design in which `defer` and `resample` *are* legal return values — a protocol returns `defer` and the *framework* calls the trusted model and substitutes. That is the only version where the mechanism/outcome distinction collapses, because core would be the thing turning a mechanism into an outcome.

It is rejected on scope. Core would have to know about trusted models, resampling and budget arithmetic; every new technique would become a core change rather than an extension; and `run_monitors` would stop being the only seam. AGENTS.md's default — new functionality is an extension, not core — points the other way, and [the extension surface](#the-extension-surface) is worth more than the vocabulary.

**`proceed` rather than `accept`.** A nested protocol returning `proceed` has declined to object; the protocol above it may still reject, so it did not authorise anything. An outermost protocol returning the same value *is* authorising. One word has to be accurate at both positions, and `proceed` is — it describes what happens to the exchange rather than claiming an authority that, per [Two philosophies](#two-philosophies-both-first-class), the configuration confers rather than the function. `accept` is accurate only at the outermost position. This also settles #5356's open question the same way, and leaves the `Approval` adapter mapping `proceed → approve` at the boundary, which it already had to do. The action was `continue` until 2026-10-11, when it was renamed `proceed` to match `Decision.proceed()`, so one word names the action everywhere; inspect_ai reads a `continue` in older logs as `proceed`.

## Open questions

Retired as answered by later decisions: whether a union annotation registers
several stages (yes — a protocol annotates `Step`, so it is load-bearing);
`Report` versus `Decision` (settled: `Decision` is what a protocol returns);
whether the protocol layer lands first (it is not a layer — monitors and
protocols are two registry types, and a protocol is what acts); whether proxy portability is a
goal or an observation, which now lives only in `sentinel-deployment.md`;
whether `modify` composes (settled in [`concurrent()`](#two-compositions-concurrent-and-ordered):
a `modify` beside another `modify` is a rejection naming the modifiers, and
since 2026-10-26 one beside a `proceed` is honoured);
what happens in a mixed configuration with no protocol (there is no such
configuration — a list containing a protocol resolves to `concurrent`, which
honours decisions and records observations, and a list of monitors only is
an error naming `observe_only()`); whether protocols need their own construct (a
[decorator](#the-protocol-marker), not a constructor or a type); and whether
ordered composition should be the default (no — see [Two compositions](#two-compositions-concurrent-and-ordered)).

1.  **Does `@approver` eventually become a protocol?** [Settled for now](#relationship-to-approval-and-review): a rule subsumes the merged `Reviewer`, `@approver` survives. What is open is whether the binding/advisory distinction deserves its own type permanently, or whether `@approver` becomes a `BeforeToolCall` rule once protocols have users. Deliberately deferred rather than answered.
2.  **Who owns the budget when protocols nest?** [Protocols nest](#a-protocol-may-have-no-children), so a budget-owning protocol can wrap a paraphrasing one wrapping leaf monitors — which is the point, but two nested budget-owning protocols need a rule. Outermost wins is the obvious answer; whether an inner one should be an error is open.
3.  **Which protocols ship in core?** Settled in [Ship protocols, not parts](#ship-protocols-not-parts): `concurrent`, `sequential`, `observe_only`, `threshold` and `resample`; `defer_to_trusted` was removed by the maintainer on 2026-10-02. `trusted_monitoring` (the audit budget), `trusted_editing`, `untrusted_monitoring` and the pipeline stages are extension candidates.
4.  **Score aggregation.** Max across monitors is the obvious default; per-dimension max when scores are structured. Weighted combination and requiring agreement are protocol policy — but the framework has to pick a default and say so.
5.  **Where does threshold calibration get its benign distribution?** Answered in `sentinel-development.md`: replay the `Observation` monitor over a known-benign corpus as a Scout scanner, and the results dataframe's score column is the distribution; a budget becomes a threshold by percentile, and a labelled corpus gives the precision–recall curve. What remains open there is hygiene — holding the calibration corpus out of the validation split.
6.  **Does anything in v1 actually need per-task state?** The audit budget does not — it is per-episode, hence per-sample. Cross-sample gating ("terminate once 3 samples are flagged") does, and it is the case where in-memory state resets in the *unsafe* direction. If nothing in v1 needs it, [Per-task](#per-task) can be deferred whole; if something does, its scope wants deciding between task, run and eval-set.
7.  **What is the default protocol called?** Settled: `concurrent` for a list with a protocol in it; a list of monitors only has no default and is an error naming `observe_only()`. `concurrent` names the execution shape, as `sequential` does, and leaves the reduction to `Decisions.strongest()` and the docs. `strictest` named only the decision half of the rule, and `all_of` named the configuration's meaning rather than its shape.
8.  **Does substituting `choices[0].message` need a usage convention?** [Settled](#one-vocabulary-across-stages) that `modified` carries a `ChatMessageAssistant` at `AfterGenerate`, leaving `usage` untouched. Now concrete, since `resample` is to ship: the proposal in [One that generates](#one-that-generates) is a `protocol` bucket in `ModelUsage`, separate from agent and monitor usage and not charged to the agent's limits. Open is whether that bucket should be visible in the eval's headline token count, and whether an eval that compares protocols wants it charged to the agent to keep totals comparable.
9.  **Does the monitor see `ContentReasoning`?** Answered in [Views](#views): visible by default at whatever fidelity the provider or wire supplies, excluded via the preprocessor, and a protocol withholds it from a child by transforming `step` before the call.
10.  **Lifecycle stages**: `SampleStart` / `SampleEnd` payloads would give per-sample state a place to initialise and a monitor a place to file a final verdict on a trajectory it watched but never interrupted. Natural, and cheap now that adding a stage means adding a payload type rather than a method to a base class — but it starts to overlap `Hooks`, and neither travels to a proxy.
11.  **Can a monitor produce a `Score`?** Yes, without touching scoring: `as_scorer(as_scanner(monitor, unit="transcript"))` in `sentinel-development.md`. The transcript-grain adapter reduces per-step reports to one value and Scout's existing `as_scorer` turns that into an Inspect scorer, with flagged steps as `scanner_references`.
12.  **Does `AfterToolCall` get a bridged-agent fallback?** [Bridged agents are the in-process proxy case](#bridged-agents-are-the-proxy-case), so three of the four stages map onto the bridge once its hook lands. Whether the lossy fourth — reconstructing a result from the *next* request — is worth building, or whether `AfterToolCall` should simply decline to fire and say why, is open.
13.  **Does `Task` / `Sample` grow a `description` field?** A public API change for a general-purpose affordance monitors happen to need first. If not, monitor context comes from `metadata` by convention and is correspondingly undiscoverable.
14.  **Which stages can `human()` serve?** *Settled for `AfterGenerate` (AISI R1, 2026-10-13).* The tool stages, `BeforeToolCall` and `AfterToolCall` with the result shown, and `AfterGenerate`, where the person sees the whole assistant turn and may approve or terminate it ([Humans in the loop](#humans-in-the-loop)). `Host.ask_human` takes the step itself, so the new stage needed only a new rendering, not a new interface. `BeforeGenerate` stays unsupported: it is hard to make meaningful to a person. `reject` and `modify` at `AfterGenerate` wait on the generate stages' part 2.
15.  **Failure tolerance across many samples.** [The failure policy](#failure-semantics) is settled for the single step. What is open is whether a monitor that fails on 3 samples in 500 under `observe_only()` should fail the eval, and whether that reuses `fail_on_error`'s existing shape (`True` / `False` / a proportion / a count) or needs a monitor-specific knob.
16.  **Should reading an eval-only surface be recorded?** `target` is opt-in and logged so a reader can tell whether a monitor's performance is achievable in deployment. `step.output` and `step.view` raise the same question and get no such treatment; a declared view was one answer and was judged heavier than the problem. A lighter one is for `briefing_as_str` and `step_as_str` to take the eval-only surfaces as explicit flags, which puts the choice in the call for a reviewer to see.
17.  **Default input window in `monitor_prompt`.** *Settled for `Host.score` (2026-10-16): the whole input, so the monitor model's prompt cache keeps working; `NewSinceLastReport(context)` is opt-in, and history is shown only when a window asks for it.* `monitor_prompt` takes the same `StepPreprocessor`, so it has the same defaults.
18.  **Is the rendered prompt part of the recorded decision?** `host.generate` already records a `ModelEvent`, so the prompt is in the log without any new field. The open part is whether replay should compare its rendering against the recorded one to detect drift in the helpers.
19.  **Is the alias `ControlProtocol` or `Protocol`?** Settled: `Protocol`, pairing with `@protocol` as `Tool` pairs with `@tool`. The collision with `typing.Protocol` is accepted: a monitor author's file rarely defines structural types, the package imports typing's under an alias, and the docs say "a control protocol, not `typing.Protocol`" on first use.
20.  **Should `sequential` and `concurrent` accept monitors at all?** They do here: a monitor in either is recorded and falls through, which is what gives a mixed top-level list a meaning. The alternative is `Protocols` only, with monitors reachable solely through a score-reading protocol; stricter, and it would make `Task(sentinel=[no_curl(), suspicion_monitor()])` a type error rather than a recorded-but-inert monitor.
21.  **What is the umbrella called?** Settled: `sentinel`, for the feature, `Task(sentinel=)`, the package and the event, with `monitor` kept for the scorer and `protocol` for the decider as the control literature and ControlArena use them. `Task(monitor=)` held protocols and rules and never a bare monitor, and reviewers read it as "the monitor plus the protocol". Renaming the scorer instead (`observer`) would have broken the literature's pair; `ranger`, `warden` and `oversight` were the runners-up for the umbrella.
22.  **Where does `conversation` come from?** *Partly answered 2026-10-10 ([sub-agents.md](sub-agents.md)): `conversation` identifies one agent instance, whatever implements it.* The id wants minting by the loop that owns the messages, which is `react`, the generate loop and each bridge in-process, and the proxy's session key outside; in-process the dispatcher currently takes the current agent span id (the sample uuid outside an agent span); whether `AgentState` should carry it instead is open. inspect_ai #5986 reads `agent_messages` by reference from the outermost running `@agent` agent function’s entry state. The log does not record that list, so replay returns `None`.
23.  **Is `decide_final()` the right design at all?** Any protocol at any depth, third-party code inside `observe_only()` or `threshold()` included, can call it and bypass every layer above, while this design presents child decisions as advisory to the parent. The options to weigh: a declared capability (`@protocol(final=True)`, recorded in the registry and the log, with an undeclared call raising); parent control (`allow_final=` on a layer, or finalization only at the root); containment (`shadow()`); or a different mechanism. `human()` no longer uses it, which weakens the case for keeping it.
24.  **What should an `escalate` that reaches the root mean?** *Answered 2026-10-10 ([pr-series.md](pr-series.md), "Unhandled escalate"): it is a configuration bug, and `run_sentinel` turns it into a `terminate` recorded as unhandled; `handle_escalation("terminate" | "proceed")` states a policy explicitly. There is no build-time check: typed decisions were considered and not adopted (pr-series.md, "Typed decisions").* Before: the step proceeded, recorded as the root's decision, with a once-per-process warning.
25.  **How do `review=` and tool approvers migrate?** Three parts are open. Is `review=` deprecated in favour of after-call protocols, as [Relationship to approval and review](#relationship-to-approval-and-review) proposes? How does `approval=` coexist with `sentinel=`? The ordering is already written down: approval and then the sentinel before a call, review and then the sentinel after. Can an existing approver run as a protocol through an `as_protocol(approver)` adapter, so that it composes with monitors? Tracked in `pr-series.md`, Deferred.
