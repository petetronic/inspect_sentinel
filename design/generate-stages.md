# Generate stages

Design for `BeforeGenerate` and `AfterGenerate`: which model calls they check, where the hook sits in inspect_ai, what each action means at a generate, what the step carries, and how the events link to the `ModelEvent`. It extends [sentinel.md](sentinel.md), "The payloads" and "One vocabulary across stages", and answers the deferred items in [pr-series.md](pr-series.md) that wait on the generate stages.

Status: landing (inspect_ai #5811, sentinel #42). Monitors observe both generate stages of every agent loop's model calls, and protocols may return `proceed`, `escalate` or `terminate` there; `reject` and `modify` are shape-check errors until part 2 (decisions 8–13). `history` keeps its meaning (decision 15, revised); the as-sent view AISI R10 asks for is a separate field, built in its own PR (2026-10-14) as `sent`; for 0.2.0 it became `history`, and the caller's list `agent_messages` (pr-series.md, "Renaming the histories"), which was redefined on 2026-10-21 as the outermost running agent function's own message list, read live at every stage (inspect_ai #5986). Decision 17's recording is dropped.

## Decisions for the maintainer {#decisions}

| # | Decision | Recommendation |
|---|---|---|
| 1 | How an agent generate is identified | **Revised 2026-10-12 (phase fence plus deny list):** any model call while the sample runs its solvers, except inside a tool's own body (unless an agent the tool runs opens its own span), a sentinel run, a compaction summary, `score()` from a solver, an approver, a reviewer or MCP sampling. Sample init and scoring are outside the fence. Coverage then matches the tool stages, which already check every agent's tool calls. A solver's direct helper call to another model is now checked. *Since inspect_ai #5990 each `ModelEvent` records the call's `origin`, task scorers are a final region, and a scaffold marks its own non-agent calls with `model_origin()` ([a](#a-which-calls), "Origins").* |
| 2 | Sub-agents and role models that are not the active model | **Revised 2026-10-12:** checked, by decision 1's agent-loop rule, whatever model they use. Reason: their tool calls are already checked at the tool stages, so leaving their turns unchecked was inconsistent and let an agent hand risky work to a sub-agent on another model. Helper calls stay out through the tool-body rule. The agent instance record (`step.agent`) and per-instance store remain later work ([sub-agents.md](sub-agents.md), #66). |
| 3 | Compaction summary calls | **Decided 2026-10-12:** not checked. The summary reaches the next `BeforeGenerate` as part of `input`. |
| 4 | Where the hook sits | **Decided 2026-10-10.** In `Model.generate`, once per call, outside the connection slot; the input preparation moves ahead of the slot so the step sees what is sent ([b](#b-where-the-hook-sits)). Built. |
| 5 | Cache hits | **Decided 2026-10-12:** both stages run on a cache hit. |
| 6 | A generate that raises | **Decided 2026-10-12:** no `AfterGenerate` (provider error, `fail_on_refusal`, a limit). |
| 7 | Dispatching a stage nothing watches | **Decided 2026-10-12:** skip it, at every stage, so a tool-only sentinel opens no span per generate. |
| 8 | `reject` before a generate | **Deferred 2026-10-12 to part 2** (a follow-up PR); illegal at the generate stages until then. Proposal: Not legal in v1; use `modify` or `terminate`. If kept, define it as a sentinel-authored refusal ([c](#c-before-generate-decisions)). |
| 9 | `modify` before a generate | **Deferred 2026-10-12 to part 2** (a follow-up PR); illegal at the generate stages until then. Proposal: Replace `input`, narrow `tools` to a subset by name, change `tool_choice`; not `model` or `config`. |
| 10 | `reject` after a generate | **Deferred 2026-10-12 to part 2** (a follow-up PR); illegal at the generate stages until then. Proposal: Replay: append the rejected message and the `message` as a user turn, regenerate, bounded at three; then terminate ([d](#d-after-generate-decisions)). |
| 11 | `modify` after a generate | **Deferred 2026-10-12 to part 2** (a follow-up PR); illegal at the generate stages until then. Proposal: Replace `choices[0].message` with a `ChatMessageAssistant`; usage, time and model unchanged; other choices dropped. |
| 12 | `Decision.modify(step, ...)` | **Deferred 2026-10-12 to part 2** (a follow-up PR); illegal at the generate stages until then. Proposal: One constructor, overloaded on the step type: `arguments=` at `BeforeToolCall`, `input=`/`tools=`/`tool_choice=` at `BeforeGenerate`, `message=` at `AfterGenerate`. |
| 13 | `SentinelEvent.modified` at the generate stages | **Deferred 2026-10-12 to part 2** (a follow-up PR); illegal at the generate stages until then. Proposal: `ChatMessageAssistant` joins the union for `AfterGenerate`; a `BeforeGenerate` modify records no copy, since the `ModelEvent` holds the request sent. A log contract change, for when `modify` lands. |
| 14 | `Host.generate` and `tool_choice` | **Decided 2026-10-12: add.** Already on `main` in both repositories (2026-10-07, "`Host.generate` takes separate model and role strings"). |
| 15 | `history` at every stage | **Revised 2026-10-12** ("agree with history decision, should move R10 to its own PR"): `history` keeps its meaning at every stage, the scaffold's list at the tool stages and the caller's list at the generate stages, as AISI R10 asked. No per-step rebuild. The as-sent view of an agent instance's run (R10; likely the instance's first model input, pending AISI) moves to its own PR as an on-demand query built on `TranscriptHistory.find_last` from inspect_ai #5831. *Built 2026-10-14 as the step field `sent`, the full list, folded in as each agent turn is made rather than queried from the transcript, with agent turns told apart by the scope test rather than marked in the transcript (pr-series.md, "The history as sent"). Renamed for 0.2.0 (2026-10-15): the as-sent list is `history` and the caller's list, the `history` of this decision, is `agent_messages` (pr-series.md, "Renaming the histories"). Superseded for `agent_messages` on 2026-10-21 (inspect_ai #5986): it is the outermost running `@agent` agent's own entry state message list, read live by reference at every stage ([sentinel.md](sentinel.md), "The agent's own messages").* The earlier ruling (`history` as the model's view, rebuilt at every step) was built and then reverted in #5811, and the `compaction` and `scorer` spans it added were removed on 2026-10-12. Marking the compaction summary call and `score()` from a solver in the transcript, for R10 and replay (R7), is deferred to that work and designed with its viewer and ACP TUI impact, since both handle span types specially; inspect_ai adds no spans for sentinel's sake. |
| 16 | `escalate` at the generate stages | **Decided 2026-10-10, built:** legal at both generate stages; an unhandled root escalate terminates through `run_sentinel` (sentinel #74). `threshold` gains `model_input` and `model_output` functions, and `handle_escalation` answers escalations at all four stages without making a stage watched on its own. inspect_ai's own fallback for an escalate that reaches it is #5812's. |
| 17 | Recording `step.input` at generate time | **Dropped 2026-10-12** ("drop 17 recording if we can use the transcript history"). At the generate stages the step carries its own call's request; at the tool stages R2 (inspect_ai #5831) finds the producing `ModelEvent` in the transcript history, which carries `input`, `tools` and `tool_choice`. |

## a. Which calls are checked {#a-which-calls}

Decided before this design: the generate stages check the agent's own model calls, and not the sentinel's own `Host.generate` calls, monitor or protocol model calls, scorers, or model-graded tools. The question is how to identify those calls robustly. Inspect had no marker for "an agent turn" (inspect_ai #5990 adds `ModelEvent.origin`, below): `react`, `basic_agent`, the `generate()` solver, `generate_loop`, the bridges and custom solvers all call `Model.generate` directly.

### Options

- **A. The active model, with exclusions.** Hook `Model.generate`; a call is the agent's when the model is the sample's active model (`self is active_model()`, the test `Model.generate` already uses to merge the task's generate config and update the message count) and the call is not inside a sentinel run, a scorer, a compaction or a tool's own body. One hook covers every loop, custom solvers included.
- **B. Agent loops opt in.** `react`, `generate()`, `generate_loop` and `bridge_generate` call a sentinel-aware generate. Exact about turns, but a custom agent or solver calling `model.generate` is never checked unless it adopts a new public API, and every loop must remember to.
- **C. Any generate in an agent context.** Hook `Model.generate` and check every call made in the solver phase outside a tool body, whatever the model. Covers sub-agents and role models, but also a solver's helper calls to a classifier or a critique model, which are not agent actions.

**Decided (decisions 1 and 2, revised 2026-10-12): C, fenced to the solvers.** A model call during the solvers is an agent's turn whatever its model, since a sub-agent's tool calls are already checked and its model calls should be too; the exclusions are a deny list of scopes (tool bodies, sentinel runs, compaction, `score()`, approvers, reviewers, MCP sampling). The earlier rules, A (active model only) and then agent spans on any model plus the active model elsewhere, are superseded; the second was replaced after a review found it checked an `on_sample_init` agent's helper.

**Decided (2026-10-13): no tool call inside a final scope is checked.** The final scopes are a sentinel run, compaction, `score()` from a solver, an approver, a reviewer and MCP sampling, where model calls are not checked either. Unlike model calls, tool calls are still checked outside the solvers. Before, every tool call was checked while a sentinel was active, so a monitor that ran a tool, as `Host.score` does through Scout's answer tool, was checked by the sentinel it belonged to, which ran the monitor again without end. Tool calls made inside a tool body are still checked, since a tool body's scope is not final.

**Revised (inspect_ai #5990): task scorers and `model_origin()` blocks are final too.** Tool calls are checked in agent, tool and setup regions. Nothing inside a final region is checked: a `model_origin()` environment or helper block, task scorers and `score()`, an approver, a reviewer, a sentinel run, a compaction summary and MCP sampling. Before #5990 a sentinel checked a task scorer's tool calls; now they are left alone, as `score()` from a solver's already were.

### The rule

Decided 2026-10-12 ("phase fence plus deny list"), after a review found that an `on_sample_init` agent's `as_tool()` helper was checked under the earlier scope rule. A generate is the agent's when all of these hold:

1. a sentinel is configured for the task and some function in it watches a generate stage;
2. the sample is running its solvers (the phase fence): inspect_ai sets it around `plan(state, generate)`, and nothing nested can change it, so sample init (including `on_sample_init` hooks) and scoring (the scoring phase and `inspect score`) are outside. A task's `setup` solvers are part of the plan and are inside;
3. the call is not in a denied scope, innermost first:
   - a tool's own body (`call_tool`, and the sandbox bridge service for a host tool) denies; an agent started inside the tool (`as_tool()`) opens an agent span, which allows again;
   - a sentinel run (`Host.generate` and a monitor calling `get_model()` itself), a compaction summary, `score()` called from a solver, an approval policy's approver, a review policy's reviewer and an MCP server's sampling request deny finally: nothing inside them, an agent span included, allows again.

Any model counts: inside the solvers, outside a denied scope, a call is the agent's whatever model it uses.

**Mechanism.** inspect_ai keeps two context variables, read by `Model.generate`: the fence (true only around the solvers), and the region a call is made in, `sentinel_scope(origin, final=)`, which defaults to `"agent"`. A tool body sets `"tool"`; an agent span sets `"agent"` where `span(type="agent")` opens, so no adapter can forget it; a sentinel run, a compaction summary, task scorers and `score()`, an approver, a reviewer, MCP sampling and a `model_origin()` block each set their own kind as final. The innermost region wins, except that nothing inside a final region changes it. An `"agent"` region outside the fence reads as `"setup"`.

**Origins (inspect_ai #5990).** Each `ModelEvent` records the region's kind as `origin`, whether or not a sentinel is configured: `"agent"`, `"environment"`, `"helper"`, `"tool"`, `"compaction"`, `"approval"`, `"review"`, `"sampling"`, `"sentinel"`, `"scorer"` or `"setup"`. Only a call recorded as `"agent"` is an agent turn: it gets the generate stages and counts toward `history`. The rule is the same live and when reading a log. `model_origin()` (in `inspect_ai.model`) is the public marker for scaffold authors. A scaffold wraps the model calls that are part of the solve but are not the evaluated agent's turns in `with model_origin("environment"):` (an auditor, a simulated user) or `with model_origin("helper"):` (the scaffold's own summarizer). Those calls get no generate stages and stay out of `history`, and the block is final, so an agent run inside it keeps its kind and its tool calls are not checked.

**Behaviour change (accepted 2026-10-12).** A custom solver's direct call to another model, outside a tool (a critique or classifier helper), is now checked. Before, outside an agent span only the active model's calls were.

### What is in and out

| Caller | Checked | Why |
|---|---|---|
| `generate()` solver, `generate_loop` | yes | inside the solvers |
| `react()` as the solver, `basic_agent()` | yes | inside the solvers |
| `handoff()`, `run()`, deepagent sub-agents, on any model | yes | each opens an agent span; its steps carry the sub-agent's conversation id |
| an agent whose model is a role (`get_model(role="untrusted")`) | yes | any model counts |
| `as_tool()` agent | yes | its agent span allows again inside the tool body |
| bridged agents (`agent_bridge()`, `sandbox_agent_bridge()`) | yes | the bridge's model calls run inside the solver's agent span; tested in process, and the sandbox host-tool exclusion is tested through the service's `call_tool` handler; the sandbox transport itself is untested |
| a custom solver calling `get_model().generate()` | yes | inside the solvers; includes helper calls such as `self_critique()`, unless marked with `model_origin()` |
| a custom solver's call to another model, outside a tool | yes | any model counts (behaviour change, above) |
| a model-graded tool calling `get_model()`, natively or as a sandbox bridge host tool | no | a tool body denies |
| scorers, `model_graded_qa()`, including `score()` called from a solver | no | task scorers, `score()` from a solver, rescoring a log and a control-channel score pass run in a final `"scorer"` region, so their tool calls are not checked either (inspect_ai #5990; before it, a task scorer's tool calls were) |
| a `model_origin("environment")` or `model_origin("helper")` block (an auditor, a simulated user, a scaffold's own summarizer), and an agent run inside it | no | the block is a final region, so its tool calls are not checked either |
| a task's `setup` solvers calling the active model | yes | they are part of the plan |
| `Host.generate`, monitor model calls, an agent a monitor runs | no | a sentinel run denies finally |
| compaction summaries | no | denied finally; decision 3 |
| a bridge filter that returns a `ModelOutput` itself | no | no `Model.generate` call; a gap to close with the bridge hook (workstream 5) |
| a tool an in-process bridged scaffold runs itself, calling inspect's model | yes | inspect never sees the tool start, so its calls look like the agent's |
| approval policies (an approver's model calls, or an agent it runs) | no | the approver runs denied finally |
| review policies (a reviewer's model calls, or an agent it runs) | no | the reviewer runs denied finally |
| MCP sampling (`get_model().generate()` from an MCP server callback) | no | the sampling request runs denied finally, whatever context the session runs in |

An agent run during sample init, such as from an `on_sample_init` hook, is not checked, however it is built: the fence is closed there.

## b. Where the hook sits {#b-where-the-hook-sits}

### Options

1. **Per attempt, inside `Model._generate`**, where `Hooks.on_before_model_generate` fires. Sees the exact request, but runs once per retry, and runs inside the connection slot: a monitor on the same model (the fallback when no `monitor` role is set) waits for a slot the agent holds, which deadlocks at `max_connections=1`, and `human()` would hold a slot while a person decides.
2. **Once per call, in `Model.generate`, outside the slot.** `BeforeGenerate` runs after the input is prepared and before the slot is acquired (so before the cache lookup); `AfterGenerate` runs after the slot is released, when the output is complete and about to be returned.
3. **In the agent loops** (option B above).

**Recommendation: 2.** The input preparation that was at the top of `_generate` (resolving tools and `tool_choice`, reasoning history, tool `model_input` handlers, media extraction, merging consecutive messages) moves into `Model._prepare_input`, called before the slot, so `step.input` is what is sent. Only `Hooks.on_before_model_generate`, which runs per attempt and may mutate the request, can still change it afterwards. The move is a behaviour change for every caller, sentinel or not: tool resolution (an MCP server's tool listing, for example) no longer holds a connection slot, so `max_connections` no longer bounds it, and `ModelEvent.timestamp` and the fallback `working_time` start after preparation rather than before it.

The step's `input`, `agent_messages`, `tools`, `config` and (after) `output` are the host's own objects, borrowed for the step, not copies: read-only by convention, including `agent_messages` at every stage. A monitor or protocol that mutates them changes the agent's state, the real request or the response the agent receives, so it must copy first to edit. The frozen dataclass prevents reassigning a field, not mutating what it holds.

### Interactions

- **Retries.** Provider retries are inside one call and are not checked again; they keep the generate's step id. A loop's own retries (`react`'s `retry_refusals`, the bridge's refusal and rejection retries) are new calls, and each is checked as a new generate with its own step id ([f](#f-events-and-the-viewer)).
- **Cache.** `BeforeGenerate` runs before the lookup and `AfterGenerate` after a hit: a cached output is still the agent's action in this sample. A monitor's own calls inherit the eval's `cache` setting and may hit the cache too.
- **Batch.** Nothing special: a slow `BeforeGenerate` delays submission to the batch.
- **Streaming.** `on_stream` deltas reach the caller before `AfterGenerate` decides. Streaming is display-only, so this does not matter for `proceed` and `terminate`. When `modify` or `reject` lands, the dispatcher must send a `StreamRetryEvent`-style boundary so a display discards the replaced deltas.
- **`fallback_models`.** Handled inside the provider call, so one call is checked once; `step.model` is the model requested, and `output.model` and `output.fallback` say which answered.
- **`cache_prompt`.** Unaffected by observation. A future `BeforeGenerate` modify that edits an early message invalidates the provider's cached prefix, which costs tokens but is not wrong.
- **Limits.** The token and turn limits are suspended inside a sentinel run, as at the tool stages. The message limit is checked before `BeforeGenerate`, so a generate the limit refuses is not checked. Time and working limits keep running, so a slow monitor counts against the sample's time. A turn or token limit raised inside the call means no `AfterGenerate`, since the output never reaches the agent.
- **Monitor failures.** As at the tool stages ([sentinel.md](sentinel.md), "Failure semantics"): a monitor that raises at a generate stage is recorded as failed, with an `error` `SentinelEvent` in the stage's span, and the generate goes ahead unless a protocol reading its observations without checking `failed` raises `MonitorFailedError` and so fails the sample.
- **`fail_on_refusal`.** The `ModelRefusalError` is raised before `AfterGenerate` runs, so a refusal configured to fail produces no after step (decision 6). A refusal returned as an output is checked like any output.
- **Failures.** A protocol that raises fails the sample with its own error; a monitor that raises follows the recorded-failure policy above. The dispatcher wraps a failure so that, inside an `as_tool()` agent, the tool call does not turn it into a tool error the model sees; the tool call unwraps it, as at the tool stages, and the sample's error handling unwraps it for a generate outside any tool.
- **Spans.** Each stage runs in its own `sentinel` span: the before span precedes the call's `ModelEvent`, the after span follows it, and monitor model calls sit inside their span. A stage that nothing in the configured tree watches opens no span (decision 7). Under a bridge with a `ModelEventSink`, the `ModelEvent` goes to the sink while the `SentinelEvent`s go to the transcript, so their order in the log depends on the sink; unverified.

### Dispatching only watched stages

Without decision 7, a sentinel of tool-stage monitors under `concurrent` or `observe_only` (both annotate `Step`) would open two empty spans per generate. The decorators therefore record, on each configured instance, the stages it or any instance passed to its factory watches. A composition that only forwards the step (`concurrent`, `sequential`, `observe_only`) contributes its children's stages and not its own `Step`; any other function contributes the stages it accepts, so a user's protocol annotated `Step` still runs everywhere. A forwarding composition whose children were not found among its arguments counts every stage it accepts, so the analysis can over-dispatch but never under-dispatch. The host asks `watched_stages(root)` before building a step. Applied to the tool stages too, it removes the empty after-call span a before-call-only sentinel opened before.

## c. Before-generate decisions {#c-before-generate-decisions}

- **`proceed`**: the request is sent unchanged.
- **`terminate`**: the request is not sent and the sample ends with an operator limit, as at the tool stages. No `ModelEvent` is recorded.
- **`escalate`**: passed to the next decider, such as `handle_escalation()` in a `sequential()`. One that reaches the root unhandled ends the sample: `run_sentinel` returns a `terminate` explained as an unhandled escalation (decision 16).
- **`modify`** (deferred to part 2, decision 9; a shape-check error in this landing). Proposed: the protocol replaces the request for this call only. What may change:
  - `input`, freely: redact a secret, annotate ("you have used 8 of 10 commands"), drop a turn. The change is not written back to the agent's history, so a persistent annotation is re-added at every generate.
  - `tools`, narrowed to a subset by name. Adding a tool the loop cannot execute produces a call that fails with "tool not found".
  - `tool_choice`, e.g. `"none"` to force a text answer.
  - not `model`: answering from another model is an `AfterGenerate` `modify` with that model's message.
  - not `config`: a protocol that needs other settings generates through `Host.generate` and substitutes at `AfterGenerate`.

  The replacement is applied to the prepared input, the list `step.input` showed, and the `ModelEvent` records it. The tool-stage policy argument for `AfterToolCall` (#5355: withholding a result makes the model's picture of the sandbox diverge from the sandbox) applies to a `modify` that removes or rewrites a tool result in `input`; the docs should say so rather than the shape check forbid it, since redacting a leaked secret from a result is a legitimate use.

- **`reject`** (deferred to part 2, decision 8; a shape-check error in this landing): the existing table has it legal, meaning "do not run this generate, and tell the agent why". There is no channel to tell the agent anything without producing an output, since the caller is waiting for a `ModelOutput`. The options:
  1. **Not legal** in v1; the use cases are `modify` (strip what should not be sent) and `terminate`. Recommended.
  2. **A sentinel-authored refusal**: return a `ModelOutput` with `stop_reason="content_filter"` and an assistant message carrying `message`, with no usage. Every loop already handles a refusal (`react` retries refusals and stops after three in a row; `fail_on_refusal` raises), and a proxy can return a refusal-shaped response. Its costs: the agent's history gains an assistant message the model did not write, which needs marking as sentinel-authored in its metadata, and a log reader cannot tell it from a provider refusal without that mark.
  3. **An exception** the loop must handle. No loop handles it, so it is `terminate` with a different name.

## d. After-generate decisions {#d-after-generate-decisions}

- **`proceed`**: the output is returned unchanged.
- **`terminate`**: the output is discarded and the sample ends with an operator limit; the `ModelEvent` stays in the log, and the assistant message never reaches the agent's messages, so its tool calls do not run.
- **`escalate`**: as before a generate: answered by a later decider, or, unhandled at the root, the sample ends (decision 16).
- **`reject`** ("discard this completion"; deferred to part 2, decision 10; a shape-check error in this landing). The options:
  1. **Replay**: append the rejected assistant message and a user message carrying `message` (or the default "not approved" text) to this request, regenerate, and check the new output; bounded at three consecutive rejections, after which the sample terminates. This is what `_bridge/_approval.py` does for a rejected tool call in a bridged agent, and what a proxy can do. The agent's loop receives only the accepted output; the rejected exchange is in the log but not in the agent's history, as in the bridge. Each regenerate is a new call with its own two stages and `ModelEvent`. Recommended.
  2. **Regenerate silently**: the same request again. The model learns nothing and may produce the same output; `resample` already covers drawing again on purpose.
  3. **A synthetic refusal**, as for `BeforeGenerate` option 2.
- **`modify`** (deferred to part 2, decision 11; a shape-check error in this landing). Proposed: `modified` is a `ChatMessageAssistant` replacing `choices[0].message`, per [sentinel.md](sentinel.md), "One vocabulary across stages". Constraints:
  - `usage`, `time` and `model` describe the call that happened and are kept; the replacement's own cost is recorded by the call that produced it (a `Host.generate` under its role).
  - Only `choices[0]` is substitutable; the other choices are dropped, so no call the monitors did not see can run.
  - `stop_reason` is recomputed: `tool_calls` when the replacement has tool calls, `stop` otherwise.
  - The replacement keeps its own message id; the step id stays the original's, and `SentinelEvent.modified` carries the replacement, so both are in the log.
  - Its tool calls run through `BeforeToolCall` like any other, so a protocol that substituted a call sees it again there; tool call ids must be unique within the message.

`resample` is an `AfterGenerate` protocol over these. It draws with `model=step.model` in a loop, scores each candidate through `run_monitors(monitors, context, replace(step, output=candidate))`, and returns `Decision.modify(step, message=best.message)` or `escalate`. Its draws run inside the sentinel run, so they are not themselves checked (rule 4), are not charged to the agent's token limit, and are recorded under the agent model's usage until the `protocol` usage bucket exists (sentinel.md open question 8). It needs `tool_choice` on `Host.generate` (decision 14).

## e. What the step carries {#e-what-the-step-carries}

| Field | `BeforeGenerate` | `AfterGenerate` |
|---|---|---|
| `model` | the requested model, `provider/name` | same |
| `conversation` | the current agent span id, or the sample uuid outside any agent span | same |
| `input` | the prepared request: the caller's messages with the configured system message, resolved reasoning history, tool `model_input` handlers, extracted media and merged consecutive messages; what the `ModelEvent` records | the same list |
| `agent_messages` | the outermost running `@agent` agent function's own message list, read live by reference; `None` without one, for bridges, in replay and under a proxy | the same live list; the agent may not have added `output` yet |
| `tools` | `ToolInfo`s sent, after `tool_choice` filtering | same |
| `tool_choice` | as sent (`"none"` when there are no tools) | same |
| `config` | the resolved `GenerateConfig`: the task's, the model's and the call's, with `max_tokens` filled | same |
| `output` | — | the `ModelOutput` returned; `output.message` is the subject |

`AfterGenerate` repeats the request so that `resample` can regenerate with exactly what the agent was given.

**`agent_messages`.** The `messages` of the `AgentState` passed to the outermost running `@agent` agent function, read live by reference at every stage (inspect_ai #5986, 2026-10-21; [sentinel.md](sentinel.md), "The agent's own messages"). A scaffold may curate this list to keep hidden side-task instructions away from monitors; it may also contain retries. Hidden input stays out only when that scaffold owns the conversation as its outermost decorated agent function. A compaction does not drop turns from the list and adds its summary message (overflow recovery also drops the reply that overflowed); `input` carries the compacted request. With `truncation` set, `react`'s overflow filter trims this list when forced compaction is not available or does not help (no compaction, a model agent as `model=`, a failed forced compaction, or a second overflow right after one), while `history` keeps the turns. It is `None` without a running `@agent` agent function, for a bridge, in replay, and under a proxy. The actual model traffic is in `input` and `history` (AISI R10; [sentinel.md](sentinel.md), "The history as sent"). inspect_ai tells the agent's turns from a compaction summary call or `score()` from a solver by the same rule as these stages, and adds no spans for sentinel's sake. Since inspect_ai #5990 each `ModelEvent` records that classification as `origin`, which replay (R7) reads to rebuild `history` from a log ([pr-series.md](pr-series.md), Deferred, "Replaying the history as sent").

**`step.input` at the tool stages.** R2 (inspect_ai #5831) finds the `ModelEvent` that produced the call in the transcript history, resident or evicted, and `step.input` is `None` when none exists. Decision 17's recording is not needed.

**Compaction.** A summary compaction's text reaches the next `BeforeGenerate` in `input`, so a monitor sees what the agent's model wrote about its own trajectory. `compaction_summary(step)` stays deferred on inspect_ai documenting the summary message metadata key.

## f. Events and the viewer {#f-events-and-the-viewer}

`SentinelEvent.stage` already declares `model_input` and `model_output` next to `tool_call` and `tool_result`; the dispatcher emits them, so no event schema or generated type changes. The stage names are the ones `human(stages=)` takes; it supports both.

`step_id` follows sentinel-development.md, "Step ids" (inspect_ai #5990):

- `model_input` and `model_output`: the generate's id, the `uuid` of its first `ModelEvent`. inspect_ai mints it before `BeforeGenerate` runs, so both stages of one generate carry the same id and join on it. A provider retry inside the call keeps the generate's id: the first attempt's `ModelEvent` has it, even when that attempt errored, and each retried attempt's event gets a uuid of its own. Each new `Model.generate()` call, such as a loop's retried refusal, is a new generate with its own id, and so is a cache hit.
- `tool_call` and `tool_result`: the tool call id, unchanged.

Linking to the `ModelEvent` needs no new field: a generate stage's `step_id` is the `uuid` of the call's first `ModelEvent`. This replaces the message ids used before #5990 (the last input message's id, with a `:N` ordinal on a repeat, at `model_input`; `output.message.id` at `model_output`). With them went two open problems: two cache hits on one request no longer share a `model_output` step id, and an output with no choices no longer has an empty one. In Inspect View the before span sits directly above the `ModelEvent` and the after span directly below it, so a viewer that renders the sentinel span inline needs no lookup; one that wants a "checked by" badge on the `ModelEvent` matches its `uuid`. Whether ts-mono renders the two new stage values well is unverified. A monitor's own model calls sit in its stage's sentinel span, listed together as at the tool stages; tying each to its instance stays deferred (pr-series.md, "Monitor model calls tied to their row").

Later contract changes, none in this landing: `SentinelEvent.modified` gains `ChatMessageAssistant` for an `AfterGenerate` modify, and a `BeforeGenerate` modify leaves `modified` unset with the validator relaxed for that stage, since the `ModelEvent` records the request sent (decision 13). If decision 8 adopts the synthetic refusal, the assistant message it produces needs a documented metadata mark.

## Scope of this landing

Built:

- `BeforeGenerate` and `AfterGenerate` in `inspect_sentinel`, exported, in `Step`, `Monitor` and `Protocol`; a function annotated with either is registered for that stage.
- The shape check allows `proceed`, `escalate` and `terminate` at the generate stages and raises "only 'proceed', 'escalate' and 'terminate' are supported at the generate stages" for `reject` and `modify`.
- `watched_stages()` in the integration contract, the forwarding mark on the three compositions, and the answers-escalations mark on `handle_escalation`.
- `threshold` and `handle_escalation` at the generate stages.
- In inspect_ai: `Model._prepare_input`, the two hooks in `Model.generate`, decision 1's agent-loop rule, the per-stage events and step ids, and the skip for unwatched stages.

Built since: `human(stages=["model_output"])` (AISI R1), offering `approve` and `terminate`; inspect_ai shows the person the whole turn as one view. Also built since: `human(stages=["model_input"])`, offering `approve` and `terminate`; inspect_ai shows the person the request as one view, with the tools it offers and the messages the model has not been sent before. Not built: `reject` and `modify` at the generate stages and `Decision.modify` (part 2, decisions 8–13), and tool-stage sentinels for bridged agents (inspect_ai #5765), and replay of `history`, the as-sent view of an agent instance's run (AISI R10, built live).
