# Implementation plan: the sidecar

How [sidecar.md](sidecar.md) becomes code. Each pull request can be reviewed on its own, has its own tests, and leaves `main` importable and passing. Proposed 2026-10-14; Envoy, incremental streaming and human review added 2026-10-17; the work after PR 7 reordered and renumbered, with streaming split into proxy and Inspect client phases, on 2026-10-19; TLS on the sidecar's listeners added as PR 16, before Middleman, on 2026-10-20.

Earlier prototype code exists for much of this. It is reused where it fits the design in `sidecar.md`, and rewritten where it does not.

## Repositories

| Repository | Owner | What lands there |
| --- | --- | --- |
| `inspect_sentinel` | Meridian | the core package, the sidecar (`inspect_sentinel.sidecar`) with its Envoy shim, and the proxy hooks, each a package of its own under `src/proxies` |
| `inspect_ai` | Meridian | Inspect's sidecar client, and the asks in `sidecar.md`, "Asked of inspect_ai" |
| `hawk` | METR | Middleman's passthrough hook, and routing `/sentinel/*` at the load balancer in front of Middleman |
| LiteLLM | the LiteLLM project | nothing. The guardrail depends only on LiteLLM's behaviour, and is tested against each LiteLLM version it supports |

## Order

After steps 1–7, the main delivery path is **8 model-facing history → 9 Envoy → 10 proxy streaming → 11 Inspect sidecar client → 12 Inspect streaming → 13 human review**. This builds and validates Envoy and incremental streaming in `inspect_sentinel` before implementing the client in `inspect_ai`. Discuss the Inspect client contract with Meridian while the proxy work proceeds. Smaller upstream asks (14) can proceed independently. Trusted run identity (15) and TLS on the sidecar's listeners (16) come before Middleman (17), whose implementation waits on Hawk/METR accepting the gateway change; identity work can proceed through Envoy and LiteLLM while that approval is pending.

```text
inspect_sentinel

1 reading traffic
        |
2 handler
        |
3 endpoint for proxies
        |
        +--------------------+
        |                    |
4 LiteLLM guardrail    5 Inspect clients in the sidecar
        |                    |
        |              +-----+------------+
        |              |                  |
        |        6 endpoint for evals   7 per-call reject and modify
        |              |                  |
        |              +--------+---------+
        |                       |
        |                       v
        |                  8 model-facing history
        |                       |
        |                       v
        |                  9 Envoy shim
        |                       |
        +-----------------------+
                                v
                   10 proxy incremental streaming
                                |
                                v
                   11 Inspect sidecar client                 inspect_ai
                                |
                                v
                   12 Inspect incremental streaming  inspect_ai
                                |
                                v
                   13 human review                  both repositories

14 smaller asks: inspect_ai, independently, after accepting each issue
15 trusted run identity: after 6 and 11, before Middleman; no Hawk dependency
16 TLS on the sidecar's listeners: after 4, 6 and 9; before Middleman
17 Middleman: inspect_sentinel and hawk, after 3 and 5; scheduled after 15 and 16
   Implementation also needs Hawk/METR accepting the passthrough hook
   Middleman streaming follows 10 and 17 and METR accepting chunk delivery
shared state: inspect_sentinel, before more than one sidecar worker or restart continuity
```

PRs 1 to 3 come first, one after another, and land as one pull request with a commit for each, because the endpoint is the first point at which anything runs. After 3, the sidecar's support for Inspect clients (5) and the guardrail (4) land together as one pull request, with a commit for each and one more for the guardrail passing the `sentinel` field on: the guardrail's tests run a real sidecar, and the field can only be tested through LiteLLM once the sidecar writes it. The endpoint for evals (6) and per-call rejection and `modify` (7) land together after that as one pull request, with commits for each, so that one review covers both.

PR 8 comes next: the sidecar fills `step.history` from the traffic it keeps, so that PR 9's model-history conformance checks run through Envoy against it. It needs nothing beyond PR 7. PR 9 then builds the Envoy shim with whole-response holding. PR 10 adds incremental streaming to the shared handler, Envoy and LiteLLM, tested with ordinary provider SDK clients and a small client of the eval endpoint. Neither needs Inspect's sidecar client or changes to `inspect_ai` source. Whole-response holding remains available for policies that need the full turn.

PR 11's technical prerequisites remain 5, 6 and 7; placing it after 10 is a delivery choice, so the proxy behavior and recovery contract are exercised before a client depends on them. From 7 it needs per-call rejection and `modify`, and also that a rejection Inspect puts back into its conversation is not judged as a tool result: before 7, the sidecar would judge each such rejection as a call that ran, and could end the run for a call that never ran. PR 11 establishes the Inspect client with whole-response holding. PR 12 then connects that client to incremental replies and interrupted streams, including transcript and sample-outcome conformance. PR 13 follows both streaming phases and adds human review through the sidecar's host, Inspect's approval UI and external review consumers.

[Shared and durable state](#follow-up-shared-and-durable-state) is delivered before the sidecar runs as more than one worker or promises that a run outlasts a restart; until then a deployment is one process. PR 14 needs nothing and can start at any time, with each upstream issue accepted first. PR 15 implements trusted run identity after registration and the Inspect client, using Envoy and LiteLLM first, without waiting for a Hawk change. PR 16 adds TLS, and client certificates, on every port the sidecar listens on, for Envoy and LiteLLM first. PR 17 is scheduled after PRs 15 and 16 and waits on Hawk/METR accepting Middleman's passthrough hook. Extending Middleman to hand over chunks also needs PR 10 and METR accepting that change; it does not block Envoy, LiteLLM, identity or the Inspect client work.

## PR 1: reading provider traffic

Repository: `inspect_sentinel`. Depends on: nothing.

- Read Anthropic Messages, OpenAI Chat Completions and OpenAI Responses requests and replies, streamed and not streamed, into Inspect's types, with inspect_ai's converters and each SDK's accumulator (`sidecar.md` section 2).
- The `format` parameter, and working the API out from the provider and the body when it is absent.
- The rules for an unreadable request or reply: unknown content, a Beta type, more than one choice, a conversation stored at the provider, a background request (sections 2 and 6).
- Tests: a replay corpus of recorded replies, with the cases that must be unreadable beside them.

## PR 2: the handler

Repository: `inspect_sentinel`. Depends on: PR 1.

- Runs, keyed by a name the proxy passes on or by how a conversation opens, and conversations within a run (section 7).
- The four steps, found in the traffic: `AfterToolCall` and `BeforeGenerate` on a request, `AfterGenerate` and `BeforeToolCall` on a reply, each tool result judged once (section 3).
- Carrying out decisions on the request or reply held: `proceed`, refusing a whole turn, a `terminate` that keeps the run ended, rejected turns put back on later requests, and three rejections in a row ending the run (section 4).
- No `agent_messages` at any step, as the core defines it behind a proxy (section 8).
- The state behind one interface, with an in-memory implementation (section 10).
- Tests: the handler driven directly, with no proxy and no HTTP.

## PR 3: the endpoint for proxies

Repository: `inspect_sentinel`. Depends on: PR 2.

- JSON-RPC 2.0 at `POST /proxy/v1`: `judge_request` and `judge_reply`, answered with `pass`, `replace` or `refuse` (section 1).
- The server command: one port by default, `GET /health`, `--max-body-bytes`, `--time-limit` and `--reject-status`.
- The schema file `proxy_endpoint.schema.json`, generated from the endpoint's models, with a test that fails when it is stale.
- Tests: the endpoint through its methods, held to its schema file.

## PR 4: the LiteLLM guardrail

Repository: `inspect_sentinel`, package `inspect_sentinel_litellm` under `src/proxies/litellm`. Depends on: PR 3, and PR 5 for the `sentinel` field. Lands with PR 5, in one pull request.

- `LiteLLMGuardrail`: `judge_request` and `judge_reply` on LiteLLM's three routes, holding a streamed reply whole, and LiteLLM's own setting for a sidecar that gives no answer (section 1).
- Passing `x-inspect-context` and `x-irid` to the sidecar without reading them, and removing both before the provider (section 7).
- Writing a refusal and a replaced reply in each of LiteLLM's forms, including the three cases where the guardrail writes the event itself so that the `sentinel` field arrives (section 1).
- Serving `/sentinel/*` through its own route in LiteLLM's app, behind LiteLLM's key check (section 1).
- The repair of a Claude `reasoning` item written through `/responses` (section 1).
- A README with its settings, and an example stack: LiteLLM, the sidecar, and an Inspect task.
- Tests: with LiteLLM's own classes; and live, with a real LiteLLM, a running sidecar and a stand-in provider, on every route, streamed and not, passed, edited and refused, for each supported LiteLLM version.

## PR 5: Inspect clients in the sidecar

Repository: `inspect_sentinel`, the sidecar and the core package. Depends on: PR 3.

- Reading `x-inspect-context`, and taking the run from its `sample_uuid` (section 7).
- Building the `sentinel` field from every step judged during an exchange: the stage, the call, the original call, the decision and the records; with what the exchange came to, and the turn withheld from the client (section 9).
- Answering an Inspect client's reply with `replace`, with the field added, and putting the field on `refuse` (sections 1 and 9). Other clients get the provider's bytes untouched.
- Trimming the field under `--no-eval-results` (section 9).
- In the core package: `Context.proxy`, with `inspect`, what `x-inspect-context` says ("Proposed for the core package").
- Tests: the field's contents for each outcome in section 9's table, for each provider format, streamed and not.

## PR 6: the endpoint for evals

Repository: `inspect_sentinel`. Depends on: PR 5.

- `GET /sentinel/info`, and JSON-RPC 2.0 at `POST /sentinel/v1`: `register_task`, `request_result`, `sample_result` and `end_sample` (section 9).
- Task registration: on unless `--no-runtime-task-registration`, replacing the last unless the task is in the tasks folder, where it is answered as already registered, with `fixed` true, and nothing is stored, `expiry` capped by `--task-expiry`, a run taking its sample's facts at its first request, and `context.eval` from them.
- The tasks folder, of tasks files each with a `tasks` list, named by `tasks_folder` or `--tasks-folder` (section 9). Producing the files is asked of the workstream that builds Inspect's sidecar client.
- `--run-expiry`, dropping registrations Inspect made once their `expiry` passes, and `request_result` answering `dropped` for a while after a request's records go.
- The configuration file: the `sentinel` stanza, a `server` stanza of the sidecar's settings, and `tasks_folder` (section 10).
- Tool views from a task's registration: inspect_ai's own viewers, the default view, and a task's own viewers where its package is installed (section 9).
- The second port and its own token, as an option.
- The schema file `eval_endpoint.schema.json`.
- Tests: each method, held to its schema file; a registration that changes, an ended run, and a run that expires.

## PR 7: per-call reject and modify

Repository: `inspect_sentinel`. Depends on: PRs 2 and 5.

- Taking a rejected call out of a reply, and changing a modified call, block by block in the provider's own document, for each provider format, streamed and not (section 4).
- Putting the call back, or the model's original call, on every later request of the conversation, and adding nothing when the request already holds it.
- Refusing the whole turn when every call in it is rejected, and ending the run after three such turns in a row. A turn in which some calls ran starts the count again (section 4).
- Tests: edited replies read back by each SDK; the put-back request equal to what Inspect sends in process.

## PR 8: model-facing history

Repository: `inspect_sentinel`. Depends on: PR 7. This is the next implementation step after PRs 6 and 7, before PR 9's transcript and model-history conformance checks run through Envoy. The core's as-sent `step.history` landed in sentinel #98, so this waits on no core change.

- Fill `step.history` from the traffic seen for each conversation, keeping the messages from before a compaction, and following Inspect's as-sent history semantics. Keep different runs, sample attempts and conversations separate.
- Keep `agent_messages` as `None`: the proxy cannot know the scaffold's own list. Do not invent history the sidecar never saw; use `None` where it cannot establish it.
- Tests: compare the history given to a monitor in process and through the sidecar before and after compaction, on retries within an attempt, and on a fresh sample attempt. A new attempt must not inherit the discarded attempt's history.

## PR 9: the Envoy shim

Repository: `inspect_sentinel`. Depends on: PRs 3, 6, 7 and 8. Does not depend on PR 11, Middleman or WASM.

- A Python gRPC `ext_proc` shim inside the sidecar, with dependencies in an optional extra. It translates Envoy's exchanges into the shared handler's `Request` and `Reply`, and applies `Pass`, `Replace` and `Refuse`; provider reading, sentinel execution and run state stay in the shared implementation.
- Anthropic Messages, OpenAI Chat Completions and OpenAI Responses, including request amendments, edited replies, and refusals. Response bodies use `FULL_DUPLEX_STREAMED` from the first version, but the shim initially holds each reply whole (section 5). Incremental release follows in PR 10.
- Envoy 1.39.1 or later. `FULL_DUPLEX_STREAMED` was added in 1.34, and releases after it fixed `ext_proc` bugs that truncated or duplicated bodies, which would corrupt a reply the shim holds and re-emits. The example and the tests pin `envoyproxy/envoy:v1.39.4`, the latest 1.39 release when the PR was built, whose `ext_proc` protos are those of 1.39.1, where the measurements in `sidecar.md` were taken. Release 1.40.0, a week old then, was not taken.
- Generated Envoy configuration and a runnable example: provider routes, request and response processing, message and body limits, timeouts long enough for sentinel decisions, and requesting uncompressed provider replies. Sidecar failures refuse traffic by default, and an unexpected content encoding is refused when it cannot be read and rewritten correctly.
- Inspect's request context reaches the handler, the `sentinel` field reaches the client on replies and refusals, and `/sentinel/*` reaches the endpoint for evals under the deployment's access controls. Exercise these with a small test client of the existing contract, without implementing Inspect's sidecar client or changing `inspect_ai` source.
- Tests with a real Envoy, the sidecar and a controlled provider, on all three formats, streamed and not, passed, edited and refused. Cover bodies above Envoy's ordinary buffer and gRPC defaults, split events, response trailers, provider errors, sidecar failures, hold limits and client disconnects. Run the model-history conformance cases through Envoy as well.

Done when the generated configuration runs the example and the same sentinel decisions, records and model-facing conversation survive an exchange through Envoy as through the existing handler.

## PR 10: proxy incremental streaming

Repository: `inspect_sentinel`. Depends on: PRs 4 and 9. Does not depend on PR 11 or require changes to `inspect_ai` source. This phase builds and validates the streaming machinery with ordinary provider SDK clients before Inspect implements a remote client.

- An incremental handler interface and a corresponding transport for proxy hooks, alongside the existing whole-exchange methods. Envoy's shim feeds it chunks directly; LiteLLM's guardrail sends chunks as they arrive. Provider decoding continues to use inspect_ai and the provider SDKs, and sentinels still judge complete steps.
- Settle and document what a sentinel sees under early release before implementing the incremental path. A monitor must not have to know that a reply was streamed: every step keeps the whole-turn meaning it has in Inspect's own process. Retain the existing whole-response behavior by default, and make early release an explicit setting. Settled in [`sidecar.md`, "What a sentinel sees under early release"](sidecar.md#what-a-sentinel-sees-under-early-release): `release: early` in the `server` stanza, and a turn's calls judged once it has ended, with the whole turn, so no sentinel ever sees part of one and none needs whole holding to be judged correctly. Judging a call before its turn ends is not built; a workload that needs it would be a monitor's own opt-in.
- Stream text and reasoning as they arrive where early release is allowed. Hold every byte of a tool call, and everything after it, until the turn has been judged, then release, modify or remove each call. Preserve event order, and establish for each of the three provider formats which of its events carry text and reasoning, rather than assuming their stream lifecycles are interchangeable. Settled in [`sidecar.md`, "Releasing a reply as it arrives"](sidecar.md#releasing-a-reply-as-it-arrives): the text and reasoning before the first call go out as they arrive, and everything from the first call on is held until the turn is judged, with the error event each SDK raises on.
- Preserve PR 7's rejection and modification semantics, the original calls restored in later model requests, and termination remembered for the run. Define what happens when every call is rejected, or the run is ended, after preceding text has gone out. Released content cannot be recalled. After the response status is committed, refusals and failures use an error event the client SDK recognises, with no successful stream ending in its place.
- Keep the `sentinel` field and request and run results correct on incremental replies and failures, including when results are trimmed. A small test client exercises the recovery contract: when a cut stream cannot carry the field, `request_result` must still expose the available decisions and original turns, or explicitly report a pending judgment, a failure to judge, dropped records or an unknown request. This validates the protocol; writing Inspect's transcript and acting on the recovered outcome belong to PR 12. Checked by `tests/sidecar/test_stream_recovery.py`: a reply released early, its field never passed on, is recovered as decided with its outcome, the turn and each decision, as pending while its turn is judged after its client went and then decided, as cut off when its client went before the turn ended, as not judged when the sentinel failed, and as dropped or unknown.
- Keep connections alive while tool calls wait, with provider-compatible keepalives. Bound queues and held bytes, apply backpressure and hold deadlines, and define cancellation and record retention when the client disconnects or a proxy stops waiting. Keepalives settled in [`sidecar.md`, "Releasing a reply as it arrives"](sidecar.md#releasing-a-reply-as-it-arrives): an SSE comment after `keepalive` seconds of silence, under `release: early` only.
- Deliver and test the incremental path through both Envoy and LiteLLM. A proxy or provider format without a validated incremental path continues to use whole-response holding. Middleman's streaming extension is a separate follow-up in `inspect_sentinel` and `hawk`, after PRs 10 and 17 and METR accepting chunk delivery. It uses the same interface and does not block this phase or PR 12; until it lands, Middleman holds replies whole.

Acceptance tests use a provider and a sentinel whose progress is controlled: preceding text must reach the client before the provider finishes or the sentinel answers, and no part of a tool call may reach it before approval. Cover multiple calls, mixed decisions, modification, rejection of every call, termination after earlier release, split and combined chunks, malformed or interrupted streams, keepalives, backpressure and disconnects. Each provider SDK must read successful edited streams and raise on refused ones. Replies released early must retain the model-facing history of the whole-response path, including the original calls and rejection results on later requests, and give each call the same step it gets when the reply is held whole; `release: whole` must demonstrably keep the reply held. Check field delivery and endpoint recovery with the test client. Actual Inspect transcript conformance is a PR 12 acceptance criterion.

## PR 11: Inspect client for the sidecar

Repository: `inspect_ai`. Depends on: PRs 5, 6 and 7 (above). Design: [inspect-client.md](inspect-client.md).

Scheduled after PR 10. The initial client uses whole-response holding; support for incremental replies follows in PR 12. While PRs 9 and 10 are built, agree the client contract with Meridian: activation and discovery failure, registration requirements, model-cache behavior, request and message identity, retries, missing results, scoring and cleanup. Review the trusted-host scope alongside #117 before implementing the client. Keep ordinary evals and in-process sentinels unchanged, and split the implementation into separately reviewable changes.

Keep the integration transparent to eval authors and preserve the existing eval and task interfaces. "Inspect client for the sidecar" describes its responsibility, without prescribing a public type or a choice users must make between local and remote execution. Its internal names, structure and any necessary deployment configuration are for Meridian to agree.

- The probe, `register_task` and the client's one setting, `auto`, `required` or `off`, at task init (section 9 and [inspect-client.md](inspect-client.md)). An answer with `fixed` true, for a task in the tasks folder, is a success.
- `complete_registration` before each sample's first epoch, with the sample's input and metadata. The sidecar's side, with its `strictness` setting, is built ahead of the client.
- Not in PR 11: the exemption for model calls that aren't the agent's ([inspect-client.md, K](inspect-client.md#k-model-calls-that-arent-the-agents)). It follows as its own change, after the client, once what Inspect signs and how the key reaches both sides are settled. It is off by default, so nothing in PR 11 depends on it.
- Any subset of a registered task's samples is valid. An eval that runs only some of the samples in the tasks folder, chosen with `--sample-id`, `--limit` or any other selection, runs them as usual.
- `x-inspect-context` and `x-irid` on each model request, only after the probe found a sentinel.
- Reading the `sentinel` field in each provider: on a reply, in the last event of a stream, and on an error, including where LiteLLM nests it (section 1).
- Acting on the field as Inspect does in process: a call taken out goes back into Inspect's conversation with an approval error, a changed call keeps the model's original, a `terminate` ends the sample, and every step is written into the transcript.
- `request_result` when the field is missing. A request the sidecar says it couldn't judge is recorded as not judged. One whose judgment is lost, unknown or can't be looked up is recorded as unavailable, and never as allowed.
- `sample_result` before scoring each sample, and `end_sample` after.
- Prevent Inspect's completion-response cache from bypassing the sidecar while the client is active. Initially disable that cache, or arrange fresh sidecar judging of a cached completion before the agent can use it. A cached `sentinel` field is not a decision for the current sample. This is separate from provider prompt caching, which still sends the model request through the proxy.
- Tests: in inspect_ai's own suite, with a stand-in sidecar; and the conformance suite in `sidecar.md` section 11, comparing Inspect's transcript in process and through a sidecar with whole-response holding. Include a pre-populated completion cache: a cached tool call cannot execute without the sidecar judging it for the current sample, and a decision cached by an earlier eval cannot authorize it. Incremental transcript conformance belongs to PR 12.

## PR 12: Inspect incremental streaming

Repository: `inspect_ai`, with coordinated changes to `inspect_sentinel` only if the agreed recovery contract needs them. Depends on: PRs 10 and 11. Extends the client built in PR 11 using the streaming machinery validated in PR 10; it does not introduce another provider decoder or proxy handler.

- Connect Inspect's sidecar client to the incremental path, preserving PR 10's explicit `release` setting and its judging of whole turns.
- Read the `sentinel` field on successful stream endings and refusals. Recover decisions through `request_result` when a stream is interrupted, ends with an error, or otherwise cannot carry the field, including trimmed results and explicit not-judged, dropped and unknown states. Define the recovery wait and retry behavior with Meridian; never infer a clean exchange from missing results.
- Apply recovered rejection, modification and termination outcomes without executing a rejected call, executing an allowed call twice, losing the model's original call, or recording a step twice. Preserve call and record identity across provider SDK retries and Inspect's own retries.
- Keep Inspect's model-facing conversation, transcript and sample outcome correct when some text or calls were released before a later refusal or termination. A later decision cannot undo a call already released. Coordinate stream errors, sample cancellation, state retrieval before scoring and `end_sample` after scoring with the lifecycle established in PR 11.
- Test in inspect_ai's own suite with a controlled sidecar, then run the real chain through Envoy and LiteLLM. Cover successful endings, interrupted streams, refusal after text was released, later termination, missing fields, result recovery, retries and disconnects. Compare decisions, records, original calls and model-facing history with the corresponding whole-response cases, accounting explicitly for the different context of an early decision. Confirm that local sentinels and evals without a remote sentinel retain their existing behavior.

Done when an Inspect eval can use the validated incremental path, recover an outcome when no terminal field arrives, and retain the expected transcript and sample outcome without duplicated decisions or tool execution. These are client guarantees that PR 10's SDK and protocol tests do not establish.

## PR 13: human review

Repositories: `inspect_sentinel` and `inspect_ai`. Depends on: PRs 10 and 12, including their Envoy, LiteLLM and Inspect client prerequisites. This is the sidecar part of [remote human surfaces](workstreams.md#13-remote-human-surfaces).

- Settle the [human-review questions](sidecar.md#human-review-questions-for-pr-13) before implementing the queue schema and surface integration: reviewer identity, context and automated assessments, call grouping and decision scope, pending age and sorting, approval validity, ownership and lifetime, reason disclosure, and audit completeness and durability.
- A sidecar implementation of `Host.ask_human` that publishes a question to a review queue and waits for the person's `HumanAnswer`, while the shared handler holds the step. Each question names its run, conversation, exchange and step, and carries the permitted choices, tool view, relevant context and escalations. Start with the two tool stages; generate-stage review still needs a supported rendering in Inspect's approval UI.
- Define the queue's discovery and answer protocol, with a schema, access controls and timestamps for showing pending age. Specify how several questions are listed, grouped and sorted, and how scores, evidence and commentary from preceding monitors and protocols are shown. External review tools consume that interface, including for agents running without an Inspect eval. Reuse existing review surfaces rather than building a new review application as part of the sidecar.
- Inspect's sidecar client discovers pending questions while its model exchange is waiting, shows them through Inspect's existing approval UI, and delivers the person's answer. An answer requires the reviewer's own credential; the agent's or eval's credential cannot approve its calls. The sidecar enforces which runs and questions a reviewer may see and answer.
- Validate an answer against the choices offered for that question and the step's legal actions. Present `terminate` wherever it is offered, distinct from rejecting one call or an explicit operation on a group. Before a call, the default choices are approve, reject and terminate; rejecting one call does not reject its siblings. Resolve each question once: retries of the same answer are idempotent, and conflicting, expired or cancelled answers cannot release a call or revive an ended run. The answer returns to the existing sentinel runner and compositions as an ordinary `HumanAnswer`.
- Keepalives cover only some waits. PR 10's keepalives work while a `BeforeToolCall` question holds a streaming reply. They cannot cover an `AfterToolCall` question, which holds the request before the provider is called, because neither Envoy nor LiteLLM can send the client anything before the request goes on. Nor can they cover a non-streaming reply, which has no stream to put them in. So a question outlives the exchange that raised it. When a review outlasts a bounded wait, the client is refused with a retryable error, and the question stays pending in the run's state. When the same request comes back, the sidecar recognises it and either waits again or applies the answer. A held reply is kept, so a retry is answered from it and does not call the provider again for a different reply. Specify the bounded wait, how a retried request is recognised, and how this interacts with the SDKs' own retries and Inspect's.
- Use PR 10's ordered release and PR 12's client recovery while a person is deciding. Specify question expiry, unanswered questions, run termination, client disconnects and sidecar restart, including how pending questions are cancelled or recovered. No missing answer implicitly approves a call; where nobody can be asked, retain the host's cautious fallback.
- Record the question, the reviewer's identity, the answer and any reason, with the final sentinel decision, so Inspect's transcript and the sidecar's request and run results retain what happened. A reviewer-only explanation stays separate from a rejection's agent-facing message, as in the existing contract.
- A runnable example and integration tests through Envoy and LiteLLM, with an external review consumer for an agent without an eval and Inspect's approval surface for an eval. Keep the queue and host independent of the proxy; the Middleman integration uses the same review path when available.

Done when a tool call stays held while its review question is visible, approval releases it once if the final sentinel decision permits it, rejection or termination is enforced with the expected model history and records, and unrelated streams can still progress. Acceptance tests cover multiple waiting calls, per-call answers within a group, terminate as a distinct option, pending age and queue ordering, assessment provenance, retries, conflicting and stale answers, reviewer access, unanswered questions, disconnects, restart and a run ending during review. A review must outlast an idle timeout between the client and the proxy at both tool stages, streamed and not: an `AfterToolCall` question on a held request, and a `BeforeToolCall` question on a held reply, where a retry must receive the reply that was held. Test the audit guarantees chosen in the design questions, including recording failures. An eval credential must demonstrably be unable to answer a review question.

## PR 14: smaller asks of inspect_ai

Repository: `inspect_ai`. Depends on: nothing. Each can be its own pull request, and each needs an accepted issue in inspect_ai first.

- Public OpenAI converters that read a dict as leniently as the SDK does.
- An Anthropic converter that says so when it meets a block it does not recognise, or a Beta type.
- Chat Completions converters that read a custom tool call, as the Responses converters do.
- An Anthropic converter that reads every block of a document whose source is a list of blocks.
- A public function that writes a message in Anthropic's format.
- Public functions that read the tools, the tool choice and the generation settings from a request, in each provider's format.
- A public function that describes a tool as the model is offered it, as a `ToolInfo`.
- A public function for each provider that turns stream events into a `ModelOutput`.
- A public way to obtain the running sample's uuid and attempt id before its first model call.
- A public way to make the default view of a tool call.

## PR 15: trusted run identity and termination scopes

Repositories: `inspect_sentinel`, `inspect_ai`, and the deployment's proxy or gateway where needed. Depends on: PRs 6 and 11. Review the identity and lifecycle contract with Meridian before PR 11, so the initial client's trusted-host scope and interfaces are explicit. Implement the stronger enforcement once registration and the Inspect client are available, before the Middleman work in PR 17. Design and test it through Envoy and LiteLLM first, without depending on Hawk/METR accepting the Middleman change. Tracked in [#117](https://github.com/meridianlabs-ai/inspect_sentinel/issues/117), following the run-identity review on PR #116; it does not block the proxy work in PRs 9 and 10.

In ordinary Inspect evals, the trusted host sets the run identity. A caller that can make its own authenticated proxy requests can instead supply a fresh `sample_uuid` and escape a prior run's termination, or claim another run's identity. Registration replay and re-registration need an explicit lifecycle contract too; the reproduced UUID-switch case does not establish that the current registration implementation clears an existing run.

- Define who may create, name, restart or resume a run, using authentication and authorization and keeping privileged registration or run-creation credentials outside the agent.
- Bind a canonical run to authorized eval, task, sample and epoch context during registration or privileged sample startup. Consider a scoped credential or a trusted proxy identity without prescribing a new token protocol. A protected model route must not treat missing identity as permission to create an anonymous run.
- Use fixed server-owned task definitions where appropriate, and distinguish permission to supply task facts from authority over a run. Preserve prior decisions for the same canonical execution across registration replay or replacement, while allowing authorized independent runs and epochs.
- Consider explicit termination scopes: the current run, a dataset sample across epochs within an eval/task, or a whole task execution within an eval. Specify authorization, effects on concurrent samples, host cancellation and scoring before adding broader scopes.
- Define how credential lifetime, retained termination state, expiry and restart interact. Do not infer shared identity by matching conversation contents, since legitimate independent epochs can have identical conversations.
- Tests: identity changes, omitted required identity and re-registration cannot evade decisions without run-creation authority; one principal cannot claim another's run; authorized independent executions stay separate; any broader termination scope covers every execution it names.

Malformed Inspect context failing closed remains a separate fix for PR #116 and is not deferred by this follow-up.

## PR 16: TLS on the sidecar's listeners

Repository: `inspect_sentinel`. Depends on: PRs 3, 4, 6 and 9. Scheduled before PR 17, so that `MiddlemanHook` supports it from its first version. It does not depend on PR 15 or on Hawk.

A proxy reaches the sidecar over plain HTTP, and Envoy reaches the Envoy shim over plain gRPC. Each is protected by where the sidecar listens: an operator makes it reachable by the proxy alone. This PR lets the sidecar serve TLS on every port it listens on, and require a client certificate from the proxies that call it. Plain listeners stay the default, for a sidecar that runs beside its proxy.

- The sidecar: a `tls` stanza inside `server`, with `cert`, `key` and `client_ca`, and the options `--tls-cert`, `--tls-key` and `--tls-client-ca`. The certificate is served on every port: the endpoints' port, the Envoy shim's port, and the endpoint for evals' own port where it has one. `client_ca` requires a client certificate signed by that CA on the ports proxies call. The endpoint for evals' own port keeps its token, since Inspect presents no client certificate.
- Envoy: the configuration generator takes the CA that signed the sidecar's certificate, and Envoy's client certificate and key, as files in Envoy's file system. It puts TLS on both of Envoy's clusters for the sidecar, checking the sidecar's host name, with `h2` for the shim's gRPC.
- LiteLLM's guardrail: an `https` URL for the sidecar, the CA to trust, and a client certificate and key.
- Certificates are read at startup, so a new certificate needs a restart. Reloading them without one, such as through Envoy's SDS, is not planned.
- Tests: a CA and certificates the tests make; Envoy and the LiteLLM guardrail calling the sidecar with mutual TLS; a client with no certificate, or one signed by another CA, refused; the endpoint for evals reached with its token over TLS.

## PR 17: Middleman

Repositories: `inspect_sentinel` and `hawk`. Technical prerequisites: PRs 3 and 5, and Hawk/METR accepting Middleman's passthrough hook. Scheduled after PRs 15 and 16; it does not hold up trusted run identity or the other proxy integrations.

- In `inspect_sentinel`: `MiddlemanHook`, a package under `src/proxies/middleman`, calling `judge_request` and `judge_reply`, over TLS with a client certificate where PR 16's settings are given, passing on and removing Inspect's two headers, and putting the `sentinel` field into Middleman's errors.
- In `hawk`: the passthrough hook in Middleman, which is METR's to accept; Middleman letting the `sentinel` field through on its errors; and routing `/sentinel/*` to the sidecar at the load balancer in front of Middleman.
- A README for the hook, including the load balancer's routing.

## Follow-up: shared and durable state

Repository: `inspect_sentinel`, with deployment configuration and Inspect lifecycle handling where needed. Deliver before running more than one sidecar worker or promising that a run's state survives a restart. The initial deployment remains one process, with no promise of restart continuity.

- Define and implement shared state and the coordination it needs: run termination, the sentinel's store and tallies, conversation amendments and judged results, registered facts, held exchanges, and request results must remain consistent when successive calls reach different workers.
- Define persistence, recovery and unavailable-state behavior before allowing a run to continue after restart. Losing state must not silently turn a previously ended run into a fresh one. Coordinate terminal-state retention and credential expiry with [PR 15](#pr-15-trusted-run-identity-and-termination-scopes).
- Tests: alternate a run's requests and replies between workers; check that termination, tallies and amendments survive the handoff and that overlapping duplicate results are judged once. Restart between exchanges and verify the chosen recovery or refusal behavior, including a run that was already terminated.

## Not planned yet

These need design first. Each names the repositories it would touch.

- A full policy for model calls that Inspect answers from its cache, which never reach the proxy. The initial remote-mode cache behavior must be agreed before PR 11; supporting cached calls with equivalent remote judgment needs further design in `inspect_ai`, and possibly `inspect_sentinel`.
- Tallies kept at the task level. `inspect_sentinel`.
