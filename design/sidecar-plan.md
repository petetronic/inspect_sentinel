# Implementation plan: the sidecar

How [sidecar.md](sidecar.md) becomes code. Each pull request can be reviewed on its own, has its own tests, and leaves `main` importable and passing. Proposed 2026-10-14; Envoy, incremental streaming and human review added 2026-10-17.

Earlier prototype code exists for much of this. It is reused where it fits the design in `sidecar.md`, and rewritten where it does not.

## Repositories

| Repository | Owner | What lands there |
|---|---|---|
| `inspect_sentinel` | Meridian | the core package, the sidecar (`inspect_sentinel.sidecar`) with its Envoy shim, and the proxy hooks, each a package of its own under `src/proxies` |
| `inspect_ai` | Meridian | `RemoteSentinel`, and the asks in `sidecar.md`, "Asked of inspect_ai" |
| `hawk` | METR | Middleman's passthrough hook, and routing `/sentinel/*` at the load balancer in front of Middleman |
| LiteLLM | the LiteLLM project | nothing. The guardrail depends only on LiteLLM's behaviour, and is tested against each LiteLLM version it supports |

## Order

```
inspect_sentinel                                              inspect_ai

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
        |                       +-------------------------->  8 RemoteSentinel

10 Middleman: inspect_sentinel and hawk, after 3 and 5          9 smaller asks, at any time

11 Envoy shim: inspect_sentinel, after 3, 6 and 7
        |
12 incremental streaming: inspect_sentinel and inspect_ai, after 4, 8 and 11
   Middleman streaming also needs 10
        |
13 human review: inspect_sentinel and inspect_ai, after 12
```

PRs 1 to 3 come first, one after another, and land as one pull request with a commit for each, because the endpoint is the first point at which anything runs. After 3, the sidecar's support for Inspect clients (5) and the guardrail (4) land together as one pull request, with a commit for each and one more for the guardrail passing the `sentinel` field on: the guardrail's tests run a real sidecar, and the field can only be tested through LiteLLM once the sidecar writes it. PR 8 needs 5, 6 and 7. From 7 it needs per-call rejection and `modify`, and also that a rejection Inspect puts back into its conversation is not judged as a tool result: before 7, the sidecar would judge each such rejection as a call that ran, and could end the run for a call that never ran. PR 9 needs nothing and can start at any time. PR 10 waits on METR accepting Middleman's passthrough hook. PR 11 builds the Envoy shim independently of Middleman. PR 12 adds incremental streaming to the shared handler, Envoy and LiteLLM, with Inspect client support from PR 8. Extending Middleman to hand over chunks also needs PR 10 and METR accepting that change; it does not block streaming through the other proxies. PR 13 follows incremental streaming and adds human review through the sidecar's host, Inspect's approval UI and external review consumers.

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

- `GET /sentinel/info`, and JSON-RPC 2.0 at `POST /sentinel/v1`: `register_task`, `request_result`, `run_result` and `end_run` (section 9).
- Task registration: on unless `--no-runtime-task-registration`, replacing the last unless the task is in the tasks folder, where it is answered as already registered, with `fixed` true, and nothing is stored, `expiry` capped by `--task-expiry`, a run taking its sample's facts at its first request, and `context.eval` from them.
- The tasks folder, of tasks files each with a `tasks` list, named by `tasks_folder` or `--tasks-folder` (section 9). Producing the files is asked of the workstream that builds `RemoteSentinel`.
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

## PR 8: `RemoteSentinel`

Repository: `inspect_ai`. Depends on: PRs 5, 6 and 7 (above).

- The probe, `register_task` and the `required` or `optional` setting at task init (section 9). An answer with `fixed` true, for a task in the tasks folder, is a success.
- `x-inspect-context` and `x-irid` on each model request, only after the probe found a sentinel.
- Reading the `sentinel` field in each provider: on a reply, in the last event of a stream, and on an error, including where LiteLLM nests it (section 1).
- Acting on the field as Inspect does in process: a call taken out goes back into Inspect's conversation with an approval error, a changed call keeps the model's original, a `terminate` ends the sample, and every step is written into the transcript.
- `request_result` when the field is missing, and recording a request as not judged when the sidecar cannot say.
- `run_result` before scoring, and `end_run` after.
- Tests: in inspect_ai's own suite, with a stand-in sidecar; and the conformance suite in `sidecar.md` section 11, comparing Inspect's transcript in process and through a sidecar.

## Follow-up: trusted run identity and termination scopes

Repositories: `inspect_sentinel`, `inspect_ai`, and the deployment's proxy or gateway where needed. Design and implement in a separate PR after PRs 6 and 8, once registration and the Inspect client are available. Tracked in [#117](https://github.com/meridianlabs-ai/inspect_sentinel/issues/117), following the run-identity review on PR #116; it does not block PRs 4 and 5.

In ordinary Inspect evals, the trusted host sets the run identity. A caller that can make its own authenticated proxy requests can instead supply a fresh `sample_uuid` and escape a prior run's termination, or claim another run's identity. Registration replay and re-registration need an explicit lifecycle contract too; the reproduced UUID-switch case does not establish that the current registration implementation clears an existing run.

- Define who may create, name, restart or resume a run, using authentication and authorization and keeping privileged registration or run-creation credentials outside the agent.
- Bind a canonical run to authorized eval, task, sample and epoch context during registration or privileged sample startup. Consider a scoped credential or a trusted proxy identity without prescribing a new token protocol. A protected model route must not treat missing identity as permission to create an anonymous run.
- Use fixed server-owned task definitions where appropriate, and distinguish permission to supply task facts from authority over a run. Preserve prior decisions for the same canonical execution across registration replay or replacement, while allowing authorized independent runs and epochs.
- Consider explicit termination scopes: the current run, a dataset sample across epochs within an eval/task, or a whole task execution within an eval. Specify authorization, effects on concurrent samples, host cancellation and scoring before adding broader scopes.
- Define how credential lifetime, retained termination state, expiry and restart interact. Do not infer shared identity by matching conversation contents, since legitimate independent epochs can have identical conversations.
- Tests: identity changes, omitted required identity and re-registration cannot evade decisions without run-creation authority; one principal cannot claim another's run; authorized independent executions stay separate; any broader termination scope covers every execution it names.

Malformed Inspect context failing closed remains a separate fix for PR #116 and is not deferred by this follow-up.

## PR 9: smaller asks of inspect_ai

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

## PR 10: Middleman

Repositories: `inspect_sentinel` and `hawk`. Depends on: PRs 3 and 5, and METR accepting Middleman's passthrough hook.

- In `inspect_sentinel`: `MiddlemanHook`, a package under `src/proxies/middleman`, calling `judge_request` and `judge_reply` and passing on Inspect's two headers. Middleman never sends a caller's `x-` headers to a provider, so the hook removes nothing. Then putting the `sentinel` field into Middleman's errors, once Middleman lets it through.
- In `hawk`: the passthrough hook in Middleman, which is METR's to accept; Middleman letting the `sentinel` field through on its errors; and routing `/sentinel/*` to the sidecar at the load balancer in front of Middleman.
- A README for the hook, including the load balancer's routing, and an example that runs a task through Middleman and the sidecar.

## PR 11: the Envoy shim

Repository: `inspect_sentinel`. Depends on: PRs 3, 6 and 7. Does not depend on Middleman or WASM.

- A Python gRPC `ext_proc` shim inside the sidecar, with dependencies in an optional extra. It translates Envoy's exchanges into the shared handler's `Request` and `Reply`, and applies `Pass`, `Replace` and `Refuse`; provider reading, sentinel execution and run state stay in the shared implementation.
- Anthropic Messages, OpenAI Chat Completions and OpenAI Responses, including request amendments, edited replies, and refusals. Response bodies use `FULL_DUPLEX_STREAMED` from the first version, but the shim initially holds each reply whole (section 5). Incremental release follows in PR 12.
- Envoy 1.39.1 or later. `FULL_DUPLEX_STREAMED` was added in 1.34, and releases after it fixed `ext_proc` bugs that truncated or duplicated bodies, which would corrupt a reply the shim holds and re-emits. The example and the tests pin the `envoyproxy/envoy:v1.39.1` image, the version the measurements in `sidecar.md` were taken on. Check for a newer release when the PR is built.
- Generated Envoy configuration and a runnable example: provider routes, request and response processing, message and body limits, timeouts long enough for sentinel decisions, and requesting uncompressed provider replies. Sidecar failures refuse traffic by default, and an unexpected content encoding is refused when it cannot be read and rewritten correctly.
- Inspect's request context reaches the handler, the `sentinel` field reaches the client on replies and refusals, and `/sentinel/*` reaches the endpoint for evals under the deployment's access controls.
- Tests with a real Envoy, the sidecar and a controlled provider, on all three formats, streamed and not, passed, edited and refused. Cover bodies above Envoy's ordinary buffer and gRPC defaults, split events, response trailers, provider errors, sidecar failures, hold limits and client disconnects. Run the model-history conformance cases through Envoy as well.

Done when the generated configuration runs the example and the same sentinel decisions, records and model-facing conversation survive an exchange through Envoy as through the existing handler.

## PR 12: incremental streaming

Repositories: `inspect_sentinel` and `inspect_ai`; `hawk` for Middleman's extension. Depends on: PRs 4, 8 and 11. Middleman's extension also depends on PR 10 and METR accepting chunk delivery.

- An incremental handler interface and a corresponding transport for proxy hooks, alongside the existing whole-exchange methods. Envoy's shim feeds it chunks directly; LiteLLM's guardrail sends chunks as they arrive. Provider decoding continues to use inspect_ai and the provider SDKs, and sentinels still judge complete steps.
- Define and document when early release is allowed. A tool-call decision made before the turn ends sees the completed call and only the assistant text available at that point; it must not silently acquire the whole-turn meaning of the existing path. Policies that need the full reply, including those watching `AfterGenerate`, keep whole-response holding. Eligibility and configuration must make this choice explicit.
- Stream text and reasoning as they arrive where early release is allowed. Hold every byte of a tool-call block until its arguments are complete and its decision is final, then release, modify or remove it. Preserve event order: content behind a held call waits for that call. Establish the safe completion boundary for each of the three provider formats rather than assuming their stream lifecycles are interchangeable.
- Preserve PR 7's rejection and modification semantics, the original calls restored in later model requests, and termination remembered for the run. Define what happens when every call is rejected after preceding text has gone out, or when a later decision terminates a turn whose earlier calls have been released. Released content cannot be recalled. After the response status is committed, refusals and failures use an error event the client SDK recognises, with no successful stream ending in its place.
- Keep the `sentinel` field, request and run results, and Inspect's records correct on incremental replies and failures. `RemoteSentinel` must recover decisions through `request_result` when a cut stream cannot carry the field, without recording a step twice or losing the original call.
- Keep connections alive while tool calls wait, with provider-compatible keepalives. Bound queues and held bytes, apply backpressure and hold deadlines, and define cancellation and record retention when the client disconnects or a proxy stops waiting.
- Deliver and test the incremental path through both Envoy and LiteLLM. Extend Middleman's hook to hand over chunks and accept incremental answers once PR 10 is in place; until then it holds replies whole. A proxy or provider format without a validated incremental path continues to use whole-response holding.

Acceptance tests use a provider and a sentinel whose progress is controlled: preceding text must reach the client before the provider finishes or the sentinel answers, and no part of a tool call may reach it before approval. Cover multiple calls, mixed decisions, modification, rejection of every call, termination after earlier release, split and combined chunks, malformed or interrupted streams, keepalives, backpressure and disconnects. Each provider SDK must read successful edited streams and raise on refused ones. Eligible fixtures must retain the model-history and Inspect transcript conformance of the whole-response path; a whole-response policy must demonstrably keep the reply held.

## PR 13: human review

Repositories: `inspect_sentinel` and `inspect_ai`. Depends on: PR 12, including its Envoy, LiteLLM and `RemoteSentinel` prerequisites. This is the sidecar part of [remote human surfaces](workstreams.md#13-remote-human-surfaces).

- Settle the [human-review questions](sidecar.md#human-review-questions-for-pr-13) before implementing the queue schema and surface integration: reviewer identity, context and automated assessments, call grouping and decision scope, pending age and sorting, approval validity, ownership and lifetime, reason disclosure, and audit completeness and durability.
- A sidecar implementation of `Host.ask_human` that publishes a question to a review queue and waits for the person's `HumanAnswer`, while the shared handler holds the step. Each question names its run, conversation, exchange and step, and carries the permitted choices, tool view, relevant context and escalations. Start with the two tool stages; generate-stage review still needs a supported rendering in Inspect's approval UI.
- Define the queue's discovery and answer protocol, with a schema, access controls and timestamps for showing pending age. Specify how several questions are listed, grouped and sorted, and how scores, evidence and commentary from preceding monitors and protocols are shown. External review tools consume that interface, including for agents running without an Inspect eval. Reuse existing review surfaces rather than building a new review application as part of the sidecar.
- `RemoteSentinel` discovers pending questions while its model exchange is waiting, shows them through Inspect's existing approval UI, and delivers the person's answer. An answer requires the reviewer's own credential; the agent's or eval's credential cannot approve its calls. The sidecar enforces which runs and questions a reviewer may see and answer.
- Validate an answer against the choices offered for that question and the step's legal actions. Present `terminate` wherever it is offered, distinct from rejecting one call or an explicit operation on a group. Before a call, the default choices are approve, reject and terminate; rejecting one call does not reject its siblings. Resolve each question once: retries of the same answer are idempotent, and conflicting, expired or cancelled answers cannot release a call or revive an ended run. The answer returns to the existing sentinel runner and compositions as an ordinary `HumanAnswer`.
- Keepalives cover only some waits. PR 12's keepalives work while a `BeforeToolCall` question holds a streaming reply. They cannot cover an `AfterToolCall` question, which holds the request before the provider is called, because neither Envoy nor LiteLLM can send the client anything before the request goes on. Nor can they cover a non-streaming reply, which has no stream to put them in. So a question outlives the exchange that raised it. When a review outlasts a bounded wait, the client is refused with a retryable error, and the question stays pending in the run's state. When the same request comes back, the sidecar recognises it and either waits again or applies the answer. A held reply is kept, so a retry is answered from it and does not call the provider again for a different reply. Specify the bounded wait, how a retried request is recognised, and how this interacts with the SDKs' own retries and Inspect's.
- Use PR 12's ordered release while a person is deciding. Specify question expiry, unanswered questions, run termination, client disconnects and sidecar restart, including how pending questions are cancelled or recovered. No missing answer implicitly approves a call; where nobody can be asked, retain the host's cautious fallback.
- Record the question, the reviewer's identity, the answer and any reason, with the final sentinel decision, so Inspect's transcript and the sidecar's request and run results retain what happened. A reviewer-only explanation stays separate from a rejection's agent-facing message, as in the existing contract.
- A runnable example and integration tests through Envoy and LiteLLM, with an external review consumer for an agent without an eval and Inspect's approval surface for an eval. Keep the queue and host independent of the proxy; the Middleman integration uses the same review path when available.

Done when a tool call stays held while its review question is visible, approval releases it once if the final sentinel decision permits it, rejection or termination is enforced with the expected model history and records, and unrelated streams can still progress. Acceptance tests cover multiple waiting calls, per-call answers within a group, terminate as a distinct option, pending age and queue ordering, assessment provenance, retries, conflicting and stale answers, reviewer access, unanswered questions, disconnects, restart and a run ending during review. A review must outlast an idle timeout between the client and the proxy at both tool stages, streamed and not: an `AfterToolCall` question on a held request, and a `BeforeToolCall` question on a held reply, where a retry must receive the reply that was held. Test the audit guarantees chosen in the design questions, including recording failures. An eval credential must demonstrably be unable to answer a review question.

## Not planned yet

These need design first. Each names the repositories it would touch.

- Model calls that Inspect answers from its cache, which never reach the proxy. `inspect_ai`, and possibly `inspect_sentinel`.
- Filling `step.history` from each conversation the sidecar keeps, with the turns before a compaction, once the core has the as-sent `step.history` (R10, sentinel #98). `inspect_sentinel`.
- Tallies kept at the task level. `inspect_sentinel`.
- A state implementation that processes share and that outlasts a restart. `inspect_sentinel`.
