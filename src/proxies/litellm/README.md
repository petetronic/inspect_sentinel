# Inspect Sentinel guardrail for LiteLLM

A guardrail for the LiteLLM proxy. LiteLLM runs the guardrail, and the guardrail calls a remote [Inspect Sentinel](https://github.com/meridianlabs-ai/inspect_sentinel) sidecar over HTTPS and applies its answer: pass, replace or refuse.

Inspect Sentinel does not run in LiteLLM. Only this guardrail does. It needs only LiteLLM and what LiteLLM's proxy already has.

## Using it

Install it into LiteLLM's image:

```bash
pip install inspect_sentinel_litellm
```

Then add it to LiteLLM's config:

```yaml
guardrails:
  - guardrail_name: inspect-sentinel
    litellm_params:
      guardrail: inspect_sentinel_litellm.LiteLLMGuardrail
      mode: [pre_call, post_call]
      default_on: true
      api_base: https://<sentinel host>:8900
```

| Setting | Meaning |
| --- | --- |
| `api_base` | the sidecar's address. The sidecar listens on port 8900 unless it is started with another |
| `mode` | both `pre_call` and `post_call`. A sentinel judges the tool calls in a reply, so the guardrail has to be shown replies as well as requests |
| `default_on` | `true` to judge every request. Without it, LiteLLM judges only a request or a key that names the guardrail |
| `unreachable_fallback` | optional: `fail_closed` refuses a call when the sidecar gives no usable answer, and `fail_open` lets it through. `fail_closed` unless set |
| `timeout` | optional: how many seconds the sidecar may take to answer. No limit unless set, because a sentinel may ask a person |
| `sidecar_headers` | optional: request headers to show the sidecar besides Inspect's own, as a list of names, or prefixes ending in `*` |
| `max_reply_bytes` | optional: the largest reply that is held to be judged. 32 MiB unless set |

An older LiteLLM loads a guardrail only from a file beside its config, and stops with `Could not find module file`. There, put this one line in a file named `sentinel_guardrail.py` beside the config, and name the class as `sentinel_guardrail.LiteLLMGuardrail`:

```python
from inspect_sentinel_litellm import LiteLLMGuardrail as LiteLLMGuardrail
```

## What it reads

Requests to LiteLLM's `/v1/messages`, `/chat/completions` and `/responses` routes, and their replies, streamed or not. A request to another route goes on unread.

A streamed reply is held whole until the sidecar has answered, so a reply that is refused is refused before any of it has gone out.

## What it sends

Each request and each held reply is one JSON-RPC 2.0 call to the sidecar, `judge_request` or `judge_reply`. A call carries the route's format, the model's name in LiteLLM, the user's id, the request's or the reply's body, and some of the request's headers:

- An Inspect eval's own two, `x-inspect-context` and `x-irid`, which say which sample a request is for. The guardrail passes them on without reading them, and removes them from what LiteLLM sends the provider.
- Any listed in `sidecar_headers`.

A call never carries a credential, and it leaves out the notes LiteLLM keeps on the caller.

The body is what LiteLLM is about to send, with one thing put right. LiteLLM writes a Claude model's thinking, asked for through `/responses`, as a `reasoning` item that OpenAI's own types reject. A client that sends that item back in a later request sends one the sidecar can't read, so the guardrail corrects the item in what it sends the sidecar. LiteLLM is still sent the item as the client sent it.

The calls and their results are described by [`proxy_endpoint.schema.json`](https://github.com/meridianlabs-ai/inspect_sentinel/blob/main/src/inspect_sentinel/sidecar/proxy_endpoint.schema.json).

## The sidecar's routes for evals

An Inspect eval knows only the base URL of the model API it calls, which here is LiteLLM's. So the guardrail adds a route to LiteLLM's app: everything under `/sentinel/` is passed on to the sidecar at `api_base`, which is where the eval talks to the sidecar. Nothing else needs setting up.

LiteLLM checks the caller's key on these routes as it does on its model routes, and a key that may call LiteLLM's OpenAI routes may call these too. The sidecar is never sent the key.

## What LiteLLM decides

The guardrail is tested with LiteLLM 1.83 and 1.105.

LiteLLM writes the error a caller is sent when the guardrail refuses, so these differ by route and by LiteLLM version. With LiteLLM 1.105, a refusal has the sentinel's message and its status, in the error format of the route. The one exception is a streamed `/responses` reply, which is answered with status 200 and a `response.failed` event that holds the message.

With LiteLLM 1.83, a streamed reply that is refused is answered with a 500 that does not hold the sentinel's message, and more of the replies that LiteLLM converts from another provider's format can't be read.
