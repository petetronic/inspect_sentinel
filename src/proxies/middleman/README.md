# Inspect Sentinel hook for Middleman

A hook for METR's Middleman. Middleman runs the hook, and the hook calls a remote [Inspect Sentinel](https://github.com/meridianlabs-ai/inspect_sentinel) sidecar over HTTPS and applies its answer: pass, replace or refuse.

Inspect Sentinel does not run in Middleman. Only this hook does. It needs only `aiohttp` and `fastapi`, which Middleman already has.

## Using it

Install it into Middleman's image:

```bash
pip install inspect_sentinel_middleman
```

Then set, in Middleman's environment:

| Variable | Meaning |
| --- | --- |
| `MIDDLEMAN_PASSTHROUGH_HOOK` | `inspect_sentinel_middleman:MiddlemanHook` |
| `INSPECT_SENTINEL_SIDECAR_URL` | the sidecar's address, as `https://<sentinel host>:8900`. The sidecar listens on port 8900 unless it is started with another |
| `INSPECT_SENTINEL_SIDECAR_HEADERS` | optional: request headers to show the sidecar besides Inspect's own, as a comma-separated list of names, or prefixes ending in `*`, such as `x-hawk-*` |

Middleman has its own settings for any hook it loads, and they apply to this one:

- `MIDDLEMAN_PASSTHROUGH_HOOK_CHANNELS` sets which traffic is judged.
- `MIDDLEMAN_PASSTHROUGH_HOOK_ON_FAILURE` sets whether a call the sidecar gives no usable answer for is refused or let through.
- `MIDDLEMAN_PASSTHROUGH_HOOK_TIMEOUT_SECONDS` sets how long an answer may take. A sentinel may ask a person, so set it as long as the slowest sentinel needs.
- `MIDDLEMAN_PASSTHROUGH_HOOK_MAX_REPLY_BYTES` sets the largest reply that is held to be judged.

## What it reads

The hook reads every request Middleman hands it and the provider's reply, streamed or not. Middleman holds a streamed reply whole until the sidecar has answered, so a refused reply is refused before any of it has gone out.

The hook tells the sidecar which API a request uses, from Middleman's route: `/v1/messages`, `/chat/completions` or `/responses`. A request on any other route goes to the sidecar without that, and the sidecar reads it by its provider. If the sidecar can't read it, Middleman's on-failure setting decides whether it is refused or let through.

## What it sends

Each request and each held reply is one JSON-RPC 2.0 call to the sidecar, `judge_request` or `judge_reply`. A call carries Middleman's id for the request, the route's API, the provider, the model's public name, the user's id, the request's or the reply's body, and some of the request's headers:

- An Inspect eval's own two headers, `x-inspect-context` and `x-irid`, which say which sample a request is for. The hook passes them on without reading them. Middleman never sends a caller's `x-` headers to a provider.
- Any headers listed in `INSPECT_SENTINEL_SIDECAR_HEADERS`.

A call never carries a credential. Middleman removes the caller's before the hook is called, and the hook is never given a provider key.

The calls and their results are described by [`proxy_endpoint.schema.json`](https://github.com/meridianlabs-ai/inspect_sentinel/blob/main/src/inspect_sentinel/sidecar/proxy_endpoint.schema.json).

## What an Inspect eval is told

When a request comes from an Inspect eval, the sidecar answers with a `sentinel` field that says what was decided.

- On a reply that goes on, the field sits beside the reply's own fields, or in the last event of a stream that the provider's SDK reads. Middleman sends the reply as the sidecar wrote it.
- On a refusal, Middleman writes the error from the refusal's status and message alone, so the field does not reach the eval. The eval then asks the sidecar what was decided.

Any other client is sent the reply, or the refusal, with no field.
