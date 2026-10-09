# Inspect Sentinel hook for Middleman

A hook for METR's Middleman. Middleman runs the hook, and the hook calls a remote [Inspect Sentinel](https://github.com/meridianlabs-ai/inspect_sentinel) endpoint over HTTPS and applies its answer: pass, replace or refuse.

Inspect Sentinel does not run in Middleman. Only this hook does. It needs only `aiohttp` and `fastapi`, which Middleman has.

## Using it

Install it into Middleman's image:

```bash
pip install inspect_sentinel_middleman
```

Then set, in Middleman's environment:

| Variable | Meaning |
| --- | --- |
| `MIDDLEMAN_PASSTHROUGH_HOOK` | `inspect_sentinel_middleman:MiddlemanHook` |
| `INSPECT_SENTINEL_SIDECAR_URL` | the Inspect Sentinel endpoint the hook calls: the sidecar's endpoint for proxies, as `https://<sentinel host>:<port>/`. The sidecar listens on port 8900 unless it is started with another |
| `INSPECT_SENTINEL_SIDECAR_HEADERS` | optional: request headers to show the sidecar besides those that say which job and sample a request belongs to, as a comma-separated list of names, or prefixes ending in `*` |

Which traffic reaches the hook, how long it may take, and whether a sidecar that can't be reached refuses a call or lets it through are Middleman's own settings: `MIDDLEMAN_PASSTHROUGH_HOOK_CHANNELS`, `MIDDLEMAN_PASSTHROUGH_HOOK_TIMEOUT_SECONDS` and `MIDDLEMAN_PASSTHROUGH_HOOK_ON_FAILURE`.

## What it sends

Each request and each held reply is one JSON message, POSTed to that endpoint. A message carries the provider, the model's public name, the user's id, the headers above, and the request's or the reply's body. It never carries a credential: Middleman removes the caller's before the hook is called, and the hook is never given a provider key.

The messages and the answers are described by [`proxy_endpoint.schema.json`](https://github.com/meridianlabs-ai/inspect_sentinel/blob/main/src/inspect_sentinel/sidecar/proxy_endpoint.schema.json).
