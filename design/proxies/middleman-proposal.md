# A passthrough hook for Middleman

Middleman is the model gateway of METR's Hawk. This is what has been proposed to Hawk so that a sentinel can see the traffic that passes through it.

**Status:** proposed, not merged. The proposal is [METR/hawk#2102](https://github.com/METR/hawk/issues/2102).

## What was proposed

Middleman gains one optional setting, `MIDDLEMAN_PASSTHROUGH_HOOK`, which names a class by its import path. Middleman creates one instance at startup and calls two optional async methods on it:

- `on_request`, with each model request before it goes to the provider.
- `on_reply`, with the provider's reply before it goes back to the caller.

Each method returns a replacement body, or `None` to send on what it was given. To refuse, it raises `HTTPException`, and the caller gets an error in the provider's own format. With the setting unset, nothing changes.

## Why this shape

- **Middleman takes no position on what the hook does or where it runs.** A hook can decide in process or ask another service. Middleman holds no message format, no schema and no second service.
- **It is scoped by traffic.** A setting lists which kinds of traffic reach the hook: eval sets, scans, or direct use. Traffic that isn't listed is forwarded and streamed as it is today.
- **The hook is told the kind of traffic**, so one hook can treat kinds differently.
- **Failure is the operator's choice.** A hook that errors or is too slow refuses the call by default, or lets it through.

## What it costs

- When a hook defines `on_reply`, a reply is held until the hook returns, so a streamed reply arrives all at once.
- A hook runs in Middleman's process, so it is reviewed and deployed as Middleman's own code is.
- A replaced request can't change the model. It still goes to the model Middleman checked the caller's access to.

## Middleman's settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `MIDDLEMAN_PASSTHROUGH_HOOK` | the class, as `package.module:ClassName` | unset, which means off |
| `MIDDLEMAN_PASSTHROUGH_HOOK_CHANNELS` | which traffic reaches the hook: any of `eval-set`, `scan`, `direct` | all |
| `MIDDLEMAN_PASSTHROUGH_HOOK_TIMEOUT_SECONDS` | the longest one call to the hook may take | 30 |
| `MIDDLEMAN_PASSTHROUGH_HOOK_ON_FAILURE` | `refuse` or `pass` when the hook fails | `refuse` |
| `MIDDLEMAN_PASSTHROUGH_HOOK_MAX_REPLY_BYTES` | the largest reply held for `on_reply` | 32 MiB |

## How a sentinel uses it

Inspect Sentinel does not run in Middleman. Only a hook does: `MiddlemanHook`, a small package of its own in [`src/proxies/middleman`](../../src/proxies/middleman). It calls the sidecar's endpoint for proxies over HTTPS and applies the answer. The sentinels run in the sidecar.

The hook is installed into Middleman by itself, as its own wheel. It needs only `aiohttp` and `fastapi`, which Middleman already has, so installing it brings in neither the inspect_sentinel package nor inspect_ai. Its [README](../../src/proxies/middleman/README.md) says how to set it up.

The messages the hook sends are described by [`proxy_endpoint.schema.json`](../../src/inspect_sentinel/sidecar/proxy_endpoint.schema.json). [`examples/always_sunny`](../../examples/always_sunny) runs the whole chain.
