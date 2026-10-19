# Always sunny, through Middleman

A sentinel that runs beside a model proxy, not inside the eval. This is the [`always_sunny`](../always_sunny) example with METR's Middleman as the proxy in place of LiteLLM.

The task is a weather reporter. It looks up Philadelphia's weather, which is heavy rain, and files a report. The sentinel has one rule: it's always sunny in Philadelphia. It rejects the rainy report, the model is told why, and it files a sunny one.

The task itself knows nothing about the sentinel. There is no `sentinel=` in `task.py`. The sentinel sits at the proxy every model call already goes through, so it sees each call the model makes whoever wrote the task.

| File | What it is |
|------|------------|
| `rule/` | the rule, `always_sunny`, in a small package, `weather_rules`, that the sidecar's image installs |
| `sentinel.yaml` | the sentinel, which names that rule |
| `task.py` | the Inspect task, `philadelphia` |
| `middleman/` | Middleman's configuration: who it lets in, and the models it serves |
| `signin.py` | a local sign-in that issues the token you call Middleman with |
| `compose.yaml` | a stack to run it all: Middleman, the sidecar, and the sign-in |

## Running the whole stack

`compose.yaml` starts Middleman with the sidecar deciding for it, so the only thing left to run is the task.

You need Docker, a provider API key, and a checkout of hawk whose Middleman has the passthrough hook. The stack looks for it in a `hawk` directory beside this repository; set `HAWK_DIR` if yours is elsewhere.

Middleman's passthrough hook is a class Middleman loads at startup and calls with each model request and reply. This repository has one for it, `MiddlemanHook`, in a small package of its own at [`src/proxies/middleman`](../../src/proxies/middleman). It sends each request and reply to the sidecar and carries out the answer. A Middleman image of your own gets it with `pip install inspect_sentinel_middleman`. This stack mounts the package into Middleman instead, and names the class in Middleman's `MIDDLEMAN_PASSTHROUGH_HOOK` setting. No sentinel runs inside Middleman.

```bash
export ANTHROPIC_API_KEY=<your provider key>
docker compose up --build
```

Middleman keeps the provider key. The stack signs a token for you and writes it to `.stack/token.txt`, and that is what the task signs in with:

```bash
ANTHROPIC_BASE_URL=http://localhost:3500/anthropic \
ANTHROPIC_API_KEY=$(cat .stack/token.txt) \
  inspect eval task.py@philadelphia --model anthropic/claude-haiku-4-5
```

For an OpenAI model, give the stack `OPENAI_API_KEY` and run:

```bash
OPENAI_BASE_URL=http://localhost:3500/openai/v1 \
OPENAI_API_KEY=$(cat .stack/token.txt) \
  inspect eval task.py@philadelphia --model openai/gpt-5-mini
```

The token works for a week, and a new one is made each time the stack starts.

To try a rule of your own, edit `rule/weather_rules/__init__.py` and `sentinel.yaml`, and rebuild the sidecar with `docker compose up --build sidecar`. The models the stack serves are listed in `middleman/models.jsonc`.

## Running with your own Middleman

Install the hook into Middleman's image:

```bash
pip install inspect_sentinel_middleman
```

Then set `MIDDLEMAN_PASSTHROUGH_HOOK=inspect_sentinel_middleman:MiddlemanHook`, and `INSPECT_SENTINEL_SIDECAR_URL` to where the sidecar listens. The hook's [README](../../src/proxies/middleman/README.md) lists its settings, and Middleman's own settings for a hook.

You also need this package with the sidecar's extras:

```bash
pip install "inspect_sentinel[sidecar,anthropic] @ git+https://github.com/meridianlabs-ai/inspect_sentinel"
```

The `anthropic` extra is for reading Anthropic's traffic. For an OpenAI model, install `openai` in its place. Then install the example's rule, from this directory:

```bash
pip install ./rule
```

Start the sidecar, from this directory:

```bash
python -m inspect_sentinel.sidecar --sentinel sentinel.yaml --reject-status 503
```

It listens on `127.0.0.1:8900`. Then run the task with the model's base URL set to Middleman.

## What to look for

The sample passes, and Inspect reports one HTTP retry. That retry is the rejection.

`--reject-status 503` is what lets the run carry on. The sidecar refuses the rainy report with a status Inspect retries by itself, and adds the rejection to the retried request, so the model sees its own report and the editor's note. With the default, 400, the refusal is final and the sample fails with the sentinel's message.

The rejected report never appears in the eval's own conversation. Only the model was told.
