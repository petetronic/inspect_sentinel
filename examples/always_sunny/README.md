# Always sunny

A sentinel that runs beside a model proxy, not inside the eval.

The task is a weather reporter. It looks up Philadelphia's weather, which is heavy rain, and files a report. The sentinel has one rule: it's always sunny in Philadelphia. It rejects the rainy report, the model is told why, and it files a sunny one.

The task itself knows nothing about the sentinel. There is no `sentinel=` in `task.py`.

That is the point of the example. The other examples here put the sentinel in the task, so it runs inside the eval. This one puts it at the proxy every model call already goes through, so the sentinel sees each call the model makes whoever wrote the task:

- A task you didn't write, or can't change, is covered as it is.
- Whoever runs the proxy chooses the rule once, for every task that uses it.
- A task can't leave the sentinel out, because it never put it in.

| File | What it is |
|------|------------|
| `rule/` | the rule, `always_sunny`, in a small package, `weather_rules`, that the sidecar's image installs |
| `sentinel.yaml` | the sentinel, which names that rule |
| `task.py` | the Inspect task, `philadelphia` |
| `litellm.yaml` | LiteLLM's configuration: the models it serves, and the guardrail that calls the sidecar |
| `compose.yaml` | a stack to run it all: LiteLLM and the sidecar |

## Running the whole stack

`compose.yaml` starts LiteLLM with the sidecar deciding for it, so the only thing left to run is the task.

You need Docker and a provider API key.

LiteLLM runs a guardrail on each model request and each reply. This repository has one for it, `LiteLLMGuardrail`, in a small package of its own at [`src/proxies/litellm`](../../src/proxies/litellm). It sends each request and reply to the sidecar and carries out the answer. The stack's LiteLLM image has it from that directory, and a LiteLLM image of your own gets it with `pip install inspect_sentinel_litellm`. `litellm.yaml` names the class and tells it where the sidecar is. No sentinel runs inside LiteLLM.

```bash
export ANTHROPIC_API_KEY=<your provider key>
docker compose up --build
```

LiteLLM keeps the provider key. The task signs in to LiteLLM with LiteLLM's own key, which in this stack is `sk-always-sunny` unless you set `LITELLM_MASTER_KEY`:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-always-sunny \
  inspect eval task.py@philadelphia --model anthropic/claude-haiku-4-5
```

For an OpenAI model, give the stack `OPENAI_API_KEY` and run:

```bash
OPENAI_BASE_URL=http://localhost:4000/v1 \
OPENAI_API_KEY=sk-always-sunny \
  inspect eval task.py@philadelphia --model openai/gpt-5-mini
```

To try a rule of your own, edit `rule/weather_rules/__init__.py` and `sentinel.yaml`, and rebuild the sidecar with `docker compose up --build sidecar`. The models the stack serves are listed in `litellm.yaml`.

## Running with your own proxy

You need a model proxy that hands its traffic to a sidecar. For LiteLLM that is the `LiteLLMGuardrail` guardrail, as `litellm.yaml` sets it up. You also need this package with the sidecar's extras:

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

It listens on `127.0.0.1:8900`. Point your proxy at it, then run the task with the model's base URL set to the proxy:

```bash
ANTHROPIC_BASE_URL=<your proxy> inspect eval task.py@philadelphia --model anthropic/claude-haiku-4-5
```

## What to look for

The sample passes, and Inspect reports one HTTP retry. That retry is the rejection.

`--reject-status 503` is what lets the run carry on. The sidecar refuses the rainy report with a status Inspect retries by itself, and adds the rejection to the retried request, so the model sees its own report and the editor's note. With the default, 400, the refusal is final and the sample fails with the sentinel's message.

The rejected report never appears in the eval's own conversation. Only the model was told.

In the stack's output, LiteLLM logs the rejection as an error with a traceback. That is LiteLLM recording that a guardrail refused a reply, and nothing has gone wrong.
