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
| `sentinels.py` | the rule, `always_sunny`, and its tally |
| `sentinel.yaml` | the sidecar's configuration, which names that rule |
| `task.py` | two Inspect tasks: `philadelphia`, the one to start with, and `stations` |
| `eval_hook.py` | an Inspect hook that `stations` needs, described [below](#telling-the-sentinel-about-the-sample) |
| `compose.yaml` | a stack to run it all: Middleman, the sidecar, and a local sign-in |

## Running the whole stack

`compose.yaml` starts Middleman with the sidecar deciding for it, so the only thing left to run is the task.

You need Docker, a provider API key, and a checkout of hawk whose Middleman has the passthrough hook. The stack looks for it in a `hawk` directory beside this repository; set `HAWK_DIR` if yours is elsewhere.

Middleman's passthrough hook is a class Middleman loads at startup and calls with each model request and reply. This repository has one for it, `MiddlemanHook`, in a small package of its own at [`src/proxies/middleman`](../../src/proxies/middleman). It sends each request and reply to the sidecar and carries out the answer. A Middleman image gets it with `pip install inspect_sentinel_middleman`. This stack mounts the package into Middleman instead, and names the class in Middleman's `MIDDLEMAN_PASSTHROUGH_HOOK` setting. No sentinel runs inside Middleman.

```bash
export ANTHROPIC_API_KEY=<your provider key>
docker compose up --build
```

Middleman keeps the provider key. The stack signs a token of its own for you and writes it to `.stack/token.txt`, and that is what the task signs in with:

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

To try a rule of your own, edit `sentinels.py` and `sentinel.yaml` and restart the sidecar with `docker compose restart sidecar`. The models the stack serves are listed in `middleman/models.jsonc`.

## Running with your own proxy

You need a model proxy that hands its traffic to a sidecar. For Middleman that is its passthrough hook with `MiddlemanHook`, as `compose.yaml` sets it up. You also need this package with the sidecar's extras:

```bash
pip install "inspect_sentinel[sidecar,anthropic] @ git+https://github.com/meridianlabs-ai/inspect_sentinel"
```

The `anthropic` extra is for reading Anthropic's traffic. For an OpenAI model, install `openai` in its place.

Start the sidecar, from this directory:

```bash
python -m inspect_sentinel.sidecar \
  --load sentinels.py --sentinel sentinel.yaml --reject-status 503
```

It listens on `127.0.0.1:8900`. Point your proxy at it, then run the task with the model's base URL set to the proxy:

```bash
ANTHROPIC_BASE_URL=<your proxy> inspect eval task.py@philadelphia --model anthropic/claude-haiku-4-5
```

Or, with an OpenAI model:

```bash
OPENAI_BASE_URL=<your proxy> inspect eval task.py@philadelphia --model openai/gpt-5-mini
```

## What to look for

The sample passes, and Inspect reports one HTTP retry. That retry is the rejection.

`--reject-status 503` is what lets the run carry on. The sidecar refuses the rainy report with a status Inspect retries by itself, and adds the rejection to the retried request, so the model sees its own report and the editor's note. With the default, 400, the refusal is final and the sample fails with the sentinel's message.

The rejected report never appears in the eval's own conversation. Only the model was told.

## Telling the sentinel about the sample

`philadelphia` needs nothing from the eval: its sentinel decides from the model call alone. The second task, `stations`, is for when that isn't enough.

`stations` has two samples. Both ask the same question, and the weather is the same heavy rain. The two samples are for different stations, and only one station insists that it's always sunny:

| Sample | Its metadata | What the sentinel does |
|--------|--------------|------------------------|
| `sunny_station` | `always_sunny: true` | rejects the rainy report, and the model files a sunny one |
| `plain_station` | `always_sunny: false` | lets the rainy report through |

The model calls of the two samples are the same. The only difference is the sample's metadata, and metadata never appears in a model call. So the eval tells the sidecar about each sample directly, before the sample's first model call. The sentinel then reads the metadata as it would inside an eval, from `context.eval.metadata`.

The sentinel also keeps a tally of the reports it rejected. Before scoring, the eval fetches that tally from the sidecar, and a scorer reads it.

### The Inspect hook is a stand-in

Inspect doesn't talk to a sentinel's sidecar by itself yet. `eval_hook.py` does that job for this example, and is meant to be replaced when Inspect does it. `task.py` imports the Inspect hook, and the Inspect hook is off unless `INSPECT_SENTINEL_EVAL_URL` is set.

The Inspect hook calls the sidecar's endpoint for evals, which is described in [design/sidecar.md](../../design/sidecar.md#the-methods). It does four things:

- When a sample starts, it registers the sample with `register_run`.
- Before each model call, it adds a header that names the run, so the sidecar can match the call to the registration. It adds a second header with an id for the call. Inspect puts an id of its own on each call, but doesn't tell an Inspect hook what it is.
- After each model call, it asks the sidecar what the sentinel recorded for the call with `request_result`, by that id, and notes the answer in the transcript.
- Before scoring, it fetches the run's tally with `run_result` and copies it into the sample's store.

### Running it

The sidecar's endpoint for evals is a second, separate endpoint, and it is off unless you give it a token. Choose any secret, and give the same one to the stack and to the eval:

```bash
export ANTHROPIC_API_KEY=<your provider key>
export INSPECT_SENTINEL_EVAL_TOKEN=<any secret you choose>
docker compose up --build
```

In another terminal, with `INSPECT_SENTINEL_EVAL_TOKEN` set to the same secret:

```bash
ANTHROPIC_BASE_URL=http://localhost:3500/anthropic \
ANTHROPIC_API_KEY=$(cat .stack/token.txt) \
INSPECT_SENTINEL_EVAL_URL=http://localhost:8901/sentinel/v1 \
  inspect eval task.py@stations --model anthropic/claude-haiku-4-5
```

### What to look for

Both scores are 1.00, and Inspect reports one HTTP retry. The retry is the rejected report at `sunny_station`.

- **`includes`** checks the answer. `sunny_station` answers "sunny" and `plain_station` answers "rain", from the same question and the same weather.
- **`editor_tally`** checks the sentinel's own count: one rejected report at `sunny_station`, and none at `plain_station`.

Each sample's transcript has notes from the Inspect hook, with what the sidecar answered: one for `register_run`, one after each model call for `request_result`, and one for `run_result`.

At `sunny_station`, the note after the model's report holds two records. The first is the report that was rejected, with the rule's reason, and the second is the report that went on. The rejected report is nowhere else in the eval's log, because Inspect was never handed it.

Now run `stations` again without `INSPECT_SENTINEL_EVAL_URL`. The Inspect hook is off, so the sentinel is told nothing about either sample, and it holds every report to the rule. `plain_station` answers "sunny" and fails, and no tally comes back for either sample.
