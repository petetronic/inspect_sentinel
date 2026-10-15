# Examples

Runnable sentinel configurations, each watching a short `react()` agent that has `bash()` and `python()` in a Docker container with no network (`compose.yaml`). `example_samples.py` holds the samples they share; each example defines its own `Task` at the bottom. Read them in this order:

| Example | What it teaches | Command |
|---------|-----------------|---------|
| `no_network.py` | A rule: a protocol with no monitors that rejects `curl`, `wget` and `pip install` | `inspect eval examples/no_network.py --model openai/gpt-5-mini` |
| `llm_suspicion.py` | A monitor that has a model score each call against each part of a policy with `context.host.score()`, under `threshold(suspicion(), reject_at=0.8)` | `inspect eval examples/llm_suspicion.py --model openai/gpt-5-mini --model-role monitor=anthropic/claude-haiku-4-5` |
| `trajectory.py` | Two functions of one monitor (`before` and `after`), returned as a `MonitorGroup` and sharing `store_as()` state, under `observe_only(failure_count())` | `inspect eval examples/trajectory.py --model openai/gpt-5-mini` |
| `nested.py` | Composition: a mapping of named children with a `concurrent()` of rules inside it, and a rule that ends the step with `decide_final()` | `inspect eval examples/nested.py --model openai/gpt-5-mini --model-role monitor=anthropic/claude-haiku-4-5` |
| `escalate_to_human.py` | A person in the loop: a rule that rejects, escalates or proceeds, ahead of `human()` in a `sequential()`, so the person is asked only about escalated calls | `inspect eval examples/escalate_to_human.py --model openai/gpt-5-mini` |
| `escalate_turn_to_human.py` | A person deciding about a whole assistant turn: a rule at `AfterGenerate` escalates a turn whose calls delete files, and `human(stages=["model_output"])` shows the person the turn's text and every call before any of them runs | `inspect eval examples/escalate_turn_to_human.py --model openai/gpt-5-mini` |

[`always_sunny/`](always_sunny) is different in kind: its sentinel runs in a sidecar beside a model proxy, LiteLLM, not in the eval, and its task has no `sentinel=`. It has its own README.

The checks are deliberately simple, to keep the examples readable. Rules are cheap and run without a model call. They match a call's text and structured arguments, and they cannot prove a call safe: a different tool, or a string built at run time, can get past one. Anything that needs an understanding of shell or code goes to an LLM monitor, or is escalated to a person, typically `sequential([rule(), llm_monitor(), human()])`. In these examples the container's `network_mode: none` is what keeps the agent off the network.

## Running

The examples need Docker running, this package (which installs `inspect-ai` 0.3.278 or later), `inspect-scout` (which `llm_suspicion.py` and `nested.py` need to render the agent's step for the monitor's model), and the model provider SDKs:

```bash
pip install inspect-sentinel inspect-scout
pip install openai anthropic
export OPENAI_API_KEY=...     # the agent
export ANTHROPIC_API_KEY=...  # the monitor role
```

(From a clone of this repository, `uv sync` installs `inspect-sentinel` and `inspect-scout` into `.venv`.) Then run the commands above from the repository root. Any capable models work; these are cheap. Each sample is capped at 20 messages. `escalate_to_human.py` waits for a person to answer each escalated call, in the approval panel of the full-screen display (or the console with `--display plain`), so run it where you can answer.

## Reading the results

The sentinel's reports are `SentinelEvent`s in each sample's transcript, next to the tool events they judged. The log viewer shows them, or read them with `read_eval_log()`:

```python
from inspect_ai.event import SentinelEvent
from inspect_ai.log import read_eval_log

log = read_eval_log("logs/<file>.eval")
for sample in log.samples or []:
    for event in sample.events:
        if isinstance(event, SentinelEvent):
            print(
                sample.id,
                event.kind,
                event.status,
                event.path,
                event.suspicion,
                event.action,
                event.explanation,
            )
```

A lone protocol is the root and records its own decision at the empty path, after its children's; a list or mapping with at least one protocol is wrapped in a root `concurrent()`. Monitors alone are a configuration error: wrap them in `observe_only()` to record them without acting, as `trajectory.py` does. What each example records:

- `no_network.py`: a `decision` at the root (`factory` `no_network`, `path` empty) for each `bash()` call, `reject` for the ones using a network command. The agent receives the decision's `message` as the tool call's error and carries on; the `explanation` is for the log only.
- `llm_suspicion.py`: an `observation` at `suspicion` with the model's `files` and `network` scores and its explanation, then `threshold`'s `decision` at the root: `reject` when either score is 0.8 or more (the `cleanup` sample's deletes outside `/work`, `fetch_data`'s download), explained with the highest score and the monitor's explanation, otherwise `proceed`. The agent is told only that the call was not approved.
- `trajectory.py`: `observation`s at `failure_count` alternating `before` and `after`. `before`'s suspicion rises by 0.2 after each call `after` saw fail.
- `nested.py`: when `protected` calls `decide_final()`, the step ends at once. The log shows `bypassed` at `guard` and at the root (their decision logic was skipped), `cancelled` at `audit` (its model call was still in flight), and then `protected`'s `reject`. A step no rule ends has decisions at `guard/network`, `guard/protected`, `guard` and the root, and an observation at `audit`.
- `escalate_to_human.py`: decisions at `triage` and the root for each call. A call that mentions `/etc` (the `add_host` sample) is rejected at `triage`. A call that deletes files with one of the listed words (the `cleanup` sample) is escalated at `triage`, then decided at `human` by the person's answer, which the root returns. Other calls proceed at `triage`, and `human` is not reached, including a delete written another way, such as `pathlib.Path.unlink()`.
