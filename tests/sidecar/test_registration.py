import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import BeforeGenerate
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Request, _state
from inspect_sentinel.sidecar._registration import (
    FileTask,
    RegistrationError,
    SampleFacts,
    TaskFacts,
    UnregisteredSampleError,
    read_tasks_file,
)
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import SAMPLE, inspect_headers, request

_TASK = TaskFacts(
    task="philadelphia",
    task_description="File the weather report.",
    metadata={"desk": "weather", "shared": "task"},
    samples=[
        SampleFacts(
            sample_id=1,
            sample_description="A rainy day.",
            sample_input="File today's report for Philadelphia.",
            metadata={"shared": "sample", "city": "Philadelphia"},
        )
    ],
)


@protocol
def reports_the_eval() -> Protocol:
    async def decide(context: Context, step: BeforeGenerate) -> Decision | None:
        given = context.eval
        if given is None:
            return Decision.proceed("None")
        return Decision.proceed(
            json.dumps(
                {
                    "task": given.task,
                    "task_description": given.task_description,
                    "sample_id": given.sample_id,
                    "epoch": given.epoch,
                    "sample_description": given.sample_description,
                    "sample_input": given.sample_input_text,
                    "metadata": given.metadata,
                }
            )
        )

    return decide


def _asked(id: str, **sample: Any) -> Request:
    # a request from the Inspect sample the header names
    return replace(request(id), headers=inspect_headers(**sample))


async def _eval_seen(handler: Handler, id: str) -> Any:
    records = await handler.records(id)
    assert records is not None
    explanation = cast(Any, records[-1].detail).report.explanation
    return None if explanation == "None" else json.loads(explanation)


@pytest.mark.anyio
async def test_a_run_of_a_registered_task_is_told_its_sample() -> None:
    handler = Handler(reports_the_eval())
    await handler.register_task("eval_789", _TASK)

    await handler.request(_asked("a"))

    assert await _eval_seen(handler, "a") == {
        "task": "philadelphia",
        "task_description": "File the weather report.",
        "sample_id": 1,
        # the epoch is the request's, as each epoch is a run of its own
        "epoch": SAMPLE["epoch"],
        "sample_description": "A rainy day.",
        "sample_input": "File today's report for Philadelphia.",
        # the sample's own metadata wins, as in Inspect's own process
        "metadata": {"desk": "weather", "shared": "sample", "city": "Philadelphia"},
    }


@pytest.mark.anyio
async def test_a_sample_input_of_chat_messages_is_kept_as_messages() -> None:
    handler = Handler(reports_the_eval())
    sample = SampleFacts.model_validate(
        {
            "sample_id": 1,
            "sample_input": [
                {"role": "system", "content": "You report the weather."},
                {"role": "user", "content": "File today's report."},
            ],
        }
    )
    await handler.register_task(
        "eval_789", _TASK.model_copy(update={"samples": [sample]})
    )

    await handler.request(_asked("a"))

    seen = await _eval_seen(handler, "a")
    assert seen["sample_input"] == "You report the weather.\nFile today's report."


@pytest.mark.anyio
@pytest.mark.parametrize(
    "sample",
    [
        {"task": "another_task"},
        {"eval_id": "another_eval"},
        {"sample_id": 2},
    ],
    ids=["another task", "another eval", "a sample the task doesn't hold"],
)
async def test_a_run_of_a_task_not_registered_is_told_no_sample(
    sample: dict[str, Any],
) -> None:
    handler = Handler(reports_the_eval())
    await handler.register_task("eval_789", _TASK)

    await handler.request(_asked("a", **sample))

    assert await _eval_seen(handler, "a") is None


@pytest.mark.anyio
async def test_a_run_keeps_the_sample_it_first_found() -> None:
    handler = Handler(reports_the_eval())
    await handler.register_task("eval_789", _TASK)
    await handler.request(_asked("a"))

    # the task is registered again, changed, while the run goes on
    await handler.register_task(
        "eval_789", _TASK.model_copy(update={"task_description": "Changed."})
    )
    await handler.request(_asked("b"))
    # a run that starts now is told the new registration
    await handler.request(_asked("c", sample_uuid="sample_124"))

    assert (await _eval_seen(handler, "b"))["task_description"] == (
        "File the weather report."
    )
    assert (await _eval_seen(handler, "c"))["task_description"] == "Changed."


@pytest.mark.anyio
async def test_a_registration_that_arrives_late_reaches_the_run() -> None:
    handler = Handler(reports_the_eval())
    await handler.request(_asked("a"))

    await handler.register_task("eval_789", _TASK)
    await handler.request(_asked("b"))

    assert await _eval_seen(handler, "a") is None
    assert (await _eval_seen(handler, "b"))["task"] == "philadelphia"


@pytest.mark.anyio
async def test_a_client_without_the_header_is_told_no_sample() -> None:
    handler = Handler(reports_the_eval())
    await handler.register_task("eval_789", _TASK)

    await handler.request(request("a"))

    assert await _eval_seen(handler, "a") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("asked", "granted"),
    [
        (None, 24 * 60 * 60),
        (7 * 24 * 60 * 60, 7 * 24 * 60 * 60),
        (365 * 24 * 60 * 60, 30 * 24 * 60 * 60),
    ],
    ids=["a day unless asked", "as asked", "no longer than the handler allows"],
)
async def test_a_registration_is_kept_as_long_as_asked_within_a_limit(
    asked: float | None, granted: float
) -> None:
    assert (
        await Handler(reports_the_eval()).register_task("eval_789", _TASK, asked)
        == granted
    )


@pytest.mark.anyio
async def test_the_limit_on_keeping_a_registration_can_be_set() -> None:
    handler = Handler(reports_the_eval(), task_expiry=60)

    assert await handler.register_task("eval_789", _TASK, 3600) == 60


class _Clock:
    now = 0.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.now


@pytest.mark.anyio
async def test_a_registration_is_kept_while_requests_name_its_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the state's own clock only: the event loop keeps the real one
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = Handler(reports_the_eval(), state=MemoryState(expiry=10**6))
    await handler.register_task("eval_789", _TASK, 100)

    # each request names the task before the registration expires
    for id, now in [("a", 90.0), ("b", 180.0), ("c", 270.0)]:
        _Clock.now = now
        await handler.request(_asked(id, sample_uuid=f"run_{id}"))
    # and then nothing names it for longer than it is kept
    _Clock.now = 371.0
    await handler.request(_asked("d", sample_uuid="run_d"))

    assert (await _eval_seen(handler, "c"))["task"] == "philadelphia"
    assert await _eval_seen(handler, "d") is None


_FILE_TASK = FileTask.model_validate(
    {**_TASK.model_dump(), "tools": {"bash": "weather_tools/bash"}}
)


@pytest.mark.anyio
async def test_a_task_registered_beforehand_is_found_for_any_eval() -> None:
    handler = Handler(reports_the_eval(), tasks={"philadelphia": _FILE_TASK})

    await handler.request(_asked("a", eval_id="any_eval"))

    assert (await _eval_seen(handler, "a"))["task"] == "philadelphia"


@pytest.mark.anyio
async def test_a_task_registered_beforehand_cant_be_replaced() -> None:
    handler = Handler(reports_the_eval(), tasks={"philadelphia": _FILE_TASK})

    with pytest.raises(RegistrationError, match="beforehand"):
        await handler.register_task(
            "eval_789", _TASK.model_copy(update={"task_description": "Changed."})
        )
    await handler.request(_asked("a"))

    assert (await _eval_seen(handler, "a"))["task_description"] == (
        "File the weather report."
    )


@pytest.mark.anyio
async def test_a_sample_a_task_registered_beforehand_doesnt_hold_is_refused() -> None:
    handler = Handler(reports_the_eval(), tasks={"philadelphia": _FILE_TASK})

    with pytest.raises(UnregisteredSampleError, match="sample 2"):
        await handler.request(_asked("a", sample_id=2))


@pytest.mark.anyio
async def test_a_sidecar_can_take_no_registrations_from_evals() -> None:
    handler = Handler(reports_the_eval(), task_registration=False)

    with pytest.raises(RegistrationError, match="takes no tasks"):
        await handler.register_task("eval_789", _TASK)


_YAML = """
tasks:
  - task: philadelphia
    task_description: File the weather report.
    metadata:
      desk: weather
    samples:
      - sample_id: 1
        sample_input: File today's report for Philadelphia.
      - sample_id: two
        sample_input:
          - role: system
            content: You report the weather.
          - role: user
            content: File today's report.
    tools:
      bash: weather_tools/bash
"""


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_a_tasks_file_is_read_as_yaml_or_json(tmp_path: Path, suffix: str) -> None:
    import yaml

    path = tmp_path / f"tasks{suffix}"
    parsed = yaml.safe_load(_YAML)
    path.write_text(_YAML if suffix == ".yaml" else json.dumps(parsed))

    tasks = read_tasks_file(path)

    task = tasks["philadelphia"]
    assert task.tools == {"bash": "weather_tools/bash"}
    assert [sample.sample_id for sample in task.samples] == [1, "two"]
    messages = task.samples[1].sample_input
    assert not isinstance(messages, str)
    assert [message.role for message in messages] == ["system", "user"]


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("tasks: [{task: a, samples: []}, {task: a, samples: []}]", "named twice"),
        ("tasks: [{task: a}]", "not a tasks file"),
        ("[oops", "not a tasks file"),
        ("tasks: [{task: a, samples: [{sample_id: 1}]}]", "not a tasks file"),
    ],
    ids=["a task twice", "no samples", "not YAML", "a sample without input"],
)
def test_a_tasks_file_that_cant_be_read(tmp_path: Path, text: str, says: str) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match=says):
        read_tasks_file(path)
