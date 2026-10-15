from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from inspect_ai.model import ChatMessage
from pydantic import BaseModel, Field, ValidationError

from .._context import EvalContext, InspectSample

TASK_EXPIRY = 24 * 60 * 60
"""How long a task Inspect registers is kept once no request has named it, unless the registration asks otherwise, in seconds."""

MAX_TASK_EXPIRY = 30 * 24 * 60 * 60
"""The longest a task Inspect registers is kept once no request has named it, unless the sidecar is told otherwise, in seconds."""


class RegistrationError(Exception):
    """A task registration the sidecar won't take."""


class UnregisteredSampleError(Exception):
    """A request for a sample of a task registered beforehand that the registration doesn't hold."""


class SampleFacts(BaseModel):
    """One sample of a registered task."""

    sample_id: str | int = Field(description="The sample's id in the dataset.")
    sample_description: str | None = Field(
        default=None, description="What is distinctive about the sample."
    )
    sample_input: str | list[ChatMessage] = Field(
        description="The sample's input: a string, or chat messages as inspect_ai writes them."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict[str, Any], description="The sample's metadata."
    )


class TaskFacts(BaseModel):
    """What a sentinel is told of a task and its samples, as `context.eval`."""

    task: str = Field(description="The task's name.")
    task_description: str | None = Field(
        default=None, description="What agents in the task are asked to do."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict[str, Any],
        description="The task's metadata. A sentinel is given it merged with each sample's own.",
    )
    samples: list[SampleFacts] = Field(description="Every sample of the task.")

    def eval_context(self, sample: InspectSample) -> EvalContext | None:
        """The sample a request names, as a sentinel is given it, or None if the task holds no such sample."""
        found = next(
            (fact for fact in self.samples if fact.sample_id == sample.sample_id),
            None,
        )
        if found is None:
            return None
        return EvalContext(
            task=self.task,
            task_description=self.task_description,
            sample_id=found.sample_id,
            epoch=sample.epoch,
            sample_description=found.sample_description,
            sample_input=found.sample_input,
            metadata={**self.metadata, **found.metadata},
        )


class FileTask(TaskFacts):
    """A task registered beforehand, in a tasks file."""

    tools: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="For a tool of the task's own that has the name of one of inspect_ai's, the tool's registry name, by the name the model sees.",
    )


class TasksFile(BaseModel):
    """A tasks file: the tasks registered with the sidecar beforehand."""

    tasks: list[FileTask] = Field(description="The tasks, each named once.")


def read_tasks_file(path: str | Path) -> dict[str, FileTask]:
    """Read a tasks file, YAML or JSON.

    Args:
        path: The file.

    Returns:
        The tasks, by name.

    Raises:
        ValueError: If the file can't be read as a tasks file, or names a task twice.
    """
    text = Path(path).read_text()
    try:
        parsed: object = yaml.safe_load(text)
        given = TasksFile.model_validate(parsed)
    except (yaml.YAMLError, ValidationError) as ex:
        raise ValueError(f"{path}: not a tasks file: {ex}") from None
    tasks: dict[str, FileTask] = {}
    for task in given.tasks:
        if task.task in tasks:
            raise ValueError(
                f"{path}: the task {task.task!r} is named twice. A request names only a task's name, so two tasks can't share one."
            )
        tasks[task.task] = task
    return tasks
