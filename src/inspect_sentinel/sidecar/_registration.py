from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from inspect_ai.model import ChatMessage
from pydantic import BaseModel, Field, ValidationError

from .._config import parse_config_file
from .._context import EvalContext, InspectSample

TASK_EXPIRY = 24 * 60 * 60
"""How long a task Inspect registers is kept once no request has named it, unless the registration asks otherwise, in seconds."""

MAX_TASK_EXPIRY = 30 * 24 * 60 * 60
"""The longest a task Inspect registers is kept once no request has named it, unless the sidecar is told otherwise, in seconds."""

MAX_TASK_REGISTRATIONS = 1000
"""How many tasks registered by evals are kept at once, unless the sidecar is told otherwise. Anything that can reach the endpoint for evals can register a task, so a registration past this is refused, rather than one still in use dropped."""


class RegistrationError(Exception):
    """A task registration the sidecar won't take."""


class RegistrationsFullError(RegistrationError):
    """A task registration the sidecar can't take now, because it keeps as many as it takes. A failure, not a refusal: one is taken again once a kept task expires."""


Strictness: TypeAlias = Literal["open", "known_tasks", "registered_only"]
"""How strict a sidecar is about a request it can't place. `open` judges every request, without the task's facts where it holds none. `known_tasks` refuses a request that names a task it doesn't hold, and judges one that names no task. `registered_only` refuses any request that doesn't name a sample it holds, including one that names no task."""


class UnregisteredSampleError(Exception):
    """A request the sidecar refuses because it can't place it: a sample of a task registered beforehand that the registration doesn't hold, or, by the sidecar's strictness, a task or sample it doesn't hold."""


class UnknownTaskError(Exception):
    """A sample's registration for a task the eval hasn't registered, or whose registration is no longer kept."""


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


class TaskInfo(BaseModel):
    """What a sentinel is told of a task, apart from its samples."""

    task: str = Field(description="The task's name.")
    task_description: str | None = Field(
        default=None, description="What agents in the task are asked to do."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict[str, Any],
        description="The task's metadata. A sentinel is given it merged with each sample's own.",
    )


class TaskFacts(TaskInfo):
    """What a sentinel is told of a task and its samples, as `context.eval`."""

    samples: list[SampleFacts] = Field(description="Every sample of the task.")

    def eval_context(self, sample: InspectSample) -> EvalContext | None:
        """The sample a request names, as a sentinel is given it, or None if the task holds no such sample."""
        found = next(
            (fact for fact in self.samples if fact.sample_id == sample.sample_id),
            None,
        )
        return _eval_context(self, found, sample) if found is not None else None


class TaskRegistration(TaskInfo):
    """What an eval registers of a task at task init: the task's facts, and which of its samples it runs. Each sample's own facts follow before the sample runs, in its `complete_registration`, so a large dataset is never sent at once."""

    sample_ids: list[str | int] = Field(
        description="The ids of the samples the eval runs, after `--limit`, `--sample-id` and the like."
    )
    epochs: int = Field(ge=1, description="How many epochs each sample runs.")


@dataclass
class RegisteredTask:
    """A task an eval registered, with the facts of each sample whose registration is complete."""

    registration: TaskRegistration
    samples: dict[str | int, SampleFacts] = field(
        default_factory=dict[str | int, SampleFacts]
    )
    sample_ids: set[str | int] = field(init=False)
    """The samples the eval runs: those the registration named, and any a sample's registration added."""

    def __post_init__(self) -> None:
        # a set, as a dataset can have many samples, and each is looked up
        self.sample_ids = set(self.registration.sample_ids)

    def eval_context(self, sample: InspectSample) -> EvalContext | None:
        """The sample a request names, as a sentinel is given it, or None if its registration isn't complete."""
        found = self.samples.get(sample.sample_id)
        if found is None:
            return None
        return _eval_context(self.registration, found, sample)


def _eval_context(
    task: TaskInfo, found: SampleFacts, sample: InspectSample
) -> EvalContext:
    # the epoch is the request's, as each epoch is a run of its own
    return EvalContext(
        task=task.task,
        task_description=task.task_description,
        sample_id=found.sample_id,
        epoch=sample.epoch,
        sample_description=found.sample_description,
        sample_input=found.sample_input,
        metadata={**task.metadata, **found.metadata},
    )


class FileTask(TaskFacts):
    """A task registered beforehand, in a tasks file in the tasks folder."""

    tools: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="For a tool of the task's own that has the name of one of inspect_ai's, the tool's registry name, by the name the model sees.",
    )


class TasksFile(BaseModel):
    """A tasks file: tasks registered with the server beforehand. The tasks folder holds any number of them."""

    tasks: list[FileTask] = Field(description="The tasks, each named once.")


def read_tasks_folder(folder: str | Path) -> dict[str, FileTask]:
    """Read every tasks file in a folder: each YAML or JSON file directly in it.

    Args:
        folder: The folder.

    Returns:
        The tasks of every file, by name.

    Raises:
        ValueError: If the folder isn't one, a file in it can't be read as a tasks file, or a task is named twice, in one file or in two.
    """
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"{folder}: not a folder of tasks files.")
    tasks: dict[str, FileTask] = {}
    found: dict[str, Path] = {}
    for file in sorted(path.iterdir()):
        if not file.is_file() or file.suffix not in _TASKS_FILES:
            continue
        for name, task in read_tasks_file(file).items():
            if name in tasks:
                raise ValueError(
                    f"{file}: the task {name!r} is named in {found[name].name} too. A request names only a task's name, so two tasks can't share one."
                )
            tasks[name] = task
            found[name] = file
    return tasks


_TASKS_FILES = frozenset({".yaml", ".yml", ".json"})


def read_tasks_file(path: str | Path) -> dict[str, FileTask]:
    """Read a tasks file, YAML or JSON.

    Args:
        path: The file.

    Returns:
        The tasks, by name.

    Raises:
        ValueError: If the file can't be read as a tasks file, or names a task twice.
    """
    # read as a sentinel file is, so that a key given twice is an error
    parsed = parse_config_file(str(path))
    try:
        given = TasksFile.model_validate(parsed)
    except ValidationError as ex:
        raise ValueError(f"{path}: not a tasks file: {ex}") from None
    tasks: dict[str, FileTask] = {}
    for task in given.tasks:
        if task.task in tasks:
            raise ValueError(
                f"{path}: the task {task.task!r} is named twice. A request names only a task's name, so two tasks can't share one."
            )
        tasks[task.task] = task
    return tasks
