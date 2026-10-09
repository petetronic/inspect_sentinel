from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from inspect_ai.core import ChatMessage, Target
from inspect_ai.util import StoreModel

from ._host import Host, active_step

SMT = TypeVar("SMT", bound=StoreModel)


@dataclass(frozen=True, kw_only=True)
class EvalContext:
    """The Inspect eval sample a monitor or protocol is running in. Constant for the sample; nothing in it came from the agent."""

    task: str
    """Task name."""

    task_description: str | None
    """What agents in this setting are asked to do: the task's `description`. None when the task sets none or the host does not provide it."""

    sample_id: str | int
    """Dataset sample identifier."""

    epoch: int
    """Epoch of the sample: 1 unless the task runs several."""

    sample_description: str | None
    """What is distinctive about this instance: the sample's `description`. None when the sample sets none or the host does not provide it."""

    sample_input: str | list[ChatMessage]
    """The sample's input: the assignment the agent was given. Judge a step against `step.input`, which is exactly what the model was sent; use this for what the agent was asked to do."""

    metadata: dict[str, Any]
    """Task metadata merged with sample metadata."""

    target: Target | None = None
    """The expected answer. None until monitors and protocols can opt in with `target=True`."""

    @property
    def sample_input_text(self) -> str:
        """`sample_input` as one string: `sample_input` itself if a string, else its messages' text joined with newlines."""
        if isinstance(self.sample_input, str):
            return self.sample_input
        return "\n".join(message.text for message in self.sample_input)


@dataclass(frozen=True, kw_only=True)
class ProxyContext:
    """The model request a monitor or protocol is running on, as a proxy described it. None of it is verified: the caller set the headers, so where an agent makes its own model calls they came from the agent."""

    provider: str
    """Whose API the request is written for, e.g. `anthropic`."""

    model: str
    """The model the request is for, as the proxy names it."""

    user: str | None
    """Who the proxy says is calling, if it knows."""

    headers: Mapping[str, str]
    """The request headers the proxy passed on, by lowercase name. Which ones is for whoever runs the proxy to choose."""


@dataclass(frozen=True, kw_only=True)
class Context:
    """What a monitor or protocol is given. Constant for a sample. Nothing in `eval` came from the agent; `proxy` says what it can't promise."""

    path: str
    """Instance path, e.g. `attempt/internet_attempt`. Empty at the top layer."""

    host: Host
    """Inference through the host's models, and asking a person. See `Host`."""

    eval: EvalContext | None
    """The task and sample being run, in an Inspect eval. None outside an eval, for example when a proxy runs the sentinel on requests that have no task, sample or epoch."""

    proxy: ProxyContext | None = None
    """The request being judged, behind a proxy. None in an eval's own process."""

    def store_as(self, model_cls: type[SMT]) -> SMT:
        """Typed view of this instance's state, namespaced by `path`.

        Renaming the instance (its mapping key) or wrapping it in another layer changes `path`, so its state moves with it. The store is the one the host supplied for the running step, so call this while the monitor or protocol runs; outside a step it raises `RuntimeError`.

        Args:
            model_cls: The `StoreModel` subclass to read and write through.
        """
        return model_cls(store=active_step("store_as()").store, instance=self.path)


def validate_instance_name(name: object) -> str:
    """Return `name` if it is a string that can be a path segment, else raise `ValueError`.

    Args:
        name: A candidate instance name; configuration may hand over a non-string key.
    """
    if not isinstance(name, str) or name == "" or "/" in name:
        raise ValueError(
            f"Instance name {name!r} must be a non-empty string without '/'."
        )
    return name
