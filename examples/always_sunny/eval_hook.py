"""An Inspect hook that connects this example's eval to the sentinel in the sidecar.

A stand-in. inspect_ai doesn't call the sidecar's endpoint for evals yet, so this Inspect hook does the part of that an Inspect hook can: it registers each sample, names the run and the request on each model call, brings back what the sentinel recorded for each call, and brings the sentinel's tally back before scoring. It can't act on a sentinel's refusal, which is still left to the retry that `--reject-status 503` relies on.

The Inspect hook is off unless `INSPECT_SENTINEL_EVAL_URL` is set.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import httpx
from inspect_ai.hooks import (
    BeforeModelGenerate,
    Hooks,
    ModelUsageData,
    SampleScoring,
    SampleStart,
    hooks,
)
from inspect_ai.log import transcript

# inspect_ai has no public way to ask which sample is running. Hawk's runner
# reads it the same way.
from inspect_ai.log._samples import sample_active
from inspect_ai.util import store
from pydantic_core import to_jsonable_python

URL = "INSPECT_SENTINEL_EVAL_URL"
TOKEN = "INSPECT_SENTINEL_EVAL_TOKEN"

# the header Hawk's runner names a run with, which Middleman passes on to the
# sidecar with each request
RUN_HEADER = "x-inspect-sample-uuid"

# a header of this example's own, for an id the hook gives each model call.
# inspect_ai puts an id of its own on each call, but doesn't tell a hook what
# it is. The stack has Middleman pass this header on, and starts the sidecar
# with it as the header a request is known by.
REQUEST_HEADER = "x-sentinel-request"

# the ids of each run's model calls that the sidecar hasn't answered for yet
_unanswered: dict[str, list[str]] = {}


class SidecarHook(Hooks):
    def enabled(self) -> bool:
        return bool(os.environ.get(URL))

    async def on_sample_start(self, data: SampleStart) -> None:
        active = sample_active()
        if active is None:
            return
        await _ask(
            "register_run",
            {
                # the id that goes on the sample's model calls, which is not
                # the uuid the eval log gives the sample
                "run": active.id,
                "eval": {
                    "task": active.task,
                    "sample_id": active.sample.id,
                    "epoch": active.epoch,
                    "sample_input": to_jsonable_python(active.sample.input),
                    "metadata": to_jsonable_python(active.sample.metadata or {}),
                },
            },
        )

    async def on_before_model_generate(self, data: BeforeModelGenerate) -> None:
        active = sample_active()
        if active is None:
            return
        await _note_requests(active.id)
        request = uuid4().hex
        _unanswered.setdefault(active.id, []).append(request)
        data.config.extra_headers = {
            **(data.config.extra_headers or {}),
            RUN_HEADER: active.id,
            REQUEST_HEADER: request,
        }

    async def on_model_usage(self, data: ModelUsageData) -> None:
        # a model call has just come back, so the sentinel has decided on it
        active = sample_active()
        if active is not None:
            await _note_requests(active.id)

    async def on_sample_scoring(self, data: SampleScoring) -> None:
        active = sample_active()
        if active is None:
            return
        await _note_requests(active.id, last=True)
        result = await _ask("run_result", {"run": active.id})
        if result is None:
            return
        # into the sample's own store, where a scorer reads it as it would
        # the state of a sentinel that ran inside the eval
        for key, value in result["store"].items():
            store().set(key, value)


# called and not used as a decorator, which inspect_ai leaves untyped
hooks(
    name="sentinel_sidecar",
    description="Registers each sample with a sentinel's sidecar, and brings its tally back.",
)(SidecarHook)


async def _note_requests(run: str, *, last: bool = False) -> None:
    """Ask the sidecar what became of a run's model calls, and note each answer in the sample's transcript.

    An answer holds what the sentinel recorded for every attempt at the call, so a report that was rejected and retried shows in the eval's log.

    Args:
        run: The run whose calls to ask about.
        last: Whether the run makes no more model calls, so that every answer is noted, settled or not.
    """
    waiting: list[str] = []
    for request in _unanswered.pop(run, []):
        result = await _ask("request_result", {"id": request}, note=False)
        if result is None:
            continue
        # a call made alongside another may still be with the sentinel
        if not last and result.get("state") in ("judging", "awaiting_reply"):
            waiting.append(request)
            continue
        transcript().info(
            {"method": "request_result", "result": result}, source="sidecar"
        )
    if waiting:
        _unanswered[run] = [*waiting, *_unanswered.get(run, [])]


async def _ask(
    method: str, params: dict[str, Any], *, note: bool = True
) -> dict[str, Any] | None:
    """Call one method of the endpoint for evals, and note the answer in the sample's transcript.

    Inspect logs an exception from a hook as a warning and carries on. A call that failed is noted in the transcript too, so that the log shows the sidecar wasn't reached.

    Args:
        method: The method to call.
        params: Its parameters.
        note: Whether to note an answer. A failure is noted either way.
    """
    try:
        result = await _call(method, params)
    except Exception as ex:
        transcript().info({"method": method, "error": str(ex)}, source="sidecar")
        return None
    if note:
        transcript().info({"method": method, "result": result}, source="sidecar")
    return result


async def _call(method: str, params: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        answered = await client.post(
            os.environ[URL],
            headers={"Authorization": f"Bearer {os.environ.get(TOKEN, '')}"},
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
    if answered.status_code != 200:
        raise RuntimeError(f"The sidecar answered with HTTP {answered.status_code}.")
    body: dict[str, Any] = answered.json()
    if "error" in body:
        raise RuntimeError(body["error"]["message"])
    return body["result"]
