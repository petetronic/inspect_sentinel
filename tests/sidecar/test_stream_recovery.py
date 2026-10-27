"""What an eval can recover from the endpoint for evals about a reply released early, when the `sentinel` field never reached it: the stream was cut, or its proxy didn't pass the field on.

A small client of the endpoint for evals stands in for the eval, and the proxy hook is driven through the endpoint for proxies. Neither reads the field.
"""

import base64
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
import httpx
import pytest
from starlette.applications import Starlette

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, _state
from inspect_sentinel.sidecar._eval_endpoint import PATH as EVAL_PATH
from inspect_sentinel.sidecar._eval_endpoint import eval_routes
from inspect_sentinel.sidecar._proxy_endpoint import PATH as PROXY_PATH
from inspect_sentinel.sidecar._proxy_endpoint import proxy_app
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import STREAM, anthropic_recorded, inspect_headers, request
from tests.sidecar.test_early_release import noting
from tests.sidecar.test_eval_endpoint import _Clock

# text, then calls to get_weather for Paris, Tokyo and Lima
_BODY = anthropic_recorded("parallel-tool-calls")
_CALLS = ["Paris", "Tokyo", "Lima"]
_IRID = "irid-1"


@dataclass(frozen=True)
class Recovered:
    """What an eval knows of a request once it has asked the sidecar."""

    state: str
    """`decided`, `pending`, `not_judged`, `dropped` or `unknown`."""

    outcome: str | None = None
    """What a decided request came to."""

    reason: str | None = None
    """Why a request was not judged."""

    turn: list[str] | None = None
    """The cities of the calls in the model's turn as it proposed it, where the client was refused it or handed it with calls taken out."""

    decided: list[str] | None = None
    """What each call judged was decided, where the sidecar sends results."""


class EvalClient:
    """A client of the endpoint for evals, as an eval asks it what became of a request whose reply didn't carry the `sentinel` field."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def recover(self, irid: str) -> Recovered:
        answered = await self._http.post(
            EVAL_PATH,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "request_result",
                "params": {"id": irid},
            },
        )
        result: dict[str, Any] = answered.json()["result"]
        if result["records"] != "held":
            return Recovered(result["records"])
        recorded = result.get("recorded")
        decided = (
            None
            if recorded is None
            else [record["report"]["action"] for record in recorded]
        )
        state = result["state"]
        if state in ("judging", "awaiting_reply"):
            return Recovered("pending", decided=decided)
        if state == "not_judged":
            return Recovered("not_judged", reason=result["reason"], decided=decided)
        turn = result.get("withheld") or result.get("proposed")
        return Recovered(
            "decided",
            outcome=result["outcome"],
            turn=None if turn is None else _cities(turn["message"]),
            decided=decided,
        )


def _cities(message: dict[str, Any]) -> list[str]:
    calls: list[dict[str, Any]] = message.get("tool_calls") or []
    return [call["arguments"]["location"] for call in calls]


class ProxyHook:
    """A proxy hook that hands a reply over as it arrives, and never passes on the `sentinel` field."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        answered = await self._http.post(
            PROXY_PATH,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        return answered.json()

    async def ask(self, irid: str = _IRID) -> None:
        asked = request(sample="sample_123")
        judged = await self.call(
            "judge_request",
            id="x",
            provider="anthropic",
            model="claude-test",
            user="user-1",
            headers=inspect_headers() | {"x-irid": irid},
            body=json.loads(asked.body),
        )
        assert judged["result"]["action"] == "pass"

    async def begin(self) -> None:
        await self.call(
            "begin_reply",
            id="x",
            provider="anthropic",
            model="claude-test",
            user="user-1",
            status=200,
            content_type=STREAM,
        )

    async def feed(self, body: bytes, size: int = 97) -> None:
        for seq, at in enumerate(range(0, len(body), size)):
            data = base64.b64encode(body[at : at + size]).decode("ascii")
            await self.call("reply_chunk", id="x", seq=seq, data=data)
        await self.call("end_reply", id="x")

    async def collect(self) -> list[dict[str, Any]]:
        # what the caller is sent, to the end
        outputs: list[dict[str, Any]] = []
        while not outputs or outputs[-1]["kind"] not in (
            "finished",
            "refuse",
            "error_event",
        ):
            answered = await self.call("next_release", id="x", after=len(outputs))
            outputs += answered["result"]["outputs"]
        return outputs

    async def stream(self) -> list[dict[str, Any]]:
        await self.ask()
        await self.begin()
        await self.feed(_BODY)
        return await self.collect()


@asynccontextmanager
async def _sidecar(
    handler: Handler, *, results: bool = True
) -> AsyncGenerator[tuple[ProxyHook, EvalClient]]:
    # both endpoints on one app, as the sidecar serves them by default
    proxy = proxy_app(handler)
    app = Starlette(
        routes=[
            *proxy.routes,
            *eval_routes(handler, max_body_bytes=1024 * 1024, results=results),
        ],
        lifespan=proxy.router.lifespan_context,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
        ) as http:
            yield ProxyHook(http), EvalClient(http)


@pytest.mark.anyio
@pytest.mark.parametrize("results", [True, False], ids=["results", "no results"])
@pytest.mark.parametrize(
    ("action", "city", "last", "outcome", "turn", "decided"),
    [
        (
            "reject",
            "Tokyo",
            "finished",
            "proceed",
            _CALLS,
            ["proceed", "reject", "proceed"],
        ),
        ("reject", "every", "error_event", "reject", _CALLS, ["reject"] * 3),
        (
            "terminate",
            "Tokyo",
            "error_event",
            "terminate",
            _CALLS,
            ["proceed", "terminate"],
        ),
    ],
    ids=["one call taken out", "every call rejected", "run ended"],
)
async def test_what_a_reply_released_early_came_to_is_recovered_without_the_field(
    action: str,
    city: str,
    last: str,
    outcome: str,
    turn: list[str] | None,
    decided: list[str],
    results: bool,
) -> None:
    handler = Handler(noting([], action, city), release="early")

    with anyio.fail_after(10):
        async with _sidecar(handler, results=results) as (proxy, eval):
            outputs = await proxy.stream()
            recovered = await eval.recover(_IRID)

    # text went out first, so the turn was refused in the stream, if at all
    assert outputs[0]["kind"] == "release" and outputs[-1]["kind"] == last
    assert recovered == Recovered(
        "decided",
        outcome=outcome,
        turn=turn,
        decided=decided if results else None,
    )


@protocol
def waits_then(action: str, asked: anyio.Event, go: anyio.Event) -> Protocol:
    """Decide `action` for the first call once `go` is set, setting `asked` when asked, and let the others go."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if asked.is_set():
            return Decision.proceed()
        asked.set()
        await go.wait()
        if action == "fail":
            raise RuntimeError("The monitor failed.")
        return Decision.reject("the first lookup", message="Not that one.")

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize("results", [True, False], ids=["results", "no results"])
async def test_a_turn_judged_when_the_client_went_is_pending_then_decided(
    results: bool,
) -> None:
    asked, go = anyio.Event(), anyio.Event()
    handler = Handler(waits_then("reject", asked, go), release="early")

    with anyio.fail_after(10):
        async with _sidecar(handler, results=results) as (proxy, eval):
            await proxy.ask()
            await proxy.begin()
            await proxy.feed(_BODY)
            await asked.wait()
            # the client goes while the sentinel judges the finished turn
            await proxy.call("abort_reply", id="x")
            pending = await eval.recover(_IRID)
            go.set()
            while (recovered := await eval.recover(_IRID)).state == "pending":
                await anyio.sleep(0.01)

    assert pending.state == "pending"
    # the judgment went on to the end, and is kept, as when held whole
    assert recovered == Recovered(
        "decided",
        outcome="proceed",
        turn=_CALLS,
        decided=["reject", "proceed", "proceed"] if results else None,
    )


@pytest.mark.anyio
async def test_a_turn_whose_client_went_before_it_ended_is_cut_off() -> None:
    handler = Handler(noting([], "reject", "Tokyo"), release="early")

    with anyio.fail_after(10):
        async with _sidecar(handler) as (proxy, eval):
            await proxy.ask()
            await proxy.begin()
            half = base64.b64encode(_BODY[: len(_BODY) // 2]).decode("ascii")
            await proxy.call("reply_chunk", id="x", seq=0, data=half)
            await proxy.call("abort_reply", id="x")
            # the stream records it as it stops, which an eval may ask before
            while (recovered := await eval.recover(_IRID)).state == "pending":
                await anyio.sleep(0.01)

    assert recovered == Recovered("not_judged", reason="cut_off", decided=[])


@pytest.mark.anyio
async def test_a_turn_the_sentinel_failed_on_mid_stream_was_not_judged() -> None:
    asked, go = anyio.Event(), anyio.Event()
    go.set()
    handler = Handler(waits_then("fail", asked, go), release="early")

    with anyio.fail_after(10):
        async with _sidecar(handler) as (proxy, eval):
            outputs = await proxy.stream()
            recovered = await eval.recover(_IRID)

    assert outputs[-1]["kind"] == "error_event"
    assert recovered == Recovered("not_judged", reason="error", decided=[])


@pytest.mark.anyio
async def test_a_request_never_seen_is_unknown_and_one_forgotten_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = Handler(
        noting([], "reject", "Tokyo"), release="early", state=MemoryState(expiry=100)
    )

    with anyio.fail_after(10):
        async with _sidecar(handler) as (proxy, eval):
            await proxy.stream()
            unknown = await eval.recover("irid-never-sent")
            _Clock.now = 150.0
            dropped = await eval.recover(_IRID)

    assert (unknown, dropped) == (Recovered("unknown"), Recovered("dropped"))
