import asyncio
import base64
import importlib
import json
import re
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import jsonschema
import pytest
import uvicorn
from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp

from inspect_sentinel.sidecar import Handler
from inspect_sentinel.sidecar._proxy_endpoint import SCHEMA_FILE, proxy_app
from tests.sidecar._traffic import (
    JSON,
    message,
    reports_the_proxy,
    weather,
    weather_rule,
)

# the hook is a package of its own, which this package neither imports nor ships
_PACKAGE = Path(__file__).parents[2] / "src/proxies/middleman"
# the one thing the hook and the sidecar share: the endpoint's description of
# its messages and answers
_SCHEMA: dict[str, Any] = json.loads(SCHEMA_FILE.read_text())
_PREFIX = "INSPECT_SENTINEL_SIDECAR_"


@dataclass(frozen=True)
class _HookRequest:
    """What Middleman gives the file for a request."""

    id: str = "request-1"
    provider: str = "anthropic"
    model: str = "claude-test"
    user: str | None = "user-1"
    channel: str = "eval-set"
    headers: dict[str, str] = field(default_factory=dict[str, str])
    body: dict[str, Any] = field(
        default_factory=lambda: {
            "messages": [{"role": "user", "content": "What is the weather?"}]
        }
    )


@dataclass(frozen=True)
class _HookReply:
    """What Middleman gives the file for a held reply."""

    body: bytes
    status: int = 200
    content_type: str = JSON
    headers: dict[str, str] = field(default_factory=dict[str, str])


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # as Middleman has it once the package is installed: importable by its
    # own name
    monkeypatch.syspath_prepend(str(_PACKAGE))  # pyright: ignore[reportUnknownMemberType]
    return importlib.import_module("inspect_sentinel_middleman")


def _make(monkeypatch: pytest.MonkeyPatch, **settings: str) -> Any:
    """The hook as Middleman makes it: no arguments, its settings in the environment."""
    for name in ("URL", "HEADERS"):
        monkeypatch.delenv(_PREFIX + name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(_PREFIX + name.upper(), value)
    return _load(monkeypatch).MiddlemanHook()


def _exchange(hook: Any, request: _HookRequest, reply: _HookReply) -> tuple[Any, Any]:
    async def run() -> tuple[Any, Any]:
        try:
            return (
                await hook.on_request(request),
                await hook.on_reply(request, reply),
            )
        finally:
            await hook.close()

    return asyncio.run(run())


@contextmanager
def _serving(app: ASGIApp) -> Generator[str]:
    """The URL of a real server for an app, since the hook makes real HTTP calls."""
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        serving.join()


@pytest.fixture
def sidecar(request: pytest.FixtureRequest) -> Iterator[str]:
    """The URL of a real sidecar, running the sentinel a test names or the weather rule."""
    sentinel = getattr(request, "param", None) or weather_rule()
    with _serving(proxy_app(Handler(sentinel))) as url:
        yield url


def _fits(definition: dict[str, Any], value: Any) -> None:
    jsonschema.validate(
        value, {**_SCHEMA, **definition}, cls=jsonschema.Draft202012Validator
    )


def _scripted(sent: list[Any], *answers: dict[str, Any]) -> Starlette:
    """A stand-in for the sidecar that keeps what it is sent and gives these answers in turn."""
    left = list(answers)

    async def exchange(http: HttpRequest) -> JSONResponse:
        sent.append(await http.json())
        return JSONResponse(left.pop(0))

    return Starlette(routes=[Route("/", exchange, methods=["POST"])])


def test_exchange_the_sentinel_allows_goes_on_as_it_is(
    sidecar: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _make(monkeypatch, url=sidecar)

    answers = _exchange(hook, _HookRequest(), _HookReply(message(weather("Oslo"))))

    assert answers == (None, None)


def test_reply_the_sentinel_rejects_is_refused_in_its_words(
    sidecar: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _make(monkeypatch, url=sidecar)

    with pytest.raises(HTTPException) as refused:
        _exchange(hook, _HookRequest(), _HookReply(message(weather("Tokyo"))))

    assert refused.value.status_code == 400
    assert (
        refused.value.detail
        == "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )
    assert refused.value.headers == {"x-sentinel-decision": "reject"}


@pytest.mark.parametrize("sidecar", [reports_the_proxy()], indirect=True)
def test_sentinel_is_given_the_headers_that_name_the_run_and_those_listed(
    sidecar: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _make(monkeypatch, url=sidecar, headers="x-irid,anthropic-*")
    sent = _HookRequest(
        headers={
            "x-inspect-epoch": "2",
            "x-irid": "request-1",
            "anthropic-beta": "beta-1",
            "user-agent": "a client",
        }
    )

    with pytest.raises(HTTPException) as refused:
        _exchange(hook, sent, _HookReply(message(weather("Oslo"))))

    shown = "[('anthropic-beta', 'beta-1'), ('x-inspect-epoch', '2'), ('x-irid', 'request-1')]"
    assert refused.value.detail.endswith(f"anthropic|claude-test|user-1|{shown}")


def test_sidecar_that_cannot_be_reached_is_a_failure_for_middleman_to_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # nothing listens on port 9, the discard port. Middleman's own setting
    # says whether a failure of this file refuses the exchange or lets it
    # through, so the file raises an error that is not a refusal.
    hook = _make(monkeypatch, url="http://127.0.0.1:9/")

    with pytest.raises(Exception) as failed:
        _exchange(hook, _HookRequest(), _HookReply(message(weather("Tokyo"))))

    assert type(failed.value).__name__ == "SidecarError"
    assert not isinstance(failed.value, HTTPException)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({}, "URL must say"),
        ({"url": "http://sidecar/", "headers": "x irid"}, "HEADERS takes"),
    ],
)
def test_setting_that_makes_no_sense_stops_the_hook_being_made(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _make(monkeypatch, **settings)


def test_close_ends_the_connection_and_can_be_asked_twice(
    sidecar: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _make(monkeypatch, url=sidecar)

    async def run() -> bool:
        await hook.on_request(_HookRequest())
        session = hook._session
        await hook.close()
        await hook.close()
        return bool(session.closed)

    assert asyncio.run(run())


def test_hook_imports_nothing_of_this_package() -> None:
    # it runs in Middleman's process, where this package is not installed
    files = list((_PACKAGE / "inspect_sentinel_middleman").glob("*.py"))
    assert files
    for file in files:
        assert not re.search(
            r"^\s*(from|import) inspect_sentinel(\.|\s)", file.read_text(), re.M
        )


def test_what_the_hook_sends_fits_the_schema_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[Any] = []
    request = _HookRequest(
        headers={
            "x-inspect-sample-uuid": "sample-1",
            "x-irid": "request-1",
            "user-agent": "a client",
        }
    )

    with _serving(_scripted(sent, {"action": "pass"}, {"action": "pass"})) as url:
        hook = _make(monkeypatch, url=url, headers="x-irid")
        _exchange(hook, request, _HookReply(message(weather("Oslo"))))

    asked, held = sent
    _fits(_SCHEMA["messages"]["request"], asked)
    _fits(_SCHEMA["messages"]["reply"], held)
    # the run is named for the sidecar, which knows nothing of Hawk's headers
    assert asked["run"] == "sample-1"
    assert asked["headers"] == {
        "x-inspect-sample-uuid": "sample-1",
        "x-irid": "request-1",
    }
    assert base64.b64decode(held["body"]) == message(weather("Oslo"))


_ADDED = {"x-checked": "yes"}
_REFUSAL = {
    "action": "refuse",
    "status": 403,
    "message": "Not allowed.",
    "headers": {"x-sentinel-decision": "reject"},
}


@pytest.mark.parametrize(
    ("phase", "answer", "expected"),
    [
        ("request", {"action": "pass"}, None),
        ("request", {"action": "replace", "body": {"messages": []}}, {"messages": []}),
        ("request", _REFUSAL, HTTPException),
        ("reply", {"action": "pass"}, None),
        (
            "reply",
            {"action": "replace", "body": base64.b64encode(b"changed").decode()},
            b"changed",
        ),
        ("reply", _REFUSAL, HTTPException),
        ("reply", {"action": "refuse", "status": 400, "message": "No."}, HTTPException),
        # response headers on an answer that lets a reply through
        ("request", {"action": "pass", "headers": _ADDED}, None),
        (
            "reply",
            {"action": "pass", "headers": _ADDED},
            _HookReply(message(weather("Oslo")), headers=_ADDED),
        ),
        (
            "reply",
            {
                "action": "replace",
                "body": base64.b64encode(b"changed").decode(),
                "headers": _ADDED,
            },
            _HookReply(b"changed", headers=_ADDED),
        ),
    ],
)
def test_every_answer_the_schema_allows_is_carried_out(
    monkeypatch: pytest.MonkeyPatch, phase: str, answer: dict[str, Any], expected: Any
) -> None:
    # the answer is one the endpoint may give
    _fits(_SCHEMA["answers"][answer["action"]], answer)
    passed = {"action": "pass"}
    answers = [answer, passed] if phase == "request" else [passed, answer]
    exchange = (_HookRequest(), _HookReply(message(weather("Oslo"))))

    with _serving(_scripted([], *answers)) as url:
        hook = _make(monkeypatch, url=url)
        if expected is HTTPException:
            with pytest.raises(HTTPException) as refused:
                _exchange(hook, *exchange)
            assert (refused.value.status_code, refused.value.detail) == (
                answer["status"],
                answer["message"],
            )
            assert refused.value.headers == answer.get("headers")
        else:
            carried_out = _exchange(hook, *exchange)
            assert carried_out[0 if phase == "request" else 1] == expected


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"action": "allow"}, id="an action that is not one"),
        pytest.param(["pass"], id="not an object"),
        pytest.param({"action": "replace", "body": "text"}, id="a request not JSON"),
    ],
)
def test_answer_the_schema_does_not_allow_is_a_failure_and_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch, answer: Any
) -> None:
    with _serving(_scripted([], answer)) as url:
        hook = _make(monkeypatch, url=url)
        with pytest.raises(Exception) as failed:
            _exchange(hook, _HookRequest(), _HookReply(b""))

    assert type(failed.value).__name__ == "SidecarError"


def test_header_that_may_not_be_added_to_a_reply_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = ({"action": "pass"}, {"action": "pass", "headers": {"set-cookie": "a=b"}})
    with _serving(_scripted([], *answers)) as url:
        hook = _make(monkeypatch, url=url)
        with pytest.raises(Exception) as failed:
            _exchange(hook, _HookRequest(), _HookReply(b""))

    assert type(failed.value).__name__ == "SidecarError"
    assert str(failed.value) == "headers that may not be added"
