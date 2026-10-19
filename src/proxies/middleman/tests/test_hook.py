"""The hook, handed requests and replies as Middleman hands them over.

These tests run in an environment of their own and import nothing of inspect_sentinel or Middleman, as the hook doesn't (`make test-middleman`). A test that needs a real sidecar starts one from the repository's own environment.
"""

import asyncio
import base64
import importlib
import inspect
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
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

_PACKAGE = Path(__file__).parents[1]
_REPOSITORY = Path(__file__).parents[4]
# the replies the sidecar's own tests replay
DATA = _REPOSITORY / "tests/sidecar/data"
# the one thing the hook and the sidecar share: the description of the
# methods and their results
_SCHEMA: dict[str, Any] = json.loads(
    (
        _REPOSITORY / "src/inspect_sentinel/sidecar/proxy_endpoint.schema.json"
    ).read_text()
)
_METHODS: dict[str, Any] = _SCHEMA["methods"]
PATH = "/proxy/v1"

_QUESTION = {"role": "user", "content": "What is the weather?"}
_REFUSAL = {"action": "refuse", "status": 403, "message": "Not allowed."}
_PASSED = {"jsonrpc": "2.0", "id": 1, "result": {"action": "pass"}}
_JSON = "application/json"
# as a provider sends it, which Middleman passes on as it is
_STREAM = "text/event-stream; charset=utf-8"
# the headers an Inspect eval sends with each model request
_INSPECT = {
    "x-inspect-context": json.dumps(
        {
            "version": 1,
            "eval_id": "eval-1",
            "task": "weather",
            "sample_id": 1,
            "epoch": 1,
            "sample_uuid": "sample-1",
            "attempt": "attempt-1",
        }
    ),
    "x-irid": "request-1",
}


@dataclass(frozen=True)
class HookRequest:
    """A request as Middleman's own `HookRequest` has it."""

    id: str
    provider: str
    model: str
    user: str | None
    channel: str
    headers: Mapping[str, str]
    body: dict[str, Any]
    endpoint: str = ""


@dataclass(frozen=True)
class HookReply:
    """A reply as Middleman's own `HookReply` has it."""

    status: int
    content_type: str
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict[str, str])


def stream_events(body: bytes) -> list[dict[str, Any]]:
    """The JSON object of each event in a stream."""
    return [
        json.loads(line[5:].strip())
        for line in body.decode().splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]


def _anthropic_whole() -> bytes:
    return json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()


@dataclass(frozen=True)
class _On:
    """One of Middleman's routes: its path, the provider it names, what a request to it holds, and its replies. Every reply here looks up the weather in Paris."""

    endpoint: str
    provider: str
    format: str
    body: dict[str, Any]
    whole: bytes
    stream: bytes

    def reply(self, streamed: bool) -> HookReply:
        if streamed:
            return HookReply(200, _STREAM, self.stream)
        return HookReply(200, _JSON, self.whole)


_MESSAGES = _On(
    "/anthropic/v1/messages",
    "anthropic",
    "anthropic-messages",
    {"model": "a-model", "max_tokens": 100, "messages": [_QUESTION]},
    _anthropic_whole(),
    (DATA / "single-tool-call.sse").read_bytes(),
)
_CHAT = _On(
    "/openai/v1/chat/completions",
    "openai",
    "openai-chat",
    {"model": "a-model", "max_tokens": 100, "messages": [_QUESTION]},
    (DATA / "openai-chat.json").read_bytes(),
    (DATA / "openai-chat.sse").read_bytes(),
)
_RESPONSES = _On(
    "/openai/v1/responses",
    "openai",
    "openai-responses",
    {"model": "a-model", "max_output_tokens": 100, "input": [_QUESTION]},
    (DATA / "openai-responses.json").read_bytes(),
    (DATA / "openai-responses.sse").read_bytes(),
)
_ROUTES = [
    pytest.param(
        route, streamed, id=f"{route.format}{', streamed' if streamed else ''}"
    )
    for route in (_MESSAGES, _CHAT, _RESPONSES)
    for streamed in (False, True)
]


def _request(on: _On, headers: Mapping[str, str] | None = None) -> HookRequest:
    """A request as Middleman hands it to a hook: its caller's headers, lowercased and without credentials."""
    return HookRequest(
        id="01J0000000000000000000000A",
        provider=on.provider,
        model="a-model",
        user="user-1",
        channel="eval-set",
        headers={name.lower(): value for name, value in (headers or {}).items()},
        body=on.body,
        endpoint=on.endpoint,
    )


def _load() -> Any:
    # as Middleman has it: installed, and importable by its own name
    return importlib.import_module("inspect_sentinel_middleman")


def _make(
    monkeypatch: pytest.MonkeyPatch, url: str | None, headers: str | None = None
) -> Any:
    """The hook as Middleman makes it, with its settings in the environment."""
    for name, value in (("URL", url), ("HEADERS", headers)):
        if value is None:
            monkeypatch.delenv(f"INSPECT_SENTINEL_SIDECAR_{name}", raising=False)
        else:
            monkeypatch.setenv(f"INSPECT_SENTINEL_SIDECAR_{name}", value)
    return _load().MiddlemanHook()


def _exchange(
    hook: Any, request: HookRequest, reply: HookReply
) -> tuple[dict[str, Any] | None, bytes | None]:
    """Put a request and its reply through the hook as Middleman does: what it returned for each."""

    async def run() -> tuple[dict[str, Any] | None, bytes | None]:
        try:
            sent = await hook.on_request(request)
            return sent, await hook.on_reply(request, reply)
        finally:
            await hook.close()

    return asyncio.run(run())


@contextmanager
def _serving(app: ASGIApp) -> Generator[str]:
    """The address of a real server for an app, since the hook makes real HTTP calls."""
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        serving.join()


_RULE = """
from inspect_sentinel import BeforeToolCall, Context, Decision, Escalations, protocol


@protocol
def weather_rule():
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == {city!r}:
            return Decision.reject("a lookup", message={message!r})
        return Decision.proceed()

    return decide
"""


def _sidecar_python() -> Path:
    # the repository's own environment, which has the sidecar
    given = os.environ.get("INSPECT_SENTINEL_PYTHON")
    python = Path(given) if given else _REPOSITORY / ".venv/bin/python"
    if not python.exists():
        if os.environ.get("GITHUB_ACTIONS"):
            raise FileNotFoundError(f"No sidecar environment at {python}.")
        pytest.skip(f"No sidecar environment at {python}: run `uv sync` first.")
    return python


@contextmanager
def _sidecar(city: str, tmp_path: Path) -> Generator[str]:
    """The address of a real sidecar, whose sentinel rejects a weather lookup of `city`."""
    rule = tmp_path / "weather_rule.py"
    rule.write_text(_RULE.format(city=city, message=f"{city} is not allowed."))
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    serving = subprocess.Popen(
        [
            str(_sidecar_python()),
            "-c",
            # the rule is registered by importing it, as an installed package's
            # are by its entry point, and the sidecar is then started as usual
            "import weather_rule\n"
            "from inspect_sentinel.sidecar.__main__ import main\n"
            "main()",
            "--sentinel",
            "weather_rule",
            "--port",
            str(port),
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=1):
                    break
            except OSError:
                if serving.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"The sidecar did not start: {serving.stderr.read().decode() if serving.stderr else ''}"
                    ) from None
                time.sleep(0.1)
        yield url
    finally:
        serving.terminate()
        serving.communicate(timeout=30)


def _result(result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "result": result}


@dataclass(frozen=True)
class _Status:
    """An answer the stand-in gives with an HTTP status other than 200."""

    status: int
    body: Any


def _scripted(calls: list[Any], *answers: Any) -> Starlette:
    """A stand-in for the sidecar that keeps the calls made to it and gives these answers in turn."""
    left = list(answers)

    async def call(http: HttpRequest) -> JSONResponse:
        calls.append(await http.json())
        answer = left.pop(0)
        if isinstance(answer, _Status):
            return JSONResponse(answer.body, status_code=answer.status)
        return JSONResponse(answer)

    return Starlette(routes=[Route(PATH, call, methods=["POST"])])


def _fits(part: dict[str, Any], value: Any) -> None:
    jsonschema.validate(value, {**_SCHEMA, **part}, cls=jsonschema.Draft202012Validator)


def _field(on: _On, streamed: bool, body: bytes) -> Any:
    """The `sentinel` field in a reply, where the provider's SDK reads it."""
    if not streamed:
        return json.loads(body).get("sentinel")
    fields = [event["sentinel"] for event in stream_events(body) if "sentinel" in event]
    assert len(fields) <= 1
    return fields[0] if fields else None


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_exchange_the_sentinel_allows_goes_on_as_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: _On, streamed: bool
) -> None:
    with _sidecar("Oslo", tmp_path) as url:
        sent = _exchange(_make(monkeypatch, url), _request(on), on.reply(streamed))

    # Middleman sends on what it has
    assert sent == (None, None)


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_reply_the_sentinel_rejects_is_refused_in_its_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: _On, streamed: bool
) -> None:
    # the recorded Responses stream is the one reply that names the country too
    city = "Paris, France" if on is _RESPONSES and streamed else "Paris"

    with _sidecar(city, tmp_path) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url), _request(on), on.reply(streamed))

    assert refused.value.status_code == 400
    assert refused.value.detail == (
        f"A sentinel rejected the call to get_weather: {city} is not allowed."
    )


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_an_inspect_eval_is_sent_the_sentinel_field_with_a_reply_that_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: _On, streamed: bool
) -> None:
    with _sidecar("Oslo", tmp_path) as url:
        sent, body = _exchange(
            _make(monkeypatch, url), _request(on, _INSPECT), on.reply(streamed)
        )

    assert sent is None
    assert body is not None
    assert _field(on, streamed, body)["outcome"] == "proceed"
    # nothing else of the reply changed
    if streamed:
        assert [
            {k: v for k, v in event.items() if k != "sentinel"}
            for event in stream_events(body)
        ] == stream_events(on.stream)
    else:
        assert {
            k: v for k, v in json.loads(body).items() if k != "sentinel"
        } == json.loads(on.whole)


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_calls_the_hook_makes_fit_the_schema_file(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    calls: list[Any] = []
    headers = {
        **_INSPECT,
        "x-hawk-job-type": "eval-set",
        "x-metr-other": "not listed",
        "user-agent": "a client",
        # Middleman removes a caller's credentials before a hook is called;
        # one that got through is still never shown
        "x-api-key": "a provider's key",
    }
    request = _request(on, headers)

    with _serving(_scripted(calls, _PASSED, _PASSED)) as url:
        # the address as an operator might write it, with a trailing slash
        hook = _make(monkeypatch, url + "/", headers="x-hawk-*")
        _exchange(hook, request, on.reply(streamed))

    asked, held = calls
    assert (asked["jsonrpc"], asked["method"]) == ("2.0", "judge_request")
    assert (held["jsonrpc"], held["method"]) == ("2.0", "judge_reply")
    told, reply = asked["params"], held["params"]
    _fits(_METHODS["judge_request"]["params"], told)
    _fits(_METHODS["judge_reply"]["params"], reply)
    expected = {
        # Middleman's own id for the request, which no caller can set
        "id": request.id,
        "provider": on.provider,
        "format": on.format,
        "model": "a-model",
        "user": "user-1",
    }
    assert {name: told[name] for name in expected} == expected
    assert {name: reply[name] for name in expected} == expected
    # Inspect's own, unread, and the one that is listed
    assert told["headers"] == {**_INSPECT, "x-hawk-job-type": "eval-set"}
    assert told["body"] == on.body
    assert reply["status"] == 200
    assert reply["content_type"] == (_STREAM if streamed else _JSON)
    assert base64.b64decode(reply["body"]) == (on.stream if streamed else on.whole)


def test_a_request_on_a_route_of_no_known_format_is_put_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    on = replace(
        _CHAT,
        endpoint="/gemini/v1beta1/publishers/google/models/a-model:generateContent",
        provider="gemini-vertex",
    )

    with _serving(_scripted(calls, _PASSED, _PASSED)) as url:
        _exchange(_make(monkeypatch, url), _request(on), on.reply(False))

    # the sidecar reads it by its provider, or answers that it can't, which
    # Middleman's own setting for a failed hook then decides
    for call in calls:
        assert "format" not in call["params"]
        assert call["params"]["provider"] == "gemini-vertex"


def test_a_request_whose_inspect_header_cant_be_read_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # anyone can write the header, so it can't be a way past the sentinel. A
    # refusal is never let through, whatever Middleman does with a failure.
    headers = {"x-inspect-context": "not json", "x-irid": "request-1"}

    with _sidecar("Oslo", tmp_path) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(
                _make(monkeypatch, url),
                _request(_CHAT, headers),
                _CHAT.reply(False),
            )

    assert refused.value.status_code == 400
    assert "x-inspect-context" in str(refused.value.detail)


def test_request_and_reply_are_replaced_as_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {**_CHAT.body, "messages": [{"role": "user", "content": "Hello."}]}
    reply = b'{"replaced": true}'
    answers = (
        _result({"action": "replace", "body": body}),
        _result({"action": "replace", "body": base64.b64encode(reply).decode()}),
    )

    with _serving(_scripted([], *answers)) as url:
        sent = _exchange(_make(monkeypatch, url), _request(_CHAT), _CHAT.reply(False))

    assert sent == (body, reply)


_FIELD: dict[str, Any] = {"version": 1, "outcome": "reject", "steps": []}


@pytest.mark.parametrize("phase", ["request", "reply"])
@pytest.mark.parametrize(
    "refusal", [_REFUSAL, _REFUSAL | {"sentinel": _FIELD}], ids=["plain", "field"]
)
def test_refusal_is_raised_as_asked(
    monkeypatch: pytest.MonkeyPatch, phase: str, refusal: dict[str, Any]
) -> None:
    answers = [_result(refusal)] if phase == "request" else [_PASSED, _result(refusal)]

    with _serving(_scripted([], *answers)) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url), _request(_CHAT), _CHAT.reply(False))

    # Middleman writes a refusal from its status and message alone
    assert (refused.value.status_code, refused.value.detail) == (403, "Not allowed.")
    assert refused.value.headers is None


@pytest.mark.parametrize("phase", ["request", "reply"])
@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(_Status(500, _PASSED), id="http-error"),
        pytest.param(
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32003, "message": "No."}},
            id="not-judged",
        ),
        pytest.param([], id="not-an-object"),
        pytest.param({"jsonrpc": "2.0", "id": 1}, id="no-result"),
        pytest.param(_result({"action": "allow"}), id="unknown-action"),
        pytest.param(
            # neither a request body nor base64
            _result({"action": "replace", "body": "not base64!"}),
            id="replaced-unusably",
        ),
    ],
)
def test_no_usable_answer_is_a_sidecar_error(
    monkeypatch: pytest.MonkeyPatch, phase: str, answer: Any
) -> None:
    calls: list[Any] = []
    answers = [answer] if phase == "request" else [_PASSED, answer]

    with _serving(_scripted(calls, *answers)) as url:
        with pytest.raises(_load().SidecarError):
            _exchange(_make(monkeypatch, url), _request(_CHAT), _CHAT.reply(False))

    assert len(calls) == len(answers)


def test_sidecar_that_cannot_be_reached_is_a_sidecar_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # nothing listens here
    hook = _make(monkeypatch, "http://127.0.0.1:9")

    with pytest.raises(_load().SidecarError):
        _exchange(hook, _request(_CHAT), _CHAT.reply(False))


def test_refusal_that_cannot_be_read_still_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreadable = _result({"action": "refuse", "status": 200, "message": "No."})

    with _serving(_scripted([], unreadable)) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url), _request(_CHAT), _CHAT.reply(False))

    assert refused.value.status_code == 503
    assert "refusal could not be read" in str(refused.value.detail)


@pytest.mark.parametrize(
    ("url", "headers", "says"),
    [
        (None, None, "URL must say"),
        ("  ", None, "URL must say"),
        ("http://sidecar", "x-hawk-*, not a name", "takes header names"),
        ("http://sidecar", "authorization", "credential"),
        ("http://sidecar", "x-api-*", "credential"),
    ],
)
def test_setting_that_makes_no_sense_stops_the_hook_being_made(
    monkeypatch: pytest.MonkeyPatch, url: str | None, headers: str | None, says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        _make(monkeypatch, url, headers)


def test_middleman_can_load_the_class(monkeypatch: pytest.MonkeyPatch) -> None:
    # as Middleman's loader takes it: made with no arguments, and each method
    # it calls is async
    hook = _make(monkeypatch, "http://sidecar")
    for name in ("on_request", "on_reply", "close"):
        assert inspect.iscoroutinefunction(getattr(hook, name))


def test_close_ends_the_connection_and_can_be_asked_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serving(_scripted([], _PASSED)) as url:
        hook = _make(monkeypatch, url)

        async def run() -> bool:
            await hook.on_request(_request(_CHAT))
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
