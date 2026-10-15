"""The guardrail, handed LiteLLM's own classes as LiteLLM's proxy hands them over.

These tests run in an environment of their own, with one LiteLLM version, and import nothing of inspect_sentinel, as the guardrail doesn't (`make test-litellm`). A test that needs a real sidecar starts one from the repository's own environment.
"""

import asyncio
import base64
import importlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest
import uvicorn
from fastapi import HTTPException
from openai.types.responses import Response, ResponseReasoningItem
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp

_PACKAGE = Path(__file__).parents[1]
_REPOSITORY = Path(__file__).parents[4]
# the replies the sidecar's own tests replay
DATA = _REPOSITORY / "tests/sidecar/data"
# the one thing the guardrail and the sidecar share: the description of the
# methods and their results
_SCHEMA: dict[str, Any] = json.loads(
    (
        _REPOSITORY / "src/inspect_sentinel/sidecar/proxy_endpoint.schema.json"
    ).read_text()
)
_METHODS: dict[str, Any] = _SCHEMA["methods"]
PATH = "/proxy/v1"

# LiteLLM's own classes, which it hands a guardrail on its OpenAI routes
_litellm: Any = importlib.import_module("litellm.types.utils")
_openai: Any = importlib.import_module("litellm.types.llms.openai")
_USER: Any = importlib.import_module("litellm.proxy._types").UserAPIKeyAuth(
    user_id="user-1"
)

_QUESTION = {"role": "user", "content": "What is the weather?"}
_REFUSAL = {"action": "refuse", "status": 403, "message": "Not allowed."}
_UNANSWERED = (503, "The sentinel sidecar gave no usable answer.")


def stream_events(body: bytes) -> list[dict[str, Any]]:
    """The JSON object of each event in a stream."""
    return [
        json.loads(line[5:].strip())
        for line in body.decode().splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]


def anthropic_recorded(name: str) -> bytes:
    """A reply stream recorded from the Anthropic API."""
    return (DATA / f"{name}.sse").read_bytes()


def message(*content: dict[str, Any]) -> bytes:
    """An Anthropic reply that was not streamed."""
    return json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": list(content),
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()


def weather(location: str, id: str = "toolu_1") -> dict[str, Any]:
    """An Anthropic tool call to get_weather."""
    return {
        "type": "tool_use",
        "id": id,
        "name": "get_weather",
        "input": {"location": location},
    }


def _recorded_events(name: str) -> list[dict[str, Any]]:
    return stream_events((DATA / name).read_bytes())


def _response_event(event: dict[str, Any]) -> Any:
    # LiteLLM has a class for each kind of event in a Responses stream
    for name in dir(_openai):
        value: Any = getattr(_openai, name)
        fields = getattr(value, "model_fields", None)
        if isinstance(value, type) and fields and "type" in fields:
            kinds = getattr(fields["type"].annotation, "__args__", ())
            if event["type"] in [getattr(kind, "value", kind) for kind in kinds]:
                return value(**event)
    raise AssertionError(f"LiteLLM has no class for a {event['type']} event.")


def _notes() -> dict[str, Any]:
    return {"additional_headers": {"x-litellm-model-group": "a-model"}}


def _messages_whole() -> Any:
    return json.loads(message(weather("Paris"))) | {"_hidden_params": _notes()}


def _messages_stream() -> list[Any]:
    # LiteLLM passes on a provider's bytes in whatever pieces they came in
    sent = anthropic_recorded("single-tool-call")
    return [sent[:100], sent[100:]]


def _chat_whole() -> Any:
    reply = _litellm.ModelResponse(
        **json.loads((DATA / "openai-chat.json").read_text())
    )
    reply._hidden_params = _notes()
    return reply


def _chat_stream() -> list[Any]:
    return [
        _litellm.ModelResponseStream(**event)
        for event in _recorded_events("openai-chat.sse")
    ]


def _responses_whole() -> Any:
    reply = _openai.ResponsesAPIResponse(
        **json.loads((DATA / "openai-responses.json").read_text())
    )
    reply._hidden_params = _notes()
    return reply


def _responses_stream() -> list[Any]:
    return [
        _response_event(event) for event in _recorded_events("openai-responses.sse")
    ]


@dataclass(frozen=True)
class _On:
    """One of LiteLLM's routes: what it calls the route, what a request to it holds, and the reply it hands a guardrail. Every reply here looks up the weather in Paris."""

    call_type: str
    provider: str
    format: str
    body: dict[str, Any]
    whole: Callable[[], Any]
    stream: Callable[[], list[Any]]
    # the field LiteLLM keeps its own notes on the caller in, on this route
    notes: str = "litellm_metadata"


_MESSAGES = _On(
    "anthropic_messages",
    "anthropic",
    "anthropic-messages",
    {"model": "a-model", "max_tokens": 100, "messages": [_QUESTION]},
    _messages_whole,
    _messages_stream,
)
_CHAT = _On(
    "acompletion",
    "openai",
    "openai-chat",
    {"model": "a-model", "max_tokens": 100, "messages": [_QUESTION]},
    _chat_whole,
    _chat_stream,
    notes="metadata",
)
_RESPONSES = _On(
    "aresponses",
    "openai",
    "openai-responses",
    {"model": "a-model", "max_output_tokens": 100, "input": [_QUESTION]},
    _responses_whole,
    _responses_stream,
)
_ROUTES = [
    pytest.param(
        route, streamed, id=f"{route.format}{', streamed' if streamed else ''}"
    )
    for route in (_MESSAGES, _CHAT, _RESPONSES)
    for streamed in (False, True)
]


def _data(on: _On, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """A request as LiteLLM hands it to a guardrail: its own fields among LiteLLM's working state."""
    notes = {"user_api_key_hash": "a-hash", "headers": headers or {}}
    return {
        **on.body,
        on.notes: notes,
        "litellm_call_id": "call-1",
        "litellm_logging_obj": object(),
        "secret_fields": object(),
        "proxy_server_request": {
            "url": "http://litellm/",
            "method": "POST",
            "headers": headers or {},
            "body": {**on.body, on.notes: notes},
        },
    }


def _load(monkeypatch: pytest.MonkeyPatch) -> Any:
    # as LiteLLM has it: installed, and importable by its own name
    return importlib.import_module("inspect_sentinel_litellm")


def _make(monkeypatch: pytest.MonkeyPatch, url: str | None, **settings: Any) -> Any:
    """The guardrail as LiteLLM makes it: its own settings for a guardrail, and those from the config."""
    told: dict[str, Any] = {
        "guardrail_name": "sentinel",
        "event_hook": ["pre_call", "post_call"],
        "default_on": True,
        "unreachable_fallback": "fail_closed",
        "api_base": url,
    }
    return _load(monkeypatch).LiteLLMGuardrail(**(told | settings))


async def _chunks(held: list[Any]) -> AsyncIterator[Any]:
    for chunk in held:
        yield chunk


def _exchange(
    guardrail: Any,
    on: _On,
    data: dict[str, Any],
    streamed: bool,
    out: list[Any] | None = None,
) -> tuple[Any, Any]:
    """Put a request and its reply through the guardrail as LiteLLM does. A streamed reply's chunks are also added to `out` as they are let go."""
    let_go: list[Any] = [] if out is None else out

    async def run() -> tuple[Any, Any]:
        try:
            sent = await guardrail.async_pre_call_hook(_USER, None, data, on.call_type)
            if not streamed:
                return sent, await guardrail.async_post_call_success_hook(
                    data, _USER, on.whole()
                )
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                _USER, _chunks(on.stream()), data
            ):
                let_go.append(chunk)
            return sent, let_go
        finally:
            await guardrail.aclose()

    return asyncio.run(run())


@contextmanager
def _serving(app: ASGIApp) -> Generator[str]:
    """The address of a real server for an app, since the guardrail makes real HTTP calls."""
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
from inspect_sentinel import BeforeToolCall, Context, Decision, protocol


@protocol
def weather_rule():
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == {city!r}:
            return Decision.reject("a lookup", message={message!r})
        return Decision.proceed()

    return decide
"""


def _sidecar_python() -> Path:
    # the repository's own environment, which has the sidecar and no LiteLLM
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


def _scripted(calls: list[Any], *answers: Any, status: int = 200) -> Starlette:
    """A stand-in for the sidecar that keeps the calls made to it and gives these answers in turn."""
    left = list(answers)

    async def call(http: HttpRequest) -> JSONResponse:
        calls.append(await http.json())
        return JSONResponse(left.pop(0), status_code=status)

    return Starlette(routes=[Route(PATH, call, methods=["POST"])])


def _fits(part: dict[str, Any], value: Any) -> None:
    jsonschema.validate(value, {**_SCHEMA, **part}, cls=jsonschema.Draft202012Validator)


def _written(reply: Any) -> Any:
    """A reply, or a chunk of one, as LiteLLM writes it for the caller."""
    return json.loads(reply.model_dump_json(exclude_none=True, exclude_unset=True))


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_exchange_the_sentinel_allows_goes_on_as_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: _On, streamed: bool
) -> None:
    data = _data(on)
    before = dict(data)

    with _sidecar("Oslo", tmp_path) as url:
        sent, reply = _exchange(_make(monkeypatch, url), on, data, streamed)

    assert sent is data
    assert {name: data[name] for name in before} == before
    if not streamed:
        # LiteLLM sends on the reply it has
        assert reply is None
    elif on is _MESSAGES:
        assert reply == on.stream()
    else:
        assert [_written(chunk) for chunk in reply] == [
            _written(chunk) for chunk in on.stream()
        ]


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_reply_the_sentinel_rejects_is_refused_in_its_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: _On, streamed: bool
) -> None:
    let_go: list[Any] = []
    # the recorded Responses stream is the one reply that names the country too
    city = "Paris, France" if on is _RESPONSES and streamed else "Paris"

    with _sidecar(city, tmp_path) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url), on, _data(on), streamed, let_go)

    assert refused.value.status_code == 400
    assert refused.value.detail == (
        f"A sentinel rejected the call to get_weather: {city} is not allowed."
    )
    # none of a streamed reply had gone out
    assert let_go == []


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_calls_the_guardrail_makes_fit_the_schema_file(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    calls: list[Any] = []
    headers = {
        "X-Inspect-Context": '{"version": 1}',
        "x-irid": "request-1",
        "x-inspect-sample-uuid": "sample-1",
        "x-hawk-job-type": "eval-set",
        "user-agent": "a client",
        "x-api-key": "a provider's key",
    }
    passed = _result({"action": "pass"})

    with _serving(_scripted(calls, passed, passed)) as url:
        guardrail = _make(monkeypatch, url, sidecar_headers=["x-hawk-*"])
        _exchange(guardrail, on, _data(on, headers), streamed)

    asked, held = calls
    assert (asked["jsonrpc"], asked["method"]) == ("2.0", "judge_request")
    assert (held["jsonrpc"], held["method"]) == ("2.0", "judge_reply")
    request, reply = asked["params"], held["params"]
    _fits(_METHODS["judge_request"]["params"], request)
    _fits(_METHODS["judge_reply"]["params"], reply)
    told = {
        "id": "call-1",
        "provider": on.provider,
        "format": on.format,
        "model": "a-model",
        "user": "user-1",
    }
    assert {name: request[name] for name in told} == told
    assert {name: reply[name] for name in told} == told
    # LiteLLM names no runs of its own: the sidecar finds the run in
    # Inspect's header
    assert "run" not in request
    # Inspect's own, unread, and the one that is listed
    assert request["headers"] == {
        "x-inspect-context": '{"version": 1}',
        "x-irid": "request-1",
        "x-hawk-job-type": "eval-set",
    }
    # the request's own fields, and none of LiteLLM's notes on the caller
    assert request["body"] == on.body
    body = base64.b64decode(reply["body"])
    assert reply["status"] == 200
    if not streamed:
        assert reply["content_type"] == "application/json"
        whole = on.whole()
        assert json.loads(body) == (
            json.loads(message(weather("Paris")))
            if on is _MESSAGES
            else json.loads(whole.model_dump_json())
        )
    else:
        assert reply["content_type"] == "text/event-stream"
        if on is _MESSAGES:
            assert body == anthropic_recorded("single-tool-call")
        else:
            assert list(stream_events(body)) == [
                _written(chunk) for chunk in on.stream()
            ]


def test_request_is_put_as_it_will_go_to_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    data = _data(_CHAT)
    # an earlier guardrail changed the request after LiteLLM recorded it
    data["messages"] = [{"role": "user", "content": "What is the weather, [NAME]?"}]

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url)
        asyncio.run(guardrail.async_pre_call_hook(_USER, None, data, "acompletion"))

    assert calls[0]["params"]["body"]["messages"] == data["messages"]


def test_request_is_replaced_in_place_and_refused_as_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = {
        "model": "a-model",
        "messages": [_QUESTION, {"role": "user", "content": "A call was rejected."}],
        "temperature": 0,
    }
    replace = {"action": "replace", "body": replacement}
    _fits(_METHODS["judge_request"]["result"], replace)
    _fits(_METHODS["judge_request"]["result"], _REFUSAL)
    data = _data(_CHAT)
    kept = {name: data[name] for name in data if name not in _CHAT.body}

    async def run(guardrail: Any) -> Any:
        try:
            sent = await guardrail.async_pre_call_hook(_USER, None, data, "acompletion")
            with pytest.raises(HTTPException) as refused:
                await guardrail.async_pre_call_hook(
                    _USER, None, _data(_CHAT), "acompletion"
                )
            return sent, refused
        finally:
            await guardrail.aclose()

    with _serving(_scripted([], _result(replace), _result(_REFUSAL))) as url:
        sent, refused = asyncio.run(run(_make(monkeypatch, url)))

    assert sent is data
    # a field that was changed, one that was added and one that was left out
    assert {name: data[name] for name in replacement} == replacement
    assert "max_tokens" not in data
    # LiteLLM's own state is as it was
    assert {name: data[name] for name in kept} == kept
    assert (refused.value.status_code, refused.value.detail) == (403, "Not allowed.")


def _changed(on: _On, streamed: bool) -> tuple[bytes, Callable[[Any], None]]:
    """A reply body to replace the route's with, and a check that what the guardrail hands LiteLLM is that reply in the form LiteLLM had it."""
    if on is _MESSAGES and streamed:
        body = b"event: ping\ndata: {}\n\n"
        return body, lambda reply: _same(reply, [body])
    if on is _MESSAGES:
        whole = json.loads(message(weather("Oslo")))
        return json.dumps(whole).encode(), lambda reply: _same(
            reply, whole | {"_hidden_params": _notes()}
        )
    if streamed:
        # the first and the last event of the reply
        events = [_written(chunk) for chunk in on.stream()]
        ends = [events[0], events[-1]]
        kinds: list[Any] = [type(chunk) for chunk in on.stream()]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in ends)

        def check(reply: Any) -> None:
            assert [_written(chunk) for chunk in reply] == ends
            assert [type(chunk) for chunk in reply] == [kinds[0], kinds[-1]]

        return (body + "data: [DONE]\n\n").encode(), check
    whole = json.loads(on.whole().model_dump_json()) | {"id": "another-id"}

    def check_whole(reply: Any) -> None:
        assert type(reply) is type(on.whole())
        assert json.loads(reply.model_dump_json()) == whole
        # LiteLLM's own notes on the call go with the replacement
        assert reply._hidden_params == _notes()

    return json.dumps(whole).encode(), check_whole


def _same(reply: Any, expected: Any) -> None:
    assert reply == expected


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_reply_is_replaced_in_the_form_litellm_had_it(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    body, check = _changed(on, streamed)
    replace = {"action": "replace", "body": base64.b64encode(body).decode()}
    _fits(_METHODS["judge_reply"]["result"], replace)

    with _serving(_scripted([], _result({"action": "pass"}), _result(replace))) as url:
        _, reply = _exchange(_make(monkeypatch, url), on, _data(on), streamed)

    check(reply)


@pytest.mark.parametrize(("on", "streamed"), _ROUTES)
def test_reply_is_refused_as_asked(
    monkeypatch: pytest.MonkeyPatch, on: _On, streamed: bool
) -> None:
    let_go: list[Any] = []
    answers = (_result({"action": "pass"}), _result(_REFUSAL))

    with _serving(_scripted([], *answers)) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url), on, _data(on), streamed, let_go)

    assert (refused.value.status_code, refused.value.detail) == (403, "Not allowed.")
    assert let_go == []


_NOT_BASE64 = {"action": "replace", "body": "not base64!"}
_NOT_A_REPLY = {"action": "replace", "body": base64.b64encode(b"[]").decode()}


@pytest.mark.parametrize(
    ("phase", "answer", "status"),
    [
        pytest.param("request", _result({"action": "allow"}), 200, id="no such action"),
        pytest.param("request", ["pass"], 200, id="not an object"),
        pytest.param("request", {"jsonrpc": "2.0", "id": 1}, 200, id="no result"),
        pytest.param(
            "request",
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32003, "message": "No."}},
            200,
            id="request not judged",
        ),
        pytest.param("request", _result({"action": "pass"}), 500, id="a server error"),
        pytest.param(
            "request",
            _result({"action": "replace", "body": "text"}),
            200,
            id="a replaced request not JSON",
        ),
        pytest.param(
            "request",
            _result({"action": "replace", "body": {"litellm_call_id": "another"}}),
            200,
            id="a replaced request that sets LiteLLM's own state",
        ),
        pytest.param(
            "reply",
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32005, "message": "No."}},
            200,
            id="reply not judged",
        ),
        pytest.param(
            "reply", _result(_NOT_BASE64), 200, id="a replaced reply not base64"
        ),
        pytest.param(
            "reply",
            _result(_NOT_A_REPLY),
            200,
            id="a replaced reply LiteLLM can't take",
        ),
    ],
)
@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_no_usable_answer_refuses_or_lets_through_as_litellm_is_set(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    answer: Any,
    status: int,
    fallback: str,
    streamed: bool,
) -> None:
    # a replaced stream on the Messages route is bytes, which are taken as they are
    on = _CHAT if answer == _result(_NOT_A_REPLY) else _MESSAGES
    calls: list[Any] = []
    passed = _result({"action": "pass"})
    answers = [answer] if phase == "request" else [passed, answer]
    data = _data(on)
    before = dict(data)

    with _serving(_scripted(calls, *answers, status=status)) as url:
        guardrail = _make(monkeypatch, url, unreachable_fallback=fallback)
        if fallback == "fail_closed":
            with pytest.raises(HTTPException) as refused:
                _exchange(guardrail, on, data, streamed)
            assert (refused.value.status_code, refused.value.detail) == _UNANSWERED
            return
        sent, reply = _exchange(guardrail, on, data, streamed)

    # it goes on as it was
    assert sent is data
    assert {name: data[name] for name in before} == before
    if streamed and on is _MESSAGES:
        assert reply == on.stream()
    elif streamed:
        assert [_written(chunk) for chunk in reply] == [
            _written(chunk) for chunk in on.stream()
        ]
    else:
        assert reply is None
    # the sidecar is not asked about the reply to a request it gave no answer on
    assert len(calls) == (1 if phase == "request" else 2)


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_sidecar_that_cannot_be_reached_refuses_or_lets_through(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    # nothing listens here
    guardrail = _make(monkeypatch, "http://127.0.0.1:9", unreachable_fallback=fallback)
    data = _data(_CHAT)

    if fallback == "fail_closed":
        with pytest.raises(HTTPException) as refused:
            _exchange(guardrail, _CHAT, data, False)
        assert (refused.value.status_code, refused.value.detail) == _UNANSWERED
    else:
        assert _exchange(guardrail, _CHAT, data, False) == (data, None)


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_refusal_that_cannot_be_read_still_refuses(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    unreadable = _result({"action": "refuse", "status": 200, "message": "No."})

    with _serving(_scripted([], unreadable)) as url:
        guardrail = _make(monkeypatch, url, unreachable_fallback=fallback)
        with pytest.raises(HTTPException) as refused:
            _exchange(guardrail, _CHAT, _data(_CHAT), False)

    assert refused.value.status_code == 503
    assert "refusal could not be read" in str(refused.value.detail)


@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_reply_too_large_to_hold_is_not_put_to_the_sidecar(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    calls: list[Any] = []
    let_go: list[Any] = []
    chunks = _MESSAGES.stream()
    settings = {"unreachable_fallback": fallback, "max_reply_bytes": len(chunks[0])}

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url, **settings)
        if fallback == "fail_closed":
            with pytest.raises(HTTPException) as refused:
                _exchange(guardrail, _MESSAGES, _data(_MESSAGES), True, let_go)
            assert (refused.value.status_code, refused.value.detail) == _UNANSWERED
        else:
            _exchange(guardrail, _MESSAGES, _data(_MESSAGES), True, let_go)

    # only the request was put, and the reply went out whole or not at all
    assert len(calls) == 1
    assert let_go == ([] if fallback == "fail_closed" else chunks)


def test_request_on_a_route_the_guardrail_does_not_read_goes_on_unasked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    embedding = _On(
        "aembedding",
        "openai",
        "",
        {"model": "a-model", "input": "hello"},
        _chat_whole,
        _chat_stream,
    )
    data = _data(embedding)

    with _serving(_scripted(calls)) as url:
        guardrail = _make(monkeypatch, url)
        whole = _exchange(guardrail, embedding, data, False)
        streamed = _exchange(guardrail, embedding, data, True)

    assert calls == []
    assert whole == (data, None)
    assert [_written(chunk) for chunk in streamed[1]] == [
        _written(chunk) for chunk in _chat_stream()
    ]


def test_sidecar_that_takes_longer_than_the_time_set_gives_no_usable_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow(http: HttpRequest) -> JSONResponse:
        await asyncio.sleep(1)
        return JSONResponse(_result({"action": "pass"}))

    app = Starlette(routes=[Route(PATH, slow, methods=["POST"])])
    with _serving(app) as url:
        with pytest.raises(HTTPException) as refused:
            _exchange(_make(monkeypatch, url, timeout=0.05), _CHAT, _data(_CHAT), False)
        unhurried = _make(monkeypatch, url)

    assert (refused.value.status_code, refused.value.detail) == _UNANSWERED
    # no limit unless one is set: a sentinel may ask a person
    assert unhurried._timeout.total is None


@pytest.mark.parametrize(
    ("settings", "says"),
    [
        pytest.param({"api_base": None}, "api_base", id="no address"),
        pytest.param({"unreachable_fallback": "maybe"}, "fail_closed", id="fallback"),
        pytest.param({"event_hook": "pre_call"}, "post_call", id="shown no replies"),
        pytest.param({"event_hook": ["post_call"]}, "pre_call", id="shown no requests"),
        pytest.param({"sidecar_headers": ["a b"]}, "'a b'", id="not a header name"),
        pytest.param(
            {"sidecar_headers": "x-irid, Authorization"},
            "credential",
            id="a credential",
        ),
        pytest.param(
            {"sidecar_headers": ["x-*"]}, "credential", id="a prefix over a credential"
        ),
    ],
)
def test_setting_that_makes_no_sense_stops_the_guardrail_being_made(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, Any], says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        _make(monkeypatch, "http://sidecar", **settings)


def test_litellm_finds_the_stream_method_on_the_class_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LiteLLM looks in the class's own namespace, so an inherited one is not called
    guardrail = _load(monkeypatch).LiteLLMGuardrail
    assert "async_post_call_streaming_iterator_hook" in vars(guardrail)


def test_aclose_ends_the_connection_and_can_be_asked_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serving(_scripted([], _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url)

        async def run() -> bool:
            await guardrail.async_pre_call_hook(
                _USER, None, _data(_CHAT), "acompletion"
            )
            session = guardrail._session
            await guardrail.aclose()
            await guardrail.aclose()
            return bool(session.closed)

        assert asyncio.run(run())


def test_guardrail_imports_nothing_of_this_package() -> None:
    # it runs in LiteLLM's process, where this package is not installed
    files = list((_PACKAGE / "inspect_sentinel_litellm").glob("*.py"))
    assert files
    for file in files:
        assert not re.search(
            r"^\s*(from|import) inspect_sentinel(\.|\s)", file.read_text(), re.M
        )


# A reply as LiteLLM 1.105 wrote it for a Claude model that was asked through
# the Responses API and thought first. Its `reasoning` item is the one OpenAI's
# types reject: it has no `summary`, and its parts are typed `output_text`.
_THINKING: dict[str, Any] = json.loads(
    (DATA / "litellm-claude-responses.json").read_text()
)
_THOUGHT: dict[str, Any] = _THINKING["output"][0]


def _thinking_whole() -> Any:
    return _openai.ResponsesAPIResponse(**_THINKING)


def _thinking_stream() -> list[Any]:
    # the event that ends a stream holds the whole reply, which is what is read
    return [_response_event({"type": "response.completed", "response": _THINKING})]


_CLAUDE_RESPONSES = _On(
    "aresponses",
    "openai",
    "openai-responses",
    {"model": "a-model", "input": [_QUESTION]},
    _thinking_whole,
    _thinking_stream,
)
_WHOLE_OR_STREAMED = pytest.mark.parametrize(
    "streamed", [False, True], ids=["whole", "streamed"]
)


def test_the_reply_the_tests_use_is_one_openais_types_reject() -> None:
    # the sidecar reads a reply as leniently as OpenAI's SDK does, so it reads
    # this one as it is; OpenAI's types, held strictly, do not
    with pytest.raises(ValidationError):
        Response.model_validate(_THINKING)


@_WHOLE_OR_STREAMED
def test_claude_thinking_asked_for_through_responses_is_judged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, streamed: bool
) -> None:
    with _sidecar("Oslo", tmp_path) as url:
        _, reply = _exchange(
            _make(monkeypatch, url),
            _CLAUDE_RESPONSES,
            _data(_CLAUDE_RESPONSES),
            streamed,
        )

    # the caller is sent the reply as LiteLLM wrote it
    if streamed:
        assert [_written(chunk) for chunk in reply] == [
            _written(chunk) for chunk in _thinking_stream()
        ]
    else:
        assert reply is None


@_WHOLE_OR_STREAMED
def test_sidecar_is_shown_claude_thinking_as_litellm_wrote_it(
    monkeypatch: pytest.MonkeyPatch, streamed: bool
) -> None:
    # the sidecar reads the item as it is, as the test above shows end to end
    calls: list[Any] = []
    passed = _result({"action": "pass"})

    with _serving(_scripted(calls, passed, passed)) as url:
        _exchange(
            _make(monkeypatch, url),
            _CLAUDE_RESPONSES,
            _data(_CLAUDE_RESPONSES),
            streamed,
        )

    shown = calls[1]["params"]
    body = base64.b64decode(shown["body"])
    written = stream_events(body)[-1]["response"] if streamed else json.loads(body)
    thought = written["output"][0]
    # unrepaired: no summary, and its part typed as LiteLLM typed it
    assert not thought.get("summary")
    assert thought["content"] == _THOUGHT["content"]


def _echoes_with_a_note(shown: list[Any]) -> Starlette:
    """A stand-in for the sidecar that keeps the request it is shown, and replaces the request with the same one and a note added to its input."""

    async def call(http: HttpRequest) -> JSONResponse:
        body = (await http.json())["params"]["body"]
        shown.append(body)
        note = {"role": "user", "content": "A call was rejected."}
        return JSONResponse(
            _result(
                {"action": "replace", "body": body | {"input": [*body["input"], note]}}
            )
        )

    return Starlette(routes=[Route(PATH, call, methods=["POST"])])


def test_request_that_sends_litellms_thinking_back_is_read_and_goes_on_as_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a client that hands an item back as it got it. One that OpenAI's types
    # take, as inspect_ai sends, is beside it.
    valid: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "content": [],
        "encrypted_content": "kept",
    }
    sent_input: list[Any] = [_QUESTION, _THOUGHT, valid, *_THINKING["output"][1:]]
    on = _On(
        "aresponses",
        "openai",
        "openai-responses",
        {"model": "a-model", "input": sent_input},
        _thinking_whole,
        _thinking_stream,
    )
    data = _data(on)
    shown: list[Any] = []

    with _serving(_echoes_with_a_note(shown)) as url:
        guardrail = _make(monkeypatch, url)

        async def run() -> None:
            try:
                await guardrail.async_pre_call_hook(_USER, None, data, on.call_type)
            finally:
                await guardrail.aclose()

        asyncio.run(run())

    # OpenAI's own types take the item the sidecar was shown, as they don't the
    # one the client sent
    ResponseReasoningItem.model_validate(shown[0]["input"][1])
    with pytest.raises(ValidationError):
        ResponseReasoningItem.model_validate(_THOUGHT)
    # an item OpenAI's types take is shown as it is
    assert shown[0]["input"][2] == valid
    # LiteLLM is sent each item as the client sent it, and what the sidecar added
    assert data["input"][:-1] == sent_input
    assert data["input"][-1] == {"role": "user", "content": "A call was rejected."}


_INSPECT = {"X-Inspect-Context": '{"version": 1}', "x-irid": "request-1"}


@pytest.mark.parametrize("call_type", ["acompletion", "aembedding"])
@pytest.mark.parametrize("fallback", ["fail_closed", "fail_open"])
def test_inspect_headers_never_reach_the_provider(
    monkeypatch: pytest.MonkeyPatch, call_type: str, fallback: str
) -> None:
    # LiteLLM sends a caller's `x-` headers on to the provider, in `headers`,
    # when its operator says to. Whether or not the sidecar answers, and
    # whether or not the route is read, Inspect's own are not sent on.
    data = _data(_CHAT, _INSPECT)
    data["headers"] = {**_INSPECT, "x-other": "kept"}
    data["extra_headers"] = {"x-irid": "request-1", "x-other": "kept"}
    # nothing listens here
    guardrail = _make(monkeypatch, "http://127.0.0.1:9", unreachable_fallback=fallback)

    async def run() -> None:
        try:
            await guardrail.async_pre_call_hook(_USER, None, data, call_type)
        except HTTPException:
            pass
        finally:
            await guardrail.aclose()

    asyncio.run(run())

    assert data["headers"] == {"x-other": "kept"}
    assert data["extra_headers"] == {"x-other": "kept"}


def test_guardrail_made_outside_litellms_proxy_adds_no_route() -> None:
    # a guardrail made by a test, or by anything but LiteLLM's proxy, loads
    # none of the proxy's server
    made = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from inspect_sentinel_litellm import LiteLLMGuardrail\n"
            "LiteLLMGuardrail(api_base='http://sidecar', guardrail_name='sentinel',"
            " event_hook=['pre_call', 'post_call'])\n"
            "print('litellm.proxy.proxy_server' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert made.stdout.strip() == "False"


def _sidecar_for_evals(seen: list[dict[str, Any]]) -> Starlette:
    """A stand-in for the sidecar's endpoint for evals that says what it was sent."""

    async def answer(http: HttpRequest) -> JSONResponse:
        seen.append(
            {
                "method": http.method,
                "path": http.url.path,
                "query": http.url.query,
                "headers": dict(http.headers),
                "body": (await http.body()).decode(),
            }
        )
        return JSONResponse({"answered": http.url.path}, status_code=201)

    return Starlette(
        routes=[Route("/sentinel/{rest:path}", answer, methods=["GET", "POST"])]
    )


def test_eval_routes_reach_the_sidecar_through_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    routes: Any = importlib.import_module("litellm.proxy._types").LiteLLMRoutes
    seen: list[dict[str, Any]] = []

    async def call(url: str) -> tuple[Any, Any]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://litellm"
        ) as client:
            probe = await client.get(
                "/sentinel/info?x=1", headers={"authorization": "Bearer sk-caller"}
            )
            asked = await client.post(
                "/sentinel/v1",
                content=b'{"jsonrpc": "2.0"}',
                headers={"authorization": "Bearer sk-caller"},
            )
            return probe, asked

    with _serving(_sidecar_for_evals(seen)) as url:
        # LiteLLM's proxy makes its guardrails once its app exists
        guardrail = _make(monkeypatch, url)
        guardrail.serve_eval_routes()
        probe, asked = asyncio.run(call(url))
        asyncio.run(guardrail.aclose())

    # ahead of LiteLLM's own routes, once however often it is asked
    paths = [getattr(route, "path", None) for route in proxy.app.router.routes]
    assert paths[0] == "/sentinel/{rest:path}"
    assert paths.count("/sentinel/{rest:path}") == 1
    assert {"/sentinel/info", "/sentinel/v1"} <= set(routes.openai_routes.value)
    assert (probe.status_code, probe.json()) == (201, {"answered": "/sentinel/info"})
    assert (asked.status_code, asked.json()) == (201, {"answered": "/sentinel/v1"})
    assert [(s["method"], s["path"], s["query"]) for s in seen] == [
        ("GET", "/sentinel/info", "x=1"),
        ("POST", "/sentinel/v1", ""),
    ]
    assert seen[1]["body"] == '{"jsonrpc": "2.0"}'
    # the caller's key was LiteLLM's to check, and the sidecar never sees it
    assert not any("authorization" in s["headers"] for s in seen)


def test_eval_route_answers_502_when_the_sidecar_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    # nothing listens here
    guardrail = _make(monkeypatch, "http://127.0.0.1:9")
    # a route added earlier in this process would pass calls to another sidecar
    proxy.app.router.routes[:] = [
        route
        for route in proxy.app.router.routes
        if getattr(route, "path", None) != "/sentinel/{rest:path}"
    ]
    guardrail.serve_eval_routes()

    async def call() -> Any:
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy.app), base_url="http://litellm"
            ) as client:
                return await client.get("/sentinel/info")
        finally:
            await guardrail.aclose()

    answered = asyncio.run(call())

    assert answered.status_code == 502
    assert answered.json() == {"error": "The sentinel sidecar could not be reached."}


@pytest.mark.parametrize(
    "raw_path",
    [
        b"/sentinel/%2e%2e/proxy/v1",
        b"/sentinel/../proxy/v1",
        b"/sentinel/v1/%2E%2E/%2e%2e/proxy/v1",
        b"/sentinel/./v1",
        b"/sentinel//v1",
    ],
)
def test_eval_route_passes_on_nothing_outside_sentinel(
    monkeypatch: pytest.MonkeyPatch, raw_path: bytes
) -> None:
    proxy: Any = importlib.import_module("litellm.proxy.proxy_server")
    seen: list[dict[str, Any]] = []

    async def asgi_call(app: Any) -> tuple[int, bytes]:
        # straight to the app, so no client tidies the path on the way
        from urllib.parse import unquote

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": unquote(raw_path.decode()),
            "raw_path": raw_path,
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"litellm"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1),
            "server": ("litellm", 80),
        }
        sent: list[dict[str, Any]] = []
        received = False

        async def receive() -> dict[str, Any]:
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        body = b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body"
        )
        return status, body

    with _serving(_sidecar_for_evals(seen)) as url:
        guardrail = _make(monkeypatch, url)
        proxy.app.router.routes[:] = [
            route
            for route in proxy.app.router.routes
            if getattr(route, "path", None) != "/sentinel/{rest:path}"
        ]
        guardrail.serve_eval_routes()
        try:
            status, body = asyncio.run(asgi_call(proxy.app))
        finally:
            asyncio.run(guardrail.aclose())

    assert (status, json.loads(body)) == (404, {"error": "Not found."})
    assert seen == []


def test_credentials_among_a_requests_fields_never_reach_the_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a caller may send LiteLLM a provider's key, or where to call, in the body
    sent_along = {
        "api_key": "sk-a-providers-key",
        "api_base": "https://elsewhere",
        "client_secret": "a-secret",
    }
    on = _On(
        "acompletion",
        "openai",
        "openai-chat",
        {**_CHAT.body, **sent_along},
        _chat_whole,
        _chat_stream,
        notes="metadata",
    )
    data = _data(on)
    calls: list[Any] = []

    with _serving(_scripted(calls, _result({"action": "pass"}))) as url:
        guardrail = _make(monkeypatch, url)

        async def run() -> None:
            try:
                await guardrail.async_pre_call_hook(_USER, None, data, on.call_type)
            finally:
                await guardrail.aclose()

        asyncio.run(run())

    assert calls[0]["params"]["body"] == _CHAT.body
    # LiteLLM still has them, to call the provider with
    assert {name: data[name] for name in sent_along} == sent_along


def test_a_replaced_stream_may_hold_any_line_separator_json_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # U+2028 and U+2029 end a line for str.splitlines, and JSON allows them
    # unescaped inside a string
    events = _recorded_events("openai-chat.sse")
    first = json.loads(json.dumps(events[0]))
    first["choices"][0]["delta"]["content"] = "Rain and wind"
    body = "".join(
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in [first, *events[1:]]
    ).encode()
    assert " ".encode() in body
    replaced = _result({"action": "replace", "body": base64.b64encode(body).decode()})

    with _serving(_scripted([], _result({"action": "pass"}), replaced)) as url:
        _, reply = _exchange(_make(monkeypatch, url), _CHAT, _data(_CHAT), True)

    assert _written(reply[0])["choices"][0]["delta"]["content"] == "Rain and wind"
    assert len(reply) == len(events)
