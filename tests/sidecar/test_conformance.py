"""The sidecar's work against Inspect's own process.

Each test runs a real Inspect eval twice against a stand-in provider. Once with the sentinel in the eval's own process, and once without it, with the provider's replies first put through a sidecar, as a model proxy would, and the eval's requests after them too. The request the model is sent after a turn with a call rejected or changed must be the same either way.
"""

import asyncio
import json
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, tool
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import Response
from starlette.routing import Route

from inspect_sentinel import BeforeToolCall, Context, Decision, Protocol, protocol
from inspect_sentinel.sidecar import Handler, Pass, Replace, Reply, Request
from inspect_sentinel.sidecar._anthropic import settled
from tests.sidecar._traffic import JSON, STREAM

_ASKED = "What is the weather in Tokyo and Paris?"


@tool
def get_weather() -> Tool:
    async def execute(location: str) -> str:
        """Get the weather in a city.

        Args:
            location: The city.
        """
        return f"Sunny in {location}."

    return execute


@protocol
def tokyo_rule(action: str) -> Protocol:
    """Reject a lookup of Tokyo, or change it to Oslo."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") != "Tokyo":
            return Decision.proceed()
        if action == "reject":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        oslo = replace(step.call, arguments={"location": "Oslo"})
        return Decision(action="modify", modified=oslo)

    return decide


def _anthropic_stream(message: dict[str, Any]) -> bytes:
    # a reply as the Messages API streams it, which Inspect's provider asks for
    def event(data: dict[str, Any]) -> str:
        return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"

    start: dict[str, Any] = {**message, "content": [], "stop_reason": None}
    written = [event({"type": "message_start", "message": start})]
    for index, block in enumerate(message["content"]):
        if block["type"] == "tool_use":
            empty: dict[str, Any] = {**block, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        else:
            empty = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        written.append(
            event(
                {"type": "content_block_start", "index": index, "content_block": empty}
            )
        )
        written.append(
            event({"type": "content_block_delta", "index": index, "delta": delta})
        )
        written.append(event({"type": "content_block_stop", "index": index}))
    stop = {"stop_reason": message["stop_reason"], "stop_sequence": None}
    written.append(
        event({"type": "message_delta", "delta": stop, "usage": {"output_tokens": 1}})
    )
    written.append(event({"type": "message_stop"}))
    return "".join(written).encode()


def _anthropic(calls: list[tuple[str, str]] | None) -> bytes:
    content: list[dict[str, Any]] = (
        [
            {
                "type": "tool_use",
                "id": id,
                "name": "get_weather",
                "input": {"location": city},
            }
            for id, city in calls
        ]
        if calls
        else [{"type": "text", "text": "Done."}]
    )
    return _anthropic_stream(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _chat(calls: list[tuple[str, str]] | None) -> bytes:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None if calls else "Done.",
    }
    if calls:
        message["tool_calls"] = [
            {
                "id": id,
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": city}),
                },
            }
            for id, city in calls
        ]
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def _responses(calls: list[tuple[str, str]] | None) -> bytes:
    output: list[dict[str, Any]] = (
        [
            {
                "type": "function_call",
                "id": f"fc_{id}",
                "call_id": id,
                "name": "get_weather",
                "arguments": json.dumps({"location": city}),
                "status": "completed",
            }
            for id, city in calls
        ]
        if calls
        else [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []}
                ],
            }
        ]
    )
    return json.dumps(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-test",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    ).encode()


@dataclass(frozen=True)
class _Api:
    """One API Inspect's providers speak: how Inspect is pointed at it, and how the stand-in replies."""

    model: str
    model_args: dict[str, Any]
    provider: str
    content_type: str
    path: str
    reply: Callable[[list[tuple[str, str]] | None], bytes]
    # what Inspect changes between requests while the conversation stays the
    # same: the Anthropic provider moves its cache marks to the latest messages
    settle: Callable[[dict[str, Any]], dict[str, Any]] = lambda body: body


def _settled(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "messages": [settled(entry) for entry in body["messages"]]}


_APIS = {
    "anthropic": _Api(
        "anthropic/claude-test",
        {},
        "anthropic",
        STREAM,
        "/v1/messages",
        _anthropic,
        _settled,
    ),
    "openai chat": _Api(
        "openai/gpt-test",
        {"responses_api": False},
        "openai",
        JSON,
        "/v1/chat/completions",
        _chat,
    ),
    "openai responses": _Api(
        "openai/gpt-test",
        {"responses_api": True},
        "openai",
        JSON,
        "/v1/responses",
        _responses,
    ),
}


@contextmanager
def _provider(api: _Api, replies: list[bytes]) -> Generator[tuple[str, list[bytes]]]:
    """A stand-in provider that gives these replies in turn, and keeps each request it was sent."""
    asked: list[bytes] = []
    left = list(replies)

    async def answer(http: HttpRequest) -> Response:
        asked.append(await http.body())
        return Response(left.pop(0), media_type=api.content_type)

    app = Starlette(routes=[Route(api.path, answer, methods=["POST"])])
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", asked
    finally:
        server.should_exit = True
        serving.join()


def _run(
    api: _Api, replies: list[bytes], log_dir: Path, sentinel: Protocol | None = None
) -> list[bytes]:
    """Run an eval whose model looks up the weather, and give back what it sent the provider."""
    with _provider(api, replies) as (url, asked):
        log = eval(
            Task(
                dataset=[Sample(input=_ASKED)],
                solver=[use_tools(get_weather()), generate()],
                sentinel=sentinel,
            ),
            model=api.model,
            model_args=api.model_args,
            model_base_url=url if api.provider == "anthropic" else f"{url}/v1",
            display="none",
            log_dir=str(log_dir),
        )[0]
    assert log.status == "success", log.error
    return asked


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stand-in")
    monkeypatch.setenv("OPENAI_API_KEY", "stand-in")


@pytest.mark.parametrize("name", _APIS)
@pytest.mark.parametrize("action", ["reject", "modify"])
def test_the_model_is_sent_what_inspect_sends_in_its_own_process(
    name: str, action: str, tmp_path: Path
) -> None:
    api = _APIS[name]
    turn = api.reply([("call_1", "Tokyo"), ("call_2", "Paris")])
    done = api.reply(None)

    # the sentinel in the eval's own process
    in_process = _run(api, [turn, done], tmp_path, tokyo_rule(action))

    # the sentinel in a sidecar, beside an eval that knows nothing of it
    handler = Handler(tokyo_rule(action))

    def judged(id: str, body: bytes) -> Any:
        return asyncio.run(
            handler.request(
                Request(id=id, provider=api.provider, model="m", user=None, body=body)
            )
        )

    assert judged("1", in_process[0]) == Pass()
    edited = asyncio.run(
        handler.reply(
            Reply(
                id="1",
                provider=api.provider,
                model="m",
                user=None,
                status=200,
                content_type=api.content_type,
                body=turn,
            )
        )
    )
    assert isinstance(edited, Replace)
    beside = _run(api, [edited.body, done], tmp_path)
    amended = judged("2", beside[1])

    assert isinstance(amended, Replace)
    assert api.settle(json.loads(amended.body)) == api.settle(json.loads(in_process[1]))
