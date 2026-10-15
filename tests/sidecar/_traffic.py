import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import AfterToolCall, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Refuse, Reply, Request

DATA = Path(__file__).parent / "data"

STREAM = "text/event-stream; charset=utf-8"
JSON = "application/json"

# Replies recorded from the OpenAI API, each calling get_weather for Paris,
# Tokyo and Lima at once.
OPENAI_RECORDED = {
    "chat": "openai-chat.json",
    "chat stream": "openai-chat.sse",
    "responses": "openai-responses.json",
    "responses stream": "openai-responses.sse",
}


def anthropic_recorded(name: str) -> bytes:
    """A reply stream recorded from the Anthropic API."""
    return (DATA / f"{name}.sse").read_bytes()


def openai_recorded(name: str) -> tuple[str, bytes]:
    """The content type and body of a reply recorded from the OpenAI API."""
    file = OPENAI_RECORDED[name]
    return (STREAM if file.endswith(".sse") else JSON), (DATA / file).read_bytes()


def message(*content: dict[str, Any], stop_reason: str = "tool_use") -> bytes:
    """An Anthropic reply that was not streamed."""
    return json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": list(content),
            "stop_reason": stop_reason,
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


def ran(
    location: str, result: str, id: str = "toolu_1", *, error: bool = False
) -> list[dict[str, Any]]:
    """A weather lookup the model made and the client's result for it: the two Anthropic messages a later request holds."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": id,
        "content": result,
    }
    if error:
        block["is_error"] = True
    return [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Looking."}, weather(location, id)],
        },
        {"role": "user", "content": [block]},
    ]


def request(
    id: str = "x",
    user_text: str = "What is the weather?",
    then: Sequence[dict[str, Any]] = (),
    *,
    sample: str | None = None,
    **fields: Any,
) -> Request:
    """An Anthropic request as a proxy hands it over, its conversation followed by `then`, from the Inspect sample whose uuid is `sample` if given."""
    return Request(
        id=id,
        headers=inspect_headers(sample_uuid=sample) if sample is not None else {},
        provider="anthropic",
        model="claude-test",
        user="user-1",
        body=json.dumps(
            {
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": user_text}, *then],
                **fields,
            }
        ).encode(),
    )


SAMPLE: dict[str, Any] = {
    "version": 1,
    "eval_id": "eval_789",
    "task": "philadelphia",
    "sample_id": 1,
    "epoch": 2,
    "sample_uuid": "sample_123",
    "attempt": "attempt_1",
}


def inspect_headers(**changes: Any) -> dict[str, str]:
    """The headers an Inspect eval sends with each model request, for `SAMPLE` with `changes`."""
    return {
        "x-inspect-context": json.dumps(SAMPLE | changes),
        "x-irid": "request_456",
    }


def plain(answer: object) -> object:
    """An answer without the `sentinel` field, to compare what any client is told."""
    return replace(answer, sentinel=None) if isinstance(answer, Refuse) else answer


def reply(
    body: bytes, *, id: str = "x", content_type: str = JSON, status: int = 200
) -> Reply:
    """An Anthropic reply as a proxy hands it over."""
    return Reply(
        id=id,
        provider="anthropic",
        model="claude-test",
        user="user-1",
        status=status,
        content_type=content_type,
        body=body,
    )


@protocol
def weather_rule(action: str = "reject", city: str = "Tokyo") -> Protocol:
    """Decide `action` for a weather lookup of `city`, and let everything else go."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") != city:
            return Decision.proceed()
        if action == "reject":
            return Decision.reject(
                f"lookup of {city}", message=f"{city} is not allowed."
            )
        if action == "terminate":
            return Decision.terminate(f"lookup of {city}")
        if action == "escalate":
            return Decision.escalate(f"lookup of {city}")
        if action == "modify":
            return Decision(action="modify", modified=step.call)
        raise RuntimeError("the rule broke")

    return decide


@protocol
def storm_rule(action: str = "terminate") -> Protocol:
    """Decide `action` for a result that reports a storm, and let everything else go."""

    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        if "storm" not in step.result.text:
            return Decision.proceed()
        if action == "terminate":
            return Decision.terminate("a storm")
        if action == "escalate":
            return Decision.escalate("a storm")
        if action == "reject":
            return Decision.reject("a storm")
        raise RuntimeError("the rule broke")

    return decide


def openai_request(api: str, then: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """An OpenAI request body for `chat` or `responses`, its conversation followed by `then`."""
    opening = {"role": "user", "content": "What is the weather?"}
    if api == "chat":
        return {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                opening,
                *then,
            ],
        }
    return {
        "model": "gpt-test",
        "instructions": "You are helpful.",
        "input": [opening, *then],
        "store": False,
    }


def openai_ran(api: str, result: str) -> list[dict[str, Any]]:
    """A weather lookup the model made and the client's result for it, as a later OpenAI request holds them."""
    arguments = json.dumps({"location": "Oslo"})
    if api == "chat":
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": arguments},
        }
        return [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": result},
        ]
    return [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": arguments,
        },
        {"type": "function_call_output", "call_id": "call_1", "output": result},
    ]
