import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import AfterToolCall, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Reply, Request

DATA = Path(__file__).parent / "data"

STREAM = "text/event-stream; charset=utf-8"
JSON = "application/json"


def recorded(name: str) -> bytes:
    """A reply stream recorded from the Anthropic API."""
    return (DATA / f"{name}.sse").read_bytes()


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
    return {
        "type": "tool_use",
        "id": id,
        "name": "get_weather",
        "input": {"location": location},
    }


def ran(
    location: str, result: str, id: str = "toolu_1", *, error: bool = False
) -> list[dict[str, Any]]:
    """A weather lookup the model made and the client's result for it: the two messages a later request holds."""
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
    run: str | None = None,
) -> Request:
    return Request(
        id=id,
        run=run,
        provider="anthropic",
        model="claude-test",
        user="user-1",
        body=json.dumps(
            {
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": user_text}, *then],
            }
        ).encode(),
    )


def reply(
    body: bytes, *, id: str = "x", content_type: str = JSON, status: int = 200
) -> Reply:
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


@protocol
def reports_the_run() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("run", message=f"{step.conversation}|{context.eval}")

    return decide


@protocol
def reports_the_proxy() -> Protocol:
    """Reject every call, telling the client what the sentinel was given of the request."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        proxy = context.proxy
        assert proxy is not None
        return Decision.reject(
            "proxy",
            message=f"{proxy.provider}|{proxy.model}|{proxy.user}|{sorted(proxy.headers.items())}",
        )

    return decide
