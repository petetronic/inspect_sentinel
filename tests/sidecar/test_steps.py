"""What a sentinel in the sidecar is given at each step, found in the traffic."""

import json
from dataclasses import replace
from typing import Any, cast

import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.tool import Tool, ToolCall, ToolCallContent, ToolCallView, tool

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Record,
    Refuse,
    Request,
    UnreadableError,
)
from inspect_sentinel.sidecar._registration import FileTask
from tests.sidecar._traffic import (
    inspect_headers,
    message,
    ran,
    reply,
    request,
    weather,
)

_TOOLS = [
    {
        "name": "get_weather",
        "description": "Look up the weather.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    },
    # a tool the provider runs
    {"type": "web_search_20250305", "name": "web_search"},
]


@protocol
def at(stage: str, action: str = "terminate") -> Protocol:
    """Decide `action` at every step of one stage."""

    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        if type(step).__name__ != stage:
            return Decision.proceed()
        return Decision(action=cast(Any, action), explanation=stage)

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["terminate", "escalate"])
async def test_request_ended_before_the_model_is_called_is_not_sent(
    action: str,
) -> None:
    handler = Handler(at("BeforeGenerate", action))

    answer = await handler.request(request())

    # an escalate nobody takes ends the run, as at any stage
    assert answer == Refuse(
        400, "A sentinel ended this run before the model was called."
    )
    with pytest.raises(UnreadableError, match="never shown"):
        await handler.reply(reply(message(weather("Oslo"))))


@pytest.mark.anyio
async def test_reply_ended_after_the_model_answered_is_not_passed_on() -> None:
    handler = Handler(at("AfterGenerate"))
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert answer == Refuse(400, "A sentinel ended this run at the model's reply.")
    records = await handler.records("x")
    assert records is not None
    # the run ended before any call in the reply was judged
    assert [record.stage for record in records] == [
        "model_input",
        "model_output",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("stage", ["BeforeGenerate", "AfterGenerate"])
async def test_reject_at_a_generate_stage_is_a_failing_protocol(stage: str) -> None:
    handler = Handler(at(stage, "reject"))

    with pytest.raises(ValueError, match="cannot reject yet"):
        await handler.request(request())
        await handler.reply(reply(message(weather("Oslo"))))


@protocol
def reports_generates() -> Protocol:
    async def before(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed(
            "|".join(
                [
                    step.model,
                    _roles(step.input),
                    str(step.agent_messages),
                    str([tool.name for tool in step.tools]),
                    str(step.tool_choice),
                    str(step.config.max_tokens),
                    str(step.config.temperature),
                ]
            )
        )

    return before


@protocol
def reports_outputs() -> Protocol:
    async def after(
        context: Context, step: AfterGenerate, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed(
            f"{_roles(step.input)}|{[tool.name for tool in step.tools]}|{step.output.message.text}"
        )

    return after


def _roles(messages: list[Any] | None) -> str:
    return str(None if messages is None else [m.role for m in messages])


def _explanations(records: list[Record] | None) -> list[str]:
    assert records is not None
    return [cast(Any, record.detail).report.explanation for record in records]


@pytest.mark.anyio
async def test_generate_stages_are_given_what_the_request_offers() -> None:
    handler = Handler([reports_generates(), reports_outputs()])
    await handler.request(request(tools=_TOOLS, max_tokens=512, temperature=0.5))

    await handler.reply(
        reply(message({"type": "text", "text": "Checking."}, weather("Oslo")))
    )

    explanations = _explanations(await handler.records("x"))
    assert (
        "anthropic/claude-test|['system', 'user']|None"
        "|['get_weather', 'web_search']|auto|512|0.5"
    ) in explanations
    assert (
        "['system', 'user']|['get_weather', 'web_search']|Checking."
    ) in explanations


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, "[]|none"),
        ({"tools": _TOOLS[:1]}, "['get_weather']|auto"),
        (
            {
                "tools": _TOOLS[:1],
                "tool_choice": {"type": "tool", "name": "get_weather"},
            },
            "['get_weather']|ToolFunction(name='get_weather')",
        ),
    ],
)
async def test_tool_choice_is_what_the_model_is_sent(
    fields: dict[str, Any], expected: str
) -> None:
    handler = Handler(reports_generates())

    await handler.request(request(**fields))

    (explanation,) = _explanations(await handler.records("x"))
    assert explanation.split("|")[3:5] == expected.split("|")


@pytest.mark.anyio
async def test_a_tool_that_cannot_be_read_makes_the_request_unreadable() -> None:
    # a step is never told the model is offered fewer tools than it is
    tools = [*_TOOLS[:1], {"type": "future_tool_20990101", "name": "future"}]

    with pytest.raises(UnreadableError, match="anthropic future_tool_20990101"):
        await Handler(reports_generates()).request(request(tools=tools))


@pytest.mark.anyio
async def test_a_tool_choice_that_cannot_be_read_is_unreadable() -> None:
    with pytest.raises(UnreadableError, match="tool choice or settings"):
        await Handler(reports_generates()).request(
            request(tool_choice={"type": "future_choice"})
        )


# the text beside each call in a reply, as `ran` writes it when the client
# sends the reply back
_LOOKING = {"type": "text", "text": "Looking."}


@protocol
def reports_histories() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        call = step.call.id if isinstance(step, BeforeToolCall | AfterToolCall) else "-"
        lists = [step.agent_messages, step.history]
        texts = [None if m is None else [message.text for message in m] for m in lists]
        return Decision.proceed(f"{type(step).__name__}|{call}|{texts[0]}|{texts[1]}")

    return decide


@pytest.mark.anyio
async def test_no_step_is_given_the_agent_messages() -> None:
    # the agent's own message list is not known behind a proxy, only what each
    # request carries, which is a step's `input`
    handler = Handler(reports_histories())
    await handler.request(request(then=ran("Oslo", "rain", "toolu_1")))

    await handler.reply(reply(message(_LOOKING, weather("Lima", "toolu_2"))))

    assert [
        explanation.split("|")[2]
        for explanation in _explanations(await handler.records("x"))
    ] == ["None"] * 4


@pytest.mark.anyio
async def test_each_step_is_given_what_its_conversation_has_sent() -> None:
    # as in Inspect's own process: the history before a generate holds what
    # earlier requests sent, and from the model's reply on, this request too
    handler = Handler(reports_histories())
    await handler.request(request("a"))
    await handler.reply(reply(message(_LOOKING, weather("Oslo", "toolu_1")), id="a"))
    await handler.request(request("b", then=ran("Oslo", "rain", "toolu_1")))
    await handler.reply(reply(message({"type": "text", "text": "Rain."}), id="b"))

    opening = ["You are helpful.", "What is the weather?"]
    histories = [
        explanation.split("|", 3)[::3]
        for explanation in _explanations(await handler.records("a"))
        + _explanations(await handler.records("b"))
    ]
    assert histories == [
        ["BeforeGenerate", "[]"],
        ["AfterGenerate", str(opening)],
        ["BeforeToolCall", str(opening)],
        # the result is judged before the request that holds it is sent
        ["AfterToolCall", str(opening)],
        ["BeforeGenerate", str(opening)],
        ["AfterGenerate", str([*opening, "Looking.", "rain"])],
    ]


@pytest.mark.anyio
async def test_a_reply_is_given_what_its_conversation_sent_while_it_waited() -> None:
    # as the core gives a step what the conversation has sent by the time the
    # step is judged, a request sent before the reply arrives included
    handler = Handler(reports_histories())
    await handler.request(request("a"))
    await handler.request(request("b", then=ran("Oslo", "rain", "toolu_1")))

    await handler.reply(reply(message({"type": "text", "text": "Rain."}), id="a"))

    (after,) = [
        explanation.split("|", 3)[3]
        for explanation in _explanations(await handler.records("a"))
        if explanation.startswith("AfterGenerate")
    ]
    assert after == str(
        ["You are helpful.", "What is the weather?", "Looking.", "rain"]
    )


_SAID: list[dict[str, Any]] = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "Bye"},
]

# a turn with a provider's own search, as a client may send it back: without
# the item's id, for which a reader makes up a new one at each read
_SEARCHED: list[dict[str, Any]] = [
    {"role": "user", "content": "Search."},
    {
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": "weather"},
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Sunny.", "annotations": []}],
    },
    {"role": "user", "content": "Thanks"},
]


# a provider's own search for a tool, sent back without its id, which a
# reader makes up a new one for at each read, as a call's
_TOOL_SEARCHED: list[dict[str, Any]] = [
    {"role": "user", "content": "Find a tool."},
    {
        "type": "tool_search_call",
        "execution": "server",
        "status": "completed",
        "arguments": {"query": "weather"},
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Found.", "annotations": []}],
    },
    {"role": "user", "content": "Thanks"},
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "again"),
    [
        ({"model": "gpt-a", "messages": _SAID}, {"model": "gpt-b"}),
        ({"model": "gpt-a", "input": _SAID}, {"model": "gpt-b"}),
        ({"model": "gpt-a", "input": _SEARCHED, "tools": [{"type": "web_search"}]}, {}),
        (
            {
                "model": "gpt-a",
                "input": _TOOL_SEARCHED,
                "tools": [{"type": "tool_search"}],
            },
            {},
        ),
    ],
    ids=[
        "chat, another model",
        "responses, another model",
        "responses, a search",
        "responses, a search for a tool",
    ],
)
async def test_a_conversation_sent_again_unchanged_adds_nothing_to_its_history(
    body: dict[str, Any], again: dict[str, Any]
) -> None:
    # what a reader fills in from the request, or makes up, is not what the
    # model was sent
    handler = Handler(reports_histories())

    for id, sent in [("a", body), ("b", body | again), ("c", body | again)]:
        await handler.request(
            Request(
                id=id,
                provider="openai",
                model=sent["model"],
                user=None,
                body=json.dumps(sent).encode(),
            )
        )

    # the third is given what the first sent, the second having added nothing
    first, third = [
        _explanations(await handler.records(id))[-1].split("|", 3)[3]
        for id in ["b", "c"]
    ]
    assert first != "[]"
    assert third == first


@pytest.mark.anyio
async def test_a_new_attempt_at_a_sample_starts_its_history_afresh() -> None:
    # the model of a new attempt never saw the discarded attempt's turns
    handler = Handler(reports_histories())

    async def turn(id: str, attempt: int, **then: Any) -> list[str]:
        headers = inspect_headers(attempt=f"attempt_{attempt}", attempt_number=attempt)
        await handler.request(replace(request(id, **then), headers=headers))
        await handler.reply(reply(message({"type": "text", "text": "Rain."}), id=id))
        return [
            explanation.split("|", 3)[3]
            for explanation in _explanations(await handler.records(id))
        ]

    await turn("a", 1)
    later = await turn("b", 1, then=ran("Oslo", "rain", "toolu_1"))
    fresh = await turn("c", 2)

    opening = ["You are helpful.", "What is the weather?"]
    assert later[0] == str(opening)
    assert fresh == ["[]", str(opening)]


@protocol
def reports_views() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        if not isinstance(step, BeforeToolCall | AfterToolCall):
            return None
        view = step.view.call
        return Decision.proceed(view.content if view is not None else "no view")

    return decide


@pytest.mark.anyio
async def test_a_tool_step_is_given_the_view_inspect_would_make() -> None:
    handler = Handler(reports_views())
    await handler.request(request(then=ran("Oslo", "rain", "toolu_1")))

    assert await handler.reply(reply(message(weather("Lima", "toolu_2")))) == Pass()

    explanations = _explanations(await handler.records("x"))
    assert len(explanations) == 2
    assert "get_weather" in explanations[0] and "Oslo" in explanations[0]
    assert "get_weather" in explanations[1] and "Lima" in explanations[1]


@pytest.mark.anyio
async def test_a_tool_the_provider_ran_is_not_judged_as_a_call() -> None:
    # the provider has run it by the time the reply arrives, so there is no
    # choice left to offer the sentinel
    handler = Handler(reports_histories())
    await handler.request(request(tools=_TOOLS))

    await handler.reply(
        reply(
            message(
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "weather in Oslo"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [],
                },
                {"type": "text", "text": "It is raining."},
                stop_reason="end_turn",
            )
        )
    )

    records = await handler.records("x")
    assert records is not None
    assert [record.stage for record in records] == ["model_input", "model_output"]


def _call(name: str, arguments: dict[str, Any]) -> bytes:
    return message(
        {"type": "tool_use", "id": "toolu_1", "name": name, "input": arguments}
    )


@tool(
    viewer=lambda call: ToolCallView(
        call=ToolCallContent(format="text", content=f"run {call.arguments['cmd']}")
    )
)
def shell_of_its_own() -> Tool:
    async def execute(cmd: str) -> str:
        """Run a command.

        Args:
            cmd: The command.
        """
        return cmd

    return execute


@tool(
    viewer=lambda call: ToolCallView(
        call=ToolCallContent(format="text", content=call.arguments["missing"])
    )
)
def breaks_its_viewer() -> Tool:
    async def execute(cmd: str) -> str:
        """Run a command.

        Args:
            cmd: The command.
        """
        return cmd

    return execute


@tool
def needs_arguments(root: str) -> Tool:
    async def execute(cmd: str) -> str:
        """Run a command.

        Args:
            cmd: The command.
        """
        return root + cmd

    return execute


def _registered(made: Tool) -> str:
    return registry_info(made).name


async def _view(
    handler: Handler,
    name: str,
    arguments: dict[str, Any] | None = None,
    **sample: Any,
) -> str:
    await handler.request(
        replace(request(), headers=inspect_headers(**sample)) if sample else request()
    )
    await handler.reply(reply(_call(name, arguments or {"cmd": "ls -la"})))
    (explanation,) = _explanations(await handler.records("x"))
    return explanation


@pytest.mark.anyio
async def test_a_call_to_one_of_inspect_ais_tools_is_given_its_view() -> None:
    view = await _view(Handler(reports_views()), "bash", {"command": "ls -la"})

    assert view == "```bash\nls -la\n```\n"


@pytest.mark.anyio
async def test_a_call_to_another_tool_is_given_the_default_view() -> None:
    view = await _view(Handler(reports_views()), "run_shell")

    assert "run_shell" in view and "ls -la" in view


def test_the_tools_looked_up_by_name_are_inspect_ais_with_viewers() -> None:
    # every tool inspect_ai registers, made as the sidecar makes one, to find
    # which have a viewer
    from inspect_ai._util.registry import registry_find
    from inspect_ai.tool._tool import TOOL_VIEWER

    from inspect_sentinel.sidecar._inspect_ai import VIEWED_TOOLS

    viewed: set[str] = set()
    for factory in registry_find(
        lambda info: info.type == "tool" and info.name.startswith("inspect_ai/")
    ):
        try:
            made = cast(Any, factory)()
            viewer = registry_info(made).metadata.get(TOOL_VIEWER)
        except Exception:
            continue
        if callable(viewer):
            viewed.add(registry_info(made).name.removeprefix("inspect_ai/"))

    assert viewed == VIEWED_TOOLS


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["web_browser_go", "made_up_1", "made_up_2"])
async def test_a_call_to_a_tool_without_a_viewer_doesnt_look_one_up(
    name: str,
) -> None:
    # The name is the client's, so the sidecar builds no tool by it, and keeps
    # nothing for it.
    from inspect_sentinel.sidecar._inspect_ai import _viewer

    before = _viewer.cache_info()

    view = await _view(Handler(reports_views()), name)

    assert name in view
    assert _viewer.cache_info() == before


def _task(tools: dict[str, str]) -> dict[str, FileTask]:
    return {
        "philadelphia": FileTask.model_validate(
            {
                "task": "philadelphia",
                "samples": [{"sample_id": 1, "sample_input": "Hi."}],
                "tools": tools,
            }
        )
    }


@pytest.mark.anyio
async def test_a_task_registered_beforehand_names_the_tool_its_calls_are_to() -> None:
    # the task's own `bash`, not inspect_ai's
    tools = {"bash": _registered(shell_of_its_own())}
    handler = Handler(reports_views(), tasks=_task(tools))

    view = await _view(handler, "bash", sample_uuid="run")

    assert view == "run ls -la"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "tool",
    [breaks_its_viewer, needs_arguments],
    ids=["a viewer that fails", "a tool that can't be made without arguments"],
)
async def test_a_view_that_cant_be_made_is_the_default(tool: Any) -> None:
    from inspect_sentinel.sidecar._inspect_ai import tool_call_view

    name = registry_info(tool).name
    handler = Handler(reports_views(), tasks=_task({"bash": name}))

    view = await _view(handler, "bash", sample_uuid="run")

    default = tool_call_view(
        ToolCall(id="toolu_1", function="bash", arguments={"cmd": "ls -la"})
    )
    assert default.call is not None
    assert view == default.call.content
