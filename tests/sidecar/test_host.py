import logging
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCall, ToolCallView

from inspect_sentinel import human, sequential
from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import AfterGenerate, BeforeGenerate, BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse, SidecarHost
from inspect_sentinel.sidecar._host import ModelUsed, RunHost, noting_models
from tests.sidecar._traffic import message, reply, request, weather


@protocol
def escalates() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        return Decision.escalate("every call")

    return decide


_CALL = BeforeToolCall(
    conversation="c",
    message="",
    message_id="m1",
    call=ToolCall(id="toolu_1", function="get_weather", arguments={}),
    view=ToolCallView(),
    input=None,
    agent_messages=None,
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        (["approve", "reject", "terminate"], "reject"),
        (["approve", "terminate"], "terminate"),
        (["approve"], "terminate"),
    ],
)
async def test_a_person_who_cannot_be_asked_is_answered_for_with_caution(
    choices: list[str], expected: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        answer = await SidecarHost().ask_human(_CALL, (), choices)

    assert answer.decision == expected
    assert answer.reason == "No person can be asked from the sidecar yet."
    assert "get_weather" in caplog.text


@pytest.mark.anyio
async def test_an_escalation_to_a_person_is_not_let_through() -> None:
    # a threshold escalates every call, and human() is asked to decide
    handler = Handler(sequential([escalates(), human(stages=["tool_call"])]))
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)


_GENERATE: dict[str, Any] = {
    "model": "anthropic/claude-test",
    "conversation": "c",
    "input": [ChatMessageUser(content="Hello")],
    "agent_messages": None,
    "tools": [],
    "tool_choice": "none",
    "config": GenerateConfig(),
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("step", "about"),
    [
        pytest.param(BeforeGenerate(**_GENERATE), "the model's request", id="request"),
        pytest.param(
            AfterGenerate(
                **_GENERATE,
                output=ModelOutput.from_content("anthropic/claude-test", "Hi."),
            ),
            "the model's turn",
            id="turn",
        ),
    ],
)
async def test_a_generate_nobody_can_be_asked_about_ends_the_run(
    step: BeforeGenerate | AfterGenerate, about: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        answer = await SidecarHost().ask_human(step, (), ["approve", "terminate"])

    # neither a request nor a turn can be rejected, so the cautious answer
    # ends the run
    assert answer.decision == "terminate"
    assert about in caplog.text


@pytest.mark.anyio
async def test_generate_needs_a_model_for_the_role_or_by_name() -> None:
    with pytest.raises(ValueError, match="no model for role 'monitor'"):
        await SidecarHost().generate("Is this suspicious?")


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_generate_uses_the_role_model() -> None:
    host = SidecarHost({"monitor": "mockllm/model"})

    output = await host.generate("Is this suspicious?")

    # mockllm names itself as the model that answered
    assert output.model == "mockllm"


@pytest.mark.anyio
# inspect_ai calls a model on trio only once it is set up to, as an eval does;
# the sidecar serves on asyncio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_a_reply_from_inspects_cache_bills_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path))
    host = SidecarHost({"monitor": "mockllm/model"})
    cached = GenerateConfig(cache=True)
    judge = ModelUsed("monitor", "mockllm/model")

    with noting_models() as noted:
        first = await host.generate("Is this suspicious?", config=cached)
        billed = noted.billed[judge]
        # the same prompt again, answered from inspect_ai's cache
        await host.generate("Is this suspicious?", config=cached)

    assert first.usage is not None
    assert billed.total_tokens == first.usage.total_tokens > 0
    assert noted.billed == {judge: billed}

    # a model whose only call was answered from the cache was called, and
    # billed nothing
    with noting_models() as noted:
        await host.generate("Is this suspicious?", config=cached)
    assert (noted.called, noted.billed) == ([judge], {})


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("roles", "model", "role"),
    [
        # the configured role, the role missing (so model= is used), and no
        # role but a model name: the same resolution generate() uses
        ({"monitor": "mockllm/model"}, None, None),
        ({}, "mockllm/model", None),
        ({"judge": "mockllm/model"}, None, "judge"),
    ],
)
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_input_budget_and_count_tokens_resolve_as_generate_does(
    roles: dict[str, str], model: str | None, role: str | None
) -> None:
    host = SidecarHost(roles)

    budget = await host.input_budget(model=model, role=role)
    count = await host.count_tokens("Is this suspicious?", model=model, role=role)

    # mockllm reports a 128k window less the output reserve, and counts tokens
    assert 0 < budget < 128_000
    assert count > 0


@pytest.mark.anyio
async def test_input_budget_needs_a_model_for_the_role_or_by_name() -> None:
    with pytest.raises(ValueError, match="no model for role 'monitor'"):
        await SidecarHost().input_budget()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_run_host_passes_the_budget_and_the_count_through() -> None:
    host = RunHost(SidecarHost({"monitor": "mockllm/model"}), None)

    assert 0 < await host.input_budget() < 128_000
    assert await host.count_tokens("Is this suspicious?") > 0
