import json
import re
from dataclasses import replace
from typing import Any, cast

import anyio
import anyio.lowlevel
import pytest
from inspect_ai.util import StoreModel

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import AfterToolCall, BeforeGenerate, BeforeToolCall, Step
from inspect_sentinel._types import Protocol, ProtocolGroup
from inspect_sentinel.sidecar import (
    Answer,
    Handler,
    Pass,
    Record,
    Refuse,
    Replace,
    TimeLimitError,
    UnreadableError,
    read_reply,
)
from tests.sidecar._traffic import (
    STREAM,
    anthropic_recorded,
    message,
    plain,
    ran,
    reply,
    request,
    storm_rule,
    weather,
    weather_rule,
)

_ENDED = Refuse(400, "A sentinel ended this run at the call to get_weather.")


@pytest.mark.anyio
async def test_request_passes() -> None:
    assert await Handler(weather_rule()).request(request()) == Pass()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "city", "expected"),
    [
        (
            "reject",
            "Tokyo",
            Refuse(
                400,
                "A sentinel rejected the call to get_weather: Tokyo is not allowed.",
            ),
        ),
        # a decision's explanation is for the log, so the client isn't told it
        ("terminate", "Tokyo", _ENDED),
        # nobody is left to take an escalate at the top, so the run is ended
        ("escalate", "Tokyo", _ENDED),
        ("reject", "Cairo", Pass()),
    ],
)
async def test_recorded_reply_is_judged_call_by_call(
    action: str, city: str, expected: Pass | Refuse
) -> None:
    handler = Handler(weather_rule(action, city))
    await handler.request(request())

    answer = await handler.reply(
        reply(anthropic_recorded("parallel-tool-calls"), content_type=STREAM)
    )

    assert answer == expected


@pytest.mark.anyio
async def test_first_refused_call_ends_the_judging() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(request())

    body = anthropic_recorded("parallel-tool-calls")
    await handler.reply(reply(body, content_type=STREAM))

    records = await handler.records("x")
    assert records is not None
    calls = (await read_reply("anthropic", STREAM, body)).message.tool_calls or []
    # Paris and Tokyo were judged; Lima, after the refusal, was not
    assert [(record.stage, record.call) for record in records] == [
        ("tool_call", calls[0].id),
        ("tool_call", calls[1].id),
    ]


@pytest.mark.anyio
async def test_reply_without_calls_passes() -> None:
    handler = Handler(weather_rule())
    await handler.request(request())

    answer = await handler.reply(
        reply(anthropic_recorded("text-only"), content_type=STREAM)
    )

    assert answer == Pass()


@pytest.mark.anyio
async def test_provider_error_passes_unread() -> None:
    handler = Handler(weather_rule())
    await handler.request(request())

    assert await handler.reply(reply(b"not a message", status=529)) == Pass()


@pytest.mark.anyio
async def test_reply_to_an_unseen_request_cannot_be_judged() -> None:
    with pytest.raises(UnreadableError, match="never shown"):
        await Handler(weather_rule()).reply(reply(message(weather("Tokyo"))))


@pytest.mark.anyio
async def test_modify_is_not_carried_out_yet() -> None:
    handler = Handler(weather_rule("modify", "Tokyo"))
    await handler.request(request())

    with pytest.raises(NotImplementedError, match="'modify'"):
        await handler.reply(reply(message(weather("Tokyo"))))


@pytest.mark.anyio
async def test_a_failing_protocol_is_not_swallowed() -> None:
    handler = Handler(weather_rule("break", "Tokyo"))
    await handler.request(request())

    with pytest.raises(RuntimeError, match="the rule broke"):
        await handler.reply(reply(message(weather("Tokyo"))))


@protocol
def takes(seconds: float) -> Protocol:
    """Wait before rejecting a call, as a monitor that asks a slow model does."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        await anyio.sleep(seconds)
        return Decision.reject("slow", message="Decided late.")

    return decide


@pytest.mark.anyio
async def test_sentinel_past_the_time_limit_is_cancelled() -> None:
    handler = Handler(takes(30), time_limit=0.05)
    await handler.request(request())

    with pytest.raises(TimeLimitError, match="reply 'x' within 0.05 seconds"):
        await handler.reply(reply(message(weather("Oslo"))))

    # the rejection it was about to make is not kept
    assert await handler.request(request("y")) == Pass()


@pytest.mark.anyio
async def test_sentinel_is_not_timed_unless_a_limit_is_set() -> None:
    within = Handler(takes(0.05), time_limit=30)
    unlimited = Handler(takes(0.05))

    for handler in (within, unlimited):
        await handler.request(request())
        answer = await handler.reply(reply(message(weather("Oslo"))))
        assert isinstance(answer, Refuse)
        assert answer.message.endswith("Decided late.")


class Seen(StoreModel):
    lookups: int = 0


@protocol
def third_lookup_ends_the_run() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        seen = context.store_as(Seen)
        seen.lookups += 1
        if seen.lookups >= 3:
            return Decision.terminate("too many lookups")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_state_is_kept_for_a_run_and_apart_from_other_runs() -> None:
    handler = Handler(third_lookup_ends_the_run())

    async def lookup(id: str, opening: str) -> Pass | Refuse:
        await handler.request(request(id, opening))
        answer = await handler.reply(reply(message(weather("Oslo")), id=id))
        assert isinstance(answer, (Pass, Refuse))
        return answer

    assert await lookup("a1", "First conversation") == Pass()
    assert await lookup("a2", "First conversation") == Pass()
    # another conversation has a count of its own
    assert await lookup("b1", "Second conversation") == Pass()
    assert isinstance(await lookup("a3", "First conversation"), Refuse)


@protocol
def reports_context() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        roles = [m.role for m in step.input or []]
        return Decision.reject(
            "context", message=f"{context.eval}|{context.path!r}|{roles}"
        )

    return decide


@pytest.mark.anyio
async def test_sentinel_is_given_no_eval_and_what_the_model_was_sent() -> None:
    handler = Handler(reports_context())
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    # a proxy's requests have no task or sample
    assert answer.message.endswith(": None|''|['system', 'user']")


def _messages(body: bytes) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", json.loads(body)["messages"])


@pytest.mark.anyio
async def test_rejection_is_added_to_the_next_request_of_the_run() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"), reject_status=503)
    first = request("a")
    await handler.request(first)

    refused = await handler.reply(
        reply(anthropic_recorded("parallel-tool-calls"), id="a", content_type=STREAM)
    )
    again = await handler.request(request("b"))

    assert refused == Refuse(
        503, "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )
    assert isinstance(again, Replace)
    sent, added = _messages(first.body), _messages(again.body)
    assert added[: len(sent)] == sent
    assistant, results = added[len(sent) :]
    assert assistant["role"] == "assistant"
    calls = [block for block in assistant["content"] if block["type"] == "tool_use"]
    assert [call["input"]["location"] for call in calls] == ["Paris", "Tokyo", "Lima"]
    assert results["role"] == "user"
    told = {
        block["tool_use_id"]: (block["is_error"], block["content"])
        for block in results["content"]
    }
    not_run = (
        "This call was not run, because another call in the same turn was rejected."
    )
    assert told == {
        calls[0]["id"]: (True, not_run),
        calls[1]["id"]: (True, "Tokyo is not allowed."),
        calls[2]["id"]: (True, not_run),
    }
    # everything else in the request is as the client sent it
    assert {k: v for k, v in json.loads(again.body).items() if k != "messages"} == {
        k: v for k, v in json.loads(first.body).items() if k != "messages"
    }


def _ids(body: bytes) -> list[str]:
    """The id of each call and each result in a request, in order."""
    return [
        block["id" if block["type"] == "tool_use" else "tool_use_id"]
        for entry in _messages(body)
        if isinstance(entry["content"], list)
        for block in cast("list[dict[str, Any]]", entry["content"])
        if block["type"] in ("tool_use", "tool_result")
    ]


@pytest.mark.anyio
async def test_rejection_is_added_to_every_later_request_of_its_conversation() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(request("a", "First conversation"))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))

    other = await handler.request(request("b", "Second conversation"))
    told = await handler.request(request("c", "First conversation"))
    # a client that comes back twice before any reply is told both times
    told_again = await handler.request(request("d", "First conversation"))
    await handler.reply(reply(message(weather("Oslo", "toolu_2")), id="d"))
    # the client's conversation goes on from the reply that passed, and holds
    # nothing of the one that didn't
    later = await handler.request(
        request("e", "First conversation", ran("Oslo", "rain", "toolu_2"))
    )

    assert other == Pass()
    assert isinstance(told, Replace)
    assert told_again == told
    assert isinstance(later, Replace)
    assert _ids(later.body) == ["toolu_1", "toolu_1", "toolu_2", "toolu_2"]


@pytest.mark.anyio
async def test_rejections_are_added_where_each_happened() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    grown = ran("Oslo", "rain", "toolu_2")
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    await handler.request(request("b"))
    await handler.reply(reply(message(weather("Oslo", "toolu_2")), id="b"))
    await handler.request(request("c", then=grown))
    await handler.reply(reply(message(weather("Tokyo", "toolu_3")), id="c"))

    again = await handler.request(request("d", then=grown))

    assert isinstance(again, Replace)
    assert _ids(again.body) == [
        "toolu_1",
        "toolu_1",
        "toolu_2",
        "toolu_2",
        "toolu_3",
        "toolu_3",
    ]


def _first_result(then: list[dict[str, Any]], **changes: Any) -> list[dict[str, Any]]:
    assistant, results = then
    return [assistant, results | {"content": [results["content"][0] | changes]}]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("first", "then", "told"),
    [
        pytest.param(
            "What is the weather?",
            _first_result(ran("Oslo", "rain"), cache_control={"type": "ephemeral"}),
            True,
            id="a cache mark where there was none",
        ),
        pytest.param(
            [
                {
                    "type": "text",
                    "text": "What is the weather?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            ran("Oslo", "rain"),
            True,
            id="a text written as a block",
        ),
        pytest.param(
            "What is the weather?",
            _first_result(ran("Oslo", "rain"), content="sun"),
            False,
            id="another conversation that opens alike",
        ),
    ],
)
async def test_a_request_continues_a_conversation_whatever_the_client_changed_in_writing_it(
    first: str | list[dict[str, Any]], then: list[dict[str, Any]], told: bool
) -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(request("a", then=ran("Oslo", "rain")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_2")), id="a"))
    later = request("b")
    body = json.loads(later.body) | {
        "messages": [{"role": "user", "content": first}, *then]
    }

    answer = await handler.request(replace(later, body=json.dumps(body).encode()))

    assert isinstance(answer, Replace if told else Pass)


@pytest.mark.anyio
async def test_every_rejected_turn_in_a_row_is_added() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"), reject_status=503)
    client = request("a")
    await handler.request(client)
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    # the client comes back with its own conversation, which holds neither
    await handler.request(request("b"))
    await handler.reply(reply(message(weather("Tokyo", "toolu_2")), id="b"))

    third = await handler.request(request("c"))

    assert isinstance(third, Replace)
    sent, added = _messages(client.body), _messages(third.body)
    assert [entry["role"] for entry in added[len(sent) :]] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert _ids(third.body) == ["toolu_1", "toolu_1", "toolu_2", "toolu_2"]


@protocol
def reports_input() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("first try", message="Tokyo is not allowed.")
        return Decision.reject("input", message=str([m.role for m in step.input or []]))

    return decide


@pytest.mark.anyio
async def test_sentinel_sees_the_rejection_the_model_was_told_of() -> None:
    handler = Handler(reports_input())
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))
    await handler.request(request("b"))

    answer = await handler.reply(reply(message(weather("Oslo")), id="b"))

    assert isinstance(answer, Refuse)
    assert answer.message.endswith("['system', 'user', 'assistant', 'tool']")


@pytest.mark.anyio
async def test_third_rejection_in_a_row_ends_the_run() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"), reject_status=503)

    async def rejected(id: str) -> Pass | Refuse:
        await handler.request(request(id))
        answer = await handler.reply(reply(message(weather("Tokyo")), id=id))
        assert isinstance(answer, (Pass, Refuse))
        return answer

    first, second, third = await rejected("a"), await rejected("b"), await rejected("c")

    assert isinstance(first, Refuse) and first.status == 503
    assert isinstance(second, Refuse) and second.status == 503
    assert third == Refuse(
        400,
        "A sentinel ended this run at the call to get_weather after 3 rejected turns in a row.",
    )
    # the run has ended, so nothing is added to what follows
    assert await handler.request(request("d")) == Pass()


@pytest.mark.anyio
async def test_a_reply_that_passes_starts_the_count_again() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))

    async def turn(id: str, city: str) -> Pass | Refuse:
        await handler.request(request(id))
        answer = await handler.reply(reply(message(weather(city)), id=id))
        assert isinstance(answer, (Pass, Refuse))
        return answer

    answers = [
        await turn(id, city)
        for id, city in [
            ("a", "Tokyo"),
            ("b", "Tokyo"),
            ("c", "Oslo"),
            ("d", "Tokyo"),
            ("e", "Tokyo"),
        ]
    ]

    assert [type(answer) for answer in answers] == [
        Refuse,
        Refuse,
        Pass,
        Refuse,
        Refuse,
    ]
    assert all(
        answer.message.startswith("A sentinel rejected")
        for answer in answers
        if isinstance(answer, Refuse)
    )


@protocol
def rejects_without_a_message() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("a reason kept for the log")

    return decide


@pytest.mark.anyio
async def test_explanation_never_reaches_the_client() -> None:
    handler = Handler(rejects_without_a_message())
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert answer == Refuse(
        400, "A sentinel rejected the call to get_weather: This call was rejected."
    )


_ENDED_AT_RESULT = Refuse(
    400, "A sentinel ended this run at the result of the call to get_weather."
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "result", "expected"),
    [
        # a decision's explanation is for the log, so the client isn't told it
        ("terminate", "storm", _ENDED_AT_RESULT),
        # nobody is left to take an escalate at the top, so the run is ended
        ("escalate", "storm", _ENDED_AT_RESULT),
        ("terminate", "rain", Pass()),
    ],
)
async def test_result_is_judged_before_the_model_is_sent_it(
    action: str, result: str, expected: Pass | Refuse
) -> None:
    handler = Handler(storm_rule(action))

    answer = await handler.request(request(then=ran("Oslo", result)))

    assert answer == expected


@pytest.mark.anyio
async def test_first_result_that_ends_the_run_ends_the_judging() -> None:
    handler = Handler(storm_rule())

    await handler.request(
        request(
            then=[
                *ran("Oslo", "rain", "toolu_1"),
                *ran("Lima", "storm", "toolu_2"),
                *ran("Cairo", "sun", "toolu_3"),
            ]
        )
    )

    records = await handler.records("x")
    assert records is not None
    # Oslo's and Lima's were judged; Cairo's, after the run was ended, was not
    assert [record.call for record in records] == ["toolu_1", "toolu_2"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "error", "match"),
    [
        ("break", RuntimeError, "the rule broke"),
        # the sentinel runner's own check: the call has already run
        ("reject", ValueError, "cannot reject"),
    ],
)
async def test_a_failing_protocol_is_not_swallowed_at_a_result(
    action: str, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        await Handler(storm_rule(action)).request(request(then=ran("Oslo", "storm")))


def _roles(messages: list[Any] | None) -> str:
    return str(None if messages is None else [m.role for m in messages])


@protocol
def reports_result() -> Protocol:
    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        return Decision.proceed(
            "|".join(
                [
                    step.message,
                    step.call.function,
                    str(step.call.arguments),
                    step.result.text,
                    str(step.result.error is not None),
                    str(step.output),
                    _roles(step.input),
                    _roles(step.history),
                    str(step.truncated),
                ]
            )
        )

    return decide


def _explanations(records: list[Record] | None) -> list[str]:
    assert records is not None
    return [cast(Any, record.detail).report.explanation for record in records]


@pytest.mark.anyio
async def test_sentinel_is_given_the_call_its_result_and_what_led_to_it() -> None:
    handler = Handler(reports_result())

    await handler.request(
        request(
            then=[
                *ran("Oslo", "rain", "toolu_1"),
                *ran("Lima", "no such city", "toolu_2", error=True),
            ],
        )
    )

    # The calls were never handed out through this sidecar, so what the model
    # was sent for them is not known. Whether the client cut a result is not
    # known either.
    assert _explanations(await handler.records("x")) == [
        "Looking.|get_weather|{'location': 'Oslo'}|rain|False||None|None|None",
        "Looking.|get_weather|{'location': 'Lima'}|no such city|True||None|None|None",
    ]


class Results(StoreModel):
    seen: int = 0
    failed: int = 0


@protocol
def second_result_ends_the_run() -> Protocol:
    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        results = context.store_as(Results)
        results.seen += 1
        if results.seen >= 2:
            return Decision.terminate("too many results")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_result_is_judged_once_however_often_it_is_sent() -> None:
    handler = Handler(second_result_ends_the_run())
    first = ran("Oslo", "rain", "toolu_1")

    assert await handler.request(request("a", then=first)) == Pass()
    # a client that tries again, and one whose conversation has grown, both
    # send the first result again
    assert await handler.request(request("b", then=first)) == Pass()
    answer = await handler.request(
        request("c", then=[*first, *ran("Lima", "sun", "toolu_2")])
    )

    assert isinstance(answer, Refuse)
    assert await handler.records("b") == []


@pytest.mark.anyio
async def test_a_result_that_ended_the_run_ends_it_each_time_it_is_sent() -> None:
    handler = Handler(storm_rule())
    stormy = ran("Oslo", "storm")

    answers = [await handler.request(request(id, then=stormy)) for id in ("a", "b")]

    assert [type(answer) for answer in answers] == [Refuse, Refuse]


@pytest.mark.anyio
async def test_a_known_call_id_with_another_result_is_judged() -> None:
    handler = Handler(storm_rule())

    assert await handler.request(request("a", then=ran("Oslo", "rain"))) == Pass()
    answer = await handler.request(request("b", then=ran("Oslo", "storm")))

    assert isinstance(answer, Refuse)


@pytest.mark.anyio
async def test_a_result_without_its_call_cannot_be_judged() -> None:
    orphan = ran("Oslo", "rain", "toolu_1")[1:]

    with pytest.raises(UnreadableError, match="doesn't hold: 'toolu_1'"):
        await Handler(storm_rule()).request(request(then=orphan))


@protocol
def failures_end_the_next_call() -> ProtocolGroup:
    async def before(context: Context, step: BeforeToolCall) -> Decision | None:
        if context.store_as(Results).failed:
            return Decision.terminate("an earlier call failed")
        return Decision.proceed()

    async def after(context: Context, step: AfterToolCall) -> Decision | None:
        if step.result.error is not None:
            context.store_as(Results).failed += 1
        return Decision.proceed()

    return ProtocolGroup(before, after)


@pytest.mark.anyio
@pytest.mark.parametrize(("error", "expected"), [(False, Pass), (True, Refuse)])
async def test_state_is_shared_by_the_steps_of_a_run(
    error: bool, expected: type[Pass | Refuse]
) -> None:
    handler = Handler(failures_end_the_next_call())
    await handler.request(request(then=ran("Oslo", "rain", "toolu_1", error=error)))

    answer = await handler.reply(reply(message(weather("Lima", "toolu_2"))))

    assert isinstance(answer, expected)
    records = await handler.records("x")
    assert records is not None
    # the request's result, then the reply's call; each is recorded for the
    # function that judged it and again as the group's decision
    assert [cast(Any, record.detail).function for record in records] == [
        "after",
        "run",
        "before",
        "run",
    ]


@protocol
def rejects_tokyo_and_ends_at_any_result() -> ProtocolGroup:
    async def before(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    async def after(context: Context, step: AfterToolCall) -> Decision | None:
        return Decision.terminate("a result")

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_what_the_model_is_told_of_a_rejection_is_not_judged_as_a_result() -> (
    None
):
    handler = Handler(rejects_tokyo_and_ends_at_any_result(), reject_status=503)
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))

    again = await handler.request(request("b"))

    assert isinstance(again, Replace)
    assert _messages(again.body)[-1]["content"][0]["type"] == "tool_result"
    assert await handler.records("b") == []


@protocol
def rejects_tokyo_and_reports_what_led_to_a_result() -> ProtocolGroup:
    async def before(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    async def after(context: Context, step: AfterToolCall) -> Decision | None:
        tools = None if step.tools is None else [tool.name for tool in step.tools]
        return Decision.proceed(f"{_roles(step.input)}|{tools}")

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_a_result_is_judged_against_what_the_model_was_sent() -> None:
    handler = Handler(rejects_tokyo_and_reports_what_led_to_a_result())
    tools = [
        {
            "name": "get_weather",
            "description": "Look up the weather.",
            "input_schema": {"type": "object"},
        }
    ]
    await handler.request(request("a", tools=tools))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    await handler.request(request("b", tools=tools))
    await handler.reply(reply(message(weather("Oslo", "toolu_2")), id="b"))

    await handler.request(
        request("c", then=ran("Oslo", "rain", "toolu_2"), tools=tools)
    )

    # the rejected turn was in what the model made the second call from,
    # recorded for the function and again as the group's decision
    assert (
        _explanations(await handler.records("c"))
        == ["['system', 'user', 'assistant', 'tool']|['get_weather']"] * 2
    )


@pytest.mark.anyio
async def test_samples_share_nothing() -> None:
    # two epochs of one sample: the same conversation, and nothing in common
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(request("a", sample="epoch-1"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))

    other = await handler.request(request("b", sample="epoch-2"))
    told = await handler.request(request("c", sample="epoch-1"))

    assert other == Pass()
    assert isinstance(told, Replace)


@pytest.mark.anyio
async def test_a_sample_is_one_run_however_its_conversations_open() -> None:
    handler = Handler(third_lookup_ends_the_run())

    async def lookup(id: str, opening: str) -> Answer:
        await handler.request(request(id, opening, sample="sample-1"))
        return await handler.reply(reply(message(weather("Oslo")), id=id))

    answers = [
        await lookup("a", "The agent's conversation"),
        await lookup("b", "A subagent's conversation"),
        await lookup("c", "The agent's conversation"),
    ]

    assert [isinstance(answer, Refuse) for answer in answers] == [False, False, True]


@pytest.mark.anyio
async def test_every_later_request_of_a_sample_that_ended_is_refused() -> None:
    handler = Handler(weather_rule("terminate", "Tokyo"))
    await handler.request(request("a", sample="sample-1"))
    ended = await handler.reply(reply(message(weather("Tokyo")), id="a"))

    again = await handler.request(request("b", sample="sample-1"))
    subagent = await handler.request(
        request("c", "A subagent's conversation", sample="sample-1")
    )
    other = await handler.request(request("d", sample="sample-2"))

    assert [plain(answer) for answer in (ended, again, subagent)] == [_ENDED] * 3
    assert other == Pass()
    # the sentinel was not asked again
    assert await handler.records("b") is None


@pytest.mark.anyio
async def test_a_sample_ended_at_a_result_is_refused_without_that_result() -> None:
    handler = Handler(storm_rule())
    ended = await handler.request(
        request("a", then=ran("Oslo", "storm"), sample="sample-1")
    )

    later = await handler.request(request("b", sample="sample-1"))

    assert isinstance(ended, Refuse)
    assert plain(later) == plain(ended)


@pytest.mark.anyio
async def test_a_sample_ended_while_the_model_answered_is_refused() -> None:
    # two conversations of one run, the second ended while the first waited
    handler = Handler(weather_rule("terminate", "Tokyo"))
    await handler.request(request("a", sample="sample-1"))
    await handler.request(request("b", "A subagent's conversation", sample="sample-1"))
    ended = await handler.reply(reply(message(weather("Tokyo")), id="b"))

    late = await handler.reply(reply(message(weather("Oslo")), id="a"))

    assert (plain(ended), plain(late)) == (_ENDED, _ENDED)


@pytest.mark.anyio
async def test_a_run_with_no_name_is_refused_where_its_conversation_continues() -> None:
    handler = Handler(weather_rule("terminate", "Tokyo"))
    looked = ran("Oslo", "rain", "toolu_1")
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Oslo")), id="a"))
    await handler.request(request("b", then=looked))
    ended = await handler.reply(reply(message(weather("Tokyo")), id="b"))

    again = await handler.request(request("c", then=looked))
    grown = await handler.request(
        request("d", then=[*looked, {"role": "user", "content": "Try another way."}])
    )
    # two conversations that open as the ended one did, and so share its key
    opening = await handler.request(request("e"))
    another = await handler.request(request("f", then=ran("Lima", "sun", "toolu_2")))

    assert (ended, again, grown) == (_ENDED, _ENDED, _ENDED)
    assert (opening, another) == (Pass(), Pass())


@pytest.mark.anyio
async def test_a_run_with_no_name_ended_on_its_first_turn_is_judged_afresh() -> None:
    # all the ended conversation holds is how it opens, which is all another
    # conversation that opens alike holds too
    handler = Handler(weather_rule("terminate", "Tokyo"))
    await handler.request(request("a"))
    ended = await handler.reply(reply(message(weather("Tokyo")), id="a"))

    asked = await handler.request(request("b"))
    answered = await handler.reply(reply(message(weather("Oslo")), id="b"))

    assert ended == _ENDED
    assert (asked, answered) == (Pass(), Pass())


@protocol
def reports_the_conversation() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("run", message=step.conversation)

    return decide


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        # a conversation is told apart within the run by how it opens
        ("sample-uuid", r"sample-uuid/[0-9a-f]{16}"),
        # how the conversation opens stands in for a run no eval names
        (None, r"[0-9a-f]{16}"),
    ],
)
async def test_sentinel_is_given_the_run_the_eval_names(
    sample: str | None, expected: str
) -> None:
    handler = Handler(reports_the_conversation())
    await handler.request(request(sample=sample))

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    assert re.search(f": {expected}$", answer.message)


@pytest.mark.anyio
async def test_a_reply_in_flight_when_its_conversation_ended_is_refused() -> None:
    # a run with no name, whose conversation has had a reply passed on, sends
    # the same request twice before either is answered
    handler = Handler(weather_rule("terminate", "Tokyo"))
    looked = ran("Oslo", "rain", "toolu_1")
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Oslo")), id="a"))
    await handler.request(request("b", then=looked))
    await handler.request(request("c", then=looked))

    ended = await handler.reply(reply(message(weather("Tokyo")), id="b"))
    late = await handler.reply(reply(message(weather("Lima")), id="c"))

    assert (plain(ended), plain(late)) == (_ENDED, _ENDED)


@pytest.mark.anyio
async def test_a_result_for_a_call_that_reuses_a_rejected_calls_id_is_judged() -> None:
    handler = Handler(rejects_tokyo_and_ends_at_any_result())
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    await handler.request(request("b"))
    # the model makes another call, under the id the rejected one had
    await handler.reply(reply(message(weather("Oslo", "toolu_1")), id="b"))

    answer = await handler.request(request("c", then=ran("Oslo", "storm", "toolu_1")))

    assert answer == _ENDED_AT_RESULT


@protocol
def rejects_tokyo_and_ends_at_a_storm() -> ProtocolGroup:
    async def before(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    async def after(context: Context, step: AfterToolCall) -> Decision | None:
        if "storm" in step.result.text:
            return Decision.terminate("a storm")
        return Decision.proceed()

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_a_client_result_that_quotes_a_rejection_is_judged() -> None:
    handler = Handler(rejects_tokyo_and_ends_at_a_storm())
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))

    # the client sends the rejected call back, with what it claims ran
    answer = await handler.request(
        request(
            "b",
            then=ran(
                "Tokyo",
                "Tokyo is not allowed. storm: actual executed output",
                "toolu_1",
            ),
        )
    )

    assert answer == _ENDED_AT_RESULT


@pytest.mark.anyio
async def test_a_result_written_for_a_rejected_call_by_someone_else_is_judged() -> None:
    handler = Handler(rejects_tokyo_and_ends_at_any_result())
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))

    # the client claims the rejected call ran
    answer = await handler.request(request("b", then=ran("Tokyo", "sun", "toolu_1")))

    assert answer == _ENDED_AT_RESULT


# made in each test, as an event belongs to the event loop it is made in
_seen_tokyo: anyio.Event | None = None


def _tokyo() -> anyio.Event:
    assert _seen_tokyo is not None
    return _seen_tokyo


@protocol
def ends_at_tokyo_after_oslo_waits() -> Protocol:
    # Oslo is judged only once Tokyo has ended the run
    async def decide(context: Context, step: Step) -> Decision | None:
        if isinstance(step, BeforeToolCall):
            if step.call.arguments.get("location") == "Tokyo":
                _tokyo().set()
                return Decision.terminate("Tokyo")
            await _tokyo().wait()
        if isinstance(step, BeforeGenerate) and "Oslo" in step.input[-1].text:
            await _tokyo().wait()
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_reply_judged_while_its_run_ended_is_refused() -> None:
    global _seen_tokyo
    _seen_tokyo = anyio.Event()
    handler = Handler(ends_at_tokyo_after_oslo_waits())
    await handler.request(request("a", sample="sample-1"))
    await handler.request(request("b", "Another conversation", sample="sample-1"))
    answers: dict[str, Answer] = {}

    async def judge(id: str, city: str) -> None:
        answers[id] = await handler.reply(reply(message(weather(city)), id=id))

    async with anyio.create_task_group() as group:
        group.start_soon(judge, "a", "Oslo")
        group.start_soon(judge, "b", "Tokyo")

    assert {id: plain(answer) for id, answer in answers.items()} == {
        "a": _ENDED,
        "b": _ENDED,
    }


@pytest.mark.anyio
async def test_a_request_judged_while_its_run_ended_is_refused() -> None:
    global _seen_tokyo
    _seen_tokyo = anyio.Event()
    handler = Handler(ends_at_tokyo_after_oslo_waits())
    await handler.request(request("a", sample="sample-1"))
    answers: dict[str, Answer] = {}

    async def ask() -> None:
        answers["b"] = await handler.request(
            request("b", "Weather in Oslo?", sample="sample-1")
        )

    async def end() -> None:
        answers["a"] = await handler.reply(reply(message(weather("Tokyo")), id="a"))

    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        group.start_soon(end)

    assert {id: plain(answer) for id, answer in answers.items()} == {
        "a": _ENDED,
        "b": _ENDED,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sentinel", "judged"),
    [
        (weather_rule(), [BeforeToolCall]),
        (storm_rule(), [AfterToolCall]),
    ],
    ids=["before a call", "after a call"],
)
async def test_a_stage_nothing_watches_is_never_judged(
    sentinel: Protocol, judged: list[type[Step]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_sentinel.sidecar import _handler

    steps: list[type[Step]] = []
    run = _handler.run_sentinel

    async def counted(protocol: Any, host: Any, step: Step) -> Decision | None:
        steps.append(type(step))
        return await run(protocol, host, step)

    monkeypatch.setattr(_handler, "run_sentinel", counted)
    handler = Handler(sentinel)
    await handler.request(request(then=ran("Oslo", "rain", "toolu_1")))

    await handler.reply(reply(message(weather("Lima", "toolu_2"))))

    assert steps == judged


_SEARCH = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "web_search",
    "input": {"query": "Paris weather"},
}


@pytest.mark.anyio
async def test_a_server_tool_answered_in_the_next_reply_is_read() -> None:
    # Anthropic can start a search in one reply, hand the client a call, and
    # give the search's result in the reply after
    handler = Handler(weather_rule())
    await handler.request(request("a"))
    first = await handler.reply(reply(message(_SEARCH, weather("Paris")), id="a"))
    then = [
        {"role": "assistant", "content": [_SEARCH, weather("Paris")]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Rain."}
            ],
        },
    ]
    await handler.request(request("b", then=then))
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://example.com/paris",
                "title": "Paris weather",
                "encrypted_content": "abc",
                "page_age": None,
            }
        ],
    }
    answered: list[dict[str, Any]] = [result, {"type": "text", "text": "Rain."}]
    second = await handler.reply(
        reply(message(*answered, stop_reason="end_turn"), id="b")
    )
    # and the conversation goes on, holding both replies
    after = [
        *then,
        {"role": "assistant", "content": answered},
        {"role": "user", "content": "Thanks."},
    ]
    third = await handler.request(request("c", then=after))

    assert (first, second, third) == (Pass(), Pass(), Pass())


@protocol
def notes_results() -> Protocol:
    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        return Decision.proceed(step.call.id)

    return decide


@pytest.mark.anyio
async def test_a_result_is_judged_in_each_conversation_it_reaches() -> None:
    handler = Handler(notes_results())
    # two conversations of one run, each holding the same call and result
    then = ran("Oslo", "Rain.")

    await handler.request(request("a", "First conversation", then, sample="run_1"))
    await handler.request(request("b", "Second conversation", then, sample="run_1"))
    # and the first again, which has judged it
    await handler.request(request("c", "First conversation", then, sample="run_1"))

    assert [len(await handler.records(id) or []) for id in "abc"] == [1, 1, 0]


class _Gate:
    judged = 0
    open: anyio.Event
    reached: anyio.Event


@protocol
def holds_the_first_result() -> Protocol:
    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        _Gate.judged += 1
        _Gate.reached.set()
        await _Gate.open.wait()
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_result_two_requests_hold_at_once_is_judged_once() -> None:
    _Gate.judged, _Gate.open, _Gate.reached = 0, anyio.Event(), anyio.Event()
    handler = Handler(holds_the_first_result())
    then = ran("Oslo", "Rain.")
    answers: dict[str, Answer] = {}

    async def ask(id: str) -> None:
        answers[id] = await handler.request(request(id, then=then))

    async with anyio.create_task_group() as group:
        group.start_soon(ask, "a")
        await _Gate.reached.wait()
        group.start_soon(ask, "b")
        # let the second request get as far as it can
        for _ in range(20):
            await anyio.lowlevel.checkpoint()
        _Gate.open.set()

    assert (_Gate.judged, answers) == (1, {"a": Pass(), "b": Pass()})


@protocol
def ends_at_a_storm_the_model_mentions() -> Protocol:
    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        if "storm" in step.message:
            return Decision.terminate("a storm")
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_result_is_judged_again_beside_what_the_model_said_anew() -> None:
    handler = Handler(ends_at_a_storm_the_model_mentions())
    then = ran("Oslo", "Rain.")
    await handler.request(request("a", then=then))
    # the same call and result, beside other words from the model
    changed = [
        {**then[0], "content": [{"type": "text", "text": "A storm."}, weather("Oslo")]},
        then[1],
    ]

    answer = await handler.request(request("b", then=changed))

    assert isinstance(answer, Refuse)


@protocol
def holds_then_ends_the_run() -> Protocol:
    async def decide(context: Context, step: AfterToolCall) -> Decision | None:
        _Gate.judged += 1
        _Gate.reached.set()
        await _Gate.open.wait()
        return Decision.terminate("a result")

    return decide


@pytest.mark.anyio
async def test_a_request_that_waited_for_a_judgment_that_ended_the_run_is_refused() -> (
    None
):
    _Gate.judged, _Gate.open, _Gate.reached = 0, anyio.Event(), anyio.Event()
    handler = Handler(holds_then_ends_the_run())
    then = ran("Oslo", "Rain.")
    answers: dict[str, Answer] = {}

    async def ask(id: str) -> None:
        answers[id] = await handler.request(request(id, then=then, sample="run_1"))

    async with anyio.create_task_group() as group:
        group.start_soon(ask, "a")
        await _Gate.reached.wait()
        group.start_soon(ask, "b")
        for _ in range(20):
            await anyio.lowlevel.checkpoint()
        _Gate.open.set()

    assert _Gate.judged == 1
    assert all(isinstance(answer, Refuse) for answer in answers.values())
