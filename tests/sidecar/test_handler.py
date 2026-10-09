import json
import re
from dataclasses import replace
from typing import Any, cast

import anyio
import pytest
from inspect_ai.util import StoreModel

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import AfterToolCall, BeforeToolCall
from inspect_sentinel._types import Protocol, ProtocolGroup
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Refuse,
    Replace,
    TimeLimitError,
    UnreadableError,
)
from inspect_sentinel.sidecar._handler import RequestResult
from tests.sidecar._traffic import (
    STREAM,
    message,
    ran,
    recorded,
    reply,
    reports_the_proxy,
    reports_the_run,
    request,
    storm_rule,
    weather,
    weather_rule,
)


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
                {"x-sentinel-decision": "reject"},
            ),
        ),
        (
            "terminate",
            "Tokyo",
            # a decision's explanation is for the log, so the client isn't told it
            Refuse(
                400,
                "A sentinel ended this run at the call to get_weather.",
                {"x-sentinel-decision": "terminate"},
            ),
        ),
        # nobody is left to take an escalate at the top, so the step goes on
        ("escalate", "Tokyo", Pass()),
        ("reject", "Cairo", Pass()),
    ],
)
async def test_recorded_reply_is_judged_call_by_call(
    action: str, city: str, expected: Pass | Refuse
) -> None:
    handler = Handler(weather_rule(action, city))
    await handler.request(request())

    answer = await handler.reply(
        reply(recorded("parallel-tool-calls"), content_type=STREAM)
    )

    assert answer == expected


@pytest.mark.anyio
async def test_first_refused_call_ends_the_judging() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(request())

    await handler.reply(reply(recorded("parallel-tool-calls"), content_type=STREAM))

    records = handler.records("x")
    assert records is not None
    # Paris and Tokyo were judged; Lima, after the refusal, was not
    assert [record.kind for record in records] == ["record", "record"]
    assert [record.factory for record in records] == ["weather_rule", "weather_rule"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(recorded("text-only"), id="no tool calls"),
    ],
)
async def test_reply_without_calls_passes(body: bytes) -> None:
    handler = Handler(weather_rule())
    await handler.request(request())

    assert await handler.reply(reply(body, content_type=STREAM)) == Pass()


@pytest.mark.anyio
async def test_provider_error_passes_unread() -> None:
    handler = Handler(weather_rule())

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
    """Wait before letting a call go, as a monitor that asks a slow model does."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        await anyio.sleep(seconds)
        return Decision.reject("slow", message="Decided late.")

    return decide


@pytest.mark.anyio
async def test_sentinel_past_the_time_limit_is_cancelled() -> None:
    handler = Handler(takes(30), time_limit=0.05)
    await handler.request(replace(request(), headers={"x-irid": "irid-1"}))

    with pytest.raises(TimeLimitError, match="reply 'x' within 0.05 seconds"):
        await handler.reply(reply(message(weather("Oslo"))))

    # no decision was reached, and the rejection it was about to make is not kept
    result = handler.request_result("irid-1")
    assert isinstance(result, RequestResult)
    assert (result.state, result.reason) == ("not_judged", "time_limit")
    assert (result.outcome, result.message) == (None, None)
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


@pytest.mark.anyio
async def test_request_is_known_by_the_header_the_handler_is_told_to_read() -> None:
    headers = {"x-irid": "irid-1", "x-sentinel-request": "chosen-1"}
    default = Handler(weather_rule())
    named = Handler(weather_rule(), request_id_header="X-Sentinel-Request")

    for handler in (default, named):
        await handler.request(replace(request(), headers=headers))

    assert isinstance(default.request_result("irid-1"), RequestResult)
    assert default.request_result("chosen-1") == "unknown"
    assert isinstance(named.request_result("chosen-1"), RequestResult)
    assert named.request_result("irid-1") == "unknown"


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
        return Decision.reject(
            "context",
            message=f"{context.eval}|{context.path!r}|{[m.role for m in step.input]}",
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
        reply(recorded("parallel-tool-calls"), id="a", content_type=STREAM)
    )
    again = await handler.request(request("b"))

    assert refused == Refuse(
        503,
        "A sentinel rejected the call to get_weather: Tokyo is not allowed.",
        {"x-sentinel-decision": "reject"},
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
    ids = [
        block.get("id") or block.get("tool_use_id")
        for entry in added[len(sent) :]
        for block in entry["content"]
        if block["type"] in ("tool_use", "tool_result")
    ]
    assert ids == ["toolu_1", "toolu_1", "toolu_2", "toolu_2"]


@protocol
def reports_input() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("first try", message="Tokyo is not allowed.")
        return Decision.reject("input", message=str([m.role for m in step.input]))

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
        {"x-sentinel-decision": "terminate"},
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
        400,
        "A sentinel rejected the call to get_weather: This call was rejected.",
        {"x-sentinel-decision": "reject"},
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "result", "expected"),
    [
        (
            "terminate",
            "storm",
            # a decision's explanation is for the log, so the client isn't told it
            Refuse(
                400,
                "A sentinel ended this run at the result of the call to get_weather.",
                {"x-sentinel-decision": "terminate"},
            ),
        ),
        # nobody is left to take an escalate at the top, so the step goes on
        ("escalate", "storm", Pass()),
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

    records = handler.records("x")
    assert records is not None
    # Oslo's and Lima's were judged; Cairo's, after the run was ended, was not
    assert [record.factory for record in records] == ["storm_rule", "storm_rule"]


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
                    str([m.role for m in step.input]),
                    str([m.role for m in step.history]),
                ]
            )
        )

    return decide


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

    records = handler.records("x")
    assert records is not None
    assert [cast(Any, record.detail).report.explanation for record in records] == [
        "Looking.|get_weather|{'location': 'Oslo'}|rain|False|"
        "|['system', 'user']|['system', 'user', 'assistant']",
        # the conversation as it stood when the second call ran
        "Looking.|get_weather|{'location': 'Lima'}|no such city|True|"
        "|['system', 'user', 'assistant', 'tool']"
        "|['system', 'user', 'assistant', 'tool', 'assistant']",
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
    assert handler.records("b") == []


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
    records = handler.records("x")
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
    assert handler.records("b") == []


@protocol
def rejects_tokyo_and_reports_what_led_to_a_result() -> ProtocolGroup:
    async def before(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    async def after(context: Context, step: AfterToolCall) -> Decision | None:
        return Decision.proceed(str([m.role for m in step.input]))

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_a_result_is_judged_against_what_the_model_was_sent() -> None:
    handler = Handler(rejects_tokyo_and_reports_what_led_to_a_result())
    await handler.request(request("a"))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    await handler.request(request("b"))
    await handler.reply(reply(message(weather("Oslo", "toolu_2")), id="b"))

    await handler.request(request("c", then=ran("Oslo", "rain", "toolu_2")))

    records = handler.records("c")
    assert records is not None
    # the rejected turn was in what the model made the second call from
    assert cast(Any, records[0].detail).report.explanation == (
        "['system', 'user', 'assistant', 'tool']"
    )


@pytest.mark.anyio
async def test_runs_the_proxy_tells_apart_share_nothing() -> None:
    # two epochs of one sample: the same conversation, and nothing in common
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(request("a", run="epoch-1"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))

    other = await handler.request(request("b", run="epoch-2"))
    told = await handler.request(request("c", run="epoch-1"))

    assert other == Pass()
    assert isinstance(told, Replace)


@pytest.mark.anyio
async def test_a_run_the_proxy_names_is_one_run_however_its_conversations_open() -> (
    None
):
    handler = Handler(third_lookup_ends_the_run())

    async def lookup(id: str, opening: str) -> Pass | Refuse:
        await handler.request(request(id, opening, run="sample-1"))
        answer = await handler.reply(reply(message(weather("Oslo")), id=id))
        assert isinstance(answer, (Pass, Refuse))
        return answer

    answers = [
        await lookup("a", "The agent's conversation"),
        await lookup("b", "A subagent's conversation"),
        await lookup("c", "The agent's conversation"),
    ]

    assert [type(answer) for answer in answers] == [Pass, Pass, Refuse]


_ENDED = Refuse(
    400,
    "A sentinel ended this run at the call to get_weather.",
    {"x-sentinel-decision": "terminate"},
)


@pytest.mark.anyio
async def test_every_later_request_of_a_named_run_that_ended_is_refused() -> None:
    handler = Handler(weather_rule("terminate", "Tokyo"))
    await handler.request(request("a", run="sample-1"))
    ended = await handler.reply(reply(message(weather("Tokyo")), id="a"))

    again = await handler.request(
        replace(request("b", run="sample-1"), headers={"x-irid": "irid-b"})
    )
    subagent = await handler.request(
        request("c", "A subagent's conversation", run="sample-1")
    )
    other = await handler.request(request("d", run="sample-2"))

    assert (ended, again, subagent) == (_ENDED, _ENDED, _ENDED)
    assert other == Pass()
    # the sentinel was not asked again, and an eval that asks is told why
    assert handler.request_result("irid-b") == RequestResult(
        "decided", None, "terminate", _ENDED.message, []
    )


@pytest.mark.anyio
async def test_a_named_run_ended_at_a_result_is_refused_without_that_result() -> None:
    handler = Handler(storm_rule())
    ended = await handler.request(
        request("a", then=ran("Oslo", "storm"), run="sample-1")
    )

    later = await handler.request(request("b", run="sample-1"))

    assert isinstance(ended, Refuse)
    assert later == ended


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


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ("sample-uuid", r"sample-uuid\|None"),
        # how the conversation opens stands in for a run the proxy doesn't name
        (None, r"[0-9a-f]{16}\|None"),
    ],
)
async def test_sentinel_is_given_the_run_the_proxy_names(
    run: str | None, expected: str
) -> None:
    handler = Handler(reports_the_run())
    await handler.request(request(run=run))

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    assert re.search(f": {expected}$", answer.message)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (
            {"x-hawk-job-type": "eval-set", "x-irid": "request-1"},
            "anthropic|claude-test|user-1|[('x-hawk-job-type', 'eval-set'), ('x-irid', 'request-1')]",
        ),
        ({}, "anthropic|claude-test|user-1|[]"),
    ],
)
async def test_sentinel_is_given_what_the_proxy_said_of_the_request(
    headers: dict[str, str], expected: str
) -> None:
    handler = Handler(reports_the_proxy())
    await handler.request(replace(request(), headers=headers))

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    assert answer.message.endswith(f": {expected}")
