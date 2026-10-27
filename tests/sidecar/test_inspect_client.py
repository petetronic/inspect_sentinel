import json
from dataclasses import replace
from typing import Any

import anyio
import pytest
from inspect_ai.util import StoreModel

from inspect_sentinel._context import Context, InspectSample, ProxyContext
from inspect_sentinel._decorators import protocol
from inspect_sentinel._protocols import concurrent
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Step,
)
from inspect_sentinel._types import Protocol, ProtocolGroup
from inspect_sentinel.sidecar import (
    Handler,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
    UnreadableError,
    read_reply,
)
from inspect_sentinel.sidecar._handler import _ENDED, SampleResult
from inspect_sentinel.sidecar._inspect_client import with_sentinel
from inspect_sentinel.sidecar._wire import stream_events
from tests.sidecar._traffic import (
    JSON,
    SAMPLE,
    STREAM,
    anthropic_recorded,
    every_call_rule,
    inspect_headers,
    message,
    openai_recorded,
    openai_request,
    ran,
    reply,
    request,
    storm_rule,
    weather,
    weather_rule,
)


def _from_inspect(asked: Request, **changes: Any) -> Request:
    return replace(asked, headers=inspect_headers(**changes))


# each calls get_weather for Paris, then Tokyo, then Lima
_EXCHANGES = [
    "anthropic",
    "anthropic stream",
    *(
        f"openai {name}"
        for name in (
            "chat",
            "chat stream",
            "responses",
            "responses stream",
        )
    ),
]

# the event of each stream the field goes in
_LAST_EVENT = {
    "anthropic stream": ("type", "message_delta"),
    "openai chat stream": ("object", "chat.completion.chunk"),
    "openai responses stream": ("type", "response.completed"),
}


def _exchange(name: str, id: str = "x") -> tuple[Request, Reply]:
    if name.startswith("anthropic"):
        if name.endswith("stream"):
            content_type, body = STREAM, anthropic_recorded("parallel-tool-calls")
        else:
            content_type, body = (
                JSON,
                message(
                    weather("Paris", "toolu_1"),
                    weather("Tokyo", "toolu_2"),
                    weather("Lima", "toolu_3"),
                ),
            )
        return _from_inspect(request(id)), reply(body, id=id, content_type=content_type)
    recorded = name.removeprefix("openai ")
    content_type, body = openai_recorded(recorded)
    asked = Request(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        body=json.dumps(openai_request(recorded.split()[0])).encode(),
        headers=inspect_headers(),
    )
    answered = Reply(
        id=id,
        provider="openai",
        model="gpt-test",
        user="user-1",
        status=200,
        content_type=content_type,
        body=body,
    )
    return asked, answered


def _field(content_type: str, body: bytes) -> dict[str, Any]:
    if content_type == JSON:
        field: dict[str, Any] = json.loads(body)["sentinel"]
        return field
    carrying = [event for event in stream_events(body) if "sentinel" in event]
    assert len(carrying) == 1
    return carrying[0]["sentinel"]


def _stages(field: dict[str, Any]) -> list[tuple[str, str]]:
    return [(step["stage"], step["decision"]["action"]) for step in field["steps"]]


@pytest.mark.anyio
@pytest.mark.parametrize("name", _EXCHANGES)
async def test_a_reply_that_goes_on_carries_every_step_judged(
    name: str,
) -> None:
    handler = Handler(weather_rule("reject", "Cairo"))
    asked, answered = _exchange(name)
    assert await handler.request(asked) == Pass()

    answer = await handler.reply(answered)

    assert isinstance(answer, Replace)
    field = _field(answered.content_type, answer.body)
    assert (field["version"], field["outcome"]) == (1, "proceed")
    assert "withheld" not in field
    assert _stages(field) == [
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
    ]
    calls = (
        await read_reply(answered.provider, answered.content_type, answered.body)
    ).message.tool_calls or []
    assert [step.get("call") for step in field["steps"]] == [
        *(call.id for call in calls),
    ]
    # the provider's SDK reads the reply as it read the provider's
    output = await read_reply(answered.provider, answered.content_type, answer.body)
    assert output.message.tool_calls == calls


@pytest.mark.anyio
@pytest.mark.parametrize("name", list(_LAST_EVENT))
async def test_a_stream_carries_the_field_in_its_last_event_the_sdk_reads(
    name: str,
) -> None:
    handler = Handler(weather_rule("reject", "Cairo"))
    asked, answered = _exchange(name)
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Replace)
    given = list(stream_events(answered.body))
    sent = list(stream_events(answer.body))
    carrying = [index for index, event in enumerate(sent) if "sentinel" in event]
    key, value = _LAST_EVENT[name]
    assert carrying == [
        max(index for index, event in enumerate(given) if event.get(key) == value)
    ]
    # nothing else in the stream changed
    sent[carrying[0]].pop("sentinel")
    assert sent == given


@pytest.mark.anyio
@pytest.mark.parametrize("name", _EXCHANGES)
async def test_a_refused_turn_carries_the_rejected_calls_and_what_was_recorded(
    name: str,
) -> None:
    asked, answered = _exchange(name)
    output = await read_reply(answered.provider, answered.content_type, answered.body)
    calls = output.message.tool_calls or []
    # the recorded streams name the cities in more than one way
    cities = [call.arguments["location"] for call in calls]
    handler = Handler(every_call_rule())
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert answer.sentinel["outcome"] == "reject"
    assert _stages(dict(answer.sentinel)) == [("tool_call", "reject")] * 3
    # the whole turn is withheld, every call in it, with what the model is told
    # of each when the sidecar adds it to the next request
    withheld = answer.sentinel["withheld"]
    assert [call["id"] for call in withheld["message"]["tool_calls"]] == [
        call.id for call in calls
    ]
    assert withheld["message"]["role"] == "assistant"
    assert withheld["results"] == {
        call.id: f"{city} is not allowed."
        for call, city in zip(calls, cities, strict=True)
    }
    rejected = answer.sentinel["steps"][1]
    assert rejected["original"]["function"] == "get_weather"
    assert rejected["original"]["arguments"] == {"location": cities[1]}
    assert rejected["original"]["id"] == rejected["call"]
    assert rejected["decision"] == {
        "action": "reject",
        "explanation": f"lookup of {cities[1]}",
        "message": f"{cities[1]} is not allowed.",
    }
    assert [
        (record["kind"], record["path"], record["name"], record["call"])
        for record in rejected["recorded"]
    ] == [("record", "", "every_call_rule", rejected["call"])]
    assert rejected["recorded"][0]["report"] == rejected["decision"]
    # every call the client is not handed has its original
    assert all("original" in step for step in answer.sentinel["steps"])


@pytest.mark.anyio
@pytest.mark.parametrize("name", _EXCHANGES)
async def test_a_turn_with_a_call_taken_out_carries_what_the_model_proposed(
    name: str,
) -> None:
    asked, answered = _exchange(name)
    output = await read_reply(answered.provider, answered.content_type, answered.body)
    calls = output.message.tool_calls or []
    city = calls[1].arguments["location"]
    handler = Handler(weather_rule("reject", city))
    await handler.request(asked)

    answer = await handler.reply(answered)

    # the turn goes on without the rejected call
    assert isinstance(answer, Replace)
    handed = await read_reply(answered.provider, answered.content_type, answer.body)
    assert [call.id for call in handed.message.tool_calls or []] == [
        calls[0].id,
        calls[2].id,
    ]
    field = _field(answered.content_type, answer.body)
    assert field["outcome"] == "proceed"
    assert _stages(field) == [
        ("tool_call", "proceed"),
        ("tool_call", "reject"),
        ("tool_call", "proceed"),
    ]
    assert field["steps"][1]["original"]["arguments"] == {"location": city}
    # the turn as the model proposed it, for the eval to put back what was
    # taken out
    proposed = field["proposed"]
    assert [call["id"] for call in proposed["message"]["tool_calls"]] == [
        call.id for call in calls
    ]
    assert proposed["results"] == {calls[1].id: f"{city} is not allowed."}
    assert "withheld" not in field


@pytest.mark.anyio
async def test_a_run_ended_at_a_call_is_the_last_step() -> None:
    handler = Handler(weather_rule("terminate", "Tokyo"))
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert answer.sentinel["outcome"] == "terminate"
    assert _stages(dict(answer.sentinel))[-1] == ("tool_call", "terminate")
    assert "original" not in answer.sentinel["steps"][-1]
    # the turn the run ended at, which the client records; none of it ran
    withheld = answer.sentinel["withheld"]
    assert [call["id"] for call in withheld["message"]["tool_calls"]] == [
        "toolu_1",
        "toolu_2",
        "toolu_3",
    ]
    assert set(withheld["results"].values()) == {_ENDED}


@protocol
def ends_at_the_reply() -> Protocol:
    """End the run at the model's reply."""

    async def decide(
        context: Context, step: AfterGenerate, escalations: Escalations
    ) -> Decision | None:
        return Decision.terminate("the reply")

    return decide


@pytest.mark.anyio
async def test_a_run_ended_at_the_reply_carries_the_turn() -> None:
    handler = Handler(ends_at_the_reply())
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert answer.sentinel["outcome"] == "terminate"
    assert _stages(dict(answer.sentinel)) == [("model_output", "terminate")]
    withheld = answer.sentinel["withheld"]
    assert len(withheld["message"]["tool_calls"]) == 3
    assert set(withheld["results"].values()) == {_ENDED}


@pytest.mark.anyio
async def test_the_rejection_that_ends_the_run_says_the_run_ended() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    answers: list[Any] = []
    for id in ("a", "b", "c"):
        await handler.request(_from_inspect(request(id)))
        answers.append(await handler.reply(reply(message(weather("Tokyo")), id=id)))

    assert [answer.sentinel["outcome"] for answer in answers] == [
        "reject",
        "reject",
        "terminate",
    ]
    # the step itself was a rejection, the third in a row
    assert _stages(answers[-1].sentinel)[-1] == ("tool_call", "reject")


@pytest.mark.anyio
async def test_a_run_ended_at_a_result_is_refused_with_the_result_step() -> None:
    handler = Handler(storm_rule())
    asked = _from_inspect(request(then=ran("Oslo", "A storm.")))

    answer = await handler.request(asked)

    assert isinstance(answer, Refuse)
    assert answer.sentinel == {
        "version": 1,
        "outcome": "terminate",
        "steps": [
            {
                "stage": "tool_result",
                "call": "toolu_1",
                "decision": {"action": "terminate", "explanation": "a storm"},
                "recorded": [
                    {
                        "kind": "record",
                        "path": "",
                        "factory": "storm_rule",
                        "name": "storm_rule",
                        "function": "decide",
                        "report": {"action": "terminate", "explanation": "a storm"},
                        "call": "toolu_1",
                    }
                ],
            }
        ],
    }


@protocol
def ends_before_the_model() -> ProtocolGroup:
    async def result(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed()

    async def before(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        return Decision.terminate("never")

    return ProtocolGroup(result, before)


@protocol
def watches_every_stage() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed()

    return decide


@pytest.mark.anyio
async def test_a_reply_carries_the_request_steps_before_its_own() -> None:
    handler = Handler(watches_every_stage())
    asked, answered = _exchange("anthropic")
    await handler.request(replace(asked, body=request(then=ran("Oslo", "Sunny.")).body))

    answer = await handler.reply(answered)

    assert isinstance(answer, Replace)
    assert _stages(_field(answered.content_type, answer.body)) == [
        ("tool_result", "proceed"),
        ("model_input", "proceed"),
        ("model_output", "proceed"),
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
        ("tool_call", "proceed"),
    ]


@pytest.mark.anyio
async def test_a_run_ended_before_the_model_is_refused_with_the_request_steps() -> None:
    handler = Handler(ends_before_the_model())

    answer = await handler.request(_from_inspect(request(then=ran("Oslo", "Sunny."))))

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert _stages(dict(answer.sentinel)) == [
        ("tool_result", "proceed"),
        ("model_input", "terminate"),
    ]


@pytest.mark.anyio
async def test_a_request_of_an_ended_run_is_refused_with_no_steps() -> None:
    handler = Handler(storm_rule())
    await handler.request(_from_inspect(request("a", then=ran("Oslo", "A storm."))))

    answer = await handler.request(_from_inspect(request("b", "Something else")))

    assert isinstance(answer, Refuse)
    # the sample is the run, however its conversations open
    assert (
        answer.message
        == "A sentinel ended this run at the result of the call to get_weather."
    )
    assert answer.sentinel == {"version": 1, "outcome": "terminate", "steps": []}


@pytest.mark.anyio
async def test_each_sample_is_a_run_of_its_own() -> None:
    handler = Handler(storm_rule())
    await handler.request(_from_inspect(request("a", then=ran("Oslo", "A storm."))))

    # another epoch of the same sample opens the same way
    answer = await handler.request(
        _from_inspect(request("b"), epoch=3, sample_uuid="sample_124")
    )

    assert answer == Pass()


def _turn(location: str, id: str, told: str) -> list[dict[str, Any]]:
    # a rejected Anthropic turn put back in a conversation, as its client may
    return [
        {"role": "assistant", "content": [weather(location, id)]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": told,
                    "is_error": True,
                }
            ],
        },
    ]


@pytest.mark.anyio
async def test_a_client_that_put_a_withheld_turn_back_is_not_told_of_it_again() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_9")), id="a"))

    # the eval put the turn back itself, from the field
    answer = await handler.request(
        _from_inspect(
            request("b", then=_turn("Tokyo", "toolu_9", "Tokyo is not allowed."))
        )
    )

    assert answer == Pass()


@pytest.mark.anyio
async def test_a_client_that_didnt_put_a_withheld_turn_back_is_told_of_it() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_9")), id="a"))

    answer = await handler.request(_from_inspect(request("b")))

    assert isinstance(answer, Replace)
    roles = [entry["role"] for entry in json.loads(answer.body)["messages"]]
    assert roles == ["user", "assistant", "user"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "value",
    [
        "not json",
        json.dumps([SAMPLE]),
        json.dumps(SAMPLE | {"version": 2}),
        json.dumps({k: v for k, v in SAMPLE.items() if k != "sample_uuid"}),
        json.dumps(SAMPLE | {"epoch": "2"}),
    ],
)
async def test_a_header_that_cant_be_read_is_refused(value: str) -> None:
    # refused, not left unjudged, which a proxy set to fail open would let go
    answer = await Handler(weather_rule()).request(
        replace(request(), headers={"x-inspect-context": value})
    )

    assert answer == Refuse(
        400, "The request's x-inspect-context header isn't one this sidecar can read."
    )


@pytest.mark.anyio
async def test_a_new_attempt_is_not_told_of_a_turn_rejected_in_the_last() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_9")), id="a"))

    # a retry of the request within the attempt is told of it
    retried = await handler.request(_from_inspect(request("b")))
    # Inspect's retry of the sample opens as the attempt it replaces did
    afresh = await handler.request(_from_inspect(request("c"), attempt="attempt_2"))

    assert isinstance(retried, Replace)
    assert afresh == Pass()


@pytest.mark.anyio
async def test_a_new_attempt_counts_its_rejected_turns_in_a_row_afresh() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))

    async def rejected(id: str, attempt: str) -> Refuse:
        await handler.request(_from_inspect(request(id), attempt=attempt))
        answer = await handler.reply(
            reply(message(weather("Tokyo", f"toolu_{id}")), id=id)
        )
        assert isinstance(answer, Refuse)
        return answer

    await rejected("a", "attempt_1")
    await rejected("b", "attempt_1")
    third = await rejected("c", "attempt_2")

    # the third rejected turn of the sample, but the first of its attempt
    assert third.message == (
        "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )


class _Lookups(StoreModel):
    count: int = 0


@protocol
def counts_lookups_and_ends_at_storms() -> Protocol:
    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        if isinstance(step, AfterToolCall) and "storm" in step.result.text:
            return Decision.terminate("a storm")
        if isinstance(step, BeforeToolCall):
            lookups = context.store_as(_Lookups)
            lookups.count += 1
            return Decision.reject("counted", message=f"Lookup {lookups.count}.")
        return None

    return decide


@pytest.mark.anyio
async def test_a_new_attempt_starts_the_runs_store_and_its_end_afresh() -> None:
    handler = Handler(counts_lookups_and_ends_at_storms())
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Oslo", "toolu_1")), id="a"))

    # each attempt has a store of its own, as the sample store is
    await handler.request(_from_inspect(request("b"), attempt="attempt_2"))
    counted = await handler.reply(reply(message(weather("Oslo", "toolu_2")), id="b"))
    # and a run a sentinel ended is ended for that attempt only
    await handler.request(
        _from_inspect(request("c", then=ran("Oslo", "A storm.")), attempt="attempt_2")
    )
    ended = await handler.sample_result("sample_123")
    afresh = await handler.request(_from_inspect(request("d"), attempt="attempt_3"))
    latest = await handler.sample_result("sample_123")

    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."
    assert ended is not None
    assert ended.outcome == "terminate"
    assert afresh == Pass()
    assert latest == SampleResult("proceed", {})


@pytest.mark.anyio
async def test_a_replaced_attempts_requests_and_replies_are_refused_unjudged() -> None:
    handler = Handler(counts_lookups_and_ends_at_storms())
    await handler.request(_from_inspect(request("a")))
    await handler.request(_from_inspect(request("b"), attempt="attempt_2"))

    # the first attempt's reply arrives after its retry began
    late = await handler.reply(reply(message(weather("Oslo", "toolu_1")), id="a"))
    # and a request of the first attempt doesn't start it again
    again = await handler.request(_from_inspect(request("c")))
    counted = await handler.reply(reply(message(weather("Lima", "toolu_2")), id="b"))

    replaced = "A later attempt at this sample replaced this one."
    assert isinstance(late, Refuse)
    assert late.message == replaced
    assert isinstance(again, Refuse)
    assert again.message == replaced
    # the late reply was never judged, so it counted nothing
    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."


# set by the test that waits in a judgment
_GATE: dict[str, anyio.Event] = {}


@protocol
def waits_at_oslo_and_ends_at_paris() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        location = step.call.arguments.get("location")
        if location == "Oslo":
            _GATE["started"].set()
            await _GATE["release"].wait()
            return None
        lookups = context.store_as(_Lookups)
        lookups.count += 1
        if location == "Paris":
            return Decision.terminate("a storm in Paris")
        return Decision.reject("counted", message=f"Lookup {lookups.count}.")

    return decide


@pytest.mark.anyio
async def test_a_judgment_that_outlives_its_attempt_leaves_the_next_alone() -> None:
    _GATE.update(started=anyio.Event(), release=anyio.Event())
    handler = Handler(waits_at_oslo_and_ends_at_paris())
    await handler.request(_from_inspect(request("a")))
    answers: dict[str, object] = {}

    async def judged_slowly() -> None:
        # Paris is judged once the next attempt has begun
        turn = message(weather("Oslo", "toolu_1"), weather("Paris", "toolu_3"))
        answers["a"] = await handler.reply(reply(turn, id="a"))

    # the retry begins while the first attempt's reply is still being judged
    async with anyio.create_task_group() as group:
        group.start_soon(judged_slowly)
        await _GATE["started"].wait()
        await handler.request(_from_inspect(request("b"), attempt="attempt_2"))
        _GATE["release"].set()
    counted = await handler.reply(reply(message(weather("Lima", "toolu_2")), id="b"))
    result = await handler.sample_result("sample_123")

    # its count and its end stay with the attempt it was part of
    assert isinstance(answers["a"], Refuse)
    assert isinstance(counted, Refuse)
    assert counted.message == "A sentinel rejected the call to get_weather: Lookup 1."
    assert result is not None
    assert result.outcome == "proceed"


@pytest.mark.anyio
async def test_a_sample_the_eval_ended_stays_ended_for_a_new_attempt() -> None:
    handler = Handler(counts_lookups_and_ends_at_storms())
    await handler.request(_from_inspect(request("a")))
    await handler.end_sample("sample_123")

    answer = await handler.request(_from_inspect(request("b"), attempt="attempt_2"))

    assert isinstance(answer, Refuse)
    assert answer.message == "This run has ended."


@protocol
def reports_the_proxy() -> Protocol:
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        proxy = context.proxy
        assert proxy is not None
        return Decision.reject("proxy", message=repr(proxy))

    return decide


@pytest.mark.anyio
async def test_sentinel_is_given_what_the_proxy_and_the_header_say() -> None:
    handler = Handler(concurrent({"reports": reports_the_proxy()}))
    # a sample id outside ASCII arrives escaped, as a header value must be ASCII
    asked = _from_inspect(request(), sample_id="Zürich")
    await handler.request(asked)

    answer = await handler.reply(reply(message(weather("Oslo"))))

    assert isinstance(answer, Refuse)
    expected = ProxyContext(
        provider="anthropic",
        model="claude-test",
        user="user-1",
        headers=asked.headers,
        inspect=InspectSample(
            eval_id="eval_789",
            task="philadelphia",
            sample_id="Zürich",
            epoch=2,
            sample_uuid="sample_123",
            attempt="attempt_1",
        ),
    )
    # given to a monitor or protocol at any depth
    assert answer.message.endswith(f": {expected!r}")


@pytest.mark.anyio
async def test_a_client_without_the_header_is_handed_the_reply_as_it_was() -> None:
    handler = Handler(weather_rule("reject", "Cairo"))
    asked, answered = _exchange("anthropic stream")
    await handler.request(replace(asked, headers={}))

    assert await handler.reply(answered) == Pass()


@pytest.mark.anyio
async def test_a_client_without_the_header_is_refused_without_the_field() -> None:
    handler = Handler(weather_rule())
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Tokyo"))))

    assert isinstance(answer, Refuse)
    assert answer.sentinel is None


@pytest.mark.anyio
async def test_a_provider_error_goes_on_as_it_was() -> None:
    handler = Handler(weather_rule())
    await handler.request(_from_inspect(request()))

    assert await handler.reply(reply(b"overloaded", status=529)) == Pass()


@pytest.mark.anyio
async def test_without_eval_results_only_what_inspect_acts_on_is_sent() -> None:
    handler = Handler(every_call_rule(), eval_results=False)
    asked, answered = _exchange("anthropic")
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert answer.sentinel["trimmed"] is True
    assert all("recorded" not in step for step in answer.sentinel["steps"])
    rejected = answer.sentinel["steps"][1]
    assert rejected["decision"] == {
        "action": "reject",
        "message": "Tokyo is not allowed.",
    }
    assert rejected["original"]["arguments"] == {"location": "Tokyo"}


def test_a_stream_keeps_its_line_ends_and_each_events_other_lines() -> None:
    stream = (
        b"event: message_start\r\n"
        b'data: {"type": "message_start"}\r\n\r\n'
        b"event: message_delta\r\n"
        b'data: {"type":\r\n'
        b'data: "message_delta"}\r\n\r\n'
        b"event: message_stop\r\n"
        b'data: {"type": "message_stop"}\r\n\r\n'
    )

    written = with_sentinel(STREAM, stream, {"version": 1, "steps": []})

    assert written == (
        b"event: message_start\r\n"
        b'data: {"type": "message_start"}\r\n\r\n'
        b"event: message_delta\r\n"
        b'data: {"type": "message_delta", "sentinel": {"version": 1, "steps": []}}\r\n\r\n'
        b"event: message_stop\r\n"
        b'data: {"type": "message_stop"}\r\n\r\n'
    )


@pytest.mark.anyio
async def test_a_reply_judged_that_cant_carry_the_field_goes_on_without_it() -> None:
    handler = Handler(weather_rule("reject", "Tokyo"))
    sent = anthropic_recorded("single-tool-call")
    # a stream whose last event never came, which is read all the same
    cut = b"".join(
        event + b"\n\n"
        for event in sent.split(b"\n\n")
        if event and b"message_delta" not in event
    )
    await handler.request(_from_inspect(request("a")))

    answer = await handler.reply(reply(cut, id="a", content_type=STREAM))
    # the turn was passed, so the conversation goes on from it
    after = await handler.request(
        _from_inspect(request("b", then=ran("Paris", "Sunny.")))
    )

    assert answer == Pass()
    assert after == Pass()
    assert await handler.records("a") is not None


def test_a_stream_with_no_event_for_the_field_cant_carry_it() -> None:
    with pytest.raises(UnreadableError, match="no event"):
        with_sentinel(STREAM, b'data: {"type": "ping"}\n\n', {"version": 1})


@protocol
def rejects_tokyo_and_ends_at_its_result() -> ProtocolGroup:
    """Reject a lookup of Tokyo, and end the run if a result for one is ever judged, as a call that never ran has none."""

    async def before(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    async def after(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.terminate("a result for Tokyo")
        return Decision.proceed()

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_a_rejection_the_client_put_back_is_not_judged_as_a_result() -> None:
    handler = Handler(rejects_tokyo_and_ends_at_its_result())
    await handler.request(_from_inspect(request("a")))
    await handler.reply(
        reply(message(weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")), id="a")
    )

    # the eval put the turn back as the model proposed it, from the field,
    # with the rejection as inspect_ai writes it
    restored = [
        {
            "role": "assistant",
            "content": [weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "Tokyo is not allowed.",
                    "is_error": True,
                },
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Sunny."},
            ],
        },
    ]
    answer = await handler.request(_from_inspect(request("b", then=restored)))

    # nothing to add, and no result for Tokyo was judged
    assert answer == Pass()


@pytest.mark.anyio
async def test_a_result_for_a_call_taken_out_is_refused() -> None:
    handler = Handler(rejects_tokyo_and_ends_at_its_result())
    await handler.request(_from_inspect(request("a")))
    await handler.reply(
        reply(message(weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")), id="a")
    )

    # a client that claims to have run the call it was never handed
    ran_anyway = [
        {
            "role": "assistant",
            "content": [weather("Tokyo", "toolu_1"), weather("Paris", "toolu_2")],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Rain."},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Sunny."},
            ],
        },
    ]

    with pytest.raises(UnreadableError, match="never ran"):
        await handler.request(_from_inspect(request("b", then=ran_anyway)))


@protocol
def changes_tokyo_and_ends_at_its_result() -> ProtocolGroup:
    """Change a lookup of Tokyo to Oslo, and end the run if a result is ever judged as Tokyo's, as only the lookup of Oslo ran."""

    async def before(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            oslo = replace(step.call, arguments={"location": "Oslo"})
            return Decision(action="modify", modified=oslo)
        return Decision.proceed()

    async def after(
        context: Context, step: AfterToolCall, escalations: Escalations
    ) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.terminate("a result judged as Tokyo's")
        return Decision.proceed()

    return ProtocolGroup(before, after)


@pytest.mark.anyio
async def test_a_changed_calls_result_is_judged_as_the_call_that_ran() -> None:
    handler = Handler(changes_tokyo_and_ends_at_its_result())
    await handler.request(_from_inspect(request("a")))
    answer = await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    assert isinstance(answer, Replace)
    field = _field(JSON, answer.body)
    assert field["steps"][0]["original"]["arguments"] == {"location": "Tokyo"}
    assert field["steps"][0]["decision"]["modified"]["arguments"] == {
        "location": "Oslo"
    }

    # the eval keeps the model's call in its conversation, as in its own
    # process, with the result of the call that ran
    answer = await handler.request(
        _from_inspect(request("b", then=ran("Tokyo", "Snow.", "toolu_1")))
    )

    # nothing to put back, and the result was judged as Oslo's
    assert answer == Pass()


def _refused_turn_put_back(told: str) -> list[dict[str, Any]]:
    # a refused turn as an eval puts it back from the field
    return [
        {"role": "assistant", "content": [weather("Tokyo", "toolu_1")]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": told,
                    "is_error": True,
                }
            ],
        },
    ]


@pytest.mark.anyio
async def test_a_refused_turns_rejection_the_client_put_back_is_not_judged() -> None:
    handler = Handler(rejects_tokyo_and_ends_at_its_result())
    await handler.request(_from_inspect(request("a")))
    refused = await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))
    assert isinstance(refused, Refuse)

    answer = await handler.request(
        _from_inspect(
            request("b", then=_refused_turn_put_back("Tokyo is not allowed."))
        )
    )

    # nothing to add, and no result for Tokyo was judged
    assert answer == Pass()


@pytest.mark.anyio
async def test_another_result_for_a_refused_call_is_unreadable() -> None:
    handler = Handler(rejects_tokyo_and_ends_at_its_result())
    await handler.request(_from_inspect(request("a")))
    await handler.reply(reply(message(weather("Tokyo", "toolu_1")), id="a"))

    with pytest.raises(UnreadableError, match="refused turn"):
        await handler.request(
            _from_inspect(request("b", then=_refused_turn_put_back("Rain.")))
        )
