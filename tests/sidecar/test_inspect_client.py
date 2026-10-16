import json
from dataclasses import replace
from typing import Any

import pytest

from inspect_sentinel._context import Context, InspectSample, ProxyContext
from inspect_sentinel._decorators import protocol
from inspect_sentinel._protocols import concurrent
from inspect_sentinel._report import Decision
from inspect_sentinel._step import AfterToolCall, BeforeGenerate, BeforeToolCall, Step
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
from inspect_sentinel.sidecar._inspect_client import with_sentinel
from inspect_sentinel.sidecar._wire import stream_events
from tests.sidecar._traffic import (
    JSON,
    SAMPLE,
    STREAM,
    anthropic_recorded,
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
async def test_a_refused_turn_carries_the_rejected_call_and_what_was_recorded(
    name: str,
) -> None:
    asked, answered = _exchange(name)
    output = await read_reply(answered.provider, answered.content_type, answered.body)
    # the recorded streams name the cities in more than one way
    city = (output.message.tool_calls or [])[1].arguments["location"]
    handler = Handler(weather_rule("reject", city))
    await handler.request(asked)

    answer = await handler.reply(answered)

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert answer.sentinel["outcome"] == "reject"
    assert _stages(dict(answer.sentinel)) == [
        ("tool_call", "proceed"),
        ("tool_call", "reject"),
        ("tool_call", "proceed"),
    ]
    # the whole turn is withheld, every call in it, with what the model is told
    # of each when the sidecar adds it to the next request
    withheld = answer.sentinel["withheld"]
    calls = output.message.tool_calls or []
    assert [call["id"] for call in withheld["message"]["tool_calls"]] == [
        call.id for call in calls
    ]
    assert withheld["message"]["role"] == "assistant"
    assert withheld["results"] == {
        calls[
            0
        ].id: "This call was not run, because another call in the same turn was rejected.",
        calls[1].id: f"{city} is not allowed.",
        calls[
            2
        ].id: "This call was not run, because another call in the same turn was rejected.",
    }
    rejected = answer.sentinel["steps"][1]
    assert rejected["original"]["function"] == "get_weather"
    assert rejected["original"]["arguments"] == {"location": city}
    assert rejected["original"]["id"] == rejected["call"]
    assert rejected["decision"] == {
        "action": "reject",
        "explanation": f"lookup of {city}",
        "message": f"{city} is not allowed.",
    }
    assert [
        (record["kind"], record["path"], record["name"], record["call"])
        for record in rejected["recorded"]
    ] == [("record", "", "weather_rule", rejected["call"])]
    assert rejected["recorded"][0]["report"] == rejected["decision"]
    # only a rejected call has an original
    assert "original" not in answer.sentinel["steps"][0]
    assert "original" not in answer.sentinel["steps"][2]


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
    assert "withheld" not in answer.sentinel


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
    async def result(context: Context, step: AfterToolCall) -> Decision | None:
        return Decision.proceed()

    async def before(context: Context, step: BeforeGenerate) -> Decision | None:
        return Decision.terminate("never")

    return ProtocolGroup(result, before)


@protocol
def watches_every_stage() -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
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
async def test_a_header_that_cant_be_read_leaves_the_request_unjudged(
    value: str,
) -> None:
    with pytest.raises(UnreadableError, match="x-inspect-context"):
        await Handler(weather_rule()).request(
            replace(request(), headers={"x-inspect-context": value})
        )


@protocol
def reports_the_proxy() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
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
    handler = Handler(weather_rule("reject", "Tokyo"), eval_results=False)
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


def test_a_stream_with_no_event_for_the_field_cant_carry_it() -> None:
    with pytest.raises(UnreadableError, match="no event"):
        with_sentinel(STREAM, b'data: {"type": "ping"}\n\n', {"version": 1})
