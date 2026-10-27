import json
import re
from dataclasses import replace

import anyio
import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import (
    Answer,
    Finished,
    Handler,
    Output,
    Refuse,
    Release,
    Replace,
    Reply,
    ReplyStart,
    ReplyStream,
    Request,
)
from inspect_sentinel.sidecar._handler import RequestResult
from inspect_sentinel.sidecar._wire import UnreadableError
from tests.sidecar._traffic import (
    STREAM,
    anthropic_recorded,
    inspect_headers,
    openai_recorded,
    openai_request,
    request,
    weather_rule,
)

_RECORDED = ["anthropic stream", "openai chat stream", "openai responses stream"]


def _exchange(name: str, *, inspect: bool = False) -> tuple[Request, Reply]:
    # a recorded reply calling get_weather for Paris, Tokyo and Lima at once,
    # and the request it answers
    headers = inspect_headers() if inspect else {}
    if name == "anthropic stream":
        asked = request(sample="sample_123") if inspect else request()
        provider, body = "anthropic", anthropic_recorded("parallel-tool-calls")
    else:
        api = name.split()[1]
        provider, (_, body) = "openai", openai_recorded(f"{api} stream")
        asked = Request(
            id="x",
            provider="openai",
            model="gpt-test",
            user="user-1",
            body=json.dumps(openai_request(api)).encode(),
            headers=headers,
        )
    answered = Reply(
        id="x",
        provider=provider,
        model=asked.model,
        user="user-1",
        status=200,
        content_type=STREAM,
        body=body,
    )
    return asked, answered


def _start(reply: Reply, status: int | None = None) -> ReplyStart:
    return ReplyStart(
        id=reply.id,
        provider=reply.provider,
        model=reply.model,
        user=reply.user,
        status=reply.status if status is None else status,
        content_type=reply.content_type,
        format=reply.format,
    )


async def _streamed(stream: ReplyStream, body: bytes, size: int) -> list[Output]:
    # run the stream, feed it `body` in chunks of `size`, and read what comes
    # out, all at once, as a proxy does
    outputs: list[Output] = []

    async def read() -> None:
        async for output in stream:
            outputs.append(output)

    async with anyio.create_task_group() as group:
        group.start_soon(stream.run)
        group.start_soon(read)
        for at in range(0, len(body), size):
            await stream.feed(body[at : at + size])
        await stream.end()
    return outputs


def _expected(answer: Answer, body: bytes) -> list[Output]:
    if isinstance(answer, Refuse):
        return [answer]
    return [Release(answer.body if isinstance(answer, Replace) else body), Finished()]


def _comparable(outputs: list[Output]) -> list[Output]:
    # inspect_ai gives a message it reads a new id each time, so a withheld
    # turn's id differs between two handlers that judged the same reply
    return [
        Release(
            re.sub(rb'("proposed": \{"message": \{"id": )"[^"]*"', rb'\1"id"', o.data)
        )
        if isinstance(o, Release)
        else _without_turn_id(o)
        if isinstance(o, Refuse)
        else o
        for o in outputs
    ]


def _without_turn_id(refused: Refuse) -> Refuse:
    if refused.sentinel is None or "withheld" not in refused.sentinel:
        return refused
    withheld = refused.sentinel["withheld"]
    message = {**withheld["message"], "id": "id"}
    sentinel = {**refused.sentinel, "withheld": {**withheld, "message": message}}
    return replace(refused, sentinel=sentinel)


@pytest.mark.anyio
@pytest.mark.parametrize("name", _RECORDED)
@pytest.mark.parametrize(
    ("action", "city"),
    [("reject", "Tokyo"), ("reject", "Cairo"), ("terminate", "Tokyo")],
    ids=["one call rejected", "nothing rejected", "run ended"],
)
@pytest.mark.parametrize("size", [1, 97, 1 << 30], ids=["bytes", "chunks", "whole"])
@pytest.mark.parametrize("inspect", [False, True], ids=["any client", "inspect"])
async def test_a_reply_held_whole_comes_out_as_reply_answers_it(
    name: str, action: str, city: str, size: int, inspect: bool
) -> None:
    asked, answered = _exchange(name, inspect=inspect)
    whole = Handler(weather_rule(action, city))
    await whole.request(asked)
    answer = await whole.reply(answered)
    streaming = Handler(weather_rule(action, city))
    await streaming.request(asked)

    outputs = await _streamed(
        streaming.reply_stream(_start(answered)), answered.body, size
    )

    assert _comparable(outputs) == _comparable(_expected(answer, answered.body))


@pytest.mark.anyio
async def test_a_provider_error_goes_out_as_it_arrives() -> None:
    asked, answered = _exchange("anthropic stream", inspect=True)
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = handler.reply_stream(_start(answered, status=529))
    error = b'{"type":"error","error":{"type":"overloaded_error"}}'

    # a part held back would leave the test waiting, so it fails fast instead
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            outputs = stream.__aiter__()
            await stream.feed(error[:10])
            # the first part is sent before the rest has arrived
            assert await outputs.__anext__() == Release(error[:10])
            await stream.feed(error[10:])
            assert await outputs.__anext__() == Release(error[10:])
            await stream.end()
            assert await outputs.__anext__() == Finished()
            with pytest.raises(StopAsyncIteration):
                await outputs.__anext__()

    assert await _not_judged(handler) == ("not_judged", "provider_error")


@pytest.mark.anyio
async def test_a_provider_error_is_not_held_to_the_limit() -> None:
    asked, answered = _exchange("anthropic stream")
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = handler.reply_stream(_start(answered, status=500), max_held_bytes=4)

    outputs = await _streamed(stream, b"an error longer than the limit", 7)

    assert b"".join(o.data for o in outputs if isinstance(o, Release)) == (
        b"an error longer than the limit"
    )
    assert outputs[-1] == Finished()


@pytest.mark.anyio
async def test_a_reply_larger_than_the_limit_is_not_judged() -> None:
    # raised, as for a reply that can't be read, so that a proxy that lets
    # through what isn't judged can, rather than refused as if decided
    asked, answered = _exchange("anthropic stream", inspect=True)
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = handler.reply_stream(_start(answered), max_held_bytes=100)

    with pytest.raises(ExceptionGroup) as raised:
        await _streamed(stream, answered.body, 40)

    assert raised.group_contains(UnreadableError, match="larger than the 100 bytes")
    assert await _not_judged(handler) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_a_slow_proxy_holds_back_the_feeding_of_a_provider_error() -> None:
    asked, answered = _exchange("anthropic stream")
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = ReplyStream(
        handler,
        _start(answered, status=500),
        max_held_bytes=1 << 30,
        max_pending_bytes=10,
    )
    fed = 0

    async def feed_all() -> None:
        nonlocal fed
        for _ in range(20):
            await stream.feed(b"x" * 5)
            fed += 1
        await stream.end()

    collected: list[Output] = []
    async with anyio.create_task_group() as group:
        group.start_soon(stream.run)
        group.start_soon(feed_all)
        outputs = stream.__aiter__()
        await anyio.wait_all_tasks_blocked()
        # nothing collected yet: the queue, the worker and the backlog are full,
        # and feeding waits
        assert fed < 10
        async for output in outputs:
            collected.append(output)

    assert b"".join(o.data for o in collected if isinstance(o, Release)) == b"x" * 100
    assert collected[-1] == Finished()


@pytest.mark.anyio
async def test_an_abort_ends_a_stream_whose_output_no_one_is_collecting() -> None:
    asked, answered = _exchange("anthropic stream")
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = ReplyStream(
        handler,
        _start(answered, status=500),
        max_held_bytes=1 << 30,
        max_pending_bytes=10,
    )

    async def feed_all() -> None:
        for _ in range(20):
            await stream.feed(b"x" * 5)
        await stream.end()

    # a stream that didn't end would leave the test waiting, so it fails fast
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            group.start_soon(feed_all)
            # the output is full, and the worker waits to deliver more
            await anyio.wait_all_tasks_blocked()
            await stream.abort()

    # what was waiting is never handed out
    assert [output async for output in stream] == []


@pytest.mark.anyio
async def test_a_reply_is_given_up_as_soon_as_it_is_fed_past_the_limit() -> None:
    asked, answered = _exchange("anthropic stream")
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = handler.reply_stream(_start(answered), max_held_bytes=100)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(stream.run)
            outputs = stream.__aiter__()
            await stream.feed(answered.body[:60])
            await stream.feed(answered.body[60:120])
            # given up before the reply has ended
            with pytest.raises(UnreadableError):
                await outputs.__anext__()
            # what comes after is dropped, without waiting
            await stream.feed(answered.body[120:])
            await stream.end()


@pytest.mark.anyio
async def test_a_reply_cut_off_before_its_end_is_recorded_as_cut_off() -> None:
    asked, answered = _exchange("anthropic stream", inspect=True)
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = handler.reply_stream(_start(answered))
    outputs: list[Output] = []

    async def read() -> None:
        async for output in stream:
            outputs.append(output)

    async with anyio.create_task_group() as group:
        group.start_soon(stream.run)
        group.start_soon(read)
        await stream.feed(answered.body[:200])
        await stream.abort()

    assert outputs == []
    assert await _not_judged(handler) == ("not_judged", "cut_off")


@protocol
def decides_when(started: anyio.Event, go: anyio.Event) -> Protocol:
    """Reject every call, once `go` is set, setting `started` when first asked."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        started.set()
        await go.wait()
        return Decision.reject("held", message="Not this one.")

    return decide


@pytest.mark.anyio
async def test_a_judgment_goes_on_after_the_client_has_gone() -> None:
    asked, answered = _exchange("anthropic stream", inspect=True)
    started, go = anyio.Event(), anyio.Event()
    handler = Handler(decides_when(started, go))
    await handler.request(asked)
    stream = handler.reply_stream(_start(answered))
    outputs: list[Output] = []

    async def read() -> None:
        async for output in stream:
            outputs.append(output)

    async with anyio.create_task_group() as group:
        group.start_soon(stream.run)
        group.start_soon(read)
        await stream.feed(answered.body)
        await stream.end()
        await started.wait()
        assert await _not_judged(handler) == ("judging", None)
        await stream.abort()
        go.set()

    # decided, and kept for the run, though nothing was sent
    assert outputs == []
    assert (await _not_judged(handler))[0] == "decided"


@protocol
def breaks() -> Protocol:
    """Fail at every call."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        raise RuntimeError("the rule broke")

    return decide


@pytest.mark.anyio
async def test_what_judging_raises_is_raised_to_the_reader() -> None:
    asked, answered = _exchange("anthropic stream")
    handler = Handler(breaks())
    await handler.request(asked)

    with pytest.raises(ExceptionGroup) as raised:
        await _streamed(handler.reply_stream(_start(answered)), answered.body, 500)

    assert raised.group_contains(RuntimeError, match="the rule broke")


@pytest.mark.anyio
async def test_bytes_after_the_end_are_a_mistake() -> None:
    asked, answered = _exchange("anthropic stream")
    handler = Handler(weather_rule())
    await handler.request(asked)
    stream = handler.reply_stream(_start(answered))
    await stream.end()

    with pytest.raises(RuntimeError, match="already been ended"):
        await stream.feed(b"more")


async def _not_judged(handler: Handler) -> tuple[str, str | None]:
    result = await handler.request_result("request_456")
    assert isinstance(result, RequestResult)
    return result.latest.state, result.latest.reason
