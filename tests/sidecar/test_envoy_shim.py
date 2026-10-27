"""The Envoy shim and Envoy's configuration, without Envoy: what Envoy sends is written by hand, for what a real Envoy can't be made to send on purpose."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import socket
import subprocess
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Pass, Replace, Reply, Request
from inspect_sentinel.sidecar._handler import RequestResult
from inspect_sentinel.sidecar._serve import (
    parse_arguments,
)
from inspect_sentinel.sidecar.envoy import (
    EnvoyShim,
    envoy_config,
    envoy_server,
    envoy_yaml,
)
from inspect_sentinel.sidecar.envoy._proto.envoy.config.core.v3 import base_pb2 as core
from inspect_sentinel.sidecar.envoy._proto.envoy.service.ext_proc.v3 import (
    external_processor_pb2 as pb,
)
from tests.sidecar._traffic import (
    every_call_rule,
    inspect_headers,
    message,
    weather,
    weather_rule,
)


@pytest.fixture
def anyio_backend() -> str:
    # gRPC's server runs on asyncio, and so does the shim
    return "asyncio"


_ANTHROPIC = (
    ("sentinel-format", "anthropic-messages"),
    ("sentinel-provider", "anthropic"),
)


class _Context:
    def __init__(self, metadata: Sequence[tuple[str, str]]) -> None:
        self._metadata = tuple(metadata)

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata


def _headers(headers: dict[str, str], *, end_of_stream: bool = False) -> pb.HttpHeaders:
    return pb.HttpHeaders(
        headers=core.HeaderMap(
            headers=[
                core.HeaderValue(key=name, raw_value=value.encode())
                for name, value in headers.items()
            ]
        ),
        end_of_stream=end_of_stream,
    )


async def _process(
    messages: Sequence[pb.ProcessingRequest],
    handler: Any = None,
    metadata: Sequence[tuple[str, str]] = _ANTHROPIC,
    max_body_bytes: int = 1024 * 1024,
) -> list[pb.ProcessingResponse]:
    async def sent() -> AsyncIterator[pb.ProcessingRequest]:
        for each in messages:
            yield each

    shim = EnvoyShim(handler or Handler(weather_rule()), max_body_bytes=max_body_bytes)
    # an exchange that is never answered would leave the test waiting
    with anyio.fail_after(10):
        return [
            answer
            async for answer in shim.Process(sent(), cast(Any, _Context(metadata)))
        ]


_REQUEST = [
    pb.ProcessingRequest(
        request_headers=_headers({"content-type": "application/json"})
    ),
    pb.ProcessingRequest(
        request_body=pb.HttpBody(
            body=b'{"model": "claude-test", "max_tokens": 1, "messages": [{"role": "user", "content": "Weather?"}]}',
            end_of_stream=True,
        )
    ),
]


@pytest.mark.anyio
async def test_empty_chunk_that_is_not_the_end_doesnt_end_the_reply() -> None:
    body = message(weather("Paris"), weather("Lima", "toolu_2"))
    middle = len(body) // 2
    answers = await _process(
        [
            *_REQUEST,
            pb.ProcessingRequest(
                response_headers=_headers(
                    {":status": "200", "content-type": "application/json"}
                )
            ),
            pb.ProcessingRequest(response_body=pb.HttpBody(body=body[:middle])),
            # Envoy 1.39 sends an empty chunk now and then mid-body
            pb.ProcessingRequest(response_body=pb.HttpBody(body=b"")),
            pb.ProcessingRequest(
                response_body=pb.HttpBody(body=body[middle:], end_of_stream=True)
            ),
        ]
    )

    released = [
        answer.response_body.response.body_mutation.streamed_response
        for answer in answers
        if answer.WhichOneof("response") == "response_body"
    ]
    # the reply whole, and then the end of the body
    assert b"".join(each.body for each in released) == body
    assert [each.end_of_stream for each in released] == [False, True]


@pytest.mark.anyio
async def test_route_that_doesnt_say_the_api_is_refused() -> None:
    answers = await _process(_REQUEST, metadata=[("sentinel-provider", "anthropic")])

    assert len(answers) == 1
    refused = answers[0].immediate_response
    assert refused.status.code == 500
    assert b"sentinel-format" in refused.body


@pytest.mark.anyio
async def test_inspects_headers_are_taken_out_of_the_request() -> None:
    answers = await _process(_REQUEST)

    assert list(answers[0].request_headers.response.header_mutation.remove_headers) == [
        "x-inspect-context",
        "x-irid",
    ]


def test_provider_at_a_path_is_refused() -> None:
    with pytest.raises(ValueError, match="has a path"):
        envoy_config(anthropic="https://gateway.example.com/anthropic")


def test_message_timeout_longer_than_envoy_allows_is_refused() -> None:
    with pytest.raises(ValueError, match="at most 3600"):
        envoy_config(message_timeout=7200)


def test_provider_left_out_has_no_routes() -> None:
    config = envoy_config(openai=None)
    manager = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][
        0
    ]["typed_config"]
    routes = manager["route_config"]["virtual_hosts"][0]["routes"]
    paths = [route["match"].get("path") for route in routes]

    assert "/v1/messages" in paths
    assert "/v1/chat/completions" not in paths
    assert "/v1/responses" not in paths
    assert "openai" not in [
        cluster["name"] for cluster in config["static_resources"]["clusters"]
    ]


_EXAMPLES = Path(__file__).parents[2] / "examples"
_ENVOY_EXAMPLES = ["always_sunny_envoy", "forecast_desk_envoy"]


@pytest.mark.parametrize("example", _ENVOY_EXAMPLES)
def test_examples_envoy_configuration_is_what_the_generator_writes(
    example: str,
) -> None:
    written = (_EXAMPLES / example / "envoy.yaml").read_text()
    # the file's own lines say how it was written, above the generator's
    generated = written[written.index("# Generated by inspect_sentinel") :]

    assert generated == envoy_yaml(sidecar_host="sidecar")


@pytest.mark.parametrize("example", _ENVOY_EXAMPLES)
def test_example_sidecar_serves_where_its_envoy_looks(example: str) -> None:
    arguments = parse_arguments(
        ["--sentinel", str(_EXAMPLES / example / "sentinel.yaml")]
    )
    clusters = {
        cluster["name"]: cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0][
            "endpoint"
        ]["address"]["socket_address"]["port_value"]
        for cluster in envoy_config(sidecar_host="sidecar")["static_resources"][
            "clusters"
        ]
    }

    assert clusters["sentinel_sidecar"] == arguments.port
    assert clusters["sentinel_shim"] == arguments.envoy_port


_BODY = b'{"model": "claude-test", "max_tokens": 1, "messages": [{"role": "user", "content": "Weather?"}]}'


def _from_inspect(**headers: str) -> pb.ProcessingRequest:
    return pb.ProcessingRequest(
        request_headers=_headers(
            {"content-type": "application/json", **inspect_headers(), **headers}
        )
    )


def _reply(*parts: bytes, **headers: str) -> list[pb.ProcessingRequest]:
    return [
        pb.ProcessingRequest(
            response_headers=_headers(
                {":status": "200", "content-type": "application/json", **headers}
            )
        ),
        *(
            pb.ProcessingRequest(
                response_body=pb.HttpBody(
                    body=part, end_of_stream=index == len(parts) - 1
                )
            )
            for index, part in enumerate(parts)
        ),
    ]


async def _not_judged(handler: Handler) -> tuple[str, str | None]:
    result = await handler.request_result("request_456")
    assert isinstance(result, RequestResult)
    return result.latest.state, result.latest.reason


@pytest.mark.anyio
async def test_compressed_request_is_refused_and_the_eval_told_it_wasnt_judged() -> (
    None
):
    handler = Handler(weather_rule())
    answers = await _process(
        [
            _from_inspect(**{"content-encoding": "gzip"}),
            pb.ProcessingRequest(
                request_body=pb.HttpBody(
                    body=b"\x1f\x8b compressed", end_of_stream=True
                )
            ),
        ],
        handler,
    )

    assert answers[-1].immediate_response.status.code == 400
    assert await _not_judged(handler) == ("not_judged", "unreadable")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("max_body_bytes", "headers"),
    [(64, {}), (1024 * 1024, {"content-encoding": "gzip"})],
    ids=["larger than the shim holds", "compressed"],
)
async def test_reply_refused_unread_tells_the_eval_it_wasnt_judged(
    max_body_bytes: int, headers: dict[str, str]
) -> None:
    handler = Handler(weather_rule())
    body = message(weather("Paris"))
    answers = await _process(
        [
            _from_inspect(),
            pb.ProcessingRequest(
                request_body=pb.HttpBody(body=_BODY, end_of_stream=True)
            ),
            *_reply(body[:100], body[100:], **headers),
        ],
        handler,
        max_body_bytes=max_body_bytes,
    )

    assert answers[-1].immediate_response.status.code == 502
    # not still awaiting the reply the client was refused
    assert await _not_judged(handler) == ("not_judged", "unreadable")


@pytest.mark.anyio
async def test_refusal_outside_the_error_statuses_is_sent_as_a_500() -> None:
    answers = await _process(
        [*_REQUEST, *_reply(message(weather("Paris")))],
        Handler(every_call_rule(), reject_status=700),
    )

    refused = answers[-1].immediate_response
    assert refused.status.code == 500
    # the body is written for the status sent
    assert json.loads(refused.body)["error"]["type"] == "api_error"


class _ReplacesEveryRequest:
    async def request(self, request: Request) -> Replace:
        return Replace(b"{}")


@pytest.mark.anyio
async def test_request_without_a_body_that_the_sidecar_replaces_is_refused() -> None:
    answers = await _process(
        [
            pb.ProcessingRequest(
                request_headers=_headers(
                    {"content-type": "application/json"}, end_of_stream=True
                )
            )
        ],
        _ReplacesEveryRequest(),
    )

    assert [answer.WhichOneof("response") for answer in answers] == [
        "immediate_response"
    ]
    assert answers[0].immediate_response.status.code == 500


def test_eval_endpoint_left_out_is_not_routed() -> None:
    config = envoy_config(eval_endpoint=False)
    manager = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][
        0
    ]["typed_config"]
    routes = manager["route_config"]["virtual_hosts"][0]["routes"]

    assert "/sentinel/" not in [route["match"].get("prefix") for route in routes]
    assert "sentinel_sidecar" in [
        cluster["name"] for cluster in config["static_resources"]["clusters"]
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("max_body_bytes", "headers"),
    [(64, {}), (1024 * 1024, {"content-encoding": "gzip"})],
    ids=["larger than the shim holds", "compressed"],
)
async def test_providers_error_passes_through_unjudged_whatever_its_size_or_encoding(
    max_body_bytes: int, headers: dict[str, str]
) -> None:
    handler = Handler(weather_rule())
    error = (
        b'{"type": "error", "error": {"type": "api_error", "message": "'
        + b"x" * 200
        + b'"}}'
    )
    replied = _reply(error[:100], error[100:], **headers)
    replied[0] = pb.ProcessingRequest(
        response_headers=_headers(
            {":status": "500", "content-type": "application/json", **headers}
        )
    )
    answers = await _process(
        [
            _from_inspect(),
            pb.ProcessingRequest(
                request_body=pb.HttpBody(body=_BODY, end_of_stream=True)
            ),
            *replied,
        ],
        handler,
        max_body_bytes=max_body_bytes,
    )

    passed = [
        answer.response_body.response.body_mutation.streamed_response
        for answer in answers
        if answer.WhichOneof("response") == "response_body"
    ]
    assert b"".join(each.body for each in passed) == error
    assert [each.end_of_stream for each in passed] == [False, True]
    assert await _not_judged(handler) == ("not_judged", "provider_error")


class _ReplacesEveryReply:
    async def request(self, request: Request) -> Pass:
        return Pass()

    async def reply(self, reply: Reply) -> Replace:
        return Replace(b"{}")


@pytest.mark.anyio
async def test_reply_without_a_body_that_the_sidecar_replaces_is_refused() -> None:
    answers = await _process(
        [
            *_REQUEST,
            pb.ProcessingRequest(
                response_headers=_headers(
                    {":status": "200", "content-type": "application/json"},
                    end_of_stream=True,
                )
            ),
        ],
        _ReplacesEveryReply(),
    )

    assert answers[-1].WhichOneof("response") == "immediate_response"
    assert answers[-1].immediate_response.status.code == 500


def _ipv6_loopback() -> bool:
    # some sandboxes and containers have IPv6 turned off
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


@pytest.mark.anyio
@pytest.mark.skipif(not _ipv6_loopback(), reason="IPv6 loopback is unavailable here")
async def test_shim_listens_on_an_ipv6_address_as_the_endpoints_do() -> None:
    # uvicorn takes `::1` bare, and gRPC only in brackets
    server = envoy_server(
        Handler(weather_rule()), host="::1", port=0, max_body_bytes=1024
    )
    await server.stop(None)


@protocol
def fails_once_released(judging: asyncio.Event, release: asyncio.Event) -> Protocol:
    """Fail at the first call, once `release` is set, setting `judging` when asked."""

    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        judging.set()
        await release.wait()
        raise RuntimeError("the rule broke")

    return decide


@pytest.mark.anyio
async def test_judgment_that_fails_after_envoy_stopped_waiting_is_logged_by_the_shim(
    caplog: pytest.LogCaptureFixture,
) -> None:
    judging, release = asyncio.Event(), asyncio.Event()
    shim = EnvoyShim(
        Handler(fails_once_released(judging, release)), max_body_bytes=1024 * 1024
    )
    reported: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, context: reported.append(context))

    async def sent() -> AsyncIterator[pb.ProcessingRequest]:
        for each in [*_REQUEST, *_reply(message(weather("Paris")))]:
            yield each
        await asyncio.Event().wait()

    async def envoy() -> None:
        async for _ in shim.Process(sent(), cast(Any, _Context(_ANTHROPIC))):
            pass

    waiting = asyncio.ensure_future(envoy())
    await judging.wait()
    waiting.cancel()
    with suppress(asyncio.CancelledError):
        await waiting
    with caplog.at_level(logging.ERROR):
        release.set()
        for _ in range(20):
            await asyncio.sleep(0)
        gc.collect()
    loop.set_exception_handler(None)

    # not left to asyncio, which reports an exception nothing retrieved
    assert reported == []
    # logged as a failure Envoy waited for is
    [logged] = caplog.records
    assert re.fullmatch(
        r"The reply '[0-9a-f]{32}' was not judged after its client had gone\.",
        logged.getMessage(),
    )
    assert logged.exc_info is not None


_CLASH = """
import sys

from google.protobuf import descriptor_pb2, descriptor_pool


def clash() -> None:
    # another copy of one of Envoy's protos, as xds-protos registers it
    descriptor_pool.Default().AddSerializedFile(
        descriptor_pb2.FileDescriptorProto(
            name="envoy/config/core/v3/socket_option.proto",
            package="envoy.config.core.v3",
            message_type=[descriptor_pb2.DescriptorProto(name="SocketOption")],
        ).SerializeToString()
    )


if sys.argv[1] == "before":
    clash()
import inspect_sentinel.sidecar.envoy
if sys.argv[1] == "after":
    clash()
"""


@pytest.mark.parametrize("backend", ["upb", "python"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_envoys_protos_load_beside_another_copy_of_them(
    when: str, backend: str
) -> None:
    # protobuf's default pool is the process's, so each case has one of its
    # own, on protobuf's compiled backend and on its pure-Python one, which
    # builds message classes another way
    loaded = subprocess.run(
        [sys.executable, "-W", "error", "-c", _CLASH, when],
        capture_output=True,
        text=True,
        env=os.environ | {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": backend},
    )

    assert loaded.returncode == 0, loaded.stderr


async def _until_blocked() -> None:
    # every task waits on something, which the shim's tasks do once settled
    for _ in range(50):
        await asyncio.sleep(0)


def _delivering() -> list[asyncio.Task[Any]]:
    return [
        task
        for task in asyncio.all_tasks()
        if "_deliver" in repr(task.get_coro()) and not task.done()
    ]


@pytest.mark.anyio(backends=["asyncio"])
async def test_nothing_is_left_delivering_once_envoy_stops_reading() -> None:
    shim = EnvoyShim(Handler(weather_rule()), max_body_bytes=1024 * 1024)

    async def sent() -> AsyncIterator[pb.ProcessingRequest]:
        for each in [*_REQUEST, *_reply(message(weather("Paris")))]:
            yield each
        await asyncio.Event().wait()

    answers = cast(
        AsyncGenerator[pb.ProcessingResponse],
        shim.Process(sent(), cast(Any, _Context(_ANTHROPIC))),
    )
    # the request's answers, and then nothing more is read: the reply's
    # answers wait to be delivered
    await answers.__anext__()
    await answers.__anext__()
    await _until_blocked()
    await answers.aclose()
    await _until_blocked()

    assert _delivering() == []


@pytest.mark.anyio(backends=["asyncio"])
async def test_a_request_sent_on_whose_reply_never_began_is_recorded_as_cut_off() -> (
    None
):
    handler = Handler(weather_rule())
    shim = EnvoyShim(handler, max_body_bytes=1024 * 1024)

    async def sent() -> AsyncIterator[pb.ProcessingRequest]:
        for each in [_from_inspect(), _REQUEST[1]]:
            yield each
        await asyncio.Event().wait()

    answers = cast(
        AsyncGenerator[pb.ProcessingResponse],
        shim.Process(sent(), cast(Any, _Context(_ANTHROPIC))),
    )
    await answers.__anext__()
    await answers.__anext__()
    # Envoy goes away before the provider's reply begins
    await answers.aclose()

    assert await _not_judged(handler) == ("not_judged", "cut_off")
