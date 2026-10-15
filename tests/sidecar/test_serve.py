import argparse
import json
import logging
import threading
import time
from dataclasses import replace

import httpx
import pytest
import uvicorn

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse
from inspect_sentinel.sidecar._proxy_endpoint import proxy_app
from inspect_sentinel.sidecar._serve import (
    _HealthCheckFilter,
    _quiet_health_checks,
    add_arguments,
    handler_from_arguments,
)
from tests.sidecar._traffic import message, reply, request, weather, weather_rule


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_health_checks_stay_out_of_the_access_log() -> None:
    # a real server, so the test reads the access log as uvicorn writes it
    server = uvicorn.Server(uvicorn.Config(proxy_app(Handler(weather_rule())), port=0))
    access = logging.getLogger("uvicorn.access")
    lines = _Lines()
    access.addHandler(lines)
    _quiet_health_checks()
    _quiet_health_checks()
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            checks = [client.get("/health"), client.get("/health?probe=1")]
            other = [
                client.post("/proxy/v1", content=b"{}"),
                client.get("/proxy/v1?next=/health"),
            ]
    finally:
        server.should_exit = True
        serving.join()
        access.removeHandler(lines)
        added = [
            kept for kept in access.filters if isinstance(kept, _HealthCheckFilter)
        ]
        for kept in added:
            access.removeFilter(kept)

    assert [answered.status_code for answered in checks] == [200, 200]
    assert [answered.status_code for answered in other] == [200, 405]
    # asking twice for quiet adds one filter
    assert len(added) == 1
    assert len(lines.lines) == 2
    assert '"POST /proxy/v1 HTTP/1.1" 200' in lines.lines[0]
    assert '"GET /proxy/v1?next=/health HTTP/1.1" 405' in lines.lines[1]


@protocol
def no_tokyo() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.arguments.get("location") == "Tokyo":
            return Decision.reject("Tokyo", message="Tokyo is not allowed.")
        return Decision.proceed()

    return decide


def _arguments(*given: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(list(given))


def test_defaults_are_one_port_and_no_limits() -> None:
    arguments = _arguments("--sentinel", "no_tokyo")

    assert (arguments.host, arguments.port) == ("127.0.0.1", 8900)
    assert arguments.reject_status == 400
    assert arguments.time_limit is None
    assert arguments.max_body_bytes == 64 * 1024 * 1024
    assert arguments.run_expiry == 24 * 60 * 60
    assert arguments.no_eval_results is False


@pytest.mark.anyio
async def test_handler_is_built_from_the_arguments() -> None:
    # a registered name, as an installed package's monitors and protocols are
    handler = handler_from_arguments(
        _arguments(
            "--sentinel",
            "no_tokyo",
            "--reject-status",
            "503",
            "--model-role",
            "monitor=mockllm/model",
        )
    )
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Tokyo"))))

    assert answer == Refuse(
        503, "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )


@pytest.mark.anyio
async def test_an_inspect_client_can_be_sent_only_what_it_acts_on() -> None:
    handler = handler_from_arguments(
        _arguments("--sentinel", "no_tokyo", "--no-eval-results")
    )
    sample = {
        "version": 1,
        "eval_id": "eval_789",
        "task": "philadelphia",
        "sample_id": 1,
        "epoch": 1,
        "sample_uuid": "sample_123",
        "attempt": "attempt_1",
    }
    await handler.request(
        replace(request(), headers={"x-inspect-context": json.dumps(sample)})
    )

    answer = await handler.reply(reply(message(weather("Tokyo"))))

    assert isinstance(answer, Refuse)
    assert answer.sentinel is not None
    assert answer.sentinel["trimmed"] is True


@pytest.mark.parametrize("role", ["monitor", "=mockllm/model", "monitor="])
def test_a_role_is_role_equals_model(role: str) -> None:
    with pytest.raises(ValueError, match="ROLE=MODEL"):
        handler_from_arguments(_arguments("--sentinel", "x", "--model-role", role))
