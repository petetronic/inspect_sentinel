import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import uvicorn

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeToolCall
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse
from inspect_sentinel.sidecar._eval_endpoint import TOKEN_VARIABLE
from inspect_sentinel.sidecar._proxy_endpoint import proxy_app
from inspect_sentinel.sidecar._registration import (
    RegistrationError,
    TaskRegistration,
    UnregisteredSampleError,
)
from inspect_sentinel.sidecar._serve import (
    _HealthCheckFilter,
    _quiet_health_checks,
    add_arguments,
    handler_from_arguments,
    parse_arguments,
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
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
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
    assert arguments.release == "whole"
    assert arguments.keepalive == 15.0
    assert arguments.time_limit is None
    assert arguments.max_body_bytes == 64 * 1024 * 1024
    assert arguments.run_expiry == 24 * 60 * 60
    assert arguments.no_eval_results is False
    assert arguments.tasks_folder is None
    assert arguments.no_runtime_task_registration is False
    assert arguments.strictness == "open"
    assert arguments.task_expiry == 30 * 24 * 60 * 60
    assert arguments.max_task_registrations == 1000
    assert (arguments.envoy_port, arguments.envoy_header) == (None, [])


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
        "attempt_number": 1,
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


@pytest.mark.anyio
async def test_tasks_registered_beforehand_are_read_from_the_tasks_folder(
    tmp_path: Path,
) -> None:
    (tmp_path / "weather.yaml").write_text(
        "tasks:\n  - task: philadelphia\n    samples:\n"
        "      - sample_id: 1\n        sample_input: Hi.\n"
    )
    handler = handler_from_arguments(
        _arguments(
            "--sentinel",
            "no_tokyo",
            "--tasks-folder",
            str(tmp_path),
            "--task-expiry",
            "60",
        )
    )

    # already registered, so nothing is kept
    assert (
        await handler.register_task(
            "eval_789", TaskRegistration(task="philadelphia", sample_ids=[], epochs=1)
        )
        is None
    )
    # a task the file doesn't hold is kept no longer than the limit set
    assert (
        await handler.register_task(
            "eval_789", TaskRegistration(task="another", sample_ids=[], epochs=1), 3600
        )
        == 60
    )


@pytest.mark.anyio
async def test_a_sidecar_can_be_told_to_take_no_registrations() -> None:
    handler = handler_from_arguments(
        _arguments("--sentinel", "no_tokyo", "--no-runtime-task-registration")
    )

    with pytest.raises(RegistrationError, match="takes no tasks"):
        await handler.register_task(
            "eval_789", TaskRegistration(task="philadelphia", sample_ids=[], epochs=1)
        )


@pytest.mark.anyio
async def test_a_sidecar_can_be_told_how_strict_to_be() -> None:
    handler = handler_from_arguments(
        _arguments("--sentinel", "no_tokyo", "--strictness", "registered_only")
    )

    # a request no eval names, such as a grader's
    with pytest.raises(UnregisteredSampleError, match="no x-inspect-context"):
        await handler.request(request())


_STANZA = """
sentinel:
  name: no_tokyo
server:
  model_roles:
    monitor: mockllm/model
  port: 9000
  reject_status: 503
  release: early
  keepalive: 5
  eval_results: false
  runtime_task_registration: false
  strictness: known_tasks
  task_expiry: 60
  max_task_registrations: 5
  envoy:
    port: 9001
    headers: [x-team]
tasks_folder: tasks
"""


def test_the_server_stanza_sets_what_the_options_would(tmp_path: Path) -> None:
    config = tmp_path / "sentinel.yaml"
    config.write_text(_STANZA)

    arguments = parse_arguments(["--sentinel", str(config)])

    assert arguments.model_role == ["monitor=mockllm/model"]
    assert (arguments.port, arguments.reject_status) == (9000, 503)
    assert arguments.release == "early"
    assert arguments.keepalive == 5
    assert (arguments.no_eval_results, arguments.no_runtime_task_registration) == (
        True,
        True,
    )
    assert arguments.strictness == "known_tasks"
    assert arguments.task_expiry == 60
    assert arguments.max_task_registrations == 5
    assert (arguments.envoy_port, arguments.envoy_header) == (9001, ["x-team"])
    # the tasks folder is found beside the configuration file
    assert arguments.tasks_folder == str(tmp_path.resolve() / "tasks")
    # the sentinel is built from the file's sentinel, which the core reads
    assert arguments.sentinel_config == {"name": "no_tokyo"}


@pytest.mark.parametrize(
    "given",
    [["--reject-status", "400"], ["--reject-status=400"]],
    ids=["option and value", "option=value"],
)
def test_an_option_given_wins_over_the_stanza_even_at_its_default(
    tmp_path: Path, given: list[str]
) -> None:
    config = tmp_path / "sentinel.yaml"
    config.write_text(_STANZA)

    arguments = parse_arguments(
        ["--sentinel", str(config), *given, "--tasks-folder", "elsewhere"]
    )

    assert arguments.reject_status == 400
    assert arguments.tasks_folder == "elsewhere"
    assert arguments.port == 9000


@pytest.mark.anyio
async def test_a_handler_is_built_from_a_file_with_a_stanza(tmp_path: Path) -> None:
    config = tmp_path / "sentinel.yaml"
    config.write_text("sentinel:\n  name: no_tokyo\nserver:\n  reject_status: 503\n")
    handler = handler_from_arguments(parse_arguments(["--sentinel", str(config)]))
    await handler.request(request())

    answer = await handler.reply(reply(message(weather("Tokyo"))))

    assert answer == Refuse(
        503, "A sentinel rejected the call to get_weather: Tokyo is not allowed."
    )


def test_a_file_without_a_stanza_is_read_as_an_eval_reads_it(tmp_path: Path) -> None:
    config = tmp_path / "sentinel.yaml"
    config.write_text("sentinel:\n  name: no_tokyo\n")

    arguments = parse_arguments(["--sentinel", str(config)])

    assert arguments.sentinel_config == str(config)
    assert arguments.reject_status == 400


def test_a_registered_name_is_found_as_an_eval_finds_it() -> None:
    assert parse_arguments(["--sentinel", "no_tokyo"]).sentinel_config == "no_tokyo"


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("sentinel:\n  name: no_tokyo\nserver:\n  colour: blue\n", "doesn't fit"),
        (
            "sentinel:\n  name: no_tokyo\nserver:\n  envoy:\n    colour: blue\n",
            "doesn't fit",
        ),
        (
            "sentinel:\n  name: no_tokyo\nserver: {}\nother: 1\n",
            "and nothing else",
        ),
        ("server:\n  port: 9000\n", "and nothing else"),
        ("sentinel:\n  name: no_tokyo\ntasks_folder: [a]\n", "a folder's path"),
        ("sentinel: [oops\n", "could not parse"),
        (
            "sentinel:\n  name: no_tokyo\n  name: no_paris\nserver:\n  port: 9000\n",
            "duplicate key 'name'",
        ),
    ],
    ids=[
        "a setting the server doesn't have",
        "a setting the Envoy shim doesn't have",
        "another key",
        "no sentinel",
        "a tasks folder that isn't a path",
        "not YAML",
        # read as the core reads a sentinel file, not with the last one winning
        "a key twice",
    ],
)
def test_a_configuration_that_cant_be_read(
    tmp_path: Path, text: str, says: str
) -> None:
    config = tmp_path / "sentinel.yaml"
    config.write_text(text)

    with pytest.raises(ValueError, match=says):
        parse_arguments(["--sentinel", str(config)])


def test_a_configuration_in_json_indented_with_tabs_is_read(tmp_path: Path) -> None:
    config = tmp_path / "sentinel.json"
    config.write_text(
        '{\n\t"sentinel": {"name": "no_tokyo"},\n\t"server": {"port": 9000}\n}\n'
    )

    arguments = parse_arguments(["--sentinel", str(config)])

    assert (arguments.sentinel_config, arguments.port) == ({"name": "no_tokyo"}, 9000)


_RULE = """
from inspect_sentinel import BeforeToolCall, Context, Decision, Escalations, protocol


@protocol
def lets_all_through():
    async def decide(
        context: Context, step: BeforeToolCall, escalations: Escalations
    ) -> Decision | None:
        return Decision.proceed()

    return decide
"""


@pytest.mark.parametrize(
    "second",
    [None, "--eval-port", "--envoy-port"],
    ids=["one port", "a port for evals", "the Envoy shim"],
)
def test_ctrl_c_stops_the_sidecar_cleanly(second: str | None, tmp_path: Path) -> None:
    (tmp_path / "lets_all_through.py").write_text(_RULE)
    free: list[int] = []
    for _ in range(1 if second is None else 2):
        with socket.socket() as found:
            found.bind(("127.0.0.1", 0))
            free.append(found.getsockname()[1])
    given = ["--port", str(free[0])]
    if second is not None:
        given += [second, str(free[1])]
    serving = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import lets_all_through\n"
            "from inspect_sentinel.sidecar.__main__ import main\n"
            "main()",
            "--sentinel",
            "lets_all_through",
            *given,
        ],
        cwd=tmp_path,
        env=os.environ | {TOKEN_VARIABLE: "s3cret"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                httpx.get(f"http://127.0.0.1:{free[0]}/health", timeout=1)
                break
            except httpx.HTTPError:
                assert serving.poll() is None and time.monotonic() < deadline
                time.sleep(0.1)
        if second == "--envoy-port":
            # the shim listens before the endpoints do
            socket.create_connection(("127.0.0.1", free[1]), timeout=5).close()
        serving.send_signal(signal.SIGINT)
        _, errors = serving.communicate(timeout=30)
    finally:
        if serving.poll() is None:
            serving.kill()

    assert serving.returncode == 0
    assert b"Traceback" not in errors


@pytest.mark.parametrize(
    ("given", "keepalive"), [(["--keepalive", "5"], 5.0), (["--keepalive", "0"], None)]
)
def test_the_handler_keeps_a_client_alive_as_the_arguments_say(
    given: list[str], keepalive: float | None
) -> None:
    handler = handler_from_arguments(
        _arguments("--sentinel", "no_tokyo", "--release", "early", *given)
    )

    assert handler.keepalive == keepalive
