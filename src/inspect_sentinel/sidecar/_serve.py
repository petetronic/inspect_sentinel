from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import anyio
import uvicorn
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette

from .._config import sentinel_from_config
from ._eval_endpoint import TOKEN_VARIABLE, eval_routes
from ._handler import Handler
from ._host import SidecarHost
from ._http import HEALTH_PATH, MAX_BODY_BYTES, health_route
from ._proxy_endpoint import proxy_app
from ._registration import MAX_TASK_EXPIRY, read_tasks_file
from ._state import RUN_EXPIRY, MemoryState


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of the sidecar's command line.

    Args:
        parser: The parser to add them to.
    """
    parser.add_argument(
        "--sentinel",
        required=True,
        help="A YAML or JSON sentinel configuration file, or the registered name of a monitor or protocol. Beside `sentinel`, the file may hold a `sidecar` stanza with the sidecar's own settings, which the options given here override.",
    )
    parser.add_argument(
        "--model-role",
        action="append",
        default=[],
        metavar="ROLE=MODEL",
        help="The model a role generates with, e.g. monitor=anthropic/claude-haiku-4-5. May be repeated.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="The address to listen on.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8900,
        help="The port to listen on.",
    )
    parser.add_argument(
        "--eval-port",
        type=int,
        default=None,
        metavar="PORT",
        help=f"Serve the endpoint for evals on a port of its own, with a token of its own read from {TOKEN_VARIABLE}, for a deployment where something other than the eval can make model calls through the proxy. By default it shares --port, and the proxy's access control.",
    )
    parser.add_argument(
        "--reject-status",
        type=int,
        default=400,
        metavar="STATUS",
        help="The HTTP status a rejected turn is refused with. 400, the default, is final to a client. One a client retries by itself, such as 503, brings it back, and the model is then told of the rejection.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="The longest the sentinel may take to judge one request or one reply, after which it is cancelled and the proxy is told it couldn't be judged. There is no limit by default: the sentinel finishes however long it takes.",
    )
    parser.add_argument(
        "--no-eval-results",
        action="store_true",
        help="Send an Inspect eval only what it acts on: each step's stage, call and decision, without the explanations, references, metadata and records. A monitor's explanation tells an agent what the monitor looks for, so use this where the eval can't be trusted.",
    )
    parser.add_argument(
        "--tasks",
        metavar="FILE",
        help="A YAML or JSON tasks file: the tasks registered beforehand, which an eval can't replace. A request for a sample of one of them that the file doesn't hold is refused.",
    )
    parser.add_argument(
        "--no-runtime-task-registration",
        action="store_true",
        help="Take no tasks registered by an eval at task init. Where an eval can't be trusted, tasks are registered beforehand with --tasks instead.",
    )
    parser.add_argument(
        "--task-expiry",
        type=float,
        default=MAX_TASK_EXPIRY,
        metavar="SECONDS",
        help="The longest a task an eval registered is kept once no request has named it, whatever the eval asks. 30 days by default.",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=MAX_BODY_BYTES,
        metavar="BYTES",
        help="The largest request or reply the proxy may hand over. A larger one is answered with a 413, which a proxy takes as one that couldn't be judged. It has to cover the largest reply the proxy holds, which is sent as base64.",
    )
    parser.add_argument(
        "--run-expiry",
        type=float,
        default=RUN_EXPIRY,
        metavar="SECONDS",
        help="How long a run's state is kept once no request has named it. A day by default.",
    )


class SidecarSettings(BaseModel):
    """The `sidecar` stanza of a sentinel configuration file: the sidecar's own settings, beside the sentinel's.

    Each key is one of the sidecar's command-line options, with underscores, and a `--no-` option is the setting turned off. Model roles are `model_roles`, as inspect_ai names them.
    """

    model_config = ConfigDict(extra="forbid")

    model_roles: dict[str, str] | None = None
    host: str | None = None
    port: int | None = None
    eval_port: int | None = None
    reject_status: int | None = None
    time_limit: float | None = None
    eval_results: bool | None = None
    tasks: str | None = None
    runtime_task_registration: bool | None = None
    task_expiry: float | None = None
    max_body_bytes: int | None = None
    run_expiry: float | None = None


# each setting of the stanza, with the option that overrides it, and what it
# sets the option to
_OPTIONS: dict[str, tuple[str, str, Callable[[Any, Path], Any]]] = {
    "model_roles": (
        "--model-role",
        "model_role",
        lambda roles, _: [f"{role}={model}" for role, model in roles.items()],
    ),
    "eval_results": ("--no-eval-results", "no_eval_results", lambda on, _: not on),
    "runtime_task_registration": (
        "--no-runtime-task-registration",
        "no_runtime_task_registration",
        lambda on, _: not on,
    ),
    # a file the stanza names is found beside the configuration file
    "tasks": ("--tasks", "tasks", lambda tasks, base: str(base / tasks)),
    **{
        name: (f"--{name.replace('_', '-')}", name, lambda value, _: value)
        for name in (
            "host",
            "port",
            "eval_port",
            "reject_status",
            "time_limit",
            "task_expiry",
            "max_body_bytes",
            "run_expiry",
        )
    },
}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the sidecar's command line, with the settings of the configuration file's `sidecar` stanza under it.

    An option given on the command line wins over the stanza, and the stanza wins over the option's default.

    Args:
        argv: The arguments, without the program's name. The process's own if not given.

    Raises:
        ValueError: If the configuration file can't be read, or its stanza holds a setting the sidecar doesn't have.
    """
    given = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m inspect_sentinel.sidecar",
        description="Run a sentinel in a sidecar, beside a model proxy.",
        # an abbreviated option would go unnoticed as overriding the stanza
        allow_abbrev=False,
    )
    add_arguments(parser)
    arguments = parser.parse_args(given)
    sentinel, settings = read_configuration(arguments.sentinel)
    arguments.sentinel_config = sentinel
    if settings is None:
        return arguments
    base = Path(arguments.sentinel).resolve().parent
    for name, value in settings.model_dump(exclude_none=True).items():
        option, dest, convert = _OPTIONS[name]
        if not any(arg == option or arg.startswith(f"{option}=") for arg in given):
            setattr(arguments, dest, convert(value, base))
    return arguments


def read_configuration(config: str) -> tuple[Any, SidecarSettings | None]:
    """Read a sentinel configuration, and its `sidecar` stanza if it has one.

    Args:
        config: A YAML or JSON sentinel configuration file, or the registered name of a monitor or protocol.

    Returns:
        What to build the sentinel from, which is `config` itself unless the file has a stanza, and the stanza's settings, or None.

    Raises:
        ValueError: If the file has keys other than `sentinel` and `sidecar`, or its stanza holds a setting the sidecar doesn't have.
    """
    path = Path(config)
    if not path.is_file():
        # a registered name, which the core loader finds
        return config, None
    try:
        content: object = yaml.safe_load(path.read_text())
    except yaml.YAMLError as ex:
        raise ValueError(f"{config}: could not parse the file: {ex}") from None
    if not isinstance(content, dict) or "sidecar" not in content:
        # the core loader reads a file without a stanza, as an eval does
        return config, None
    content = cast("dict[str, Any]", content)
    if set(content) != {"sentinel", "sidecar"}:
        raise ValueError(
            f"{config}: a sentinel configuration file with a sidecar stanza holds only `sentinel` and `sidecar`."
        )
    try:
        settings = SidecarSettings.model_validate(content["sidecar"] or {})
    except ValidationError as ex:
        raise ValueError(f"{config}: the sidecar stanza doesn't fit: {ex}") from None
    return content["sentinel"], settings


def handler_from_arguments(arguments: argparse.Namespace) -> Handler:
    """Build the handler the arguments describe.

    Args:
        arguments: Parsed arguments that `add_arguments` defined.

    Raises:
        ValueError: If a `--model-role` is not `ROLE=MODEL`, or the sentinel configuration or the tasks file is invalid.
    """
    roles: dict[str, str] = {}
    for role in arguments.model_role:
        name, sep, model = str(role).partition("=")
        if not sep or not name or not model:
            raise ValueError(f"--model-role takes ROLE=MODEL, not {role!r}.")
        roles[name] = model
    return Handler(
        sentinel_from_config(getattr(arguments, "sentinel_config", arguments.sentinel)),
        host=SidecarHost(roles),
        state=MemoryState(expiry=arguments.run_expiry),
        reject_status=arguments.reject_status,
        time_limit=arguments.time_limit,
        eval_results=not arguments.no_eval_results,
        tasks=read_tasks_file(arguments.tasks) if arguments.tasks else None,
        runtime_task_registration=not arguments.no_runtime_task_registration,
        task_expiry=arguments.task_expiry,
    )


def apps(
    handler: Handler, arguments: argparse.Namespace
) -> list[tuple[Starlette, int]]:
    """The apps to serve, each with its port: the endpoints for proxies and for evals together, or each on a port of its own.

    Args:
        handler: The handler a proxy's requests and replies are put to.
        arguments: Parsed arguments that `add_arguments` defined.

    Raises:
        ValueError: If the endpoint for evals has a port of its own, and no token is set for it.
    """
    proxy = proxy_app(handler, max_body_bytes=arguments.max_body_bytes)
    results = not arguments.no_eval_results
    if arguments.eval_port is None:
        evals = eval_routes(
            handler, max_body_bytes=arguments.max_body_bytes, results=results
        )
        return [(Starlette(routes=[*proxy.routes, *evals]), arguments.port)]
    token = os.environ.get(TOKEN_VARIABLE, "")
    if not token:
        raise ValueError(
            f"The endpoint for evals has a port of its own, and needs a token of its own in {TOKEN_VARIABLE}."
        )
    evals = eval_routes(
        handler,
        max_body_bytes=arguments.max_body_bytes,
        token=token,
        results=results,
    )
    return [
        (proxy, arguments.port),
        (Starlette(routes=[*evals, health_route()]), arguments.eval_port),
    ]


def serve(handler: Handler, arguments: argparse.Namespace) -> None:
    """Serve the endpoints for proxies and for evals.

    Args:
        handler: The handler a proxy's requests and replies are put to.
        arguments: Parsed arguments that `add_arguments` defined.
    """
    _quiet_health_checks()
    servers = [
        uvicorn.Server(uvicorn.Config(app, host=arguments.host, port=port))
        for app, port in apps(handler, arguments)
    ]
    anyio.run(_serve_all, servers)


async def _serve_all(servers: list[uvicorn.Server]) -> None:
    async def until_one_stops(server: uvicorn.Server) -> None:
        await server.serve()
        # one signal stops only the server that caught it
        for other in servers:
            other.should_exit = True

    async with anyio.create_task_group() as group:
        for server in servers:
            group.start_soon(until_one_stops, server)


class _HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn's access record holds the client, the method, the path with
        # its query, the HTTP version and the status
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        return str(args[2]).partition("?")[0] != HEALTH_PATH


def _quiet_health_checks() -> None:
    # a probe arrives every few seconds for as long as the sidecar runs, and
    # its line in the access log would bury the proxy's requests
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(kept, _HealthCheckFilter) for kept in access.filters):
        access.addFilter(_HealthCheckFilter())
