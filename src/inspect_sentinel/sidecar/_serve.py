from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import anyio
import uvicorn
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette

from .._config import parse_config_file, sentinel_from_config
from ._eval_endpoint import TOKEN_VARIABLE, eval_routes
from ._exchange import MAX_BODY_BYTES
from ._handler import Handler
from ._host import SidecarHost
from ._http import HEALTH_PATH, health_route
from ._proxy_endpoint import proxy_app
from ._registration import MAX_TASK_EXPIRY, MAX_TASK_REGISTRATIONS, read_tasks_folder
from ._reply_stream import KEEPALIVE_SECONDS
from ._state import RUN_EXPIRY, MemoryState


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of the sidecar's command line.

    Args:
        parser: The parser to add them to.
    """
    parser.add_argument(
        "--sentinel",
        required=True,
        help="A YAML or JSON configuration file, or the registered name of a monitor or protocol. Beside `sentinel`, the file may hold a `server` stanza with the server's own settings, which the options given here override, and a `tasks_folder`.",
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
        "--envoy-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Serve the Envoy shim, which Envoy's ext_proc filter calls over gRPC, on this port, for a sidecar behind Envoy. Needs the `envoy` extra. `python -m inspect_sentinel.sidecar.envoy` writes Envoy's configuration for it.",
    )
    parser.add_argument(
        "--envoy-header",
        action="append",
        default=[],
        metavar="NAME",
        help="A request header the Envoy shim shows the sentinel, besides Inspect's own. May be repeated. A credential is never shown.",
    )
    parser.add_argument(
        "--release",
        choices=["whole", "early"],
        default="whole",
        help="How a streamed reply is passed on. whole, the default, holds every reply until the sentinel has judged it. early passes on the text and reasoning before the turn's first tool call as they arrive, and holds the rest of the turn until it has ended and is judged. The sentinel is given the whole turn either way.",
    )
    parser.add_argument(
        "--keepalive",
        type=float,
        default=KEEPALIVE_SECONDS,
        metavar="SECONDS",
        help=f"Under --release early, how long a client may be sent nothing while a streamed reply is held, after which it is sent a keepalive that every provider SDK skips, so that something between them doesn't close a quiet connection. {KEEPALIVE_SECONDS:g} by default, and 0 for none. A keepalive sends the status, so a turn refused after one is refused with an error event in the stream.",
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
        help="Send an Inspect eval only what it acts on: each step's stage, call and decision, and what the turn came to, without the explanations, references, metadata, records or the sentinel's store. A monitor's explanation tells an agent what the monitor looks for, so use this where the eval can't be trusted.",
    )
    parser.add_argument(
        "--tasks-folder",
        metavar="FOLDER",
        help="A folder of YAML or JSON tasks files, each with a `tasks` list: the tasks registered beforehand, which an eval can't replace: its registration of one is answered as already registered. A request for a sample of one of them that its file doesn't hold is refused.",
    )
    parser.add_argument(
        "--no-runtime-task-registration",
        action="store_true",
        help="Take no tasks registered by an eval at task init. Where an eval can't be trusted, tasks are registered beforehand with --tasks-folder instead.",
    )
    parser.add_argument(
        "--strictness",
        choices=["open", "known_tasks", "registered_only"],
        default="open",
        help="How strict the sidecar is about a request it can't place. open, the default, judges every request, without the task's facts where it holds none. known_tasks refuses a request whose x-inspect-context names a task the sidecar doesn't hold, and judges one without the header. registered_only refuses any request that doesn't name a sample the sidecar holds, including one without the header, such as a grader's or a client's other than an eval's.",
    )
    parser.add_argument(
        "--task-expiry",
        type=float,
        default=MAX_TASK_EXPIRY,
        metavar="SECONDS",
        help="The longest a task an eval registered is kept once no request has named it, whatever the eval asks. 30 days by default.",
    )
    parser.add_argument(
        "--max-task-registrations",
        type=int,
        default=MAX_TASK_REGISTRATIONS,
        metavar="COUNT",
        help=f"How many tasks registered by evals are kept at once. Anything that can reach the endpoint for evals can register a task, so a registration past this is refused, rather than one still in use dropped. {MAX_TASK_REGISTRATIONS} by default.",
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


class EnvoySettings(BaseModel):
    """The `envoy` stanza inside `server`: the Envoy shim's settings, which the options that start `--envoy-` set too."""

    model_config = ConfigDict(extra="forbid")

    port: int | None = None
    """The port the Envoy shim serves on, `--envoy-port`."""

    headers: list[str] | None = None
    """The request headers the shim shows the sentinel besides Inspect's own, `--envoy-header`."""


class ServerSettings(BaseModel):
    """The `server` stanza of a configuration file: the server's own settings, beside the sentinel.

    Each key is one of the command-line options, with underscores, and a `--no-` option is the setting turned off. Model roles are `model_roles`, as inspect_ai names them. The Envoy shim's settings are in a stanza of their own, `envoy`.
    """

    model_config = ConfigDict(extra="forbid")

    model_roles: dict[str, str] | None = None
    host: str | None = None
    port: int | None = None
    eval_port: int | None = None
    envoy: EnvoySettings | None = None
    reject_status: int | None = None
    time_limit: float | None = None
    eval_results: bool | None = None
    runtime_task_registration: bool | None = None
    strictness: Literal["open", "known_tasks", "registered_only"] | None = None
    task_expiry: float | None = None
    max_task_registrations: int | None = None
    max_body_bytes: int | None = None
    run_expiry: float | None = None
    release: Literal["whole", "early"] | None = None
    keepalive: float | None = None


# each setting of the stanza, with the option that overrides it, and what it
# sets the option to
_OPTIONS: dict[str, tuple[str, str, Callable[[Any, Path], Any]]] = {
    "model_roles": (
        "--model-role",
        "model_role",
        lambda roles, _: [f"{role}={model}" for role, model in roles.items()],
    ),
    "envoy_headers": ("--envoy-header", "envoy_header", lambda names, _: list(names)),
    "eval_results": ("--no-eval-results", "no_eval_results", lambda on, _: not on),
    "runtime_task_registration": (
        "--no-runtime-task-registration",
        "no_runtime_task_registration",
        lambda on, _: not on,
    ),
    **{
        name: (f"--{name.replace('_', '-')}", name, lambda value, _: value)
        for name in (
            "host",
            "port",
            "eval_port",
            "envoy_port",
            "reject_status",
            "time_limit",
            "strictness",
            "task_expiry",
            "max_task_registrations",
            "max_body_bytes",
            "run_expiry",
            "release",
            "keepalive",
        )
    },
}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line, with the configuration file's `server` stanza and `tasks_folder` under it.

    An option given on the command line wins over the file, and the file wins over the option's default. The tasks folder is found beside the file.

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
    sentinel, settings, tasks_folder = read_configuration(arguments.sentinel)
    arguments.sentinel_config = sentinel

    def overridden(option: str) -> bool:
        return any(arg == option or arg.startswith(f"{option}=") for arg in given)

    base = Path(arguments.sentinel).resolve().parent
    if tasks_folder is not None and not overridden("--tasks-folder"):
        arguments.tasks_folder = str(base / tasks_folder)
    if settings is not None:
        for name, value in _flat(settings.model_dump(exclude_none=True)):
            option, dest, convert = _OPTIONS[name]
            if not overridden(option):
                setattr(arguments, dest, convert(value, base))
    return arguments


def _flat(settings: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    # each setting by the name of its option, a stanza's inside it by the
    # stanza's name and its own: `envoy`'s `port` is `envoy_port`
    for name, value in settings.items():
        if name == "envoy":
            for inner, set_to in cast("dict[str, Any]", value).items():
                yield f"{name}_{inner}", set_to
        else:
            yield name, value


def read_configuration(config: str) -> tuple[Any, ServerSettings | None, str | None]:
    """Read a configuration file: the sentinel, the `server` stanza and the `tasks_folder`.

    Args:
        config: A YAML or JSON configuration file, or the registered name of a monitor or protocol.

    Returns:
        What to build the sentinel from, which is `config` itself unless the file holds more than a sentinel; the `server` stanza's settings, or None; and the tasks folder as the file names it, or None.

    Raises:
        ValueError: If the file has keys other than `sentinel`, `server` and `tasks_folder`, or its stanza holds a setting the server doesn't have.
    """
    path = Path(config)
    if not path.is_file():
        # a registered name, which the core loader finds
        return config, None, None
    # read as the core loader reads a sentinel file, so that a key given
    # twice is an error here too, rather than the last one silently winning
    content = parse_config_file(str(path))
    if not isinstance(content, dict) or set(cast("dict[str, Any]", content)) == {
        "sentinel"
    }:
        # the core loader reads a file of only a sentinel, as an eval does
        return config, None, None
    content = cast("dict[str, Any]", content)
    if "sentinel" not in content or not set(content) <= _KEYS:
        raise ValueError(
            f"{config}: a configuration file holds `sentinel`, and may hold `server` and `tasks_folder`, and nothing else."
        )
    try:
        settings = ServerSettings.model_validate(content.get("server") or {})
    except ValidationError as ex:
        raise ValueError(f"{config}: the server stanza doesn't fit: {ex}") from None
    folder = content.get("tasks_folder")
    if folder is not None and not isinstance(folder, str):
        raise ValueError(f"{config}: `tasks_folder` is a folder's path.")
    return content["sentinel"], settings, folder


_KEYS = frozenset({"sentinel", "server", "tasks_folder"})


def handler_from_arguments(arguments: argparse.Namespace) -> Handler:
    """Build the handler the arguments describe.

    Args:
        arguments: Parsed arguments that `add_arguments` defined.

    Raises:
        ValueError: If a `--model-role` is not `ROLE=MODEL`, or the sentinel configuration or a tasks file is invalid.
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
        tasks=read_tasks_folder(arguments.tasks_folder)
        if arguments.tasks_folder
        else None,
        runtime_task_registration=not arguments.no_runtime_task_registration,
        strictness=arguments.strictness,
        task_expiry=arguments.task_expiry,
        max_task_registrations=arguments.max_task_registrations,
        release=arguments.release,
        keepalive=arguments.keepalive or None,
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
        # with the proxy app's lifespan, in which its reply streams run
        return [
            (
                Starlette(
                    routes=[*proxy.routes, *evals],
                    lifespan=proxy.router.lifespan_context,
                ),
                arguments.port,
            )
        ]
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
    """Serve the endpoints for proxies and for evals, and the Envoy shim if it has a port.

    Args:
        handler: The handler a proxy's requests and replies are put to.
        arguments: Parsed arguments that `add_arguments` defined.

    Raises:
        ValueError: If the Envoy shim has a port, and the `envoy` extra isn't installed.
    """
    _quiet_health_checks()
    servers = [
        uvicorn.Server(uvicorn.Config(app, host=arguments.host, port=port))
        for app, port in apps(handler, arguments)
    ]
    shim = _envoy_shim(handler, arguments)
    try:
        anyio.run(_serve_all, servers, shim)
    except KeyboardInterrupt:
        # a server that stops for Ctrl-C raises it again once it has stopped,
        # and stopping is what was asked, as `uvicorn.run` takes it
        pass


class _GrpcServer(Protocol):
    async def start(self) -> None: ...

    async def stop(self, grace: float | None) -> None: ...


def _envoy_shim(
    handler: Handler, arguments: argparse.Namespace
) -> Callable[[], _GrpcServer] | None:
    # the shim's server, made in the event loop it serves in
    if arguments.envoy_port is None:
        return None
    try:
        from .envoy import envoy_server
    except ImportError as ex:
        raise ValueError(
            f"--envoy-port needs the `envoy` extra: pip install 'inspect_sentinel[envoy]' ({ex})."
        ) from None
    return lambda: envoy_server(
        handler,
        host=arguments.host,
        port=arguments.envoy_port,
        max_body_bytes=arguments.max_body_bytes,
        headers=arguments.envoy_header,
    )


_SHIM_GRACE = 5.0
"""How long the Envoy shim's open streams are given to finish when the sidecar stops, in seconds."""


async def _serve_all(
    servers: list[uvicorn.Server], shim: Callable[[], _GrpcServer] | None = None
) -> None:
    async def until_one_stops(server: uvicorn.Server) -> None:
        await server.serve()
        # one signal stops only the server that caught it
        for other in servers:
            other.should_exit = True

    envoy = shim() if shim is not None else None
    if envoy is not None:
        await envoy.start()
    try:
        async with anyio.create_task_group() as group:
            for server in servers:
                group.start_soon(until_one_stops, server)
    finally:
        if envoy is not None:
            # the HTTP servers have stopped, on a signal, and the shim with them
            with anyio.CancelScope(shield=True):
                await envoy.stop(_SHIM_GRACE)


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
