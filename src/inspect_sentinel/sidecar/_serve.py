from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .._config import sentinel_from_config
from .._types import Sentinels
from ._eval_endpoint import eval_app
from ._handler import REQUEST_ID_HEADER, Handler
from ._host import SidecarHost
from ._http import HEALTH_PATH, MAX_BODY_BYTES
from ._proxy_endpoint import proxy_app

EVAL_TOKEN = "INSPECT_SENTINEL_EVAL_TOKEN"
"""The environment variable that holds the credential evals present. The endpoint for evals is served only when it is set."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of the sidecar's command line.

    Args:
        parser: The parser to add them to.
    """
    parser.add_argument(
        "--sentinel",
        required=True,
        help="A YAML or JSON sentinel configuration file, or the registered name of a monitor or protocol.",
    )
    parser.add_argument(
        "--load",
        action="append",
        default=[],
        metavar="FILE",
        help="A Python file to import first, so the monitors and protocols it defines can be named. May be repeated.",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        metavar="ROLE=MODEL",
        help="The model a role generates with, e.g. monitor=anthropic/claude-haiku-4-5. May be repeated.",
    )
    parser.add_argument(
        "--reject-status",
        type=int,
        default=400,
        metavar="STATUS",
        help="The HTTP status a rejected reply is refused with. 400, the default, is final to a client. One a client retries by itself, such as 503, brings it back, and the model is then told of the rejection.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="The address to listen on for the proxy's requests and replies.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8900,
        help="The port to listen on for the proxy's requests and replies.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="The longest the sentinel may take to judge one request or one reply, after which it is cancelled and the proxy is told it couldn't be judged. There is no limit by default: the sentinel finishes however long it takes.",
    )
    parser.add_argument(
        "--request-id-header",
        default=REQUEST_ID_HEADER,
        metavar="NAME",
        help="The request header that holds the id a client gives a request, which an eval asks about that request by. The default is the one inspect_ai sends. The proxy has to pass the header on.",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=MAX_BODY_BYTES,
        metavar="BYTES",
        help="The largest request or reply the proxy may hand over. A larger one is answered with a 413, which a proxy takes as one that couldn't be judged. It has to cover the largest reply the proxy holds.",
    )
    parser.add_argument(
        "--eval-host",
        default="127.0.0.1",
        help=f"The address the endpoint for evals listens on. It is served only when {EVAL_TOKEN} is set.",
    )
    parser.add_argument(
        "--eval-port",
        type=int,
        default=8901,
        help="The port the endpoint for evals listens on.",
    )
    parser.add_argument(
        "--no-eval-results",
        action="store_true",
        help="Serve evals no results, only registration. For where a caller of the endpoint for evals isn't trusted to read what a monitor recorded.",
    )


def handler_from_arguments(arguments: argparse.Namespace) -> Handler:
    """Build the handler the shared arguments describe.

    Args:
        arguments: Parsed arguments that `add_arguments` defined.

    Raises:
        ValueError: If a `--role` is not `ROLE=MODEL`, or the sentinel configuration is invalid.
    """
    roles: dict[str, str] = {}
    for role in arguments.role:
        name, sep, model = str(role).partition("=")
        if not sep or not name or not model:
            raise ValueError(f"--role takes ROLE=MODEL, not {role!r}.")
        roles[name] = model
    return Handler(
        load_sentinel(arguments.sentinel, arguments.load),
        host=SidecarHost(roles),
        reject_status=arguments.reject_status,
        time_limit=arguments.time_limit,
        request_id_header=arguments.request_id_header,
    )


def serve(handler: Handler, arguments: argparse.Namespace) -> None:
    """Serve the endpoint for proxies, and beside it the endpoint for evals if a token for it is set.

    The two listen on ports of their own, so an operator controls who reaches each: proxies alone for the one, and evals for the other.

    Args:
        handler: The handler a proxy's requests and replies are put to, which the endpoint for evals reads and briefs.
        arguments: Parsed arguments that `add_arguments` defined.
    """
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                proxy_app(handler, max_body_bytes=arguments.max_body_bytes),
                host=arguments.host,
                port=arguments.port,
            )
        )
    ]
    token = os.environ.get(EVAL_TOKEN)
    if token:
        servers.append(
            uvicorn.Server(
                uvicorn.Config(
                    eval_app(
                        handler,
                        token=token,
                        results=not arguments.no_eval_results,
                    ),
                    host=arguments.eval_host,
                    port=arguments.eval_port,
                )
            )
        )

    _quiet_health_checks()

    async def run() -> None:
        await asyncio.gather(*(server.serve() for server in servers))

    asyncio.run(run())


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


def load_sentinel(config: str, load: Sequence[str] = ()) -> Sentinels:
    """Import the files that define monitors and protocols, then build the sentinel a configuration names.

    Args:
        config: A YAML or JSON sentinel configuration file, or a registered monitor or protocol name.
        load: Python files to import first.

    Raises:
        FileNotFoundError: If a file in `load` does not exist.
        ValueError: If the configuration is invalid.
    """
    for file in load:
        path = Path(file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No such file to load: {file}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"{file} can't be imported as a Python module.")
        module = importlib.util.module_from_spec(spec)
        # a file may import the files beside it, as it would when run directly
        sys.path.insert(0, str(path.parent))
        try:
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(path.parent))
    return sentinel_from_config(config)
