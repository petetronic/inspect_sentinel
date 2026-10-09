from __future__ import annotations

from starlette.requests import Request as HttpRequest
from starlette.responses import PlainTextResponse
from starlette.routing import Route

HEALTH_PATH = "/health"
"""The path that answers a health check, on every port the sidecar listens on."""

MAX_BODY_BYTES = 64 * 1024 * 1024
"""The largest request or reply a proxy may hand over, unless set otherwise. Middleman holds a reply of up to 32 MiB by default and sends it as base64, which is a third larger."""


async def read_within(http: HttpRequest, limit: int) -> bytes | None:
    """Read a request's body, giving up as soon as it is known to be larger than a limit.

    Args:
        http: The request.
        limit: The largest body to read, in bytes.

    Returns:
        The body, or None if it is larger than `limit`. No more than `limit` bytes are held either way.
    """
    declared = http.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in http.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


def health_route() -> Route:
    """A route that answers `ok` to a GET of `HEALTH_PATH`, for whatever restarts the sidecar to probe.

    It asks for no credential and checks nothing: the sentinel is loaded before the sidecar listens, so an answer means the sidecar is ready, and a sentinel that blocks the process stops the answer too.
    """

    async def health(http: HttpRequest) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return Route(HEALTH_PATH, health, methods=["GET"])
