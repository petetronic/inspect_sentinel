"""A hook for METR's Middleman that calls a remote Inspect Sentinel endpoint over HTTPS.

Middleman's `MIDDLEMAN_PASSTHROUGH_HOOK` setting names the class as `inspect_sentinel_middleman:MiddlemanHook`. Middleman creates one at startup, then calls `on_request` with each model request before it goes to a provider, and `on_reply` with the provider's reply, held whole, before any of it goes to the caller. Each is put to the sidecar as a JSON-RPC 2.0 call, `judge_request` or `judge_reply`, whose result is carried out: pass, replace or refuse. Inspect Sentinel does not run in Middleman: only this hook does, and the sentinels run in the sidecar it calls.

The class runs inside Middleman's process, so this package stands alone: it imports nothing from `inspect_sentinel`, and needs only `aiohttp` and `fastapi`, which Middleman has. The calls it makes and the results it reads are described by `proxy_endpoint.schema.json` in `inspect_sentinel`'s sidecar.

It is set through the environment, read once when Middleman creates it:

- `INSPECT_SENTINEL_SIDECAR_URL`: the sidecar's address for proxies, as `https://<sentinel host>:<port>`. Required.
- `INSPECT_SENTINEL_SIDECAR_HEADERS`: request headers the sidecar is shown besides Inspect's own, as a comma-separated list of names, or prefixes ending in `*`. None unless set.

Which traffic reaches the class, how large a reply may be held, how long an answer may take, and whether a sidecar that gives no usable answer refuses an exchange or lets it through are Middleman's own settings for the class it loads. A sidecar that gives no usable answer is raised as a `SidecarError`, which Middleman treats as it treats any failure of that class.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from collections.abc import Mapping
from typing import Any, Protocol, cast

import aiohttp
from fastapi import HTTPException

logger = logging.getLogger("inspect_sentinel_middleman")

_PREFIX = "INSPECT_SENTINEL_SIDECAR_"
# where the sidecar serves the methods this calls, and the version of them
_PATH = "/proxy/v1"
# the headers an Inspect eval puts on a model request: which sample it is for,
# and an id for the request. The sidecar is always shown them, unread.
# Middleman sends a provider none of a caller's `x-` headers.
_INSPECT_HEADERS = frozenset({"x-inspect-context", "x-irid"})
# never shown to the sidecar, whatever is listed. Middleman leaves the
# caller's own out of the headers a hook is shown.
_CREDENTIALS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
    }
)
# what a body on Middleman's route is written in, by the end of the route's
# path. A request on any other route is put to the sidecar without a format,
# which then reads it by its provider, or answers that it can't.
_FORMATS = {
    "/v1/messages": "anthropic-messages",
    "/v1/chat/completions": "openai-chat",
    "/v1/responses": "openai-responses",
}


class SidecarError(Exception):
    """The sidecar could not be asked, or its answer could not be read or carried out."""


class HookRequest(Protocol):
    """A passthrough request as Middleman gives it to a hook. This names what `MiddlemanHook` reads of it, so that this package need not import Middleman."""

    @property
    def id(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def user(self) -> str | None: ...
    @property
    def headers(self) -> Mapping[str, str]: ...
    @property
    def body(self) -> dict[str, Any]: ...
    @property
    def endpoint(self) -> str: ...


class HookReply(Protocol):
    """A provider's reply, held whole, as Middleman gives it to a hook. This names what `MiddlemanHook` reads of it."""

    @property
    def status(self) -> int: ...
    @property
    def content_type(self) -> str: ...
    @property
    def body(self) -> bytes: ...


class MiddlemanHook:
    """Puts each of Middleman's passthrough requests and replies to a sentinel sidecar.

    Raises:
        ValueError: A setting in the environment makes no sense, which stops Middleman starting.
    """

    def __init__(self) -> None:
        base = _setting("URL")
        if not base:
            raise ValueError(
                f"{_PREFIX}URL must say where the sentinel sidecar listens."
            )
        self._url = base.rstrip("/") + _PATH
        self._shown_headers = _listed(_setting("HEADERS"))
        # made on first use, when Middleman's event loop is running
        self._session: aiohttp.ClientSession | None = None

    async def on_request(self, request: HookRequest) -> dict[str, Any] | None:
        """Put a request to the sidecar. Middleman sends on the body returned, or the request as it is for None.

        Raises:
            HTTPException: The request is refused. Middleman answers the caller with it in the provider's own error format.
            SidecarError: The sidecar gave no usable answer.
        """
        answer = await self._ask(
            "judge_request",
            _told(request)
            | {
                "headers": {
                    name: value
                    for name, value in request.headers.items()
                    if self._shows(name.lower())
                },
                "body": request.body,
            },
        )
        if answer["action"] == "pass":
            return None
        body = answer.get("body")
        if not isinstance(body, dict):
            raise _failed("a replaced request that is not a JSON object")
        return cast("dict[str, Any]", body)

    async def on_reply(self, request: HookRequest, reply: HookReply) -> bytes | None:
        """Put a held reply to the sidecar. Middleman sends the caller the body returned, or the reply as it is for None.

        Raises:
            HTTPException: The reply is refused. Middleman answers the caller with it in the provider's own error format.
            SidecarError: The sidecar gave no usable answer.
        """
        answer = await self._ask(
            "judge_reply",
            _told(request)
            | {
                "status": reply.status,
                "content_type": reply.content_type,
                "body": base64.b64encode(reply.body).decode("ascii"),
            },
        )
        if answer["action"] == "pass":
            return None
        body = answer.get("body")
        try:
            if not isinstance(body, str):
                raise ValueError
            return base64.b64decode(body, validate=True)
        except (binascii.Error, ValueError):
            raise _failed("a replaced reply that is not base64") from None

    async def close(self) -> None:
        """Close the connection to the sidecar. Middleman calls this as it shuts down."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _shows(self, name: str) -> bool:
        # what the sidecar is shown of the caller's headers: Inspect's own, and
        # any that are listed
        return name not in _CREDENTIALS and (
            name in _INSPECT_HEADERS
            or any(_covers(pattern, name) for pattern in self._shown_headers)
        )

    async def _ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        # the result returned is to pass or to replace; a refusal is raised
        if self._session is None:
            self._session = aiohttp.ClientSession()
        # one call to a request, so the call's own id tells nothing apart
        call = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        try:
            # no time limit of this class's own: Middleman limits how long a
            # call to it may take, and cancels one that takes longer
            async with self._session.post(
                self._url, json=call, timeout=aiohttp.ClientTimeout(total=None)
            ) as response:
                if response.status != 200:
                    raise _failed(f"status {response.status}")
                read: Any = await response.json()
        except (aiohttp.ClientError, TypeError, ValueError) as ex:
            raise _failed(type(ex).__name__) from None

        if not isinstance(read, dict):
            raise _failed("an answer that is not a JSON object")
        answered = cast("dict[str, Any]", read)
        error: Any = answered.get("error")
        if error is not None:
            # the sidecar did not judge it, and its code says why
            code: Any = (
                cast("dict[str, Any]", error).get("code")
                if isinstance(error, dict)
                else None
            )
            raise _failed(f"error {code}" if isinstance(code, int) else "an error")
        answer = answered.get("result")
        if not isinstance(answer, dict):
            raise _failed("an answer with no result")
        answer = cast("dict[str, Any]", answer)
        if answer.get("action") == "refuse":
            raise _refusal(answer)
        if answer.get("action") not in ("pass", "replace"):
            raise _failed("an answer with no action this class knows")
        return answer


def _setting(name: str) -> str:
    return os.environ.get(_PREFIX + name, "").strip()


def _listed(headers: str) -> tuple[str, ...]:
    patterns = tuple(
        name.strip().lower() for name in headers.split(",") if name.strip()
    )
    for pattern in patterns:
        if not re.fullmatch(r"[a-z0-9_-]+\*?", pattern):
            raise ValueError(
                f"{_PREFIX}HEADERS takes header names, or prefixes ending in '*', not {pattern!r}."
            )
        if any(_covers(pattern, name) for name in _CREDENTIALS):
            raise ValueError(
                f"{_PREFIX}HEADERS may not list {pattern!r}, which would show the sidecar a credential."
            )
    return patterns


def _covers(pattern: str, name: str) -> bool:
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def _told(request: HookRequest) -> dict[str, Any]:
    # what the sidecar is told of a request, with the request and its reply
    # alike. Middleman makes the id itself, so no caller can make two requests
    # in flight share one.
    told: dict[str, Any] = {
        "id": request.id,
        "provider": request.provider,
        "model": request.model,
        "user": request.user,
    }
    for ending, format in _FORMATS.items():
        if request.endpoint.endswith(ending):
            told["format"] = format
    return told


def _refusal(answer: dict[str, Any]) -> HTTPException:
    status = answer.get("status", 400)
    message = answer.get("message")
    if (
        not isinstance(status, int)
        or not 400 <= status <= 599
        or not isinstance(message, str)
    ):
        # a refusal that can't be carried out as asked still refuses,
        # whatever Middleman does with a failure
        logger.warning("The sentinel sidecar's refusal could not be read.")
        return HTTPException(503, "The sentinel sidecar's refusal could not be read.")
    # Middleman writes a refusal from its status and message alone, so the
    # `sentinel` field an Inspect eval is sent with one has nowhere to go
    return HTTPException(status, message)


def _failed(reason: str) -> SidecarError:
    # the reason is one of this module's own short strings, never anything a
    # request, a reply or the sidecar's answer held. Middleman logs only that
    # the class failed, so the reason is logged here.
    logger.warning("The sentinel sidecar gave no usable answer (%s).", reason)
    return SidecarError(reason)
