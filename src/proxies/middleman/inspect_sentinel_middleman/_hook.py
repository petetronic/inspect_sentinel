"""A hook for Middleman that calls a remote Inspect Sentinel endpoint over HTTPS.

Middleman's `MIDDLEMAN_PASSTHROUGH_HOOK` setting names `MiddlemanHook` by its import path, `inspect_sentinel_middleman:MiddlemanHook`. Middleman creates one at startup, then calls `on_request` with each request and `on_reply` with each held reply. Each is POSTed as JSON to the sidecar's endpoint for proxies, whose answer is carried out: pass, replace or refuse. Inspect Sentinel does not run in Middleman: only this hook does, and the sentinels run in the sidecar it calls.

The class runs inside Middleman's process, so this package stands alone: it imports nothing from `inspect_sentinel`, and needs only `aiohttp` and `fastapi`, which Middleman has. The messages it sends and the answers it reads are described by `proxy_endpoint.schema.json` in `inspect_sentinel`'s sidecar.

It is set through the environment, read once when Middleman creates it:

- `INSPECT_SENTINEL_SIDECAR_URL`: where the sidecar's endpoint for proxies listens. Required.
- `INSPECT_SENTINEL_SIDECAR_HEADERS`: request headers the sidecar is shown besides those that say which job and sample a request belongs to, as a comma-separated list of names, or prefixes ending in `*`. None unless set.

Which traffic reaches the class, how long an answer may take, and whether a sidecar that can't be reached refuses an exchange or lets it through are Middleman's own settings for the class it loads. A sidecar that gives no usable answer is raised as a `SidecarError`, which Middleman treats as it treats any failure of that class.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import aiohttp
from fastapi import HTTPException

logger = logging.getLogger("inspect_sentinel_middleman")

_PREFIX = "INSPECT_SENTINEL_SIDECAR_"
# the headers a Hawk runner puts on a model request to say which job, task
# and sample it belongs to
_CORRELATION_PREFIXES = ("x-metr-", "x-hawk-", "x-inspect-", "x-scout-")
# the one of them that names a sample's run, which the sidecar keeps a
# sentinel's state by
_RUN_HEADER = "x-inspect-sample-uuid"
# a header the sidecar asks to have added to a response is its own to name,
# but not one that changes how the response is framed or read
_ADDED_HEADER = re.compile(r"x-[a-z0-9_-]+")


class SidecarError(Exception):
    """The sidecar could not be asked, or its answer could not be read."""


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


class HookReply(Protocol):
    """A provider's reply, held whole, as Middleman gives it to a hook. This names what `MiddlemanHook` reads of it."""

    @property
    def status(self) -> int: ...
    @property
    def content_type(self) -> str: ...
    @property
    def body(self) -> bytes: ...


@dataclass(frozen=True)
class _Settings:
    url: str
    headers: tuple[str, ...]


def _read_settings() -> _Settings:
    def setting(name: str) -> str:
        return os.environ.get(_PREFIX + name, "").strip()

    url = setting("URL")
    if not url:
        raise ValueError(f"{_PREFIX}URL must say where the sentinel sidecar listens.")
    headers = tuple(
        name.strip().lower() for name in setting("HEADERS").split(",") if name.strip()
    )
    for pattern in headers:
        if not re.fullmatch(r"[a-z0-9_-]+\*?", pattern):
            raise ValueError(
                f"{_PREFIX}HEADERS takes header names, or prefixes ending in '*', not {pattern!r}."
            )
    return _Settings(url=url, headers=headers)


class MiddlemanHook:
    """Puts each of Middleman's passthrough requests and replies to a sentinel sidecar.

    Raises:
        ValueError: A setting in the environment makes no sense, which stops Middleman starting.
    """

    def __init__(self) -> None:
        self._settings = _read_settings()
        # made on first use, when Middleman's event loop is running
        self._session: aiohttp.ClientSession | None = None

    async def on_request(self, request: HookRequest) -> dict[str, Any] | None:
        """Put a request to the sidecar. Middleman sends on the body returned, or the request as it is for None.

        Raises:
            HTTPException: The request is refused. Middleman answers the client with it in the provider's own error format.
            SidecarError: The sidecar gave no usable answer.
        """
        answer = await self._ask(
            _message(request, "request")
            | {
                "run": request.headers.get(_RUN_HEADER),
                "headers": self._shown(request.headers),
                "body": request.body,
            }
        )
        if answer["action"] == "pass":
            return None
        body = answer.get("body")
        if not isinstance(body, dict):
            raise _failed("a replaced request that is not a JSON object")
        return cast(dict[str, Any], body)

    async def on_reply(
        self, request: HookRequest, reply: HookReply
    ) -> HookReply | bytes | None:
        """Put a held reply to the sidecar. Middleman sends the client the body returned, or the reply as it is for None. When the sidecar names response headers, the reply is returned with them and Middleman adds them.

        Raises:
            HTTPException: The reply is refused. Middleman answers the client with it in the provider's own error format.
            SidecarError: The sidecar gave no usable answer.
        """
        answer = await self._ask(
            _message(request, "reply")
            | {
                "status": reply.status,
                "content_type": reply.content_type,
                "body": base64.b64encode(reply.body).decode("ascii"),
            }
        )
        headers = _added_headers(answer)
        if headers is None:
            raise _failed("headers that may not be added")
        replaced: bytes | None = None
        if answer["action"] == "replace":
            body = answer.get("body")
            try:
                if not isinstance(body, str):
                    raise ValueError
                replaced = base64.b64decode(body, validate=True)
            except (binascii.Error, ValueError):
                raise _failed("a replaced reply that is not base64") from None
        if not headers:
            return replaced
        # Middleman's own reply, with what the sidecar asked to change
        changes: dict[str, Any] = {"headers": headers}
        if replaced is not None:
            changes["body"] = replaced
        return cast(HookReply, dataclasses.replace(cast(Any, reply), **changes))

    async def close(self) -> None:
        """Close the connection to the sidecar. Middleman calls this as it shuts down."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _shown(self, headers: Mapping[str, str]) -> dict[str, str]:
        # what the sidecar is shown of the caller's headers: those that say
        # which job and sample a request belongs to, and any that are listed
        return {
            name: value
            for name, value in headers.items()
            if name.startswith(_CORRELATION_PREFIXES)
            or any(_covers(pattern, name) for pattern in self._settings.headers)
        }

    async def _ask(self, message: dict[str, Any]) -> dict[str, Any]:
        # the answer returned is to pass or to replace; a refusal is raised
        if self._session is None:
            self._session = aiohttp.ClientSession()
        try:
            # no time limit of this class's own: Middleman limits how long a
            # call to it may take, and cancels one that takes longer
            async with self._session.post(
                self._settings.url,
                json=message,
                timeout=aiohttp.ClientTimeout(total=None),
            ) as response:
                if response.status != 200:
                    raise _failed(f"status {response.status}")
                read: Any = await response.json()
        except (aiohttp.ClientError, ValueError) as ex:
            raise _failed(type(ex).__name__) from None

        if not isinstance(read, dict):
            raise _failed("an answer that is not a JSON object")
        answer = cast(dict[str, Any], read)
        if answer.get("action") == "refuse":
            raise _refusal(answer)
        if answer.get("action") not in ("pass", "replace"):
            raise _failed("an answer with no action this class knows")
        return answer


def _message(request: HookRequest, phase: str) -> dict[str, Any]:
    return {
        "version": 1,
        "phase": phase,
        "id": request.id,
        "provider": request.provider,
        "model": request.model,
        "user": request.user,
    }


def _covers(pattern: str, name: str) -> bool:
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def _added_headers(answer: dict[str, Any]) -> dict[str, str] | None:
    # the response headers an answer names, or None when they can't be added
    headers = cast(dict[Any, Any], answer.get("headers", {}))
    if not isinstance(headers, dict) or not all(  # pyright: ignore[reportUnnecessaryIsInstance]
        isinstance(name, str)
        and _ADDED_HEADER.fullmatch(name)
        and isinstance(value, str)
        for name, value in headers.items()
    ):
        return None
    return cast(dict[str, str], headers)


def _refusal(answer: dict[str, Any]) -> HTTPException:
    status = answer.get("status", 400)
    message = answer.get("message")
    headers = _added_headers(answer)
    if (
        not isinstance(status, int)
        or not 400 <= status <= 599
        or not isinstance(message, str)
        or headers is None
    ):
        # a refusal that can't be carried out as asked still refuses,
        # whatever Middleman does with a failure
        logger.warning("The sentinel sidecar's refusal could not be read.")
        return HTTPException(503, "The sentinel sidecar's refusal could not be read.")
    return HTTPException(status, message, headers=headers or None)


def _failed(reason: str) -> SidecarError:
    # the reason is one of this module's own short strings, never anything a
    # request, a reply or the sidecar's answer held. Middleman logs only that
    # the class failed, so the reason is logged here.
    logger.warning("The sentinel sidecar gave no usable answer (%s).", reason)
    return SidecarError(reason)
