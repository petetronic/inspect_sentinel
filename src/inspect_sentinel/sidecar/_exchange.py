from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias


@dataclass(frozen=True, kw_only=True)
class Request:
    """A model request a proxy is about to send, as the proxy handed it over."""

    id: str
    """Joins the request to its reply. Chosen by the proxy."""

    provider: str
    """Whose API the body is written for, e.g. `anthropic`."""

    model: str
    """The model the request is for, as the proxy names it."""

    user: str | None
    """Who the proxy says is calling, if it knows."""

    body: bytes
    """The request body, in the provider's own format."""

    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    """The request's headers the proxy passed on, by lowercase name. The caller set them, so they are claims."""

    run: str | None = None
    """What tells this run from every other, if the proxy was told. Without it, requests whose conversations open alike are taken for one run, which two epochs of a sample are not."""


@dataclass(frozen=True, kw_only=True)
class Reply:
    """A provider's reply to a `Request`, before the proxy passes any of it on."""

    id: str
    """The `id` of the request this answers."""

    provider: str
    """Whose API the body is written in."""

    model: str
    """The model the request was for, as the proxy names it."""

    user: str | None
    """Who the proxy says is calling, if it knows."""

    status: int
    """The provider's HTTP status."""

    content_type: str
    """The provider's `content-type`, which says whether the body is a stream."""

    body: bytes
    """The whole reply body, in the provider's own format."""


DECISION_HEADER = "x-sentinel-decision"
"""The response header that names the decision a refusal carries out, `reject` or `terminate`. A client that knows of sentinels acts on it, and any other client ignores it."""


@dataclass(frozen=True)
class Pass:
    """Send it on as it is."""

    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    """Response headers to add for the client when this answers a reply. None by default."""


@dataclass(frozen=True)
class Replace:
    """Send this body on in its place."""

    body: bytes
    """The replacement, in the provider's own format."""

    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    """Response headers to add for the client when this answers a reply. None by default."""


@dataclass(frozen=True)
class Refuse:
    """Don't send it on. Answer the client with an error."""

    status: int
    """The HTTP status for the client."""

    message: str
    """What the client is told. A client that carries on hands this to the model."""

    headers: Mapping[str, str]
    """Response headers for the client, which the endpoint for proxies passes on without reading. `DECISION_HEADER` is among them."""


Answer: TypeAlias = Pass | Replace | Refuse
"""What a proxy does with a request or a reply: what was decided, never how to do it."""
