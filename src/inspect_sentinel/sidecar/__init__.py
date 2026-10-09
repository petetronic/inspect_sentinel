"""A host for sentinels that runs beside a model proxy, outside any eval.

A proxy hands the sidecar each model request and reply. The sidecar reads them into Inspect's types, puts each step to a sentinel, and answers what the proxy should do. `Handler` is the part that judges. Nothing of this package runs in a proxy. A proxy runs a hook, which is a small package of its own, and the hook calls the sidecar's endpoint for proxies over HTTPS.
"""

from ._exchange import (
    DECISION_HEADER,
    Answer,
    Pass,
    Refuse,
    Replace,
    Reply,
    Request,
)
from ._handler import Handler, TimeLimitError
from ._host import ListRecorder, Record, SidecarHost
from ._wire import UnreadableError

__all__ = [
    "Answer",
    "Handler",
    "ListRecorder",
    "Pass",
    "Record",
    "DECISION_HEADER",
    "Refuse",
    "Replace",
    "Reply",
    "Request",
    "SidecarHost",
    "TimeLimitError",
    "UnreadableError",
]
