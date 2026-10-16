"""Run a sentinel beside a model proxy, on the traffic the proxy hands over.

The sidecar described in `design/sidecar.md`. A proxy hands over each model request and reply in the provider's own format, as it crossed the wire. The sidecar reads it into Inspect's types, finds the steps a sentinel judges in it, and answers what the proxy should do: pass it on, replace it, or refuse it. Each provider's SDK is an optional extra (`anthropic`, `openai`), imported the first time a body in its format is read.

`python -m inspect_sentinel.sidecar` serves the endpoint for proxies, with the `sidecar` extra installed. A proxy hook calls it with JSON-RPC 2.0 at `/proxy/v1`, as `proxy_endpoint.schema.json` beside this file describes. It also serves the endpoint for evals, at `/sentinel/v1` with a probe at `/sentinel/info`, where an Inspect eval registers its tasks and asks what was decided, as `eval_endpoint.schema.json` describes.
"""

from ._exchange import Answer, Pass, Refuse, Replace, Reply, Request
from ._handler import Handler, ModifyError, TimeLimitError
from ._host import Record, SidecarHost, Stage
from ._wire import Format, UnreadableError, read_reply, read_request

__all__ = [
    "Answer",
    "Format",
    "Handler",
    "ModifyError",
    "Pass",
    "Record",
    "Refuse",
    "Replace",
    "Reply",
    "Request",
    "SidecarHost",
    "Stage",
    "TimeLimitError",
    "UnreadableError",
    "read_reply",
    "read_request",
]
