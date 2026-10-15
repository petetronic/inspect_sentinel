"""A guardrail for the LiteLLM proxy that calls a remote Inspect Sentinel endpoint over HTTPS.

LiteLLM's `guardrails` setting names the class as `inspect_sentinel_litellm.LiteLLMGuardrail`.
"""

from importlib.metadata import PackageNotFoundError, version

from ._guardrail import LiteLLMGuardrail, SidecarError

try:
    __version__ = version("inspect_sentinel_litellm")
except PackageNotFoundError:
    # imported from a checkout that isn't installed
    __version__ = "unknown"

__all__ = [
    "LiteLLMGuardrail",
    "SidecarError",
    "__version__",
]
