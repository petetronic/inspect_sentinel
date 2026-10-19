"""A hook for METR's Middleman that calls a remote Inspect Sentinel endpoint over HTTPS.

Middleman's `MIDDLEMAN_PASSTHROUGH_HOOK` setting names the class as `inspect_sentinel_middleman:MiddlemanHook`.
"""

from importlib.metadata import PackageNotFoundError, version

from ._hook import HookReply, HookRequest, MiddlemanHook, SidecarError

try:
    __version__ = version("inspect_sentinel_middleman")
except PackageNotFoundError:
    # imported from a checkout that isn't installed
    __version__ = "unknown"

__all__ = [
    "HookReply",
    "HookRequest",
    "MiddlemanHook",
    "SidecarError",
    "__version__",
]
