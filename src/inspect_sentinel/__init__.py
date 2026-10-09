"""Monitors and control protocols that watch agents and intervene in their execution.

A *monitor* observes a step of an agent's execution and reports a suspicion score. A *protocol* decides what happens at that step, optionally after consulting monitors. See `design/` for the design these will implement.
"""

from ._context import Context, EvalContext, ProxyContext
from ._decorators import monitor, protocol
from ._final import decide_final
from ._host import Host, HumanAnswer
from ._portable import PortabilityError
from ._protocols import concurrent, human, observe_only, sequential, threshold
from ._report import (
    Action,
    Decision,
    Failed,
    Observation,
    Report,
    Reported,
    Suspicion,
)
from ._results import Decisions, MonitorFailedError, Observations, Reports
from ._rules import (
    call_text,
    find_words,
    result_text,
    tool_matches,
)
from ._runner import run_children, run_monitors, run_protocols
from ._step import AfterToolCall, BeforeToolCall, Step
from ._types import (
    Monitor,
    MonitorGroup,
    Monitors,
    Protocol,
    ProtocolGroup,
    Protocols,
    Sentinel,
    Sentinels,
)

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    "Action",
    "AfterToolCall",
    "BeforeToolCall",
    "Context",
    "Decision",
    "Decisions",
    "EvalContext",
    "Failed",
    "Host",
    "HumanAnswer",
    "Monitor",
    "MonitorFailedError",
    "MonitorGroup",
    "Monitors",
    "Observation",
    "Observations",
    "PortabilityError",
    "Protocol",
    "ProtocolGroup",
    "Protocols",
    "ProxyContext",
    "Report",
    "Reported",
    "Reports",
    "Sentinel",
    "Sentinels",
    "Step",
    "Suspicion",
    "call_text",
    "concurrent",
    "decide_final",
    "find_words",
    "human",
    "monitor",
    "observe_only",
    "protocol",
    "result_text",
    "run_children",
    "run_monitors",
    "run_protocols",
    "sequential",
    "threshold",
    "tool_matches",
    "__version__",
]
