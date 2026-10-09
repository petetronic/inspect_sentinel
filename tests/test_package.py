import anyio
import pytest

import inspect_sentinel
import inspect_sentinel._integration


def test_version_is_exposed() -> None:
    assert isinstance(inspect_sentinel.__version__, str)
    assert inspect_sentinel.__version__


@pytest.mark.anyio
async def test_async_tests_run() -> None:
    await anyio.sleep(0)


def test_author_facing_names_are_exported() -> None:
    expected = {
        "BeforeToolCall",
        "AfterToolCall",
        "Step",
        "Observation",
        "Decision",
        "Suspicion",
        "Action",
        "Context",
        "ProxyContext",
        "Host",
        "HumanAnswer",
        "Report",
        "Reported",
        "Sentinel",
        "Sentinels",
        "Protocol",
        "Monitor",
        "Monitors",
        "MonitorGroup",
        "ProtocolGroup",
        "Protocols",
        "monitor",
        "protocol",
        "Decisions",
        "Observations",
        "Reports",
        "run_children",
        "run_monitors",
        "run_protocols",
        "observe_only",
        "concurrent",
        "sequential",
        "human",
        "threshold",
        "decide_final",
        "PortabilityError",
    }
    assert expected <= set(inspect_sentinel.__all__)
    for name in expected:
        assert hasattr(inspect_sentinel, name)


def test_integration_names_are_not_exported() -> None:
    for name in (
        "HostContext",
        "Recorder",
        "step_types",
        "Stage",
        "Final",
        "validate_decision_shape",
        "resolve_sentinel",
        "SentinelConfig",
        "SentinelEntry",
        "sentinel_from_config",
        "config_from_sentinel",
        "validate_instance_name",
        "PRECEDENCE",
        "named_children",
        "run_sentinel",
    ):
        assert name not in inspect_sentinel.__all__


def test_single_child_runners_are_gone() -> None:
    for name in ("run_monitor", "run_protocol"):
        assert name not in inspect_sentinel.__all__
        assert not hasattr(inspect_sentinel, name)


def test_integration_module_exports_the_host_surface() -> None:
    assert set(inspect_sentinel._integration.__all__) == {
        "Recorder",
        "HostContext",
        "Sentinels",
        "sentinel_from_config",
        "config_from_sentinel",
        "validate_instance_name",
        "resolve_sentinel",
        "step_types",
        "run_sentinel",
    }


def test_wire_types_come_from_inspect_ai() -> None:
    from inspect_ai.core import SentinelAction, SentinelSuspicion

    assert inspect_sentinel.Action is SentinelAction
    assert inspect_sentinel.Suspicion is SentinelSuspicion
