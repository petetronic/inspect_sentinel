"""A rule for a sentinel that runs beside a proxy: it's always sunny in Philadelphia."""

from inspect_sentinel import BeforeToolCall, Context, Decision, Protocol, protocol


@protocol
def always_sunny() -> Protocol:
    """Reject any report of Philadelphia's weather that isn't sunny."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.function != "report_weather":
            return None
        city = str(step.call.arguments.get("city", ""))
        conditions = str(step.call.arguments.get("conditions", ""))
        if "philadelphia" in city.lower() and "sunny" not in conditions.lower():
            return Decision.reject(
                f"reported {conditions!r} for Philadelphia",
                message="It's always sunny in Philadelphia.",
            )
        return Decision.proceed()

    return decide
