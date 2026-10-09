"""A rule for a sentinel that runs beside a proxy: it's always sunny in Philadelphia."""

from inspect_ai.util import StoreModel

from inspect_sentinel import BeforeToolCall, Context, Decision, Protocol, protocol


class Tally(StoreModel):
    """What the rule has done with the reports of one run."""

    reports: int = 0
    rejected: int = 0


@protocol
def always_sunny() -> Protocol:
    """Reject any report of Philadelphia's weather that isn't sunny, unless the sample says its station has no such rule."""

    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        if step.call.function != "report_weather":
            return None
        tally = context.store_as(Tally)
        tally.reports += 1

        # a sample's metadata never appears in a model call, so the sentinel
        # has it only when the eval told the sidecar about the sample. Told
        # nothing, it holds every report to the rule.
        applies = context.eval is None or bool(
            context.eval.metadata.get("always_sunny", True)
        )

        city = str(step.call.arguments.get("city", ""))
        conditions = str(step.call.arguments.get("conditions", ""))
        if (
            applies
            and "philadelphia" in city.lower()
            and "sunny" not in conditions.lower()
        ):
            tally.rejected += 1
            return Decision.reject(
                f"reported {conditions!r} for Philadelphia",
                message="It's always sunny in Philadelphia.",
            )
        return Decision.proceed()

    return decide
