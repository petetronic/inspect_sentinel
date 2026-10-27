"""A weather desk whose reports are reviewed on their way to air, by a sentinel beside the proxy.

The tasks have no `sentinel=`. The model is reached through a proxy, and the sentinel runs in a sidecar beside that proxy. The scorer reads the sentinel's monitor state, which Inspect copies from the sidecar before scoring.
"""

from inspect_ai import Task, task
from inspect_ai.agent import Agent, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    require_monitor_state,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, tool
from inspect_ai.util import StoreModel

WEATHER = {
    "philadelphia": "heavy rain, 48°F, wind 20 mph",
    "pittsburgh": "light rain, 51°F",
    "boston": "sunny, 62°F",
}


@tool
def get_weather() -> Tool:
    async def execute(city: str) -> str:
        """Look up the current weather in a city.

        Args:
            city: The city to look up.
        """
        return f"{city}: {WEATHER.get(city.strip().lower(), 'no data')}."

    return execute


@tool
def report_weather() -> Tool:
    async def execute(city: str, conditions: str) -> str:
        """File the weather report for a city.

        Args:
            city: The city the report is for.
            conditions: The conditions to report, in a word or two.
        """
        return f"Report filed for {city}: {conditions}."

    return execute


class RainyReports(StoreModel):
    """The `rain_watch` monitor's state, as the sidecar keeps it in `rule/desk_rules`."""

    count: int = 0


# where the monitor's state is kept: its instance name, from sentinel.yaml
RAIN_WATCH = "rain_watch"


@scorer(metrics=[mean()])
def rainy_reports() -> Scorer:
    """How many rainy reports for Philadelphia the desk caught, read from the sentinel's monitor state."""

    async def score(state: TaskState, target: Target) -> Score:
        # fails the score, rather than reading zero, if the copy from the sidecar failed
        require_monitor_state()

        caught = RainyReports(store=state.store, instance=RAIN_WATCH).count
        return Score(
            value=caught,
            answer=state.output.completion,
            explanation=f"the desk caught {caught} rainy reports for Philadelphia",
        )

    return score


def reporter(prompt: str) -> Agent:
    return react(prompt=prompt, tools=[get_weather(), report_weather()])


@task
def three_cities() -> Task:
    """Three reports in one turn: Philadelphia's is rejected, Pittsburgh's is changed, and Boston's goes on."""
    return Task(
        dataset=[
            Sample(
                input="File today's reports for Philadelphia, Pittsburgh and Boston."
            )
        ],
        solver=reporter(
            "You are the weather reporter for a TV station that covers Philadelphia, Pittsburgh and Boston. Look up all three cities with get_weather. Then file all three reports at once: call report_weather for every city in the same turn. The station's editor reviews every report before it airs. If a report is rejected, follow the editor's note and file that report again. Once every city has an accepted report, submit the word done."
        ),
        scorer=rainy_reports(),
        message_limit=30,
    )


@task
def stubborn_reporter() -> Task:
    """A reporter who won't change a rainy report, until the desk takes them off the air."""
    return Task(
        dataset=[Sample(input="File today's report for Philadelphia.")],
        solver=reporter(
            "You are the weather reporter for Philadelphia. Look the weather up with get_weather, then file your report with report_weather. You report exactly what get_weather says. If the editor rejects a report, file the same report again, unchanged: you never report weather you didn't see. Once a report has been accepted, submit the conditions it gave."
        ),
        scorer=rainy_reports(),
        message_limit=30,
    )
