"""The readings graph: an instance's decode tok/s over time, from the measurements store
(the request readings the router and `mnm try` took; the store keeps a speed reading only until
it has enough for a build and settings) and the nightly's reports (one row a night, whatever the
store kept), one line per engine build,
drawn as inline SVG on the instance page and System → Nightly. No script and no library: each
point carries a `<title>` (its date, build, settings key and speed) that the browser shows on
hover, behind a hit target larger than the mark.

The terminal skin is one colour (`mnm.css`): the newest build is the bright ink and solid,
earlier builds dimmer and dashed, each named in the legend, so a build is never told by colour
alone. The newest `BUILDS` builds are drawn; the readings table under the graph lists them all."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from makenomistakes.local_time import local
from makenomistakes.measurements import Reading, SpeedUnit
from makenomistakes.nightly.report import NightlyReport, StepName
from makenomistakes.readings_filter import Here

WIDTH = 640
HEIGHT = 200
LEFT, RIGHT, TOP, BOTTOM = 44, 12, 12, 26  # room for the axis labels
BUILDS = 4
DASHES = ("", "6 4", "2 4", "8 3 2 3")  # newest first: solid, then three dash patterns
SAME_SECONDS = 300  # a request reading this close to a nightly row of its build is that row


@dataclass(frozen=True)
class Speed:
    """One decode reading, from either source."""

    at: datetime
    tok_s: float
    build: str
    settings: str
    nightly: bool  # from a nightly report (else a request reading)


def from_readings(readings: list[Reading]) -> list[Speed]:
    return [
        Speed(r.at, r.speed, r.build or "unknown", r.settings, nightly=False)
        for r in readings
        if r.speed and r.speed_unit is SpeedUnit.TOKENS
    ]


def from_reports(name: str, reports: list[NightlyReport], here: Here | None = None) -> list[Speed]:
    """The nightly's rows of `name` (on `here` when given)."""
    found = []
    for report in reports:
        step = report.step(StepName.TRY)
        for row in step.rows if step is not None else []:
            if here is not None and not here.owns(row.machine):
                continue
            if row.name == name and row.speed and row.unit is SpeedUnit.TOKENS:
                at = report.finished_at or report.started_at
                found.append(
                    Speed(at, row.speed, row.build or "unknown", row.settings or "?", True)
                )
    return found


def merged(
    readings: list[Reading], reports: list[NightlyReport], name: str, here: Here | None = None
) -> list[Speed]:
    """Both sources (this box's, `here`: the readings come filtered), a request reading the
    nightly's own row already stands for left out."""
    nightly = from_reports(name, reports, here)

    def known(speed: Speed) -> bool:
        return any(
            n.build == speed.build and abs((n.at - speed.at).total_seconds()) < SAME_SECONDS
            for n in nightly
        )

    return [*nightly, *(s for s in from_readings(readings) if not known(s))]


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    title: str


@dataclass(frozen=True)
class Series:
    build: str
    points: list[Point]
    dash: str
    newest: bool

    @property
    def path(self) -> str:
        return " ".join(f"{p.x:.1f},{p.y:.1f}" for p in self.points)


@dataclass(frozen=True)
class Tick:
    at: float  # px
    label: str


@dataclass(frozen=True)
class Graph:
    name: str
    series: list[Series]
    y_ticks: list[Tick]
    x_ticks: list[Tick]
    readings: int
    width: int = WIDTH
    height: int = HEIGHT
    left: int = LEFT
    plot_right: int = WIDTH - RIGHT
    plot_bottom: int = HEIGHT - BOTTOM

    @property
    def summary(self) -> str:
        """What a screen reader hears for the whole graph."""
        builds = ", ".join(s.build for s in self.series)
        return (
            f"{self.name}: decode speed of {self.readings} readings over time, one line per "
            f"engine build ({builds})"
        )


def _nice(top: float) -> float:
    """A round axis top at or above `top`: 1, 2 or 5 times a power of ten."""
    if top <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(top))
    return float(next(step * power for step in (1, 2, 5, 10) if step * power >= top))


def speed_graph(name: str, speeds: list[Speed]) -> Graph | None:
    """The graph of a chat-like instance's tok/s readings; None with fewer than two."""
    if len(speeds) < 2:
        return None
    timed = sorted(speeds, key=lambda s: s.at)
    builds: list[str] = []
    for speed in reversed(timed):  # newest build first
        if speed.build not in builds:
            builds.append(speed.build)
    shown = builds[:BUILDS]
    kept = [s for s in timed if s.build in shown]
    first, last = kept[0].at.timestamp(), kept[-1].at.timestamp()
    span = max(last - first, 1.0)
    top = _nice(max(s.tok_s for s in kept) * 1.1)
    width, height = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(speed: Speed) -> float:
        return LEFT + width * (speed.at.timestamp() - first) / span

    def y(value: float) -> float:
        return TOP + height * (1 - value / top)

    def title(s: Speed) -> str:
        what = "nightly" if s.nightly else "a request"
        return (
            f"{local(s.at):%Y-%m-%d %H:%M}: {s.tok_s:.1f} tok/s ({what}), build {s.build}, "
            f"settings {s.settings}"
        )

    series = [
        Series(
            build,
            [Point(x(s), y(s.tok_s), title(s)) for s in kept if s.build == build],
            DASHES[index],
            newest=index == 0,
        )
        for index, build in enumerate(shown)
    ]
    y_ticks = [Tick(y(v), f"{v:g}") for v in (0, top / 2, top)]
    begin, end = local(kept[0].at), local(kept[-1].at)
    when = "%H:%M" if begin.date() == end.date() else "%m-%d"
    x_ticks = [Tick(LEFT, f"{begin:{when}}"), Tick(LEFT + width, f"{end:{when}}")]
    return Graph(name, series, y_ticks, x_ticks, len(kept))
