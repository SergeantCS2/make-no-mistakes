"""A candidate's score is a reading: written through `measurements.reading_for` with
its box stamped, on the song instance that sang it, keyed by that instance's settings and by
the scorer, so `mnm calibration` can say "this box's fast tier scores 6.9 ± 0.4 over 12
candidates" and a new song model's candidates are compared with the old one's on one scale.

A score is relative between candidates of one scorer (never a Suno figure); the GUIDE
says so.
"""

from __future__ import annotations

import contextlib
import statistics
from dataclasses import dataclass

from makenomistakes.instances import InstanceService
from makenomistakes.measurements import Kind, Reading, reading_for
from makenomistakes.reading_stamp import stamp_for
from makenomistakes.recipes import Recipe
from makenomistakes.studio.tools_answers import ScoreAnswer


def record(
    service: InstanceService,
    recipe: Recipe,
    answer: ScoreAnswer,
    audio_seconds: float | None,
    scored_run: str,
) -> None:
    """The score of the candidate `scored_run` as a reading of `recipe` (the song instance),
    with this box's stamp."""
    build = None
    with contextlib.suppress(Exception):  # a reading is a hint: Docker silent is no build
        build = service.supervisor.status(recipe.name).build
    service.measurements.add(
        recipe.name,
        reading_for(
            recipe,
            Kind.SCORE,
            build,
            score=answer.score,
            scorer=answer.scorer,
            axes=answer.axes,
            audio_seconds=audio_seconds,
            scored_run=scored_run,
            stamp=stamp_for(service, recipe),
        ),
    )


@dataclass(frozen=True)
class ScoreSummary:
    """One instance's candidates under one scorer and one set of settings."""

    instance: str
    scorer: str
    settings: str
    scores: tuple[float, ...]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores)

    @property
    def spread(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    def words(self) -> str:
        """`6.90 ± 0.40 over 12 candidates (5.9 to 7.6)`."""
        count = len(self.scores)
        many = "candidate" if count == 1 else "candidates"
        span = f" ({min(self.scores):.1f} to {max(self.scores):.1f})" if count > 1 else ""
        return f"{self.mean:.2f} ± {self.spread:.2f} over {count} {many}{span}"


def summaries(by_name: dict[str, list[Reading]]) -> list[ScoreSummary]:
    """Every instance's scores (this box's readings, as the caller filtered them), grouped by
    scorer and settings, in the order the readings were taken; a candidate scored again by
    the same scorer counts once, at its newest score."""
    groups: dict[tuple[str, str, str], dict[str, float]] = {}
    for name, readings in by_name.items():
        for at, reading in enumerate(readings):
            if reading.kind is Kind.SCORE and reading.score is not None:
                key = (name, reading.scorer or "unknown", reading.settings)
                groups.setdefault(key, {})[reading.scored_run or f"#{at}"] = reading.score
    return [
        ScoreSummary(instance, scorer, settings, tuple(scores.values()))
        for (instance, scorer, settings), scores in groups.items()
    ]
