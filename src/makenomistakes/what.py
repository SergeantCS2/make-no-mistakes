"""`mnm what`: the hits (`what_sources`) graded (`what_grade`), the answer the CLI and the
web page show. With `--for <use>` (no words) the starter menu's picks for the use graded,
three of them: those that fit comfortably first, then the tighter ones, each with its line;
only curated rows are ever picks (mnm ranks none of a source's hits)."""

from __future__ import annotations

from dataclasses import dataclass, field

from makenomistakes.config import Settings
from makenomistakes.fit import Budget, FitLevel
from makenomistakes.hub.civitai import SourcesHub
from makenomistakes.model_uses import Use
from makenomistakes.what_grade import Graded, Planner, Verdict, grade_all, plan_hit
from makenomistakes.what_sources import find

PICKS = 3  # `--for`'s three picks
_LEVELS = (FitLevel.COMFORTABLE, FitLevel.TIGHT, FitLevel.OFFLOAD)


@dataclass
class Answer:
    rows: list[Graded] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # a source that did not answer
    left_out: int = 0  # hits the planner read as another type than the use asked


def answer(  # noqa: PLR0913 - the search's inputs, and a swappable planner
    sources: SourcesHub,
    settings: Settings,
    budget: Budget,
    words: str,
    use: Use | None,
    limit: int,
    *,
    picks: bool = False,
    planner: Planner = plan_hit,
) -> Answer:
    found = find(sources, words, use, limit)
    graded = grade_all(found.hits, settings, budget, planner)
    rows = [g for g in graded if g.verdict is not Verdict.OTHER_TYPE]
    done = Answer(rows, found.notes, len(graded) - len(rows))
    if picks:
        done.rows = sorted((r for r in rows if r.hit.curated), key=_pick_order)[:PICKS]
    return done


def _pick_order(row: Graded) -> tuple[int, int]:
    """Comfortable first, then tight, then offload, then what does not fit; at one level the
    profile's own quant before a smaller one; the menu's order within (the sort is stable)."""
    if row.pick is None:
        return (len(_LEVELS), 0)
    level = row.pick.fit.level
    rank = _LEVELS.index(level) if level in _LEVELS else len(_LEVELS)
    return (rank, 0 if row.verdict is Verdict.FITS else 1)
