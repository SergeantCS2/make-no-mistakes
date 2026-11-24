"""A quality tune's readings: each measured quant a `Kind.QUALITY` reading of
this box, through `reading_for` with a stamp like every reading, keyed by the candidate's
serving recipe (its files and settings). `measured_quants` gives what this box measured of a
repo's quants: `mnm add`, `mnm fit` and `mnm what` show it beside the planner's speed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from makenomistakes.instances import InstanceService
from makenomistakes.measurements import Kind, MeasurementStore, Reading, reading_for
from makenomistakes.quality.figures import QualityFigures
from makenomistakes.reading_stamp import stamp_for
from makenomistakes.readings_filter import every_here
from makenomistakes.recipes import Recipe

QUALITY_CAPTION = (
    "`mnm tune <name> --quality`: KL divergence from the reference (lower drifts less), how often "
    "the most likely token stayed the reference's, perplexity (one model's quants compare only "
    "among themselves) and the decode measured beside them."
)


@dataclass(frozen=True)
class QuantScore:
    figures: QualityFigures
    at: datetime

    @property
    def words(self) -> str:
        """`KLD 0.0120 ± 0.0004 against Q8_0, the same top token 95.1%, PPL 9.87; 31.2 tok/s
        measured (prose)`."""
        decode = self.figures.decode
        spread = " (spread)" if self.figures.decode_noisy else ""
        speed = f"; {decode:.1f} tok/s measured{spread}" if decode else ""
        return f"{self.figures.words}{speed} ({self.figures.text})"


def write(
    service: InstanceService, recipe: Recipe, figures: QualityFigures, build: str | None
) -> None:
    """One quant's reading, on the instance's measurements (the candidate's recipe keys it).
    Its decode stays in the figures: the reading's `speed` is a request's to every learner."""
    reading = reading_for(
        recipe,
        Kind.QUALITY,
        build,
        stamp=stamp_for(service, recipe),
        quality=figures,
        model_file=figures.file,
    )
    service.measurements.add(recipe.name, reading)


def measured_quants(store: MeasurementStore, repo_id: str) -> dict[str, QuantScore]:
    """The newest score of each of a repo's quants on this box, whatever instance it was
    measured on (one quant's scores against one reference on one text are its own)."""
    return measured_by_repo(store).get(repo_id, {})


def measured_by_repo(store: MeasurementStore) -> dict[str, dict[str, QuantScore]]:
    """Every repo's measured quants on this box, read in one pass (`mnm what`'s rows)."""
    found: dict[str, dict[str, QuantScore]] = {}
    for readings in every_here(store).values():
        for reading in readings:
            figures = reading.quality
            if reading.kind is not Kind.QUALITY or figures is None:
                continue
            quants = found.setdefault(figures.repo_id, {})
            seen = quants.get(figures.quant)
            if seen is None or reading.at > seen.at:
                quants[figures.quant] = QuantScore(figures, reading.at)
    return found


def newest_scores(readings: list[Reading]) -> list[QuantScore]:
    """An instance's newest score of each quant on each reference and text (this box's
    readings, as `readings_filter` gives them), the most faithful first: `mnm calibration`'s
    and the web's quality table."""
    found: dict[tuple[str, str, str, str], QuantScore] = {}
    for reading in readings:
        figures = reading.quality
        if reading.kind is not Kind.QUALITY or figures is None:
            continue
        key = (figures.repo_id, figures.quant, figures.reference, figures.text_sha256)
        seen = found.get(key)
        if seen is None or reading.at > seen.at:
            found[key] = QuantScore(figures, reading.at)
    return sorted(found.values(), key=lambda s: (s.figures.repo_id, s.figures.drift))
