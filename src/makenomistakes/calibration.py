"""Measured numbers only ever make an estimate safer.

A reading larger than the estimate applies at once. One that is smaller needs
`MIN_READINGS` readings and lowers the estimate by at most `MAX_LOWER`. "Make no mistakes"
beats squeezing out the last 200 MiB.

Two figures come from the readings:
- **An instance's own figure**: the largest VRAM it was measured holding on this build with
  these settings. *Fits now* and room-making use it (`InstanceVram.estimate`).
- **An engine's correction**: the median of measured-over-estimated across that engine's
  readings. The planner scales the card by it (`plan.make_plan`), and so does auto-fix.
- **A picture family's speed**: the median of a picture's or clip's seconds over the
  fit model's, per DiT family (`speed_correction`), from `MIN_READINGS` warm readings on
  (with fewer warm ones, from `MIN_READINGS` readings cold ones included, said so);
  never capped (time is no safety margin). The first picture after a start (`cold`) also
  moves the weights onto the card, so it over-reads the time: it counts only while there
  are fewer than `MIN_READINGS` warm ones, and the note says so. The planner's s/image and
  s/clip take it; no VRAM figure depends on it.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.hub.kit_types import FAMILY_LABELS, ImageFamily
from makenomistakes.measurements import Reading
from makenomistakes.private_ids import masked as masked_id

MIN_READINGS = 3
MAX_LOWER = 0.10
RECENT = 50  # an engine's correction rests on its newest readings, across instances
# A reading this far from the fit model's figure, either way, is not the instance's own (a
# card shared in a way nothing could see): kept in the file, never used.
PLAUSIBLE = (0.25, 4.0)


def plausible(reading: Reading) -> bool:
    ratio = reading.ratio
    return ratio is None or PLAUSIBLE[0] <= ratio <= PLAUSIBLE[1]


def safer(estimate: int, measured: list[int]) -> int:
    """The figure to plan with, from the estimate and what was measured (the rule above)."""
    if not measured:
        return estimate
    largest = max(measured)
    if largest >= estimate:
        return largest
    if len(measured) < MIN_READINGS:
        return estimate
    return max(largest, int(estimate * (1 - MAX_LOWER)))


@dataclass(frozen=True)
class Figure:
    bytes: int
    readings: int

    @property
    def basis(self) -> str:
        return f"measured on this card, {self.readings} reading{'s' * (self.readings != 1)}"


def instance_figure(
    readings: Iterable[Reading], settings: str, build: str | None, estimate: int | None
) -> Figure | None:
    """This instance's own measured figure for these settings on this build, under the
    rule; None when nothing matches (another build or other settings: the estimate stands)."""
    measured = [
        r.vram_bytes
        for r in readings
        if r.vram_bytes
        and r.settings == settings
        and build is not None
        and r.build == build
        and plausible(r)
    ]
    if not measured:
        return None
    figure = safer(estimate, measured) if estimate else max(measured)
    return Figure(figure, len(measured))


# The types one fit model prices: chat, coding and vision chat are llama.cpp's (or vLLM's)
# same estimate; every other type has its own (embeddings' whole-input logits, sd.cpp's
# pictures and clips).
_SAME_FIT = {Modality.CODING: Modality.CHAT, Modality.VISION_CHAT: Modality.CHAT}


def fit_group(modality: Modality) -> Modality:
    """The type whose correction a type shares."""
    return _SAME_FIT.get(modality, modality)


@dataclass(frozen=True)
class Correction:
    """Multiply an engine's estimates for one fit model (`fit_group`) by `ratio`: 1.0 means
    none. One engine runs different fit models (sd.cpp's pictures and clips, llama.cpp's
    chat and embeddings), so each has its own."""

    engine: Engine
    group: Modality
    ratio: float = 1.0
    readings: int = 0

    @property
    def label(self) -> str:
        return f"{self.engine.value} {self.group.value}"

    def apply(self, estimate: int) -> int:
        return int(estimate * self.ratio)

    @property
    def note(self) -> str | None:
        """What the plan table prints; None when nothing is corrected."""
        if self.readings == 0:
            return None
        if self.ratio == 1.0 and self.readings < MIN_READINGS:
            return (
                f"{self.label}: no correction yet ({self.readings} reading"
                f"{'s' * (self.readings != 1)} on this card; a smaller one counts from "
                f"{MIN_READINGS})"
            )
        if self.ratio == 1.0:
            return f"{self.label}: estimates hold on this card ({self.readings} readings)"
        return (
            f"{self.label}: estimates x{self.ratio:.2f}, measured on this card "
            f"({self.readings} reading{'s' * (self.readings != 1)})"
        )


def correction(
    engine: Engine, modality: Modality, readings: Iterable[Reading], backend: str | None = None
) -> Correction:
    """The correction of the engine's fit model for this type, from its newest readings that
    carry both figures, across builds and instances: it is the fit model's error on this
    card, which an engine update seldom moves (an instance's own figure, which it does,
    waits for the same build). With `backend`, only the readings naming it (an AMD card
    runs one engine on ROCm and on Vulkan, two builds' errors): a plan asks
    `measurements.readings_backend`, never the box's own backend, which a CUDA 13 box's
    `cuda12` images never name; an older reading names none and counts for any."""
    group = fit_group(modality)
    ratios = [
        r.ratio
        for r in readings
        if r.engine is engine
        and (backend is None or r.backend is None or r.backend == backend)
        and fit_group(r.modality) is group
        and r.ratio is not None
        and plausible(r)
    ]
    ratios = ratios[-RECENT:]
    if not ratios:
        return Correction(engine, group)
    median = statistics.median(ratios)
    if median >= 1.0:
        ratio = median
    elif len(ratios) < MIN_READINGS:
        ratio = 1.0
    else:
        ratio = max(median, 1 - MAX_LOWER)
    return Correction(engine, group, ratio, len(ratios))


@dataclass(frozen=True)
class SpeedCorrection:
    """Multiply the fit model's s/image (or s/clip) for one DiT family by `ratio`."""

    family: ImageFamily
    ratio: float = 1.0
    readings: int = 0  # the readings a correction is made from
    cold: int = 0  # first requests after a start: counted only while warm ones are too few
    with_cold: bool = False  # made from first requests too (fewer than 3 warm ones)
    ignored: int = 0  # readings outside `PLAUSIBLE` times the estimate: counted nowhere

    @property
    def label(self) -> str:
        return f"{FAMILY_LABELS.get(self.family, self.family.value)} speed"

    @property
    def note(self) -> str | None:
        """What the plan prints beside the time; None when nothing was measured."""
        if self.readings == 0 and self.cold == 0:
            return None
        count = f"{self.readings} reading{'s' * (self.readings != 1)}"
        if self.with_cold:
            return (
                f"{self.label}: estimates x{self.ratio:.2f}, measured on this card ({count}, "
                f"{self.cold} of them first pictures after a start: those also load the "
                "weights, so the correction leans slow)"
            )
        if self.readings < MIN_READINGS:
            first = (
                f"; {self.cold} first picture{'s' * (self.cold != 1)} after a start not counted, "
                "as they also load the weights"
                if self.cold
                else ""
            )
            return (
                f"{self.label}: no correction yet ({count} on this card; it counts from "
                f"{MIN_READINGS}{first})"
            )
        return f"{self.label}: estimates x{self.ratio:.2f}, measured on this card ({count})"

    @property
    def ignored_words(self) -> str | None:
        """How many readings were left out as implausible (`mnm calibration`, the web)."""
        if not self.ignored:
            return None
        low, high = PLAUSIBLE
        many = f"{self.ignored} reading{'s' * (self.ignored != 1)}"
        return f"{many} ignored: outside {low:g} to {high:g}x the estimate"


def speed_correction(family: ImageFamily, readings: Iterable[Reading]) -> SpeedCorrection:
    """The family's picture-speed correction from its newest timed readings (the rule in
    the module's docstring)."""
    ratios = [
        (r.cold, ratio)
        for r in readings
        if r.family is family and (ratio := r.speed_ratio) is not None
    ]
    timed = [(cold, ratio) for cold, ratio in ratios if PLAUSIBLE[0] <= ratio <= PLAUSIBLE[1]]
    ignored = len(ratios) - len(timed)
    warm = [ratio for cold, ratio in timed if not cold][-RECENT:]
    if len(warm) >= MIN_READINGS:  # warm readings alone, once there are enough
        left = sum(1 for cold, _ratio in timed if cold)
        median = round(statistics.median(warm), 3)
        return SpeedCorrection(family, median, len(warm), left, ignored=ignored)
    every = [ratio for _cold, ratio in timed][-RECENT:]
    cold = len(every) - len(warm)
    if len(every) < MIN_READINGS:  # never from one reading
        return SpeedCorrection(family, 1.0, len(warm), cold, ignored=ignored)
    # The router parks between uses, so nearly every picture is a cold one: counting warm
    # ones only, no correction would ever apply. A cold reading over-reads the time (it
    # also moved the weights in), so a correction from them stays on the slow side.
    ratio = round(statistics.median(every), 3)
    return SpeedCorrection(family, ratio, len(every), cold, with_cold=True, ignored=ignored)


def speed_corrections(readings: list[Reading]) -> list[SpeedCorrection]:
    """One per DiT family that was timed: `mnm calibration`, the export and the planner."""
    families = sorted({r.family for r in readings if r.family is not None and r.speed_ratio})
    found = [speed_correction(family, readings) for family in families]
    return [c for c in found if c.readings or c.cold or c.ignored]


def corrections(readings: list[Reading]) -> list[Correction]:
    """One per engine and fit model that has readings: what `mnm calibration` and the export
    show."""
    kinds = sorted({(r.engine, fit_group(r.modality)) for r in readings})
    found = [correction(engine, group, readings) for engine, group in kinds]
    return [c for c in found if c.readings]


def every_reading(by_name: dict[str, list[Reading]]) -> list[Reading]:
    """All instances' readings, oldest first (the order `correction` takes its newest from)."""
    return sorted((r for found in by_name.values() for r in found), key=lambda r: r.at)


class CorrectionRow(BaseModel):
    engine: Engine
    group: Modality
    ratio: float
    readings: int


class SpeedRow(BaseModel):
    family: ImageFamily
    ratio: float
    readings: int
    cold: int  # first requests after a start, not counted
    ignored: int = 0  # outside `PLAUSIBLE` times the estimate


class MachineSummary(BaseModel):
    """The box an export was written on: its fingerprint id and one line."""

    id: str | None = None
    words: str | None = None


class CalibrationExport(BaseModel):
    """`mnm calibration export`: every reading and correction in one file, to paste into a
    chat; the fit constants are re-fitted from it. Every box's readings are in it, each
    naming its `machine`; the corrections are this box's. The cards' uuids never are
    (`private_ids.masked`)."""

    written_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    machine: MachineSummary | None = None
    gpu: str | None = None
    vram_total_bytes: int | None = None
    corrections: list[CorrectionRow]
    speed_corrections: list[SpeedRow] = Field(default_factory=list)
    readings: dict[str, list[Reading]]


def export(
    by_name: dict[str, list[Reading]],
    gpu: str | None,
    total: int | None,
    here: dict[str, list[Reading]] | None = None,
    machine: MachineSummary | None = None,
) -> CalibrationExport:
    """`by_name` is every reading kept; `here` this box's (`readings_filter.every_here`), which
    the corrections are made of (all of them when not given)."""
    every = every_reading(here if here is not None else by_name)
    rows = [
        CorrectionRow(engine=c.engine, group=c.group, ratio=c.ratio, readings=c.readings)
        for c in corrections(every)
    ]
    speeds = [
        SpeedRow(
            family=c.family, ratio=c.ratio, readings=c.readings, cold=c.cold, ignored=c.ignored
        )
        for c in speed_corrections(every)
    ]
    return CalibrationExport(
        gpu=gpu,
        vram_total_bytes=total,
        corrections=rows,
        speed_corrections=speeds,
        machine=machine,
        readings={name: [masked(r) for r in found] for name, found in by_name.items()},
    )


def masked(reading: Reading) -> Reading:
    """A reading as an export carries it: a card's uuid masked (a hardware
    identifier is a secret); a slot or a name stays."""
    if not reading.cards:
        return reading
    return reading.model_copy(update={"cards": [masked_id(c) for c in reading.cards]})
