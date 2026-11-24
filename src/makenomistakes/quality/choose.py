"""Which quant a quality tune keeps (quality over speed, above the user's floor). Pure over
what the tune measured.

Among the quants whose measured decode meets the floor, the one whose answers drift least from
the reference's (the lowest mean KLD; the reference's own is 0). Two whose KLDs differ by less
than twice their combined error are the same quality as far as the text can tell: the faster
of them is kept. None over the floor keeps nothing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from makenomistakes.quality.figures import QualityFigures

NOISE = 2.0  # KLDs closer than this many combined errors are the same quality


@dataclass(frozen=True)
class Measured:
    quant: str
    figures: QualityFigures | None  # None: its quality run failed or never ran
    decode: float | None  # median tok/s at its serving plan; None: no speed reading
    serves: bool = True  # False: a reference only measured (never kept)
    noisy: bool = False  # its speed readings spread (or something else used the card)
    build: str | None = None  # the `full` image its quality run read it on

    @property
    def scored(self) -> QualityFigures | None:
        """Its figures with the decode the tune measured beside them (what a reading keeps)."""
        if self.figures is None:
            return None
        return self.figures.model_copy(update={"decode": self.decode, "decode_noisy": self.noisy})


def eligible(rows: list[Measured], floor: float) -> list[Measured]:
    return [
        m for m in rows
        if m.serves and m.figures is not None and m.decode is not None and m.decode >= floor
    ]  # fmt: skip


def choose(rows: list[Measured], floor: float) -> Measured | None:
    """The quant to keep; None when none meets the floor with its quality known."""
    found = eligible(rows, floor)
    if not found:
        return None
    best = min(found, key=lambda m: (_drift(m), -(m.decode or 0.0)))
    tied = [m for m in found if _drift(m) - _drift(best) <= NOISE * _err(m, best)]
    return max(tied, key=lambda m: m.decode or 0.0)


def fastest(rows: list[Measured]) -> Measured | None:
    """The fastest quant measured (what a tune with none over the floor says it read)."""
    timed = [m for m in rows if m.serves and m.decode is not None]
    return max(timed, key=lambda m: m.decode or 0.0, default=None)


def _drift(m: Measured) -> float:
    return m.figures.drift if m.figures is not None else math.inf


def _err(a: Measured, b: Measured) -> float:
    ea = a.figures.drift_err if a.figures is not None else 0.0
    eb = b.figures.drift_err if b.figures is not None else 0.0
    return math.hypot(ea, eb)
