"""The quality tune's choice on a recipe (recipe schema 27): the quant it
kept, its scores against the tune's reference, the speed it measured and the floor it met.
None on a recipe never quality-tuned, or planned since at another quant (`replan` drops it)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from makenomistakes.quality.figures import QualityFigures


class QualityTuned(BaseModel):
    quant: str  # the quant the tune kept (the recipe's own since)
    was: str  # the quant it had before the tune
    figures: QualityFigures  # the kept quant's scores, its measured decode among them
    floor: float = Field(gt=0)  # the floor it met (tok/s)
    use: Literal["chat", "agents"]
    measured: int = Field(ge=1)  # the quants the tune measured
    build: str | None = None  # the `full` image's short id the scores were taken on
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    made_for: datetime | None = None  # the queue entry it was saved for (`Tuned.made_for`)

    @model_validator(mode="after")
    def _timed(self) -> Self:
        if self.figures.decode is None:
            raise ValueError("a kept quant's decode was measured")
        return self

    @property
    def decode(self) -> float:
        return self.figures.decode or 0.0  # set: `_timed`

    @property
    def summary(self) -> str:
        """`Q5_K_M: KLD 0.0120 ± 0.0004 against Q8_0, the same top token 95.1%, PPL 9.87, 31.2
        tok/s (floor 15)`; the reference kept: `Q8_0, the reference itself (PPL 9.80), …`."""
        speed = f"{self.decode:.1f} tok/s (floor {self.floor:g})"
        if self.figures.is_reference:
            return f"{self.quant}, the reference itself (PPL {self.figures.ppl:.2f}), {speed}"
        return f"{self.quant}: {self.figures.words}, {speed}"

    def lines(self) -> list[tuple[str, str]]:
        """`mnm show`'s lines for the block."""
        moved = f"from {self.was}" if self.was != self.quant else f"kept {self.quant}"
        on = f"{self.measured} quants measured on {self.figures.text}"
        return [
            ("Quality", f"{self.summary} ({moved}; {on})"),
            ("Quality on", f"build {self.build or '?'}, {self.at:%Y-%m-%d %H:%M} UTC"),
        ]
