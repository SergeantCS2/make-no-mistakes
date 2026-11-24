"""The queue's readings summed: each instance's `Kind.QUEUE` windows on this box, per
settings (a second slot is new settings, so part two's policy compares the rows), worded once
for `mnm calibration` and System → Calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.model_meta import context_of
from makenomistakes.queue_figures import QueueFigures

if TYPE_CHECKING:
    from makenomistakes.measurements import Reading

QUEUE_CAPTION = (
    "A slot wait is the router's count (the engines take requests in order); a load wait, past "
    "a second; llama.cpp's own wait, its time at the engine outside its prompt and generation; "
    "the context held, the tokens one request held."
)


@dataclass(frozen=True)
class QueueSummary:
    """One instance's windows on this box under one settings, summed."""

    name: str
    windows: int
    figures: QueueFigures
    per_request: int | None  # the tokens one request may hold with these settings

    @property
    def waited_share(self) -> float:
        """The share of the requests that reached the engine and waited for a slot."""
        forwarded = self.figures.forwarded
        return self.figures.slot_waits.n / forwarded if forwarded else 0.0

    def requests_words(self) -> str:
        figures = self.figures
        windows = f"{self.windows} window{'s' * (self.windows != 1)}"
        refused = figures.requests - figures.forwarded
        shut = f" ({refused:,} never reached it)" if refused else ""
        return f"{figures.forwarded:,}{shut} in {windows}, at most {figures.peak} at once"

    def slot_words(self) -> str:
        waits = self.figures.slot_waits
        if not waits.n:
            return "none"
        share = f"{waits.n:,} of {self.figures.forwarded:,} ({self.waited_share:.0%})"
        return f"{share}, longest {waits.top:.1f} s, mean {waits.mean or 0:.1f} s"

    def load_words(self) -> str:
        waits = self.figures.load_waits
        if not waits.n:
            return "none"
        return f"{waits.n:,}, longest {waits.top:.1f} s, mean {waits.mean or 0:.1f} s"

    def engine_words(self) -> str:
        """llama.cpp's own: answers whose time at the engine was more than their work."""
        timed, waits = self.figures.timed, self.figures.engine_waits
        if not timed:
            return "-"
        if not waits.n:
            return f"none of {timed:,} answers"
        return f"{waits.n:,} of {timed:,} answers, longest {waits.top:.1f} s"

    def context_words(self) -> str:
        held = self.figures.context
        if not held.n:
            return "-"
        limit = f" of the {self.per_request:,} a request may hold" if self.per_request else ""
        return f"at most {held.top:,.0f} tokens{limit} (mean {held.mean or 0:,.0f})"


def summaries(by_name: dict[str, list[Reading]]) -> list[QueueSummary]:
    """Each instance's queue readings summed per settings key (as every reading is compared),
    its newest settings first; the context a request may hold is the newest window's."""
    found: list[QueueSummary] = []
    for name, readings in by_name.items():
        groups: dict[str, tuple[int, QueueFigures, int | None]] = {}
        for reading in reversed(readings):  # newest first: its row first, its context's limit
            figures = reading.queue
            if figures is None:
                continue
            before = groups.get(reading.settings)
            if before is None:
                groups[reading.settings] = (1, figures, context_of(reading.args))
            else:
                count, merged, limit = before
                groups[reading.settings] = (count + 1, merged.merged(figures), limit)
        found += [QueueSummary(name, n, f, limit) for n, f, limit in groups.values()]
    return found
