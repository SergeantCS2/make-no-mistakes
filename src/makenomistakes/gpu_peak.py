"""GPU memory during a request (`mnm try --measure`) and on demand (`mnm vram`).

`PeakSampler` reads nvidia-smi's used memory every quarter second while a request runs: the
number the image fit model is calibrated against. Both take their reader and
clock as arguments so tests never touch a GPU.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

ReadUsed = Callable[[], int | None]  # GPU memory in use now (bytes), None when unreadable
# Every chosen card's memory in use now, by nvidia-smi's index (empty: unreadable)
ReadCards = Callable[[], dict[int, int]]


class PeakSampler:
    """GPU memory in use, read every `every` seconds on a thread while a request runs."""

    def __init__(
        self, read_used: ReadUsed, every: float = 0.25, read_cards: ReadCards | None = None
    ) -> None:
        self._read = read_used
        self._cards = read_cards  # several cards: each one sampled too
        self._every = every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.readings: list[int] = []
        self.card_readings: list[dict[int, int]] = []

    def __enter__(self) -> PeakSampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()

    @property
    def peak(self) -> int | None:
        return max(self.readings) if self.readings else None

    @property
    def card_peaks(self) -> dict[int, int]:
        """Each card's highest reading; empty when only one card was sampled."""
        peaks: dict[int, int] = {}
        for reading in self.card_readings:
            for index, used in reading.items():
                peaks[index] = max(peaks.get(index, 0), used)
        return peaks

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            self._sample()

    def _sample(self) -> None:
        used = self._read()
        if used is not None:
            self.readings.append(used)
        cards = self._cards() if self._cards is not None else {}
        if cards:
            self.card_readings.append(cards)


def vram_readings(
    read_used: ReadUsed, count: int, every: float, sleep: Callable[[float], None] = time.sleep
) -> list[int]:
    """`count` readings `every` seconds apart (unreadable ones are skipped)."""
    readings: list[int] = []
    for index in range(count):
        if index:
            sleep(every)
        used = read_used()
        if used is not None:
            readings.append(used)
    return readings
