"""When a reading of the card is one instance's own.

nvidia-smi gives the card's total in use, not each process's (WSL reports no per-process
memory), so a reading is the card's change over a window. It counts only when, from the
window's start to its end, nothing else changed the card: no other instance had a request in
flight, no request arrived at the router (one that began and ended inside the window would
be invisible to the busy count alone), no other start or stop is in progress at its end, and
nothing on the box started or stopped meanwhile (`box_events`). Anything unknown counts as
not alone; on a box without a card there is no card to read, and the rest decides."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes import box_events
from makenomistakes.backend_now import no_card
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.router_client import Traffic


@dataclass(frozen=True)
class Window:
    used: int | None  # the card's memory in use when the window opened
    traffic: Traffic | None
    stamp: int | None
    # The whole read `used` came from (every card, `utilization.gpu`), when the card
    # is read through one: a reading's stamp says what else held the card then
    snapshot: MemorySnapshot | None = None


class Card:
    """What a window reads: the card, the router's traffic, the box's events."""

    def __init__(
        self,
        settings: Settings,
        used: Callable[[], int | None],
        traffic: Callable[[], Traffic | None],
        snapshot: Callable[[], MemorySnapshot] | None = None,
    ) -> None:
        self._settings = settings
        self.used = used
        self.traffic = traffic
        self._snapshot = snapshot  # read instead of `used` when the window opens (one read)

    def open(self) -> Window:
        if self._snapshot is None:
            return Window(self.used(), self.traffic(), box_events.stamp(self._settings))
        snap = self._snapshot()
        used = snap.gpu.vram_used_bytes if snap.gpu is not None else None
        return Window(used, self.traffic(), box_events.stamp(self._settings), snap)

    def alone(self, name: str, began: Window, own: int = 0) -> bool:
        """Whether nothing but `name` touched the card since `began`; `own` requests arrived
        meanwhile are the caller's (`mnm try --measure` opens its window before it asks)."""
        now = self.traffic()
        if began.traffic is None or now is None or (began.used is None and not no_card()):
            return False
        # A start in progress when the window opened either ended inside it (the stamp moved)
        # or still runs at its end (its marker is there).
        return (
            began.traffic.only(name)
            and now.only(name)
            and now.arrivals == began.traffic.arrivals + own
            and box_events.stamp(self._settings) == began.stamp
            and not box_events.others_changing(self._settings)
        )

    def grew(self, name: str, began: Window) -> int | None:
        """What the card grew by since `began`, if `name` was alone on it; None otherwise and
        for a card that did not grow (nothing measurable)."""
        after = self.used()
        if after is None or began.used is None or not self.alone(name, began):
            return None
        grown = after - began.used
        return grown if grown > 0 else None
