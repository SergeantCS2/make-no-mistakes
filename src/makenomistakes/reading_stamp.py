"""What a reading is stamped with: the box it was taken on, the cards, and what else
held the card before it.

A browser playing a video on Windows holds part of the card: nvidia-smi's memory in use
carries it, every plan is sized with that much less free, and no reading would say so
(`measure_window` watches only mnm's own traffic). A writer samples the card before its
start or request (`card_before`) and stamps the reading after (`stamp_for`): the card's memory
in use then, less what mnm's running instances hold as the
room verdicts price them (`InstanceService.running_vram`), and `utilization.gpu`. ComfyUI's
own memory is not subtracted (asking it needs Docker and its API): it reads as outside.

The box is `machine.json`'s current fingerprint (`readings_filter.here_of`), the one every
consumer filters by. Nothing here fails the reading it stamps: a part that cannot be read is
None."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.fit.budget import BUSY_CARD_BYTES
from makenomistakes.hardware.fingerprint import stable_id_of
from makenomistakes.hardware.probe import Gpu, MemorySnapshot
from makenomistakes.measure_window import Window
from makenomistakes.measurements import Reading, Stamp, gpu_name_of
from makenomistakes.readings_filter import Here, here_of
from makenomistakes.recipes.cards import recipe_cards

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService
    from makenomistakes.recipes import Recipe

GIB = 2**30
BUSY_PERCENT = 20  # a card this busy before the request was working for something else


@dataclass(frozen=True)
class CardBefore:
    """The card as a writer found it before its start or request (None: unreadable)."""

    snapshot: MemorySnapshot | None
    own_running: bool  # the instance already held its memory then (a request, not a start)


def card_before(service: InstanceService, own_running: bool = True) -> CardBefore:
    try:
        return CardBefore(service.memory(), own_running)
    except Exception:  # a reading is a hint
        return CardBefore(None, own_running)


def window_before(window: Window, own_running: bool = True) -> CardBefore:
    """The card as a window opened on it read it (no second read of nvidia-smi)."""
    return CardBefore(window.snapshot, own_running)


def instance_cards(snapshot: MemorySnapshot, recipe: Recipe | None) -> list[Gpu]:
    """The cards the instance runs on, in nvidia-smi's order (one card: the first chosen)."""
    first = snapshot.gpu
    if recipe is None or len(snapshot.cards) < 2:
        return [first] if first is not None else []
    chosen = recipe_cards(recipe, listed=[g.index for g in snapshot.cards]) or ()
    return [c for i in sorted(chosen) if (c := snapshot.card(i)) is not None]


def stamp_for(service: InstanceService, recipe: Recipe, before: CardBefore | None = None) -> Stamp:
    """The stamp of a reading of `recipe` taken now (`before`: the card before it)."""
    snapshot = before.snapshot if before is not None else None
    if snapshot is None:
        with contextlib.suppress(Exception):
            snapshot = service.memory()
    try:
        here = here_of(service.settings.expanded_state_dir())
    except Exception:  # a service that cannot say its state folder: no box named
        here = Here()
    if snapshot is None:
        return Stamp(machine=here.id)
    cards = instance_cards(snapshot, recipe)
    ids = tuple(stable_id_of(c.uuid, c.bus_id, c.name, c.index, c.unique_id) for c in cards)
    outside = busy = None
    if before is not None and before.snapshot is not None:
        outside = _outside(service, recipe, before.snapshot, before.own_running)
        found = [c.busy_percent for c in cards if c.busy_percent is not None]
        busy = max(found) if found else None
    return Stamp(gpu_name_of(snapshot, recipe), here.id, ids, outside, busy)


def _outside(
    service: InstanceService, recipe: Recipe, snapshot: MemorySnapshot, own_running: bool
) -> int | None:
    """The chosen cards' memory in use less what mnm's running instances hold (the reading's
    own instance only when it was running then); never below 0 (an estimate above what the
    card says)."""
    if not snapshot.cards:
        return None
    used = sum(c.vram_used_bytes for c in snapshot.cards)
    try:
        held = service.running_vram(in_ram=False)  # what mnm holds on the cards
    except Exception:
        return None
    if not own_running:
        held.pop(recipe.name, None)
    return max(used - sum(held.values()), 0)


def beside_something(reading: Reading) -> bool:
    """Whether something outside mnm held or worked the card before the reading."""
    held = reading.outside_bytes is not None and reading.outside_bytes > BUSY_CARD_BYTES
    working = reading.gpu_busy is not None and reading.gpu_busy > BUSY_PERCENT
    return held or working


def beside_words(reading: Reading) -> str | None:
    """`beside something else (3.1 GiB held, 45% busy)`; None when the card was mnm's."""
    if not beside_something(reading):
        return None
    parts = []
    if reading.outside_bytes is not None:
        parts.append(f"{reading.outside_bytes / GIB:.1f} GiB held")
    if reading.gpu_busy is not None:
        parts.append(f"{reading.gpu_busy}% busy")
    return f"beside something else ({', '.join(parts)})"
