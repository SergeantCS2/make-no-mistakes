"""`mnm studio models` and System → Studio's Models table: every studio model of the
ladder (`studio/music_ladder.py`) against this box now, the smallest card that holds it, and
its time: this box's own once it has readings (`music_speed`), else `fit.music`'s estimate
for this card. The room is what `mnm add` plans in: the card's free VRAM with what mnm's
running models hold counted free, or on a box that runs every model in RAM the RAM's."""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.fit.budget import Budget
from makenomistakes.fit.music import TimeBasis, music_time, rate_key
from makenomistakes.hub.facts import Engine
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.measurements import readings_backend
from makenomistakes.music_speed import MusicRate, rates_here
from makenomistakes.plan_budget import planning_budget
from makenomistakes.studio.music_ladder import (
    MUSIC_LADDER,
    MusicRung,
    card_holds,
    card_words,
    need_words,
    verdict_words,
)

GIB = 2**30


@dataclass(frozen=True)
class ModelRow:
    rung: MusicRung
    needs: str
    here: str  # this box's word now
    holds: bool  # this box holds it now
    card: str  # the smallest common card that holds it alone
    time: str  # seconds of work a second of audio, and where the figure came from


@dataclass(frozen=True)
class Models:
    room: str  # the line above the table: what this box has free, and of what
    rows: list[ModelRow]
    in_ram: bool = False  # a box that runs every model in RAM


def models(service: InstanceService) -> Models:
    """Every rung against this box now; a store or Docker that cannot be read leaves the
    verdicts unknown and says why, never failing the page."""
    try:
        found = planning_budget(service)
    except InstanceError as exc:
        return _unread(str(exc))
    first = found.cards[0] if found is not None and len(found.cards) > 1 else None
    budget = found.one_card() if found is not None else None  # audio.cpp runs on one card
    where = "the RAM" if budget is not None and budget.no_card else "the card"
    ran = readings_backend(Engine.AUDIOCPP, budget.backend, budget.arch) if budget else None
    learned = rates_here(service.settings.expanded_state_dir(), ran)
    unread = f"unknown ({where} cannot be read)" if budget is None else ""
    rows = [
        ModelRow(
            rung=rung,
            needs=need_words(rung, where),
            here=verdict_words(rung, budget.vram_free_bytes) if budget is not None else unread,
            holds=budget is not None and card_holds(rung, budget.vram_free_bytes),
            card=card_words(rung),
            time=_time(rung, budget, learned),
        )
        for rung in MUSIC_LADDER
    ]
    if budget is None:
        room = "This box: neither the card nor the RAM can be read right now (`mnm doctor`)."
    else:
        of = f"card {first.index} ({first.name})" if first else where
        room = f"This box: {budget.vram_free_bytes / GIB:.1f} GiB of {of} free"
        if budget.held_by:
            room += ", counting what mnm's running models hold as free (a start parks them)"
        room += "." if first is None else ": each studio model runs on that card alone."
    return Models(room=room, rows=rows, in_ram=budget is not None and budget.no_card)


def _unread(why: str) -> Models:
    rows = [
        ModelRow(rung=r, needs=need_words(r), here="unknown", holds=False, card=card_words(r),
                 time="-")
        for r in MUSIC_LADDER
    ]  # fmt: skip
    return Models(room=f"This box: its room could not be read now ({why}).", rows=rows)


def _time(rung: MusicRung, budget: Budget | None, learned: dict[str, MusicRate]) -> str:
    if rung.file is None or budget is None:
        return "-"
    rates = {key: rate.rate for key, rate in learned.items()}
    time = music_time(rung.family, budget.device, rung.file, rates)
    if time.basis is TimeBasis.MEASURED:
        found = learned[rate_key(rung.family.value, rung.file)]
        return f"{time.rate:.2f} s (this box, {found.readings} readings)"
    return f"{time.rate:.2f} s (estimate)"
