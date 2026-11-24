"""Whether a draft (`mnm set <name> draft=<instance>|mtp`) leaves the instance room on the
card: on an RTX 4070, coder38 alone loaded 10.37 GiB of 11.5 (near the
limit) and read 21.4 tok/s; with its MTP head it loaded 11.30 GiB and read 8.46, llama.cpp
spilling at the card's edge. The room is what the card can give the instance: the VRAM free
now plus what mnm's running instances hold (the router parks them for it), as `mnm add`
plans. Over that, or tight while the model alone is already tight, the draft is refused;
tight beside a model that fits comfortably alone is a warning (`mnm try --speed` decides)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.draft import DraftError, draft_name
from makenomistakes.fit.estimate import FitLevel, headroom_level
from makenomistakes.ram_room import runs_on_processor
from makenomistakes.recipes.model import Recipe

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

GIB = 2**30


@dataclass(frozen=True)
class DraftRoom:
    need: int  # the instance with its draft, bytes on the card
    alone: int  # the instance without one
    room: int  # what the card can give it
    level: FitLevel  # the draft's
    alone_level: FitLevel

    @property
    def refused(self) -> bool:
        if self.level is FitLevel.NO_FIT:
            return True
        return self.level is FitLevel.TIGHT and self.alone_level is not FitLevel.COMFORTABLE


def draft_room(service: InstanceService, recipe: Recipe) -> DraftRoom | None:
    """`recipe` (its draft set) against the card; None when there is no draft or a figure
    cannot be had (no card, files unread: nothing is refused on a guess)."""
    if recipe.draft is None:
        return None
    room = card_room(service, recipe)
    need = service.vram_estimate(recipe).bytes
    alone = service.vram_estimate(recipe.model_copy(update={"draft": None})).bytes
    if room is None or need is None or alone is None:
        return None
    return DraftRoom(need, alone, room, headroom_level(need, room), headroom_level(alone, room))


def card_room(service: InstanceService, recipe: Recipe) -> int | None:
    """What the card can give `recipe`: the VRAM free now plus what mnm's running instances
    hold there (the router parks them for it), never more than the card; on the processor's
    image the RAM and what runs in it; None with neither read. The one figure a
    draft and a MoE cache are judged against."""
    snapshot = service.memory()
    here = runs_on_processor(recipe)
    pool = snapshot.ram if here else snapshot.gpu
    if pool is None:
        return None
    held = sum(service.running_vram(in_ram=here).values())
    return min(pool.vram_total_bytes, pool.vram_total_bytes - pool.vram_used_bytes + held)


def checked(service: InstanceService, before: Recipe, after: Recipe) -> Recipe:
    """`after` when its draft is unchanged or leaves room; DraftError with the figures when
    the draft would push the card to its edge."""
    if after.draft is None or after.draft == before.draft:
        return after
    found = draft_room(service, after)
    if found is None or not found.refused:
        return after
    what = "its MTP head" if after.draft.is_mtp else f"the draft {draft_name(after.draft)}"
    if found.level is FitLevel.NO_FIT:
        verdict = "it does not fit"
    else:
        verdict = "it would fit with almost no room"
    alone = (
        "the model alone fits near the limit"
        if found.alone_level is FitLevel.TIGHT
        else "the model alone fits"
        if found.alone_level is FitLevel.COMFORTABLE
        else "the model alone does not fit either"
    )
    raise DraftError(
        f"{after.name} with {what} needs about {found.need / GIB:.1f} GiB on the card and the "
        f"card can give it {found.room / GIB:.1f} GiB (free now plus what mnm's running models "
        f"hold): {verdict}; {alone} ({found.alone / GIB:.1f} GiB). At the card's edge "
        "llama.cpp spills and the draft loses more than it gains (the box's coder38: 21.4 "
        "tok/s alone, 8.5 with its head). Nothing saved."
    )


def tight_note(service: InstanceService, recipe: Recipe) -> str | None:
    """The warning a tight draft beside a comfortable model gets once saved."""
    found = draft_room(service, recipe)
    if found is None or found.level is not FitLevel.TIGHT or found.refused:
        return None
    return (
        f"Tight: {recipe.name} with its draft needs about {found.need / GIB:.1f} of the "
        f"{found.room / GIB:.1f} GiB the card can give it. `mnm try {recipe.name} --speed` "
        "decides whether it gains; `mnm set "
        f"{recipe.name} draft=` runs it alone."
    )
