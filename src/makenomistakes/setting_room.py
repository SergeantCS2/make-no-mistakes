"""Settings judged against the model's files and the card when they are saved, beside the
draft's room (`draft_room`):

- The context: `mnm set qwen3-8b ctx=65536` would save a context above what the model was
  trained for (40,960); llama-server caps each slot there itself ("the slot context (65536)
  exceeds the training context of the model (40960) - capping") and the recipe would keep
  65,536. A per-request context above the model's own is refused, naming it.
- The MoE cache: `mnm set coder36 moe-cache=1024` would save a cache with no room judged. A
  cache is judged as a draft is: against what the card can give the instance (free now plus
  what mnm's running models hold); over it, or tight while the model alone is already tight,
  refused with the largest cache that stays comfortable; tight beside a comfortable model, a
  warning.

Both judge only a value this save changed: the web form posts every field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.controls import SettingsError
from makenomistakes.draft_room import card_room
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.fit.estimate import COMFORTABLE_HEADROOM, FitLevel, headroom_level
from makenomistakes.model_meta import serving_context, serving_slots
from makenomistakes.recipes.model import Recipe

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

GIB = 2**30
MIB = 2**20
CACHE_STEP_MIB = 64  # the largest comfortable cache is said in steps of this


def checked_settings(service: InstanceService, before: Recipe, after: Recipe) -> Recipe:
    """`after`, or SettingsError when a changed context or MoE cache cannot stand."""
    _check_context(service, before, after)
    found = cache_room(service, after) if cache_changed(before, after) else None
    if found is not None and found.refused:
        raise SettingsError(found.refusal(after.name))
    return after


def _check_context(service: InstanceService, before: Recipe, after: Recipe) -> None:
    if not isinstance(after.args, LlamaCppArgs | VllmArgs):
        return
    asked, was = serving_context(after), serving_context(before)
    if asked is None or asked == was or (was is not None and asked < was):
        return  # only a context this save raised is judged
    trained = service.native_context(after)
    if trained is None or asked <= trained:
        return  # not downloaded (the start judges), or within the model's own
    slots = serving_slots(after)
    shared = f" ({slots} slots share it)" if slots > 1 else ""
    if isinstance(after.args, LlamaCppArgs):
        raise SettingsError(
            f"ctx={after.args.ctx} gives each request {asked:,} tokens{shared}, more than "
            f"{after.name}'s model was trained for ({trained:,}, its GGUF's context length): "
            f"llama-server would cap it there and run at {trained:,}, and answers past it "
            f"degrade. `mnm set {after.name} ctx={trained * slots}` is the most it takes. "
            "Nothing saved."
        )
    raise SettingsError(
        f"ctx={asked} is more than {after.name}'s model was trained for ({trained:,}, its "
        f"config.json): vLLM refuses to start with it. `mnm set {after.name} ctx={trained}` "
        "is the most it takes. Nothing saved."
    )


def cache_changed(before: Recipe, after: Recipe) -> bool:
    if not isinstance(after.args, LlamaCppArgs) or not after.args.moe_cache_mib:
        return False
    old = before.args.moe_cache_mib if isinstance(before.args, LlamaCppArgs) else None
    return (old or 0) < after.args.moe_cache_mib


@dataclass(frozen=True)
class CacheRoom:
    cache: int  # the cache asked for, bytes
    need: int  # the instance with it, bytes on the card
    alone: int  # without it
    room: int  # what the card can give the instance
    level: FitLevel
    alone_level: FitLevel

    @property
    def refused(self) -> bool:
        if self.level is FitLevel.NO_FIT:
            return True
        return self.level is FitLevel.TIGHT and self.alone_level is not FitLevel.COMFORTABLE

    @property
    def largest_comfortable_mib(self) -> int:
        """The largest cache (in CACHE_STEP_MIB steps) that leaves the instance comfortable;
        0 when even none does."""
        spare = int((1 - COMFORTABLE_HEADROOM) * self.room) - self.alone
        return max(spare // MIB // CACHE_STEP_MIB * CACHE_STEP_MIB, 0)

    def figures(self, name: str) -> str:
        return (
            f"{name} with a {self.cache // MIB:,} MiB expert cache needs about "
            f"{self.need / GIB:.1f} GiB on the card ({self.alone / GIB:.1f} GiB without it) "
            f"and the card can give it {self.room / GIB:.1f} GiB (free now plus what mnm's "
            "running models hold)"
        )

    def advice(self, name: str) -> str:
        largest = self.largest_comfortable_mib
        if largest == 0:
            return f"No cache leaves {name} comfortable on this card now."
        return f"`mnm set {name} moe-cache={largest}` is the largest that stays comfortable."

    def refusal(self, name: str) -> str:
        verdict = "it does not fit" if self.level is FitLevel.NO_FIT else "it would fit tight"
        return (
            f"{self.figures(name)}: {verdict}. At the card's edge llama.cpp spills into "
            "shared memory and every token slows down (measured with Qwen3.6-35B-A3B on an RTX "
            f"4070 under WSL2: 28.3 tok/s, 5.8 with the card full). {self.advice(name)} Nothing "
            "saved."
        )


def cache_room(service: InstanceService, recipe: Recipe) -> CacheRoom | None:
    """`recipe`'s MoE cache against the card; None with no cache, no card read or a figure
    unknown (nothing refused on a guess)."""
    if not isinstance(recipe.args, LlamaCppArgs) or not recipe.args.moe_cache_mib:
        return None
    room = card_room(service, recipe)
    need = service.vram_estimate(recipe).bytes
    alone = service.vram_estimate(recipe.with_args(moe_cache_mib=None)).bytes
    if room is None or need is None or alone is None:
        return None
    level, alone_level = headroom_level(need, room), headroom_level(alone, room)
    return CacheRoom(recipe.args.moe_cache_mib * MIB, need, alone, room, level, alone_level)


def cache_note(service: InstanceService, recipe: Recipe) -> str | None:
    """The warning a tight cache beside a comfortable model gets once saved."""
    found = cache_room(service, recipe)
    if found is None or found.level is not FitLevel.TIGHT or found.refused:
        return None
    return (
        f"Tight: {found.figures(recipe.name)}. {found.advice(recipe.name)} "
        f"`mnm try {recipe.name} --speed` decides whether it gains."
    )
