"""What a settings save says after it: the settings that changed and when each applies, the
MoE cache's note, the tune the save dropped, a new draft's cost and its tight-draft warning.
`mnm set` and the web's settings form say the same lines from here."""

from __future__ import annotations

from dataclasses import dataclass, field

from makenomistakes.controls import Control, controls_for, current_value
from makenomistakes.draft import EXPERIMENT
from makenomistakes.draft_room import tight_note
from makenomistakes.instances import InstanceService
from makenomistakes.measurements import settings_key
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.llamacpp_cache import keeps_slots
from makenomistakes.setting_room import cache_changed, cache_note
from makenomistakes.tune.override import dropped
from makenomistakes.tune.store import QualityQueued, TuneStore, TuneStoreError

ROUTER_KEYS = frozenset({"idle", "keep"})  # the router's own: it is told of a change at once


@dataclass(frozen=True)
class Line:
    text: str
    warn: bool = False


@dataclass(frozen=True)
class Saved:
    changed: list[Control] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)  # after the per-setting "a → b" lines

    @property
    def router_moved(self) -> bool:
        """`idle` or `keep` changed: a client's `keep_alive` then goes (the form
        posts every field, so a save of only `kv` must not forget it)."""
        return any(c.key in ROUTER_KEYS for c in self.changed)


def slots_line(before: Recipe, after: Recipe) -> str | None:
    """`cache=` turned on or off: what happens to the saved prompts."""
    was, now = keeps_slots(before), keeps_slots(after)
    if was and not now:
        return "Its saved prompts were deleted; the engine keeps none on disk from now on."
    if now and not was:
        return (
            "From its next start on, the slot it used last is saved to disk when it unloads "
            "and put back when it starts again."
        )
    return None


def saved(service: InstanceService, before: Recipe, after: Recipe) -> Saved:
    changed = [  # every key: a hidden one may have been set
        c
        for c in controls_for(before.modality, before.engine)
        if current_value(before, c) != current_value(after, c)
    ]
    if not changed:
        return Saved(changed, [Line("No change: those are the current settings.")])
    now = [c.key for c in changed if c.live]  # the router's: it reads them per request
    later = [c.key for c in changed if not c.live]
    if later:
        said = f"Saved. {', '.join(later)} takes effect on the next `mnm start {after.name}`" + (
            f"; {', '.join(now)} applies now." if now else "."
        )
    else:
        said = f"Saved. {', '.join(now)} applies now (the router reads it as it goes)."
    lines = [Line(said)]
    if (cache := cache_note(service, after)) is not None and cache_changed(before, after):
        lines.append(Line(cache, warn=True))
    if (gone := dropped(before, after)) is not None:  # the user's setting wins
        lines.append(Line(gone))
    if (kept := slots_line(before, after)) is not None:
        lines.append(Line(kept))
    if (doomed := doomed_plan(service, after)) is not None:
        lines.append(Line(doomed, warn=True))
    if after.draft is not None and after.draft != before.draft:
        lines.append(Line(draft_note(after)))
        if (tight := tight_note(service, after)) is not None:
            lines.append(Line(tight, warn=True))
    return Saved(changed, lines)


def doomed_plan(service: InstanceService, recipe: Recipe) -> str | None:
    """A quality tune queued on the settings before this change is dropped when it runs, its
    downloads with it: said where the change is made, not only in the journal later."""
    try:
        queued = TuneStore(service.settings.expanded_state_dir()).queued(recipe.name)
    except TuneStoreError:
        return None
    if not isinstance(queued, QualityQueued) or settings_key(recipe) == queued.quality.settings:
        return None
    return (
        f"{recipe.name}'s queued quality tune was planned on its settings before this change: "
        f"it is dropped when it runs, its downloads with it (`mnm tune {recipe.name} --quality` "
        f"plans it again; `mnm tune {recipe.name} --cancel` drops it now)."
    )


def draft_note(recipe: Recipe) -> str:
    """What a new draft costs and what is not known yet: its size beside the model's,
    and that its speed is the box's reading, not the planner's."""
    draft = recipe.draft
    if draft is None:
        return ""
    size = f"{draft.size_bytes / 2**30:.2f} GiB, every layer on the GPU"
    if draft.is_mtp:
        what = (
            f"The MTP head ({size}, the KV of its own MTP blocks at ctx "
            f"{recipe.llama_args.ctx:,}, and on a hybrid model a recurrent-state row per "
            "drafted token)"
        )
    else:
        what = f"The draft ({size}, its KV cache at ctx {recipe.llama_args.ctx:,})"
    return f"{what} is priced in Fits now. " + EXPERIMENT.format(name=recipe.name)
