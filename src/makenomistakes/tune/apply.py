"""A tune's result on the recipe, and the words that read it back.

The chosen settings go through `Recipe.with_args` (validated as any setting is) and the draft
through `draft.with_draft`; the plan's VRAM estimate is kept (Fits now reads the files and
the settings anyway, and the trials' readings are keyed by the new settings as usual). The
recipe re-read under the store's lock must still be the one the tune started from: an
user's `mnm set` meanwhile wins and the result is not saved."""

from __future__ import annotations

from makenomistakes.backend_setting import BackendSettingError, with_backend
from makenomistakes.controls import SettingsError
from makenomistakes.draft import DraftError, with_draft
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.instance_settings import numa_facts, placed_again
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine import planned_here, stamped
from makenomistakes.measurements import settings_key
from makenomistakes.numa_setting import NumaSettingError, with_numa
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.tuned import (
    BACKEND_FIELDS,
    NUMA_FIELDS,
    PLACEMENT_FIELDS,
    TUNED_KEYS,
    Tuned,
    TuneKey,
)


class TuneChangedError(InstanceError):
    """The recipe changed while the tune ran: nothing is saved."""


def with_tune(service: InstanceService, recipe: Recipe, tuned: Tuned) -> Recipe:
    """`recipe` with the tune's settings and its block, the plan's estimate kept."""
    special = PLACEMENT_FIELDS | BACKEND_FIELDS | NUMA_FIELDS
    plain = {f: v for f, v in tuned.changes.items() if f not in special}
    changed = recipe
    numa = tuned.changes.get("numa")
    if isinstance(numa, str):  # First, so a threads count the tune chose wins
        try:
            changed = with_numa(changed, numa, lambda now: numa_facts(service, now))
        except NumaSettingError as exc:
            raise InstanceError(f"the tune's numa could not be set: {exc}") from exc
    changed = changed.with_args(**plain) if plain else changed
    chosen = tuned.changes.get("backend")
    if isinstance(chosen, str):  # The build the tune measured faster here
        try:
            changed = with_backend(changed, chosen)
        except BackendSettingError as exc:
            raise InstanceError(f"the tune's backend could not be set: {exc}") from exc
    placing = {TUNED_KEYS[f]: str(v) for f, v in tuned.changes.items() if f in PLACEMENT_FIELDS}
    if placing:  # Expert blocks per card, placed again from the files
        try:
            changed = placed_again(service, changed, placing)
        except SettingsError as exc:
            raise InstanceError(f"the tune's experts per card could not be set: {exc}") from exc
    if tuned.draft is not None:
        raw = "" if tuned.draft == "off" else tuned.draft
        try:
            changed = with_draft(changed, raw, service.find, service.models_dir)
        except DraftError as exc:
            raise InstanceError(f"the tune's draft could not be set: {exc}") from exc
    return changed.model_copy(
        update={"tuned": tuned, "vram_estimate_bytes": recipe.vram_estimate_bytes}
    )


def save_tune(service: InstanceService, name: str, started_from: str, tuned: Tuned) -> Recipe:
    """Save it unless the recipe moved since the tune started (`started_from`: its
    `settings_key` then)."""

    here = planned_here(service.settings, service.memory())  # The box it measured on

    def change(current: Recipe) -> Recipe:
        if settings_key(current) != started_from:
            raise TuneChangedError(
                f"{name}'s settings changed while it was tuned: nothing saved; "
                f"`mnm tune {name}` tunes again from them."
            )
        tuned_now = with_tune(service, current, tuned)
        return stamped(tuned_now, here)

    return service.change(name, change)


def stale_word(recipe: Recipe, now: TuneKey) -> str | None:
    """`another build` when the tune no longer holds here; None while it does."""
    return recipe.tuned.key.differs(now) if recipe.tuned is not None else None


def tuned_word(recipe: Recipe, now: TuneKey | None) -> str | None:
    """The Settings cell's word: `tuned +18%`, `tuned for another build`; None untuned."""
    tuned = recipe.tuned
    if tuned is None:
        return None
    if now is not None and (why := tuned.key.differs(now)) is not None:
        return f"tuned for {why}"
    return f"tuned {tuned.gain:+.0%}"


def key_or_none(
    service: InstanceService, recipe: Recipe, snapshot: MemorySnapshot | None
) -> TuneKey | None:
    """The key now, or None when the box cannot say (a word without the stale check)."""
    from makenomistakes.tune.key import key_now  # noqa: PLC0415 - nvidia-smi only when asked

    try:
        return key_now(service, recipe, snapshot)
    except Exception:  # Docker or nvidia-smi away: the word stays the saved one
        return None


def shown(tuned: Tuned) -> list[tuple[str, str]]:
    """`mnm show`'s lines for the block."""
    score = tuned.score
    parts = [f"decode {score.decode:.1f} tok/s"]
    if score.prompt_2k:
        parts.append(f"prompt {score.prompt_2k:.0f} tok/s at 2k")
    if score.prompt_16k:
        parts.append(f"{score.prompt_16k:.0f} at 16k")
    planned = f"; mnm's plan measured {tuned.seed.decode:.1f} tok/s, score {tuned.seed.value:.1f}"
    return [
        ("Tuned", f"{tuned.summary} ({tuned.gain:+.0%} over mnm's plan, {tuned.trials} trials)"),
        ("Tuned score", f"{score.value:.1f} ({score.mode}): {', '.join(parts)}{planned}"),
        ("Tuned on", f"build {tuned.key.build or '?'}, {tuned.at:%Y-%m-%d %H:%M} UTC"),
    ]
