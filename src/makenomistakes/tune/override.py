"""The user's setting wins: a save that changes a setting the tune searched (or one that
changes what it measured: the context, the slots, flash attention, the draft) drops the
recipe's `tuned` block, and says so; one that changes the KV cache's precision drops its
`quality` block too (a quant's score is measured with it).
A leaf module: `controls.apply_settings` imports it, which the tuner itself imports."""

from __future__ import annotations

from makenomistakes.recipes.model import Recipe
from makenomistakes.recipes.tuned import TUNED_KEYS

MEASURED_WITH = frozenset({*TUNED_KEYS, "ctx", "parallel", "flash_attn"})
SCORED_WITH = frozenset({"kv_type"})  # What a quality tune's scores were measured with
DROPPED = "The tune of {name} is dropped: your setting wins; `mnm tune {name}` tunes again from it."
QUALITY_DROPPED = (
    "The quality tune of {name} is dropped: its scores were measured with the KV cache it had; "
    "`mnm tune {name} --quality` measures again."
)


def untuned_if_changed(before: Recipe, after: Recipe) -> Recipe:
    """`after` without its tune when the save moved what the tune chose or measured with."""
    after = _unscored_if_changed(before, after)
    if after.tuned is None:
        return after
    moved = any(
        getattr(before.args, name, None) != getattr(after.args, name, None)
        for name in MEASURED_WITH
    )
    if moved or before.draft != after.draft or before.image != after.image:
        return after.model_copy(update={"tuned": None})
    return after


def _unscored_if_changed(before: Recipe, after: Recipe) -> Recipe:
    if after.quality is None:
        return after
    moved = any(
        getattr(before.args, name, None) != getattr(after.args, name, None) for name in SCORED_WITH
    )
    return after.model_copy(update={"quality": None}) if moved else after


def dropped(before: Recipe, after: Recipe) -> str | None:
    """The line a save prints when it dropped the tune or the quality tune's record."""
    said = []
    if before.tuned is not None and after.tuned is None:
        said.append(DROPPED.format(name=after.name))
    if before.quality is not None and after.quality is None:
        said.append(QUALITY_DROPPED.format(name=after.name))
    return " ".join(said) or None
