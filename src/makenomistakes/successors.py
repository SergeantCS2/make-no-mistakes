"""Which newer model a saved instance has: the family table first (`families.py`),
then the Hub's own word where a card carries it: `new_version` on the instance's repo, the
table on its card's `base_model` (a fine-tune or another publisher's quant of a listed model),
`new_version` on that base. A "deprecated" or "superseded" tag is said, never followed: the
Hub names no successor with it. Nothing is guessed from names alone.

`find` takes the Hub as a lookup (`HubClient.model_info`, None when it cannot say), so the
nightly asks each repo once a night (`cached`) and any lookup can stand in."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from makenomistakes.families import READ, Offer, Placed, place, turbo_quant
from makenomistakes.hub.civitai import is_civitai
from makenomistakes.hub.client import HubError, HubModel
from makenomistakes.hub.elsewhere import elsewhere
from makenomistakes.hub.facts import Engine
from makenomistakes.recipes import Recipe

Lookup = Callable[[str], HubModel | None]
TABLE = f"the family table (read {READ.isoformat()})"
_MARKS = ("deprecated", "superseded")


class Successor(BaseModel):
    repo: str  # what `mnm upgrade` plans
    label: str  # "Qwen3.5-9B", or the repo for the Hub's own word
    quant: str | None = None  # a size label inside the same repo (whisper.cpp)
    offer: Offer = Offer.NEWER
    via: str  # where it was read
    note: str = ""

    @property
    def target(self) -> str:
        """The repo, and the size inside it where that is the change: `ggerganov/whisper.cpp
        (large-v3-turbo)`."""
        return f"{self.repo} ({self.quant})" if self.quant else self.repo


class NoSuccessor(BaseModel):
    words: str  # why not, in whole sentences


def cached(lookup: Lookup) -> Lookup:
    """One Hub question per repo for the lookup's life (the nightly's night)."""
    seen: dict[str, HubModel | None] = {}

    def ask(repo: str) -> HubModel | None:
        if repo not in seen:
            seen[repo] = lookup(repo)
        return seen[repo]

    return ask


def hub_lookup(model_info: Callable[[str], HubModel]) -> Lookup:
    """`HubClient.model_info`, None when the Hub cannot say (not found, private, offline)."""

    def ask(repo: str) -> HubModel | None:
        try:
            return model_info(repo)
        except (HubError, OSError):
            return None

    return ask


def find(recipe: Recipe, lookup: Lookup | None) -> Successor | NoSuccessor:
    """The newest member after the instance's; `lookup` None: the table alone (no Hub)."""
    vllm = recipe.engine is Engine.VLLM
    quant = recipe.quant if recipe.engine is Engine.WHISPERCPP else None
    placed = place(recipe.repo_id, quant)
    found = _from_table(placed, vllm, recipe.quant)
    if found is not None:
        return found
    table_words = _newest_words(recipe, placed)
    if lookup is None or is_civitai(recipe.repo_id) or elsewhere(recipe.repo_id) is not None:
        return NoSuccessor(words=table_words)
    return _from_hub(recipe, lookup, vllm, table_words)


def _from_table(placed: list[Placed], vllm: bool, old_quant: str) -> Successor | None:
    """A newer member first; a faster one (whisper's turbo) only when no newer one is known."""
    for offer in (Offer.NEWER, Offer.FASTER):
        for spot in placed:
            newest = spot.newest
            if spot.family.offer is not offer or newest is None:
                continue
            repo = newest.repo_for(vllm)
            if repo is None:
                continue
            family = spot.family
            return Successor(
                repo=repo,
                label=newest.label,
                quant=turbo_quant(old_quant, newest.quant) if newest.quant else None,
                offer=offer,
                via=f"{TABLE}: {family.name}, {family.maker}",
                note=family.note,
            )
    return None


def _newest_words(recipe: Recipe, placed: list[Placed]) -> str:
    if not placed:
        return (
            f"No newer family member known for {recipe.name} ({recipe.repo_id}): it is not in "
            f"{TABLE}."
        )
    spot = placed[0]
    note = f" {spot.family.note}" if spot.family.note else ""
    if spot.newest is not None:  # a newer member, but none in this engine's format
        return (
            f"{recipe.name} ({spot.member.label}) has a newer family member "
            f"({spot.newest.label}), but the table lists no {recipe.engine.value} build of "
            f"it.{note}"
        )
    return (
        f"No newer family member known for {recipe.name}: {spot.member.label} is the newest "
        f"of its family in {TABLE}.{note}"
    )


def _from_hub(
    recipe: Recipe, lookup: Lookup, vllm: bool, table_words: str
) -> Successor | NoSuccessor:
    model = lookup(recipe.repo_id)
    if model is None:
        return NoSuccessor(words=f"{table_words} The Hub could not be asked about it.")
    if (newer := model.card_data.new_version) is not None and newer != recipe.repo_id:
        return Successor(
            repo=newer,
            label=newer,
            via=f"the Hub: {recipe.repo_id}'s card names {newer} as its newer version",
        )
    base = model.card_data.base_model
    if base is not None and base != recipe.repo_id:
        found = _from_table(place(base), vllm, recipe.quant)
        if found is not None:
            via = f"{found.via}, through the card's base model {base}"
            return found.model_copy(update={"via": via})
        upstream = lookup(base)
        if upstream is not None and (newer := upstream.card_data.new_version) is not None:
            return Successor(
                repo=newer,
                label=newer,
                via=f"the Hub: {recipe.repo_id}'s base model {base} names {newer} as its "
                "newer version",
            )
    marked = [t for t in model.tags if any(word in t.lower() for word in _MARKS)]
    if marked:
        return NoSuccessor(
            words=f"{table_words} The Hub marks {recipe.repo_id} {marked[0]}, and its card "
            "names no newer version."
        )
    return NoSuccessor(words=f"{table_words} Its Hub card names no newer version either.")
