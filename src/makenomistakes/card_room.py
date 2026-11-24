"""Room per card: where an instance's VRAM lands when `MNM_GPU` names
several cards, and the fit-now verdicts and room-making over those cards.

The rule is the simplest that is right per card: a one-card instance's estimate lands on
its card, a llama.cpp instance across cards is split by its `-ts` (evenly without one),
and a vLLM instance with tensor parallel holds its per-card estimate on each card. A model
held on card 1 never blocks a start on card 0. Room-making for an instance on several cards
judges each of its cards with what each holder holds on that card, and a set's
fit-together check judges each chosen card with the members' shares of it; a placement's
shares are its per-card estimate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from makenomistakes.capacity import FitNow, Verdict, VramEstimate, fit_now
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.fit.cards import shares
from makenomistakes.fit.estimate import GIB
from makenomistakes.hardware.gpu_select import CardChoice, chosen
from makenomistakes.hardware.probe import RAM_INDEX, Gpu, MemorySnapshot
from makenomistakes.ram_room import runs_on_processor
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.cards import placed_here, recipe_cards

_WORST = (FitNow.UNKNOWN, FitNow.NO_FIT, FitNow.NEAR_LIMIT, FitNow.FITS)


def placed(
    recipe: Recipe, estimate: VramEstimate, choice: CardChoice | None = None
) -> VramEstimate:
    """The estimate with its bytes per card (`VramEstimate.on`); unchanged on a one-card
    box. An instance on the processor's image sits in the RAM (`ram_room`), whatever cards
    the box has."""
    if estimate.bytes is not None and runs_on_processor(recipe):
        return estimate.model_copy(update={"on": ((RAM_INDEX, estimate.bytes),)})
    try:
        choice = choice or chosen()
    except ValueError:  # MNM_GPU unreadable: the probe says so, the estimate stays whole
        return estimate
    if estimate.bytes is None or (not choice.several and not recipe.gpus):
        return estimate
    cards = recipe_cards(recipe, choice)  # None: `all`, unresolved: the first card's stands
    if cards is None or (cards == (choice.first,) and not choice.several):
        return estimate
    ordered = tuple(sorted(cards))
    if isinstance(recipe.args, VllmArgs) and recipe.args.tensor_parallel and len(cards) > 1:
        on = tuple((c, estimate.bytes) for c in ordered)  # the estimate is each worker's
        total = estimate.bytes * len(cards)
        return estimate.model_copy(update={"on": on, "bytes": total})
    args = recipe.args
    split = args.tensor_split if isinstance(args, LlamaCppArgs) else None
    placement = placed_here(recipe, ordered) if isinstance(args, LlamaCppArgs) else None
    if placement is not None:  # where its cards are now
        split = tuple(max(p.bytes, 1) for p in sorted(placement.cards, key=lambda p: p.index))
    parts = shares(estimate.bytes, split, len(ordered))
    return estimate.model_copy(update={"on": tuple(zip(ordered, parts, strict=True))})


def _rank(level: FitNow) -> int:
    return _WORST.index(level) if level in _WORST else 0


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def fit_on_cards(estimate: VramEstimate, snapshot: MemorySnapshot) -> Verdict:
    """Each card's share judged on that card alone (`fit_now`'s own rules); the verdict is
    the worst card's, named."""
    found: list[tuple[Gpu, int, Verdict]] = []
    for index, size in estimate.on:
        card = snapshot.card(index)
        if card is None and index == RAM_INDEX:
            return Verdict(level=FitNow.UNKNOWN, note="Runs in RAM, which cannot be read now.")
        if card is None:
            return Verdict(
                level=FitNow.UNKNOWN,
                note=f"Runs on card {index}, which nvidia-smi does not list: check MNM_GPU "
                "and `mnm set <name> gpus=`.",
            )
        alone = estimate.model_copy(update={"bytes": size, "on": ()})
        found.append((card, size, fit_now(alone, view(snapshot, card), running=False)))
    worst = min(found, key=lambda f: _WORST.index(f[2].level))
    card, _size, verdict = worst
    if card.in_ram and len(found) == 1:  # the RAM's own words say where
        return verdict
    if len(found) == 1 or verdict.level is not FitNow.FITS:
        return verdict.model_copy(update={"note": f"{_on(card).capitalize()}: {verdict.note}"})
    each = ", ".join(f"{_gib(size)} {_on(c)}" for c, size, _ in found)
    free = ", ".join(f"{_gib(c.vram_free_bytes)} {_on(c)}" for c, _, _ in found)
    return Verdict(level=FitNow.FITS, note=f"Needs about {each}; free right now: {free}.")


def _on(card: Gpu) -> str:
    return "in RAM" if card.in_ram else f"on card {card.index}"


def view(snapshot: MemorySnapshot, card: Gpu) -> MemorySnapshot:
    """The snapshot as if `card` were the only one."""
    return snapshot.model_copy(update={"gpus": [card], "chosen": ()})


def _cards_of(estimate: VramEstimate, snapshot: MemorySnapshot) -> dict[int, int]:
    first = snapshot.gpu
    if estimate.on:
        return dict(estimate.on)
    return {first.index: estimate.bytes or 0} if first is not None else {}


class RoomView(NamedTuple):
    """`capacity.with_room`'s inputs: the estimate, the snapshot, each holder's bytes on the
    instance's cards, and (several cards) each holder's bytes per card."""

    estimate: VramEstimate
    snapshot: MemorySnapshot
    held: dict[str, int]
    held_on: dict[str, dict[int, int]] | None = None


def room_view(
    estimate: VramEstimate, snapshot: MemorySnapshot, held: dict[str, VramEstimate]
) -> RoomView:
    """`capacity.with_room`'s inputs over the cards `estimate` runs on. With nothing placed
    per card it is today's: the snapshot and every holder's bytes. Placed per card:
    the instance's share on each of its cards judged on that card, each holder's bytes per
    card (`held_on`), so stopping a holder frees only the cards it is on."""
    if not estimate.on and not any(e.on for e in held.values()):
        return RoomView(estimate, snapshot, {n: e.bytes or 0 for n, e in held.items()})
    mine = _cards_of(estimate, snapshot)
    if not any(snapshot.card(i) is not None for i in mine):
        return RoomView(estimate, snapshot, {})
    placed = estimate if estimate.on else estimate.model_copy(update={"on": tuple(mine.items())})
    per_card = {name: _cards_of(e, snapshot) for name, e in held.items()}
    on_mine = {name: sum(b for i, b in on.items() if i in mine) for name, on in per_card.items()}
    return RoomView(placed, snapshot, on_mine, per_card)


def short_cards(estimate: VramEstimate, snapshot: MemorySnapshot) -> set[int]:
    """The cards on which the instance's share does not fit comfortably now."""
    found = set()
    for index, size in estimate.on:
        card = snapshot.card(index)
        alone = estimate.model_copy(update={"bytes": size, "on": ()})
        if (
            card is not None
            and fit_now(alone, view(snapshot, card), False).level is not FitNow.FITS
        ):
            found.add(index)
    return found


def lighter_on(snapshot: MemorySnapshot, freed: list[dict[int, int]]) -> MemorySnapshot:
    """The snapshot with each card's use (and the RAM's) less what `freed` gives back
    on it."""
    back: dict[int, int] = {}
    for each in freed:
        for index, size in each.items():
            back[index] = back.get(index, 0) + size

    def less(g: Gpu) -> Gpu:
        return g.model_copy(
            update={"vram_used_bytes": max(g.vram_used_bytes - back.get(g.index, 0), 0)}
        )

    ram = less(snapshot.ram) if snapshot.ram is not None else None
    if not isinstance(snapshot.gpus, list):
        return snapshot.model_copy(update={"ram": ram})
    return snapshot.model_copy(update={"gpus": [less(g) for g in snapshot.gpus], "ram": ram})


Waiting = list[tuple[str, VramEstimate]]


def together_on_cards(
    waiting: Waiting, snapshot: MemorySnapshot, judge: Callable[[Waiting, MemorySnapshot], Verdict]
) -> Verdict | None:
    """A set's waiting members judged card by card: on each chosen card, each member's
    share of it as an estimate of its own (`judge`: `capacity`'s rule for one card); the worst
    card's verdict, named. None on one card, or with nothing placed per card (the pooled
    check stands)."""
    cards = list(snapshot.cards)
    on_ram = any(i == RAM_INDEX for _, e in waiting for i, _ in e.on)
    if on_ram and snapshot.ram is not None:  # The RAM beside the cards
        cards.append(snapshot.ram)
    if on_ram and snapshot.ram is None:
        return Verdict(level=FitNow.UNKNOWN, note="Runs in RAM, which cannot be read now.")
    if not any(e.on for _, e in waiting) or (len(cards) < 2 and not on_ram):
        return None
    found: list[tuple[Gpu, Verdict]] = []
    for card in cards:
        here = [
            (name, estimate.model_copy(update={"bytes": size, "on": ()}))
            for name, estimate in waiting
            if (size := _cards_of(estimate, snapshot).get(card.index, 0))
        ]
        if here:
            found.append((card, judge(here, view(snapshot, card))))
    if not found:
        return None
    card, verdict = min(
        found, key=lambda f: _WORST.index(f[1].level) if f[1].level in _WORST else 0
    )
    if card.in_ram and len(found) == 1:
        return verdict
    return verdict.model_copy(update={"note": f"{_on(card).capitalize()}: {verdict.note}"})


def pooled(snapshot: MemorySnapshot) -> MemorySnapshot:
    """Several chosen cards as one (a set's fit-together check); one card as it is."""
    cards = snapshot.cards
    if len(cards) < 2:
        return snapshot
    together = cards[0].model_copy(
        update={
            "vram_total_bytes": sum(c.vram_total_bytes for c in cards),
            "vram_used_bytes": sum(c.vram_used_bytes for c in cards),
        }
    )
    return view(snapshot, together)


def instance_cards(recipe: Recipe, snapshot: MemorySnapshot) -> list[Gpu]:
    """The cards the instance runs on as nvidia-smi reads them now, in nvidia-smi's order
    (`start_fixes` reads them all); the first chosen card when they cannot be told."""
    try:
        cards = recipe_cards(recipe, listed=[g.index for g in snapshot.cards])
    except ValueError:  # MNM_GPU unreadable: the probe says so
        cards = None
    found = [g for i in sorted(cards or ()) if (g := snapshot.card(i)) is not None]
    return found or ([snapshot.gpu] if snapshot.gpu is not None else [])
