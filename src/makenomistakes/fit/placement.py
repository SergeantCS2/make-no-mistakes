"""Per-card expert placement: a mixture-of-experts model that does not fit
whole on several cards keeps whole blocks on each card and fills each card's room after them
with routed experts, the rest in RAM; one `-ot` list says where (`card_placement`).

In llama.cpp, tensors follow their block's card unless `-ot` says otherwise, and experts are
not spread across cards
by themselves; `--n-cpu-moe N` beside a `-ts` split takes the first N blocks' experts off the
first card only, so that card is left under-filled while the others hold their blocks'
experts in full (llama.cpp #15263). So:

- the cards are ranked (`fit.cards.ranked`): the main card first (the widest link: prompt
  processing copies the RAM-held experts to it), an x1 riser last;
- every block's dense part (its weights, KV and recurrent state) goes on the cards in runs,
  each card's run in proportion to its usable room (its free VRAM at the fit level, less its
  CUDA context and compute buffer, the main card's copy buffer and the input table, the last
  card's output head); a riser takes as many blocks as it holds and no experts;
- then each card takes its own blocks' experts, first blocks first, whole blocks only;
- then any card with room left takes other blocks' experts (largest room first), only where
  that beats RAM by the speed model: a block's active experts read at the card's bandwidth
  plus a crossing to that card and back (`HOP_SECONDS` each), against the RAM rate
  (`experts=cards` takes them regardless, `experts=ram` keeps every expert in RAM);
- the rest stay in RAM.

What each card holds is a planner estimate: llama.cpp's own `llama-fit-params` cannot split
across devices it does not have, so a reading on the box replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from makenomistakes.card_placement import CardPlace, Placement
from makenomistakes.fit.cards import CardBudget, ranked
from makenomistakes.fit.cpu_rates import EXPERT_TOUCH_SECONDS
from makenomistakes.fit.device_row import HOP_SECONDS, RTX_4070, DeviceRow
from makenomistakes.fit.estimate import (
    COMFORTABLE_HEADROOM,
    DEFAULT_GPU_BANDWIDTH,
    Fit,
    FitLevel,
    MemoryEstimate,
    decode_efficiency,
)
from makenomistakes.gguf.reader import GgufHeader

Mode = Literal["auto", "ram", "cards"]

__all__ = ["HOP_SECONDS", "Mode", "Placed", "decode_seconds", "place_experts", "refill"]


@dataclass(frozen=True)
class _Price:
    """What a token's reads cost on this box: the RAM-held experts' rate and the card's row
    (its efficiency, launch term and crossing)."""

    rate: float
    device: DeviceRow


@dataclass(frozen=True)
class Placed:
    """A placement with its fit and its decode time per token."""

    placement: Placement
    fit: Fit
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return 1.0 / self.seconds


@dataclass(frozen=True)
class _Parts:
    """What goes where, block by block, from the estimate and the header."""

    dense: tuple[float, ...]  # each block's dense weights, KV and recurrent state
    dense_weights: float  # one block's dense weights (what a token reads of it)
    experts: tuple[int, ...]  # each block's routed experts as stored
    shares: tuple[float, ...]  # each block's active share of them
    used: tuple[int, ...]  # each block's used experts a token (each touched in RAM)
    fixed: int  # a card's CUDA context and compute buffer
    head: int  # the output head (last card) and the input table (main card), each
    projector: int
    copy: int  # the main card's buffer for a big batch's copies of RAM-held experts

    @classmethod
    def of(cls, header: GgufHeader, estimate: MemoryEstimate) -> _Parts | None:
        experts = header.expert_layer_bytes()
        blocks = len(experts)
        if not blocks or not any(experts):
            return None
        head = estimate.head_bytes
        weights = max(estimate.weights_bytes - 2 * head - sum(experts), 0) / blocks
        holds_kv = header.kv_layers()
        kv_blocks = sum(holds_kv) or blocks
        recurrent = header.recurrent_layers or (False,) * blocks
        state_blocks = sum(recurrent) or blocks
        dense = tuple(
            weights
            + (estimate.kv_bytes / kv_blocks if holds_kv[b] or not any(holds_kv) else 0)
            + (estimate.state_bytes / state_blocks if recurrent[b] else 0)
            for b in range(blocks)
        )
        return cls(
            dense=dense,
            dense_weights=weights,
            experts=experts,
            shares=header.expert_shares(),
            used=header.experts_used(),
            fixed=estimate.overhead_bytes + estimate.compute_bytes,
            head=head,
            projector=estimate.projector_bytes,
            copy=max(experts) // 3,  # one of gate, up and down (`ffn_*_exps`)
        )


def place_experts(  # noqa: PLR0913 - the card's row by keyword
    header: GgufHeader,
    estimate: MemoryEstimate,
    cards: tuple[CardBudget, ...],
    ram_free_bytes: int,
    expert_rate: float,
    mode: Mode = "auto",
    *,
    ordered: bool = False,
    device: DeviceRow = RTX_4070,
) -> Placed | None:
    """The placement at the best fit level: every card comfortable (a quarter of its free
    VRAM to spare), else every card tight; None when the cards cannot hold every block's
    dense part, nothing would stay in RAM (it fits whole: no placement needed), or RAM
    cannot hold what stays there. `estimate` is the one-card estimate (`per_card` not
    applied: each card's context and buffer are priced here). `ordered`: the cards are in
    the order to use already (a saved placement's), else they are ranked."""
    parts = _Parts.of(header, estimate)
    if parts is None or len(cards) < 2:
        return None
    use = list(cards) if ordered else ranked(list(cards))
    for fraction in (1 - COMFORTABLE_HEADROOM, 1.0):
        placed = _attempt(header, parts, use, fraction, mode, _Price(expert_rate, device))
        if placed is not None and placed.placement.ram_bytes <= ram_free_bytes:
            return placed
    return None


def refill(
    header: GgufHeader,
    estimate: MemoryEstimate,
    placement: Placement,
    expert_rate: float,
    mode: Mode = "auto",
    counts: tuple[int, ...] | None = None,
    *,
    device: DeviceRow = RTX_4070,
) -> Placed:
    """A saved placement placed again (`mnm set experts=` / `card-experts=`, the tuner): by
    `mode` against each card's free VRAM when it was planned, or with `counts` blocks'
    experts on each card (own blocks first, then the lowest others) on the same runs of
    blocks, whether or not the estimate says they fit (Fits now says so; a start that runs
    out of memory is the reading). ValueError, worded, when it cannot be."""
    parts = _Parts.of(header, estimate)
    if parts is None:
        raise ValueError("its header names no routed experts to place")
    cards = [
        CardBudget(
            index=c.index,
            name="",
            free_bytes=c.free,
            total_bytes=c.free,
            bandwidth=c.bandwidth or DEFAULT_GPU_BANDWIDTH,
            lanes=c.lanes,
            stable_id=c.stable_id,  # Placed again, on the same cards
        )
        for c in placement.cards
    ]
    if counts is None:
        placed = place_experts(
            header, estimate, tuple(cards), 2**62, expert_rate, mode, ordered=True, device=device
        )
        if placed is None:
            raise ValueError(
                "its blocks do not fit these cards with any expert in RAM: `mnm replan` "
                "plans it for the cards as they are now"
            )
        return placed
    if len(counts) != len(cards):
        raise ValueError(
            f"card-experts takes one number per card ({len(cards)}), in the order "
            f"`mnm show` lists them ({', '.join(f'GPU{c.index}' for c in cards)})"
        )
    runs = [range(c.first, c.first + c.count) for c in placement.cards]
    with_experts = [b for b, size in enumerate(parts.experts) if size]
    if sum(counts) > len(with_experts):
        raise ValueError(
            f"the model has {len(with_experts)} blocks with experts, not {sum(counts)}"
        )
    held: list[list[int]] = [[] for _ in cards]
    taken: set[int] = set()
    for k, card in enumerate(cards):
        if counts[k] and card.riser:
            raise ValueError(f"GPU{card.index} is on an x1 riser: layers only, no experts")
        own = [b for b in runs[k] if parts.experts[b]]
        for b in [*own, *(b for b in with_experts if b not in runs[k])]:
            if len(held[k]) >= counts[k]:
                break
            if b not in taken:
                held[k].append(b)
                taken.add(b)
    price = _Price(expert_rate, device)
    return _build(header, parts, cards, runs, held, _extras(parts, len(cards)), price)


def _extras(parts: _Parts, count: int) -> list[int]:
    """Beyond each card's context and buffer: the main card's copy buffer, projector and the
    input table (it rides along, as the one-card estimate counts it), the last card's head."""
    extras = [0] * count
    extras[0] += parts.head + parts.projector + parts.copy
    extras[-1] += parts.head
    return extras


def _attempt(
    header: GgufHeader,
    parts: _Parts,
    cards: list[CardBudget],
    fraction: float,
    mode: Mode,
    price: _Price,
) -> Placed | None:
    count = len(cards)
    extras = _extras(parts, count)
    usable = [int(c.free_bytes * fraction) - parts.fixed - extras[k] for k, c in enumerate(cards)]
    if min(usable) < 0:
        return None
    runs = _runs(parts.dense, usable, [c.riser for c in cards])
    if runs is None:
        return None
    left = [usable[k] - sum(parts.dense[b] for b in runs[k]) for k in range(count)]
    held: list[list[int]] = [[] for _ in cards]
    if mode != "ram":
        for k, card in enumerate(cards):  # own blocks' experts, first blocks first
            for b in runs[k]:
                if not card.riser and 0 < parts.experts[b] <= left[k]:
                    held[k].append(b)
                    left[k] -= parts.experts[b]
        _hops(parts, cards, held, left, mode, price, decode_efficiency(header, price.device))
    placed = _build(header, parts, cards, runs, held, extras, price)
    if not placed.placement.in_ram:
        return None  # every expert on the cards: the whole fit's case, not a placement
    return placed


def _build(
    header: GgufHeader,
    parts: _Parts,
    cards: list[CardBudget],
    runs: list[range],
    held: list[list[int]],
    extras: list[int],
    price: _Price,
) -> Placed:
    """The placement, its fit and its decode time from each card's run and experts."""
    placed = {b for blocks in held for b in blocks}
    in_ram = tuple(b for b, size in enumerate(parts.experts) if size and b not in placed)
    places = tuple(
        CardPlace(
            index=card.index,
            stable_id=card.stable_id,
            first=runs[k].start,
            count=len(runs[k]),
            experts=tuple(sorted(held[k])),
            bytes=int(
                parts.fixed
                + extras[k]
                + sum(parts.dense[b] for b in runs[k])
                + sum(parts.experts[b] for b in held[k])
            ),
            free=card.free_bytes,
            bandwidth=card.bandwidth,
            lanes=card.lanes,
        )
        for k, card in enumerate(cards)
    )
    ram = sum(parts.experts[b] for b in in_ram)
    placement = Placement(cards=places, in_ram=in_ram, ram_bytes=ram, copy_bytes=parts.copy)
    on_cards = sum(p.bytes for p in places)
    fit = Fit(
        level=FitLevel.OFFLOAD,
        gpu_layers=len(parts.experts),
        headroom_bytes=sum(c.free_bytes for c in cards) - (on_cards + ram),
        gpu_bytes=on_cards,
        room_bytes=sum(c.free_bytes for c in cards),
        cpu_moe_layers=max(len(in_ram), 1),  # every block on the cards (`ngl_for`)
    )
    return Placed(placement, fit, decode_seconds(header, parts, placement, cards, price))


def _runs(dense: tuple[float, ...], usable: list[int], risers: list[bool]) -> list[range] | None:
    """Each card's run of blocks in rank order: a riser as many as it holds (leaving one for
    each other card), the others in proportion to their usable room (the main card first on
    a tie), then moved one block at a time across a boundary while a card overflows."""
    blocks, count = len(dense), len(usable)
    if blocks < count:
        return None
    average = sum(dense) / blocks or 1.0
    counts = [0] * count
    others = [k for k in range(count) if not risers[k]] or list(range(count))
    for k in range(count):
        if risers[k] and k not in others:
            counts[k] = max(min(int(usable[k] // average), blocks - len(others)), 0)
    rest = blocks - sum(counts)
    room = sum(usable[k] for k in others) or 1
    shares = {k: rest * usable[k] / room for k in others}
    for k in others:
        counts[k] = int(shares[k])
    for k in sorted(others, key=lambda k: (-(shares[k] - int(shares[k])), k))[
        : rest - sum(counts[k] for k in others)
    ]:
        counts[k] += 1
    for _ in range(blocks * count):
        runs = _as_runs(counts)
        over = [k for k in range(count) if sum(dense[b] for b in runs[k]) > usable[k]]
        if not over:
            return runs
        k = over[0]
        target = next(
            (
                j
                for j in (k + 1, k - 1)
                if 0 <= j < count and _spare(dense, usable, runs, j) >= average
            ),
            None,
        )
        if target is None or counts[k] == 0:
            return None
        counts[k] -= 1
        counts[target] += 1
    return None


def _as_runs(counts: list[int]) -> list[range]:
    runs, start = [], 0
    for n in counts:
        runs.append(range(start, start + n))
        start += n
    return runs


def _spare(dense: tuple[float, ...], usable: list[int], runs: list[range], k: int) -> float:
    return usable[k] - sum(dense[b] for b in runs[k])


def _hops(
    parts: _Parts,
    cards: list[CardBudget],
    held: list[list[int]],
    left: list[float],
    mode: Mode,
    price: _Price,
    efficiency: float,
) -> None:
    """Room left on a card takes other blocks' experts, the largest room first, the lowest
    block first, where reading them there and crossing twice beats reading them from RAM."""
    placed = {b for blocks in held for b in blocks}
    for b, size in enumerate(parts.experts):
        if not size or b in placed:
            continue
        active = size * parts.shares[b]
        options = sorted(
            (k for k in range(len(cards)) if not cards[k].riser and size <= left[k]),
            key=lambda k: (-left[k], k),
        )
        for k in options:
            card_time = active / (cards[k].bandwidth * efficiency) + 2 * price.device.hop_seconds
            ram_time = active / price.rate + parts.used[b] * EXPERT_TOUCH_SECONDS
            if mode == "cards" or card_time < ram_time:
                held[k].append(b)
                left[k] -= size
                break


def decode_seconds(
    header: GgufHeader,
    parts: _Parts,
    placement: Placement,
    cards: list[CardBudget],
    price: _Price,
) -> float:
    """A token's seconds: each card reads its blocks' dense weights (the last also the output
    head) and the active experts it holds at its own bandwidth, one card after another; the
    RAM-held active experts come at the expert rate; a launch cost per block; two crossings
    for each block whose experts sit on another card."""
    bandwidth = {c.index: c.bandwidth for c in cards}
    efficiency = decode_efficiency(header, price.device)
    seconds = 0.0
    for place in placement.cards:
        read = place.count * parts.dense_weights + sum(
            parts.experts[b] * parts.shares[b] for b in place.experts
        )
        if place is placement.cards[-1]:
            read += parts.head
        seconds += read / (bandwidth[place.index] * efficiency)
    ram = sum(parts.experts[b] * parts.shares[b] for b in placement.in_ram)
    touched = sum(parts.used[b] for b in placement.in_ram)
    blocks = len(parts.experts)
    return (
        seconds
        + ram / price.rate
        + touched * EXPERT_TOUCH_SECONDS
        + blocks * price.device.launch_seconds
        + 2 * placement.hops * price.device.hop_seconds
    )
