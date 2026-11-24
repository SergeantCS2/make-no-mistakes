"""A budget across several cards: what each card has free, how llama.cpp's
layer split shares a plan among them, and what the split costs.

llama.cpp's layer split (`-sm layer`, `docs/multi-gpu.md`, `src/llama-model.cpp`, read
2026-10-22) gives each card a run of whole layers, in device order, by `-ts`'s proportions
of the `-ngl` window. The planner sizes the plan against the cards' summed free VRAM and
splits it in proportion to each card's free VRAM, so every card keeps the same headroom.
Each card holds its own CUDA context and compute buffer (llama.cpp allocates a compute
buffer per backend), so a plan on N cards prices N of each (unmeasured: the split readings
wait for the hardware).

Decode is pipelined: each token passes every card's layers in turn, so the times add. A
card's share of the streamed bytes is read at its own bandwidth, and the effective
bandwidth is the shares' weighted harmonic mean: a slow card costs in proportion to the
layers it holds, and the split already gives the smaller (free VRAM) card fewer. Two equal
cards decode as fast as one (they hold twice the model, not twice the speed).
"""

from __future__ import annotations

from pydantic import BaseModel

MIB = 2**20


class CardBudget(BaseModel):
    """One chosen card in a plan's budget: nvidia-smi's index, and what it has."""

    index: int
    name: str
    free_bytes: int
    total_bytes: int
    bandwidth: int
    # Its PCIe link as nvidia-smi read it (`hardware.probe.Gpu.lanes`); None unknown
    lanes: int | None = None
    pcie_gen: int | None = None  # the link's top generation (`pcie.link.gen.max`)
    # The card itself (its uuid), which a placement keeps to find it after a move
    stable_id: str | None = None
    # mnm's running instances on this card counted free (`plan_budget.across_cards`), and
    # what they hold here: what a plan on this card alone names (`Budget.one_card`)
    held_by: tuple[str, ...] = ()
    held_bytes: int = 0

    @property
    def riser(self) -> bool:
        """On one lane: layers only, never experts, placed last."""
        return self.lanes is not None and self.lanes <= 1

    @property
    def link_rank(self) -> int:
        """Lanes times generation: the wider and newer link carries prompt processing's
        copies of the RAM-held experts; 0 when nvidia-smi could not say."""
        return (self.lanes or 0) * (self.pcie_gen or 1)


def combined_bandwidth(cards: list[CardBudget]) -> int:
    """The layer split's effective bandwidth: each card's share of the layers (by its free
    VRAM) read at its own rate, one after another."""
    free = sum(c.free_bytes for c in cards)
    if not free:
        return min(c.bandwidth for c in cards)
    seconds = sum((c.free_bytes / free) / c.bandwidth for c in cards)
    return int(1 / seconds)


def tensor_split(cards: list[CardBudget]) -> tuple[int, ...]:
    """`-ts`: each card's free MiB in nvidia-smi's order (the container's order too:
    `CUDA_DEVICE_ORDER=PCI_BUS_ID`); llama.cpp normalises the proportions."""
    return tuple(max(c.free_bytes // MIB, 1) for c in sorted(cards, key=lambda c: c.index))


def ranked(cards: list[CardBudget]) -> list[CardBudget]:
    """The cards in the order a placement uses them: the main card first (the widest
    link, then the most free VRAM, then the lowest index), an x1 riser always last."""
    return sorted(cards, key=lambda c: (c.riser, -c.link_rank, -c.free_bytes, c.index))


def main_gpu(cards: list[CardBudget]) -> int:
    """`-mg`: the main card (`ranked`'s first: the widest link, then the most free VRAM,
    then the lowest index), by its place in nvidia-smi's order. Without link readings it is
    the card with the most free VRAM."""
    ordered = sorted(cards, key=lambda c: c.index)
    return ordered.index(ranked(cards)[0])


def shares(total: int, split: tuple[int, ...] | None, count: int) -> list[int]:
    """`total` bytes across `count` cards by a `-ts` split (even without one)."""
    weights = list(split) if split and len(split) == count else [1] * count
    whole = sum(weights)
    parts = [total * w // whole for w in weights]
    parts[-1] += total - sum(parts)
    return parts
