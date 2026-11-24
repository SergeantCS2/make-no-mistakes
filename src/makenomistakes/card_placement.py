"""Where a mixture-of-experts model's parts sit across several cards: the
typed placement a recipe saves and the llama.cpp flags it renders. The planning is
`fit.placement`; this module is a leaf (`engine_args` imports it, and `engine_args` must
import nothing from the `fit` package).

llama.cpp (read at d854d98: `src/llama-model.cpp`, `common/arg.cpp`, `common/common.h`):

- `--device CUDA1,CUDA0` lists the cards in the order the model uses them; `-ts` gives each
  of them, in that order, its share of the `-ngl` window (the blocks, then the output layer):
  layer `il` goes to the first card whose cumulative share exceeds `il / (blocks + 1)`. Shares
  given as whole block counts, the last card's plus one for the output layer, put exactly
  those blocks on each card. A big batch's operations on weights kept in RAM run on the first
  listed card (`ggml_backend_sched`'s offload goes to the lowest backend that takes the op),
  which is why the main card comes first.
- `-ot <regex>=<buffer>,…` is one comma-joined list, first match wins (`std::regex_search`
  on each tensor's name). `--n-cpu-moe N` is `blk\\.i\\.ffn_(up|down|gate|gate_up)_(ch|)exps
  =CPU` for i < N, the same expert pattern this module writes (`LLM_FFN_EXPS_REGEX`).
- Buffer names are `CUDA<k>` by the card's place among the cards the container sees (Docker
  exposes them in nvidia-smi's order, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), whatever `--device`'s
  order. A CPU build lists no CUDA buffer and refuses those names.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

EXPERTS = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"  # llama.cpp's LLM_FFN_EXPS_REGEX


class CardPlace(BaseModel):
    """One card's part: its run of blocks (dense weights, KV), the blocks whose routed experts
    it holds, and what the plan put on it."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)  # nvidia-smi's, when it was planned
    # The card itself (its uuid, `fingerprint.stable_id_of`), which follows it from
    # slot to slot; None in a placement saved before mnm kept card ids
    stable_id: str | None = None
    first: int = Field(ge=0)  # the first block of its run
    count: int = Field(ge=0)  # blocks in its run
    experts: tuple[int, ...] = ()  # blocks whose routed experts it holds
    bytes: int = Field(ge=0)  # estimated: a box reading replaces it
    # what `mnm set experts=` and the tuner place it again with: its free VRAM when it was
    # planned, its memory bandwidth and its PCIe lanes (an x1 riser holds no experts)
    free: int = Field(ge=0)
    bandwidth: int = Field(default=0, ge=0)
    lanes: int | None = None

    @property
    def blocks(self) -> range:
        return range(self.first, self.first + self.count)


class Placement(BaseModel):
    """Every chosen card's part, in the order llama.cpp uses them (the main card first, an
    x1 riser last), and what stays in RAM."""

    model_config = ConfigDict(frozen=True)

    cards: tuple[CardPlace, ...] = Field(min_length=2)
    in_ram: tuple[int, ...] = ()  # blocks whose routed experts stay in RAM
    ram_bytes: int = Field(default=0, ge=0)
    # on the main card for a big batch's copies of the RAM-held experts (the largest of one
    # block's expert tensors): estimated, counted in its `bytes` and in Fits now
    copy_bytes: int = Field(default=0, ge=0)

    @property
    def counts(self) -> tuple[int, ...]:
        """How many blocks' experts each card holds, in its order (`card-experts=`)."""
        return tuple(len(c.experts) for c in self.cards)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(c.index for c in self.cards)

    @property
    def tensor_split(self) -> tuple[int, ...]:
        """`-ts` in `--device` order: each card's blocks, the last card's plus the output
        layer, so the split lands on exactly these blocks."""
        counts = [c.count for c in self.cards]
        counts[-1] += 1
        return tuple(counts)

    @property
    def hops(self) -> int:
        """Blocks whose experts sit on another card than their dense weights: a token
        crosses to that card and back at each."""
        return sum(1 for c in self.cards for b in c.experts if b not in c.blocks)

    def remapped(self, live: Mapping[str, int] | None) -> Placement | None:
        """This placement on the cards where they are now: each place's index moved
        to its card's (`live`: stable id -> nvidia-smi index, this box as last recorded);
        as saved when no place names its card (planned before mnm kept card ids) or the
        box's cards are not known; None when a card it was placed on is not there."""
        if live is None or not all(c.stable_id for c in self.cards):
            return self
        if any(c.stable_id not in live for c in self.cards):
            return None
        moved = tuple(c.model_copy(update={"index": live[c.stable_id or ""]}) for c in self.cards)
        return self.model_copy(update={"cards": moved})

    def on(self, index: int) -> CardPlace | None:
        return next((c for c in self.cards if c.index == index), None)

    def overrides(
        self, positions: dict[int, int], lazy: tuple[str, ...] = (), prefix: str = "CUDA"
    ) -> str:
        """`-ot`'s one value: tensors read from disk first (an Engram table), then each
        card's experts, then every other expert to the CPU. `prefix` is llama.cpp's name for
        the backend's devices (`backend.device_name`: `CUDA`, `ROCm`, `Vulkan`)."""
        rules = [f"{name}=CPU" for name in lazy]
        for card in self.cards:
            if card.experts:
                blocks = "|".join(str(b) for b in sorted(card.experts))
                rules.append(f"blk\\.({blocks}){EXPERTS}={prefix}{positions[card.index]}")
        rules.append(f"{EXPERTS}=CPU")
        return ",".join(rules)

    def argv(
        self, cards: tuple[int, ...], lazy: tuple[str, ...] = (), prefix: str = "CUDA"
    ) -> list[str]:
        """`--device`, `-ts`, `-mg` and `-ot` for the instance's cards (nvidia-smi indices);
        the caller checked they are this placement's cards."""
        positions = {index: place for place, index in enumerate(sorted(cards))}
        return [
            "--device",
            ",".join(f"{prefix}{positions[i]}" for i in self.indices),
            "-ts",
            ",".join(str(n) for n in self.tensor_split),
            "-mg",
            "0",  # the first `--device` card: the main card
            "-ot",
            self.overrides(positions, lazy, prefix),
        ]

    def lines(self, gib: int = 2**30) -> list[str]:
        """`GPU0: blocks 0-27 + experts of 0-9 (14.2 GiB)` per card, then RAM's line."""
        found = []
        for card in self.cards:
            run = f"blocks {_span((card.first, card.first + card.count - 1))}"
            experts = f" + experts of {_spans(card.experts)}" if card.experts else ""
            found.append(f"GPU{card.index}: {run}{experts} ({card.bytes / gib:.1f} GiB)")
        if self.in_ram:
            found.append(
                f"experts of {_spans(self.in_ram)} in RAM ({self.ram_bytes / gib:.1f} GiB)"
            )
        return found


def _span(edge: tuple[int, int]) -> str:
    return str(edge[0]) if edge[0] == edge[1] else f"{edge[0]}-{edge[1]}"


def _spans(blocks: tuple[int, ...]) -> str:
    """`0-9, 12, 14-15`."""
    runs: list[tuple[int, int]] = []
    for b in sorted(blocks):
        if runs and b == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((b, b))
    return ", ".join(_span(r) for r in runs)
