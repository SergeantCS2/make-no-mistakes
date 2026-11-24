"""What a plan is made against: this machine's free VRAM and RAM, the cards, and the
corrections its readings taught (`plan.live_budget` builds one; `--vram` makes another)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from makenomistakes.backend import Backend
from makenomistakes.fit.cards import CardBudget
from makenomistakes.fit.cpu_rates import CPU_BANDWIDTH, CPU_EXPERT_RATE, CpuSide
from makenomistakes.fit.device_row import RTX_4070, DeviceRow
from makenomistakes.fit.devices import DEFAULT_GPU_TFLOPS, device_for
from makenomistakes.fit.ram_floor import OS_RAM_MARGIN, margin_for
from makenomistakes.hardware.fingerprint import SHARED_MEMORY
from makenomistakes.hub.kit_types import ImageFamily

# A desktop holds 1-2 GiB of the card; above this, another program does, and a plan made now
# is sized around it for good (beside a program holding 8 GiB, qwen3-8b would be saved with 7
# of 36 layers on the GPU). The doctor warns above the same line.
BUSY_CARD_BYTES = 2 * 1024**3
# Under WSL the driver spills past the card's top into the PC's memory instead of failing a
# start, and a model that spills crawls: zimage at 11,808 of the 4070's 12,282 MiB (474 MiB
# short of its top) spilled, Bolt peaked at 11,994 (288 MiB short) and decoded 2.1 tok/s
# against about 12 planned. A llama.cpp plan on a WSL card keeps this much of the free VRAM
# unplanned (`Budget.plan_free_bytes`), whatever its level.
WSL_SPILL_MARGIN = 512 * 1024**2
# Plans made in RAM (the CPU backend, Mesa's software device): their row says unified so the
# floor counts the RAM once, but they are no card's pool (no firm margin, no `load=none`, no
# vLLM)
IN_RAM_FAMILIES = frozenset({"cpu", "lavapipe"})


class Budget(BaseModel):
    vram_free_bytes: int
    ram_free_bytes: int
    # The machine's whole RAM (under WSL the VM's), what the RAM floor holds a model to;
    # None for `--ram` overrides, where the free figure stands in
    ram_total_bytes: int | None = None
    gpu_bandwidth: int
    gpu_compute: float = DEFAULT_GPU_TFLOPS * 1e12  # FLOP/s, for diffusion models
    vram_total_bytes: int | None = None  # set by the live probe; None for `--vram` overrides
    # mnm's own running instances, counted as free (`plan_budget.planning_budget`): a new
    # instance is planned for the card, and each one is started on its own or in a workflow
    held_by: tuple[str, ...] = ()
    held_bytes: int = 0
    # The engine's measured correction (`calibration`): `vram_free_bytes` is the card's
    # free VRAM divided by it, so every fit decision prices the estimate times the ratio.
    correction: float = 1.0
    correction_note: str | None = None
    reported_free_bytes: int | None = None  # the card's own figure, once corrected
    # What a plan leaves free at the card's top (`WSL_SPILL_MARGIN` on a WSL card, set by
    # `plan.budget_for`); 0 for another machine (`--vram`) and a native box
    spill_margin_bytes: int = 0
    # Picture and clip time measured on this card (`calibration.speed_correction`):
    # a DiT family's s/image and s/clip are the fit model's times this ratio, said in its note.
    speed: dict[ImageFamily, float] = Field(default_factory=dict)
    speed_notes: dict[ImageFamily, str] = Field(default_factory=dict)
    # A music model's seconds of work a second of audio measured on this box
    # (`music_speed`), by `family/file`; the plan takes it before the device's estimate
    music_rates: dict[str, float] = Field(default_factory=dict)
    # Each chosen card when `MNM_GPU` names several (`fit.cards`); the fields above are
    # their sums then (bandwidth: the layer split's). Empty on one card, as before.
    cards: tuple[CardBudget, ...] = ()
    # The rate RAM-held experts are read at on this box (`machine_facts`), else the
    # constant fitted on an RTX 4070 machine
    expert_rate: float = CPU_EXPERT_RATE
    # The card's row in the device table (`fit.devices.device_for`), with this box's
    # learned figures over it once it has readings; a budget built by hand gets the 4070's
    device: DeviceRow = RTX_4070
    # What whole RAM-held blocks are read at (`fit.cpu_rates.layer_rate`): this box's
    # probe x `LLAMA_RAM_SHARE`, else the RTX 4070 machine's 55 GB/s
    layer_rate: float = CPU_BANDWIDTH
    # What the system keeps beside the models on this box's RAM platform (a server
    # board more than a desktop, `hardware.ram_platforms`), and how the RAM-held weights are
    # priced, in words, for the plan's notes ("priced at 50 GB/s for an EPYC 7532 with 4 CCDs
    # until this box measures"); None for a budget that did not read the box
    ram_margin: int = OS_RAM_MARGIN
    ram_note: str | None = None
    # A box without a card, the RAM the pool every model is planned in
    # (`fit.cpu_plan.cpu_budget`): its threads and NUMA mode for the recipe; None on a card
    cpu: CpuSide | None = None
    # The backend the plan's images are for and the box's architecture (`uname -m`),
    # decided from the box when the live probe built the budget; None for a budget built by
    # hand or for another machine (`--vram`): the recipe keeps the images every box ran before
    backend: Backend | None = None
    arch: str | None = None
    # A plan for another machine (`--vram`, `plan.explicit_budget`): said, never read off a
    # missing total (a box without a card has none either, and its RAM floor would name
    # "the machine `--vram` describes")
    elsewhere: bool = False

    @property
    def no_card(self) -> bool:
        """A plan made in RAM: the CPU backend, or Vulkan on Mesa's software device."""
        return self.device.family in IN_RAM_FAMILIES

    @property
    def pool(self) -> bool:
        """One pool: the card plans in the machine's RAM (a GB10, an APU's GTT)."""
        return self.device.memory in SHARED_MEMORY and self.device.family not in IN_RAM_FAMILIES

    @property
    def margin(self) -> int:
        """What the system keeps beside the models: 8 GiB on a unified pool, else the RAM
        platform's (4 GiB, 8 on a server board)."""
        return margin_for(self.device.memory) if self.pool else self.ram_margin

    def speed_ratio(self, family: ImageFamily) -> float:
        return self.speed.get(family, 1.0)

    @property
    def spill_words(self) -> str | None:
        """The margin in words, under the budget line."""
        if not self.spill_margin_bytes:
            return None
        return (
            f"Plans keep {self.spill_margin_bytes / 2**30:.1f} GiB of that free: under WSL "
            "the driver spills past the card's top into the PC's memory, and a model that "
            "spills crawls."
        )

    @property
    def plan_free_bytes(self) -> int:
        """What a llama.cpp plan may put on the card: the free VRAM less the spill margin
        (priced like it, by the correction: the margin is the card's)."""
        return self.vram_free_bytes - int(self.spill_margin_bytes / self.correction)

    @property
    def card_free_bytes(self) -> int:
        """The free VRAM the card itself reports (what the plan table shows)."""
        return (
            self.vram_free_bytes if self.reported_free_bytes is None else self.reported_free_bytes
        )

    @property
    def vram_used_bytes(self) -> int | None:
        """The VRAM in use that this budget does not count as free. For a plan's budget
        (`plan_budget.planning_budget`) that is what is left once mnm's running instances'
        estimates are counted as free: other programs, or more than those estimates."""
        if self.vram_total_bytes is None:
            return None
        return max(self.vram_total_bytes - self.card_free_bytes, 0)

    @property
    def busy_bytes(self) -> int | None:
        """That VRAM when it is more than a desktop holds (another program), else None; `busy_card`
        words it."""
        used = self.vram_used_bytes
        # unified memory: the pool's use beyond the system's margin is the busy part,
        # what other processes hold (their RSS is the pool's pages)
        beyond = self.margin if self.device.memory in SHARED_MEMORY else 0
        return used if used is not None and used - beyond > BUSY_CARD_BYTES else None

    def one_card(self) -> Budget:
        """The first chosen card alone (`MNM_GPU`'s first): what an engine that runs on one
        card is planned against (`recipes.cards.ONE_CARD_ENGINES`: sd.cpp, whisper.cpp,
        audio.cpp), never the cards' sum (a music plan on two 12 GiB cards would save BF16,
        which neither holds). Unchanged on one card."""
        return self.on_card(self.cards[0]) if len(self.cards) > 1 else self

    def on_card(self, card: CardBudget) -> Budget:
        """One of `cards` alone: its free and total VRAM, bandwidth and row (the budget's
        `device` is the first card's, priced for the backend), its share of the spill margin,
        and only what mnm's instances hold on it counted free and named."""
        device = self.device if card.name == self.cards[0].name else device_for(card.name)
        reported = self.reported_free_bytes
        return self.model_copy(
            update={
                "vram_free_bytes": card.free_bytes,
                "vram_total_bytes": card.total_bytes,
                "reported_free_bytes": None
                if reported is None
                else int(card.free_bytes * self.correction),
                "gpu_bandwidth": card.bandwidth,
                "device": device,
                "gpu_compute": device.tflops * 1e12,
                "spill_margin_bytes": self.spill_margin_bytes // max(len(self.cards), 1),
                "held_by": card.held_by,
                "held_bytes": card.held_bytes,
                "cards": (),
            }
        )

    def with_more(self, extra: int) -> Budget:
        """`extra` bytes more of the place this budget plans in (`mnm why`'s *more room*), with
        the fields that read that place moved together (an edit that left the card's own
        reported figure and a pool's RAM behind would read the room as held by other programs,
        keep a vLLM share the same in bytes and leave a pool's RAM floor refusing): the free
        and total, the card's own figure where a correction priced it, the one card in `cards`;
        on a pool or the RAM (`pool`, `no_card`) the RAM's total too, which the RAM floor reads.
        Several cards are refused: which card the room goes to is a placement question."""
        if len(self.cards) > 1:
            raise ValueError("more room on several cards is a placement question")
        priced = int(extra / self.correction)  # the free figure is the card's over the ratio
        update: dict[str, object] = {
            "vram_free_bytes": self.vram_free_bytes + priced,
            "cards": tuple(
                c.model_copy(
                    update={
                        "free_bytes": c.free_bytes + priced,
                        "total_bytes": c.total_bytes + extra,
                    }
                )
                for c in self.cards
            ),
        }
        if self.vram_total_bytes is not None:
            update["vram_total_bytes"] = self.vram_total_bytes + extra
        if self.reported_free_bytes is not None:
            update["reported_free_bytes"] = self.reported_free_bytes + extra
        if (self.pool or self.no_card) and self.ram_total_bytes is not None:
            update["ram_total_bytes"] = self.ram_total_bytes + extra
        return self.model_copy(update=update)

    def corrected(self, ratio: float, note: str | None) -> Budget:
        """Priced for an engine whose estimates run `ratio` times too low (or high)."""
        if self.correction != 1.0:  # priced once: the note stays the one for its ratio
            return self
        if ratio == 1.0:
            return self.model_copy(update={"correction_note": note}) if note else self
        return self.model_copy(
            update={
                "vram_free_bytes": int(self.vram_free_bytes / ratio),
                # each card alike, or placement and vLLM's split plan uncorrected
                "cards": tuple(
                    c.model_copy(update={"free_bytes": int(c.free_bytes / ratio)})
                    for c in self.cards
                ),
                "reported_free_bytes": self.vram_free_bytes,
                "correction": ratio,
                "correction_note": note,
            }
        )


def pool_load(budget: Budget) -> Literal["none"] | None:
    """llama.cpp's `--load-mode` a plan writes: `none` on a pool (a GB10, an APU's GTT),
    where a mapped file's page cache sits in the same memory as the weights the engine copies
    from it, and the GB10's mmap loads read slow (gpt-oss-120b ready in 22 s with `--no-mmap`
    against about 90 s); None elsewhere (the engine's own `auto`). `none` in RAM too: on the
    CPU backend room is MemTotal less MemAvailable (`ram_room`), and a mapped model's weights
    sit in the page cache, which MemAvailable counts as free, so a running model would read as
    holding nothing; Mesa's software device copies its weights into host memory as a pool
    does."""
    return "none" if budget.pool or budget.no_card else None
