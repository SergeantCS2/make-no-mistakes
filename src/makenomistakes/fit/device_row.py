"""One card family's figures as the speed and memory models read them,
and the 4070's: every per-card constant mnm was fitted with is a field of `DeviceRow`, and
`RTX_4070` holds the 4070's values, fitted to its readings. The table of every other card is
`fit.devices`; this module is a leaf so the estimate modules can take a row without
importing the table.

The constants below keep their names for every reader that imported them: they are the 4070's,
and `fit/devices.py` is the table."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from makenomistakes.fit.picture_factors import FAMILY_SPEEDS, SPEED_MODEL
from makenomistakes.hardware.fingerprint import MemoryKind, Vendor
from makenomistakes.hub.kit_types import ImageFamily

MIB = 1024 * 1024
# The 4070's: `fit/devices.py` is the table. Fraction of peak bandwidth a decode step achieves
# over the bytes it streams (the file less its input embedding table), fitted to two readings
# on an RTX 4070 (75.4 and 325 tok/s, every block on the GPU) with the launch term below.
GPU_EFFICIENCY = 0.91
# The codebook i-quants (IQ1-IQ3) decode at about half of that on the card: Qwen3.8-27B
# UD-IQ3_XXS, every block on the 4070, read 21.35 tok/s against 35 at 0.91 (10.4 GB streamed
# a token in 41.3 ms after the launches, 253 GB/s). IQ4_XS/NL keep
# 0.91, unmeasured.
IQ_GRID_EFFICIENCY = 0.50
LAYER_OVERHEAD_SECONDS = 0.086e-3  # kernel launches per layer per token, fitted on the 4070
RUNTIME_OVERHEAD_BYTES = 256 * MIB  # CUDA context + scratch as nvidia-smi sees it under WSL
VLLM_DECODE_EFFICIENCY = 0.85  # vLLM: of the bandwidth over the weights, per decoded token
# One crossing between cards per token, a constant until a box measures it. From llama.cpp
# discussion #22183 (Vulkan on an RX 7800 XT + RX 580): moving 11 blocks' experts from RAM to
# a second card took decode from 35.4 to 38.2 tok/s, 2.1 ms a token, where the bytes alone
# save 14.5 ms: 12.4 ms over 22 crossings. For CUDA it is likely 5-10x too high; it stays the
# default, said as unmeasured wherever it decides a placement.
HOP_SECONDS = 0.56e-3
UNMEASURED = "unmeasured"
FOUR_070_SOURCE = "an RTX 4070 under WSL2"


class ChosenBy(StrEnum):
    """How `fit.devices.device_for` found the row: what the plan says about it."""

    NAME = "name"  # the card's own row
    FAMILY = "family"  # the family's default by compute capability or gfx target
    VENDOR = "vendor"  # nothing known about the card but its maker


class Learned(BaseModel):
    """This box's own decode figures (`machine_facts`, `calibration`), over the row's."""

    model_config = ConfigDict(frozen=True)

    efficiency: float = Field(gt=0)
    launch_seconds: float | None = None  # learned only from two model sizes
    readings: int = 0
    at: str = ""  # the date learned, as ISO


class DeviceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    match: tuple[str, ...]  # name words, longest first wins (`fit.devices.matches`)
    vendor: Vendor
    family: str  # "ada", "ampere", "blackwell", "hopper", "turing", "gb10", "rdna3" ...
    memory: MemoryKind = MemoryKind.DISCRETE
    bandwidth: int  # bytes a second: the vendor's figure until a box reads it
    tflops: float  # dense fp16 without the matrix units: what the picture model scales by
    efficiency: float = GPU_EFFICIENCY
    # A mixture of experts' own decode efficiency where the row knows one (the GB10's
    # benches read 0.64 MoE against 0.72 dense); None: the dense one for both
    moe_efficiency: float | None = None
    iq_grid_efficiency: float = IQ_GRID_EFFICIENCY
    launch_seconds: float = LAYER_OVERHEAD_SECONDS
    context_bytes: int = RUNTIME_OVERHEAD_BYTES
    picture_speeds: dict[ImageFamily, float] = Field(
        default_factory=lambda: dict(FAMILY_SPEEDS[SPEED_MODEL])
    )
    vllm_decode_efficiency: float = VLLM_DECODE_EFFICIENCY
    hop_seconds: float = HOP_SECONDS
    hop_source: str = UNMEASURED
    source: str  # where the bandwidth came from: a box, a vendor page, a family default
    # the efficiencies' own basis: the RTX 4070 machine's readings, or theirs borrowed
    efficiency_source: str = "family default: the 4070's efficiencies"
    name: str = ""  # the name as the plan says it ("RTX 3090")
    compute: str | None = None  # NVIDIA compute capability ("8.6")
    gfx: str | None = None  # AMD gfx target ("gfx1100")
    memory_gb: int | None = None  # this variant's memory, when the line has several
    hf_name: str | None = None  # the part in Hugging Face's table (`fit.hf_hardware`)
    chosen_by: ChosenBy = ChosenBy.NAME
    learned: Learned | None = None
    # AMD's PCI device ids (the probe's last word for a name), and the compute a
    # backend reaches where it differs (`for_backend`): the R9700 reads about 62 TFLOPS on HIP
    # and 43 on Vulkan, never the marketing matrix figure
    pci_ids: tuple[int, ...] = ()
    backend_tflops: dict[str, float] = Field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """The efficiencies are this kind of card's own readings (the 4070's, or learned)."""
        return self.learned is not None or self.efficiency_source == FOUR_070_SOURCE

    @property
    def source_word(self) -> str:
        """The doctor's one word: measured, family or unknown."""
        if self.measured:
            return "measured"
        return "unknown" if self.chosen_by is ChosenBy.VENDOR else "family"

    def for_backend(self, backend: str | None) -> DeviceRow:
        """The row as `backend` (a `backend.Backend` value) runs the card: its compute where
        the row knows one for it, else unchanged."""
        found = self.backend_tflops.get(backend or "")
        return self if found is None else self.model_copy(update={"tflops": found})

    def with_learned(self, learned: Learned | None) -> DeviceRow:
        """This box's own figures over the row's (this box's readings only)."""
        if learned is None:
            return self
        update: dict[str, object] = {
            "efficiency": learned.efficiency,
            "moe_efficiency": None,  # this box's readings stand for both
            "learned": learned,
        }
        if learned.launch_seconds is not None:
            update["launch_seconds"] = learned.launch_seconds
        return self.model_copy(update=update)


RTX_4070 = DeviceRow(
    match=("rtx 4070",),
    vendor=Vendor.NVIDIA,
    family="ada",
    bandwidth=504 * 10**9,
    tflops=29.0,
    source=FOUR_070_SOURCE + "; 504 GB/s from NVIDIA's RTX 4070 page",
    efficiency_source=FOUR_070_SOURCE,
    name="RTX 4070",
    compute="8.9",
    memory_gb=12,
    hf_name="RTX 4070",
)
