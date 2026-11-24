"""What the CPU side of a decode costs on the box: the rate RAM-held routed experts
are read at, the cost of each used expert held there, and the rate whole blocks left in RAM
are read at. A leaf: `fit.estimate` and `fit.placement` read it, `machine_facts` overlays what
this box measured.

**The experts.** The active share of a block's experts once had a floor of 0.05, which priced
Qwen3.6-35B's 8 of 256 experts as if 12.8 were read, DeepSeek-V4.1-Flash's 6 of 384 as 19. The
share is the real one now, and what the floor stood in for is a cost per used expert held in
RAM (`EXPERT_TOUCH_SECONDS`: llama.cpp's CPU path launches and synchronises one small matrix
product per used expert per block, whatever its size). Two unknowns, fitted to two models with
experts in RAM, measured on an RTX 4070 under WSL2:

- Qwen3-30B-A3B Q3_K_M, ctx 32,768, the experts of 28 of 48 layers in RAM: 28.6 tok/s.
  8 of 128 experts a token (0.0625, above the old floor, so unchanged by it):
  5.858 ms of card reads and launches, 515.3 MB from RAM, 28 x 8 = 224 experts touched.
- Qwen3.6-35B-A3B UD-Q4_K_M, ctx 32,768, 29 of 40 layers' experts in RAM: 25.2 tok/s, then
  26.8 (28 layers) and 28.3 in later runs; 8 of 256 (0.03125: the floor had read 1.6x the
  bytes). 8.373 ms on the card, 447.0 MB from RAM, 29 x 8 = 232 touched.

The two models touch almost the same number of experts (224 and 232), so the two readings
cannot tell a rate from a touch cost on their own: any rate from 18 to 130 GB/s with its touch
fits the 30B exactly. The rate is held at 30 GB/s, half that machine's ~60 GB/s (llama.cpp
reads RAM-held experts at 0.45-0.5 of peak on native Linux), and the touch
cost fitted to the 30B's reading: (1/28.6 - 5.858 ms - 515.3 MB / 30 GB/s) / 224 = 53.3 us. The
35B then reads 1 / (8.373 + 14.90 + 12.37 ms) = 28.1 tok/s at 29 layers, inside the readings'
25.2 to 28.3 (the floor at 17.7 GB/s said 20.4; without the floor and without the touch, 29.7).
Every learned rate (`machine_facts`) is relearned under this model (`EXPERT_MODEL`): a rate
learned under the floor is not used."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# The rate RAM-held experts are read at, and the cost of each used expert held in RAM: the
# RTX 4070 machine's, fitted above; `machine_facts` replaces the rate with this box's own readings.
CPU_EXPERT_RATE = 30.0 * 10**9
EXPERT_TOUCH_SECONDS = 53.3e-6
EXPERT_MODEL = 2  # 1 (None in an older file): the 0.05 floor and 17.7 GB/s; 2: the touch cost
# What whole blocks left in RAM (`-ngl` below the block count) are read at: an estimate for a
# dual-channel DDR5 desktop, never measured; a box's own probe replaces it (`layer_rate`).
CPU_BANDWIDTH = 55 * 10**9
# The share of the probe's bandwidth llama.cpp reads RAM-held weights at: 0.40-0.55 of peak on
# native Linux with a GPU doing attention (a 12600K with DDR5-6000 read 48 of 96 GB/s;
# servers 0.25-0.40). The probe counts a copy's read and write, as STREAM's copy
# does, so its figure is close to the peak a read sees.
LLAMA_RAM_SHARE = 0.5


def layer_rate(ram_gbps: float | None) -> float:
    """Bytes a second whole RAM-held blocks are read at: the box's probe x `LLAMA_RAM_SHARE`
    when it measured one, else the RTX 4070 machine's figure."""
    if not ram_gbps:
        return CPU_BANDWIDTH
    return ram_gbps * 1e9 * LLAMA_RAM_SHARE


class CpuSide(BaseModel):
    """What a CPU plan renders beyond the profile: llama.cpp's threads and NUMA mode."""

    model_config = ConfigDict(frozen=True)

    threads: int | None = None  # the physical cores (`-t`); None leaves llama.cpp's default
    numa_nodes: int | None = None

    @property
    def numa(self) -> str | None:
        """`distribute` on a box of several NUMA nodes (`--numa`), else None."""
        return "distribute" if self.numa_nodes and self.numa_nodes > 1 else None
