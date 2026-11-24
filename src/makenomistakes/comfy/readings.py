"""A ComfyUI run's readings: how long it took, the GPU memory in use at its peak
(and how much above what was in use before it), the RAM at its peak, and the size of the
model files it loads; before the run, a word when those files are larger than the card, or
than the card and the free RAM together.

The figures are nvidia-smi's and /proc/meminfo's, sampled every half second while the run
goes, so they are what the card and the box held, not what ComfyUI says it allocated.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from makenomistakes import comfyui
from makenomistakes.comfy.models import link_name
from makenomistakes.comfy.report import ModelNeed
from makenomistakes.gpu_peak import PeakSampler, ReadUsed
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import (
    PROC_MEMINFO,
    MemorySnapshot,
    parse_meminfo,
    probe_memory,
    read_text_or_empty,
)
from makenomistakes.try_figures import Figure

GIB = 2**30
EVERY = 0.5  # seconds between samples: a run is minutes long


def vram_used() -> int | None:
    gpu = probe_memory().gpu
    return gpu.vram_used_bytes if gpu is not None else None


def ram_used() -> int | None:
    """RAM in use (total less available), from /proc/meminfo alone."""
    found = parse_meminfo(read_text_or_empty(PROC_MEMINFO))
    if found is None or found.available_bytes is None:
        return None
    return found.total_bytes - found.available_bytes


class Readings:
    """Around a run: `with Readings(...) as readings:`, then `readings.numbers()`."""

    def __init__(
        self,
        vram: ReadUsed = vram_used,
        ram: ReadUsed = ram_used,
        clock: Callable[[], float] = time.monotonic,
        every: float = EVERY,
    ) -> None:
        self._vram = PeakSampler(vram, every)
        self._ram = PeakSampler(ram, every)
        self._read_vram = vram
        self._clock = clock
        self.before: int | None = None
        self.seconds = 0.0
        self._started = 0.0

    def __enter__(self) -> Readings:
        self.before = self._read_vram()
        self._started = self._clock()
        self._vram.__enter__()
        self._ram.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._vram.__exit__(*exc)
        self._ram.__exit__(*exc)
        self.seconds = max(self._clock() - self._started, 0.0)

    def numbers(self, model_bytes: int = 0) -> dict[str, float]:
        found: dict[str, float] = {}
        if self._vram.peak is not None:
            found[Figure.VRAM_PEAK_GIB.value] = round(self._vram.peak / GIB, 2)
            if self.before is not None:
                added = max(self._vram.peak - self.before, 0)
                found[Figure.VRAM_ADDED_GIB.value] = round(added / GIB, 2)
        if self._ram.peak is not None:
            found[Figure.RAM_PEAK_GIB.value] = round(self._ram.peak / GIB, 2)
        if model_bytes:
            found[Figure.MODEL_FILES_GIB.value] = round(model_bytes / GIB, 2)
        return found


def model_bytes(models_dir: Path, needs: list[ModelNeed]) -> int:
    """The model files a workflow loads, as linked where ComfyUI looks (missing ones: 0)."""
    total = 0
    for need in needs:
        path = models_dir / comfyui.SUBDIR / (need.folder or "checkpoints") / link_name(need)
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def size_warning(files: int, memory: MemorySnapshot) -> str | None:
    """A word before the run when its files are larger than the card (ComfyUI keeps part in
    RAM: slower) or than the card and the free RAM together (it may run out of memory)."""
    gpu = memory.gpu
    if gpu is None or not files:
        return None
    free_ram = memory.memory.available_bytes if memory.memory is not None else None
    sizes = f"{files / GIB:.1f} GiB of model files"
    if free_ram is not None and files > gpu.vram_total_bytes + free_ram:
        return (
            f"{sizes}: more than the card ({gpu.vram_total_bytes / GIB:.1f} GiB) and the free "
            f"RAM ({free_ram / GIB:.1f} GiB) together. ComfyUI may run out of memory; a "
            f"smaller quant of the model, or {_more_memory()}, helps."
        )
    if files > gpu.vram_total_bytes:
        return f"{sizes}, more than the card holds: ComfyUI keeps part in RAM, which is slower."
    return None


def _more_memory() -> str:
    """What gives ComfyUI more room on this platform: WSL's own setting, or RAM and swap."""
    return "more WSL memory (`mnm doctor`)" if platforms.current().is_wsl else "more RAM or swap"
