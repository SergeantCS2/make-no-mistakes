"""ComfyUI's HTTP answers mnm reads, typed at the edge (v0.42.1, `server.py`)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Device(BaseModel):
    name: str = ""
    type: str = ""
    vram_total: int = 0
    vram_free: int = 0
    torch_vram_total: int = 0  # what PyTorch holds: ComfyUI's own share of the card
    torch_vram_free: int = 0


class System(BaseModel):
    comfyui_version: str = ""
    python_version: str = ""
    pytorch_version: str = ""
    ram_total: int = 0
    ram_free: int = 0


MIB = 2**20
GIB = 2**30
# What ComfyUI holds on a CUDA card beyond PyTorch's pool, which is all `system_stats`
# reports (`torch_vram_total`: PyTorch's reserved bytes). Measured on an RTX 4070:
# after a film, nvidia-smi read 3,573 MiB in use with the desktop at about 1,214 and nothing
# else of mnm's loaded, while the pool read 0.1 GiB: about 2.2 GiB more. Two parts:
CUDA_CONTEXT_BYTES = 512 * MIB  # the CUDA context and its libraries, from ComfyUI's start
# ComfyUI v0.42's dynamic VRAM ("comfy-aimdo", on by default on CUDA) keeps its cast and
# staging buffers outside PyTorch's allocator after a run; the rest of that 2.2 GiB.
DYNAMIC_VRAM_BYTES = int(1.7 * GIB)
# What a free gives back of those buffers, as nvidia-smi measured it on an RTX 4070: 544 MiB
# (the pool's 0.1 GiB included), not the 1.7 GiB a reading of the source suggested: room is
# judged on what comes back.
FREED_DYNAMIC_BYTES = 544 * MIB - int(0.1 * GIB)  # the pool, 0.1 GiB, counted apart


class SystemStats(BaseModel):
    """`GET /system_stats`: the health check, and what ComfyUI holds."""

    system: System = Field(default_factory=System)
    devices: list[Device] = Field(default_factory=list)

    @property
    def held_bytes(self) -> int:
        """What its PyTorch pool holds of the card(s): what `/free` gives back, so what a
        plan may count as free with the buffers `freeable` adds. A CPU build lists one `cpu` device
        whose "VRAM" is the machine's RAM: not counted."""
        return sum(d.torch_vram_total for d in self.devices if d.type != "cpu")

    @property
    def on_cuda(self) -> bool:
        return any(d.type == "cuda" for d in self.devices)

    def held_estimate(self, ran: bool) -> int:
        """What ComfyUI holds of the card in all, as the user reads it (the Status panel,
        `mnm ls`, the status bar): the pool, plus its CUDA context, plus, once a run has gone
        since its start or its last free, the dynamic-VRAM buffers (`DYNAMIC_VRAM_BYTES`,
        a reading on an RTX 4070). Its CPU build holds none."""
        if not self.on_cuda:
            return self.held_bytes
        return self.held_bytes + CUDA_CONTEXT_BYTES + (DYNAMIC_VRAM_BYTES if ran else 0)

    def freeable(self, ran: bool) -> int:
        """What a free (`POST /free`, every start's `release_vram`) gives back of
        `held_estimate`: the pool, and the dynamic-VRAM buffers once a run has gone; never
        the CUDA context, held while ComfyUI runs. Read from v0.42.1's source: the prompt
        worker wakes on the flag (`PromptQueue.set_flag` notifies) and unloads every model,
        the dynamic ones too (`model_management.free_memory` →
        `ModelPatcher.partially_unload` → comfy-aimdo 0.5.6's `ModelVBAR.free_memory`, which
        unmaps and releases the pages), resets the executor's cache (the models' VBARs go
        with it) and empties PyTorch's; the cast buffers are reset after every node anyway.
        nvidia-smi on an RTX 4070 saw less come back than that reading suggests (544 MiB):
        `FREED_DYNAMIC_BYTES`."""
        if not self.on_cuda:
            return self.held_bytes
        return self.held_bytes + (FREED_DYNAMIC_BYTES if ran else 0)

    def held_method(self, ran: bool) -> str:
        """How `held_estimate` was reached, for the panel's hover."""
        if not self.on_cuda:
            return "PyTorch's pool (ComfyUI's own count)"
        parts = [
            f"PyTorch's pool {self.held_bytes / GIB:.1f} GiB (ComfyUI's own count)",
            f"its CUDA context, about {CUDA_CONTEXT_BYTES / GIB:.1f} GiB",
        ]
        if ran:
            parts.append(
                f"its dynamic-VRAM buffers after a run, about {DYNAMIC_VRAM_BYTES / GIB:.1f} GiB"
                " (measured on an RTX 4070)"
            )
        back = (
            f"the pool and about {FREED_DYNAMIC_BYTES / GIB:.1f} GiB of these buffers"
            if ran
            else "the pool"
        )
        return " + ".join(parts) + f"; Free VRAM returns {back}, the context stays while it runs"


class Queue(BaseModel):
    """`GET /queue`: the job running and the ones waiting (each a list ComfyUI builds)."""

    queue_running: list[list[object]] = Field(default_factory=list)
    queue_pending: list[list[object]] = Field(default_factory=list)

    @property
    def busy(self) -> bool:
        return bool(self.queue_running or self.queue_pending)
