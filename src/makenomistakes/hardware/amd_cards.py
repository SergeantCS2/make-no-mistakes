"""What each AMD generation can run in mnm's engine images, keyed by gfx target
(`hardware.cards` holds NVIDIA's by compute capability). Data, cited; the doctor words it.

Read 2026-10-26: llama.cpp's `server-rocm` is built for `gfx908;gfx90a;
gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1150;gfx1200;gfx1201` (`.devops/
rocm.Dockerfile`, `backend.ROCM_TARGETS`) and `server-vulkan` runs any Vulkan 1.x device;
sd.cpp, whisper.cpp and audio.cpp publish Vulkan images only; vLLM's ROCm image lists
gfx1100/1101/1200/1201/1150/1151 (`Dockerfile.rocm_base`'s `PYTORCH_ROCM_ARCH`), never run by
mnm (planned behind a box reading); PyTorch's ROCm 7.2 wheels (ComfyUI) cover RDNA 2 to 4 and
the APUs (`manywheel/build.sh`). gfx906 (MI50, Radeon VII) and the smaller RDNA 2/3 chips
are outside the ROCm image's list: Vulkan only. CDNA (MI100 to MI350) is out of mnm's scope:
its cards run ROCm's full stack, which mnm neither plans nor tests; Vulkan runs them as any
card. Every cell is from documentation: nothing here ran on an AMD card."""

from __future__ import annotations

from pydantic import BaseModel

from makenomistakes.backend import ROCM_TARGETS
from makenomistakes.hardware.cards import COMFYUI, Floor, Runs
from makenomistakes.hub.facts import Engine


class AmdGeneration(BaseModel):
    name: str
    targets: tuple[str, ...]  # gfx targets, exact
    engines: dict[str, Floor]
    note: str = ""


_VULKAN = Floor(runs=Runs.RUNS, note="its Vulkan image")
_BOTH = Floor(runs=Runs.RUNS, note="Vulkan by default; ROCm when chosen (`MNM_BACKEND=rocm`)")
_VULKAN_ONLY = Floor(
    runs=Runs.WITHOUT, note="Vulkan only: llama.cpp's ROCm image is not built for it"
)
_VLLM_PLANNED = Floor(
    runs=Runs.WITHOUT, note="its ROCm image lists the card; planned, unverified until a box runs it"
)
_VLLM_NO = Floor(runs=Runs.REFUSED, note="vLLM's ROCm image is not built for it")
_COMFY_ROCM = Floor(runs=Runs.RUNS, note="PyTorch's ROCm 7.2 wheels (planned, unverified)")
_COMFY_NO = Floor(runs=Runs.REFUSED, note="PyTorch's ROCm wheels are not built for it")
_REST = {Engine.SDCPP: _VULKAN, Engine.WHISPERCPP: _VULKAN, Engine.AUDIOCPP: _VULKAN}


def _row(
    name: str, targets: tuple[str, ...], vllm: Floor, comfy: Floor, note: str = ""
) -> AmdGeneration:
    llama = _BOTH if all(t in ROCM_TARGETS for t in targets) else _VULKAN_ONLY
    return AmdGeneration(
        name=name,
        targets=targets,
        engines={Engine.LLAMACPP: llama, Engine.VLLM: vllm, COMFYUI: comfy, **_REST},
        note=note,
    )


GENERATIONS: tuple[AmdGeneration, ...] = (
    _row("RDNA 4", ("gfx1200", "gfx1201"), _VLLM_PLANNED, _COMFY_ROCM),
    _row("RDNA 3", ("gfx1100", "gfx1101"), _VLLM_PLANNED, _COMFY_ROCM),
    _row("RDNA 3 (RX 7600)", ("gfx1102",), _VLLM_NO, _COMFY_ROCM),
    _row(
        "RDNA 3.5 APU",
        ("gfx1151", "gfx1150"),
        _VLLM_PLANNED,
        _COMFY_ROCM,
        note="an APU: it plans in its GTT, the machine's RAM",
    ),
    _row("RDNA 2", ("gfx1030",), _VLLM_NO, _COMFY_ROCM),
    _row(
        "RDNA 2/3, smaller chips and APUs",
        ("gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036", "gfx1103"),
        _VLLM_NO,
        _COMFY_NO,
    ),
    _row("GCN 5 (MI50, MI60, Radeon VII)", ("gfx906", "gfx900"), _VLLM_NO, _COMFY_NO),
)
CDNA = ("gfx908", "gfx90a", "gfx942", "gfx950")
OUT_OF_SCOPE = (
    "CDNA (Instinct MI100 to MI350) is outside mnm's scope: its full ROCm stack is neither "
    "planned nor tested; the Vulkan images run it as any card"
)
_CDNA = AmdGeneration(
    name="CDNA",
    targets=CDNA,
    engines={
        Engine.LLAMACPP: Floor(runs=Runs.WITHOUT, note=OUT_OF_SCOPE),
        Engine.VLLM: Floor(runs=Runs.UNKNOWN, note=OUT_OF_SCOPE),
        COMFYUI: Floor(runs=Runs.UNKNOWN, note=OUT_OF_SCOPE),
        **_REST,
    },
    note=OUT_OF_SCOPE,
)
_UNKNOWN = Floor(runs=Runs.UNKNOWN, note="its gfx target was not read")


def generation(gfx: str | None) -> AmdGeneration | None:
    """The row for a gfx target; None when it was not read or mnm has no row for it."""
    if not gfx:
        return None
    return next((g for g in (*GENERATIONS, _CDNA) if gfx in g.targets), None)


def floor(gfx: str | None, engine: str) -> Floor:
    found = generation(gfx)
    return found.engines[engine] if found is not None else _UNKNOWN


def rocm_runs(gfx: str | None) -> bool:
    """llama.cpp's ROCm image is built for this target (`MNM_BACKEND=rocm` may choose it)."""
    return gfx is not None and gfx in ROCM_TARGETS


def generation_words(gfx: str | None) -> str:
    """`gfx1201 (RDNA 4)`; `gfx not read`."""
    if not gfx:
        return "gfx not read"
    found = generation(gfx)
    return f"{gfx} ({found.name})" if found is not None else gfx
