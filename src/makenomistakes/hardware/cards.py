"""What each NVIDIA card generation can run in mnm's engine images, and which mixes of cards
are a bad idea. Data, cited; the doctor words it.

Read on 2026-10-22:
- llama.cpp `server-cuda` is CUDA 12.8 with ggml's default architectures: SASS for 8.6/8.9/
  12.0, PTX for 5.0/6.1/7.0/7.5/8.0/9.0, so Pascal and Volta run by PTX JIT; `server-cuda13`
  has no 5.0/6.1/7.0 (.devops/cuda.Dockerfile, ggml/src/ggml-cuda/CMakeLists.txt). Without
  tensor cores (Pascal) flash attention takes its tile/vec kernels and a 6.1 card gets FP32
  cuBLAS (ggml-cuda/common.cuh `fast_fp16_available`, fattn.cu).
- vLLM needs compute capability 7.5 (docs, gpu.cuda.inc.md; CMakeLists.txt's
  CUDA_SUPPORTED_ARCHS for nvcc >= 12.8 start at 7.5); FP8 W8A8 needs Ada (8.9) or newer
  (docs.vllm.ai features/quantization); FlashAttention-2 and bf16 need 8.0 (unverified).
- ComfyUI runs on PyTorch's CUDA 13 wheels, which cover 7.5 and newer ("CUDA 13.x does not
  support compute capabilities below Turing", dev-discuss.pytorch.org, the 2.15 notice).
- whisper.cpp `main-cuda` is CUDA 13.0 built for 7.5/8.0/8.6/9.0 (.devops/main-cuda.Dockerfile).
- stable-diffusion.cpp `master-cuda` is CUDA 12.6.3 with ggml's default list (no 12.0 SASS):
  Pascal and Volta by PTX, Blackwell by PTX JIT from 9.0 (unverified)
  (docker/Dockerfile.cuda, .github/workflows/build.yml).
- audio.cpp's image was not read: unknown, never refused.
- The 580 driver branch is the last for Maxwell, Pascal and Volta (NVIDIA's deprecation
  schedule, forums.developer.nvidia.com/t/60588); 590 drops Pascal (archlinux.org news,
  2025-12-20). One such card holds the whole box at 580.

Read on 2026-10-25: the RTX 5080 is compute capability 12.0 (NVIDIA's CUDA GPUs page),
Blackwell like the 5090, 5070 Ti and PRO 6000; 16 GB of GDDR7 on a 256-bit bus, 960 GB/s
(Caseking's listing; `fit.devices` carries the bandwidth). On AM5 the CPU gives the graphics
cards x16 or x8/x8 of PCIe 5.0; a third or fourth card sits on the chipset at x4, sharing one
uplink (TechSpot's AM5 board guide). Prompt processing copies the experts kept in RAM to the
main card over its link, so the main card is the one on the widest link (`fit.cards.main_gpu`);
an x1 riser card holds layers only.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from makenomistakes.hardware.probe import Gpu
from makenomistakes.hub.facts import Engine

COMFYUI = "comfyui"  # not an `Engine`: mnm builds it
TURING = 7.5


class Runs(StrEnum):
    RUNS = "runs"
    WITHOUT = "runs, with limits"  # runs, without what `note` says
    REFUSED = "refused"
    UNKNOWN = "not known"


class Floor(BaseModel):
    runs: Runs
    note: str = ""


class Generation(BaseModel):
    name: str
    lowest: float  # compute capability, inclusive
    engines: dict[str, Floor]


_PTX = "by PTX (compiled at load), no tensor cores"
_CUDA13 = "its image is CUDA 13, which needs compute capability 7.5"
_UNKNOWN = Floor(runs=Runs.UNKNOWN, note="its image's architectures were not checked")

GENERATIONS: tuple[Generation, ...] = (
    Generation(
        name="Blackwell",
        lowest=10.0,
        engines={
            Engine.LLAMACPP: Floor(runs=Runs.RUNS),
            Engine.VLLM: Floor(runs=Runs.RUNS),
            COMFYUI: Floor(runs=Runs.RUNS),
            Engine.WHISPERCPP: Floor(runs=Runs.RUNS, note="by PTX from 9.0"),
            Engine.SDCPP: Floor(runs=Runs.RUNS, note="by PTX from 9.0 (unverified)"),
            Engine.AUDIOCPP: _UNKNOWN,
        },
    ),
    Generation(
        name="Ada or Hopper",
        lowest=8.9,
        engines={
            Engine.LLAMACPP: Floor(runs=Runs.RUNS),
            Engine.VLLM: Floor(runs=Runs.RUNS),
            COMFYUI: Floor(runs=Runs.RUNS),
            Engine.WHISPERCPP: Floor(runs=Runs.RUNS),
            Engine.SDCPP: Floor(runs=Runs.RUNS),
            Engine.AUDIOCPP: _UNKNOWN,
        },
    ),
    Generation(
        name="Ampere",
        lowest=8.0,
        engines={
            Engine.LLAMACPP: Floor(runs=Runs.RUNS),
            Engine.VLLM: Floor(runs=Runs.WITHOUT, note="FP8 weights (Ada and newer)"),
            COMFYUI: Floor(runs=Runs.RUNS),
            Engine.WHISPERCPP: Floor(runs=Runs.RUNS),
            Engine.SDCPP: Floor(runs=Runs.RUNS),
            Engine.AUDIOCPP: _UNKNOWN,
        },
    ),
    Generation(
        name="Turing",
        lowest=7.5,
        engines={
            Engine.LLAMACPP: Floor(runs=Runs.RUNS),
            Engine.VLLM: Floor(
                runs=Runs.WITHOUT, note="FP8, and FlashAttention-2 and bf16 (unverified)"
            ),
            COMFYUI: Floor(runs=Runs.RUNS),
            Engine.WHISPERCPP: Floor(runs=Runs.RUNS),
            Engine.SDCPP: Floor(runs=Runs.RUNS),
            Engine.AUDIOCPP: _UNKNOWN,
        },
    ),
    Generation(
        name="Volta",
        lowest=7.0,
        engines={
            Engine.LLAMACPP: Floor(runs=Runs.WITHOUT, note="its CUDA 12 image only, by PTX"),
            Engine.VLLM: Floor(runs=Runs.REFUSED, note="vLLM needs compute capability 7.5"),
            COMFYUI: Floor(runs=Runs.REFUSED, note=_CUDA13),
            Engine.WHISPERCPP: Floor(runs=Runs.REFUSED, note=_CUDA13),
            Engine.SDCPP: Floor(runs=Runs.WITHOUT, note="by PTX"),
            Engine.AUDIOCPP: _UNKNOWN,
        },
    ),
    Generation(
        name="Pascal",
        lowest=6.0,
        engines={
            Engine.LLAMACPP: Floor(
                runs=Runs.WITHOUT, note=f"its CUDA 12 image only, {_PTX}; FP32 matmuls on 6.1"
            ),
            Engine.VLLM: Floor(runs=Runs.REFUSED, note="vLLM needs compute capability 7.5"),
            COMFYUI: Floor(runs=Runs.REFUSED, note=_CUDA13),
            Engine.WHISPERCPP: Floor(runs=Runs.REFUSED, note=_CUDA13),
            Engine.SDCPP: Floor(runs=Runs.WITHOUT, note=_PTX),
            Engine.AUDIOCPP: _UNKNOWN,
        },
    ),
)
_OLDER = Generation(
    name="older than Pascal",
    lowest=0.0,
    engines={
        name: Floor(runs=Runs.REFUSED, note="older than Pascal: no engine image is built for it")
        for name in (*Engine, COMFYUI)
    },
)


def capability(gpu: Gpu) -> float | None:
    try:
        return float(gpu.compute_cap) if gpu.compute_cap else None
    except ValueError:
        return None


def generation(gpu: Gpu) -> Generation | None:
    """The card's row by compute capability; None when nvidia-smi did not say."""
    cc = capability(gpu)
    if cc is None:
        return None
    return next((g for g in GENERATIONS if cc >= g.lowest), _OLDER)


def floor(gpu: Gpu, engine: str) -> Floor:
    if is_amd(gpu):  # By gfx target
        from makenomistakes.hardware import amd_cards  # noqa: PLC0415 - it imports this module

        return amd_cards.floor(gpu.gfx_target, engine)
    found = generation(gpu)
    return found.engines[engine] if found is not None else _UNKNOWN


def is_amd(gpu: Gpu) -> bool:
    return gpu.is_amd


_PRO_WORDS = ("tesla", "titan", "dgx")


def is_consumer(gpu: Gpu) -> bool:
    return "geforce" in gpu.name.lower()


def is_pro(gpu: Gpu) -> bool:
    """Tesla, Titan and DGX cards: never mixed with GeForce ones."""
    lowered = gpu.name.lower()
    return any(word in lowered for word in _PRO_WORDS)


def equal(cards: list[Gpu]) -> bool:
    """Same name and VRAM: what vLLM's tensor parallel asks (each worker must report the
    same number of KV blocks, vllm/v1/core/kv_cache_utils.py)."""
    return len({(c.name, c.vram_total_bytes) for c in cards}) == 1


def mix_problems(cards: list[Gpu]) -> list[str]:
    """What is wrong with these cards together (empty for one card or a good pair)."""
    if len(cards) < 2:
        return []
    found = []
    if any(is_amd(c) for c in cards) and not all(is_amd(c) for c in cards):
        found.append(
            "NVIDIA and AMD cards together: an instance runs on one maker's cards (their "
            "images differ), so mnm never splits one across both; pin each instance to one "
            "maker's cards"
        )
    if any(is_consumer(c) for c in cards) and any(is_pro(c) for c in cards):
        found.append(
            "GeForce cards beside Tesla/Titan/DGX ones: their drivers, power and cooling "
            "differ, and mnm never plans such a mix (by design)"
        )
    old = [c for c in cards if (cc := capability(c)) is not None and cc < TURING]
    if old and len(old) < len(cards):
        names = ", ".join(c.name for c in old)
        found.append(
            f"{names} (older than Turing) beside newer cards: it runs by PTX, no tensor "
            "cores, CUDA 13 images refuse it (vLLM, ComfyUI, whisper.cpp), and a layer split "
            "goes at its pace; it also holds the whole box at driver 580"
        )
    return found


def card_line(gpu: Gpu) -> str:
    """`0: NVIDIA GeForce RTX 4070, 12.0 GiB, compute 8.9 (Ada or Hopper)`, and `, x16 PCIe
    4.0` when nvidia-smi read the link."""
    link = link_words(gpu)
    lanes = f", {link}" if link else ""  # The PCIe link, when nvidia-smi says
    if is_amd(gpu):  # The gfx target, and an APU's GTT
        from makenomistakes.hardware.amd_cards import generation_words  # noqa: PLC0415

        size = f"{gpu.vram_total_bytes / 2**30:.1f} GiB" + (" of GTT" if gpu.carve_out else "")
        if not gpu.carve_out and gpu.lanes is None:
            lanes = ", link unknown"
        return f"{gpu.index}: {gpu.name}, {size}, {generation_words(gpu.gfx_target)}{lanes}"
    row = generation(gpu)
    cc = f"compute {gpu.compute_cap} ({row.name})" if row else "compute capability unknown"
    return f"{gpu.index}: {gpu.name}, {gpu.vram_total_bytes / 2**30:.1f} GiB, {cc}{lanes}"


LABELS = {
    Engine.LLAMACPP: "llama.cpp",
    Engine.VLLM: "vLLM",
    COMFYUI: "ComfyUI",
    Engine.WHISPERCPP: "whisper.cpp",
    Engine.SDCPP: "sd.cpp",
    Engine.AUDIOCPP: "audio.cpp",
}


def refusals(gpu: Gpu) -> list[str]:
    """`vLLM refused (…)` and `llama.cpp runs, with limits (…)` for what this card cannot run
    well; nothing about an engine it runs."""
    if is_amd(gpu):
        from makenomistakes.hardware.amd_cards import generation as amd_generation  # noqa: PLC0415

        found = amd_generation(gpu.gfx_target)
        engines = found.engines if found is not None else {}
        return [
            f"{LABELS[engine]} {floor.runs.value} ({floor.note})"
            for engine, floor in engines.items()
            if floor.runs in (Runs.REFUSED, Runs.WITHOUT)
        ]
    row = generation(gpu)
    if row is None:
        return []
    return [
        f"{LABELS[engine]} {found.runs.value} ({found.note})"
        for engine, found in row.engines.items()
        if found.runs in (Runs.REFUSED, Runs.WITHOUT)
    ]


RISER_LANES = 1  # a card on one lane: layers only, never experts
CHIPSET_LANES = 4  # a card at x4 (on AM5: the chipset's shared uplink)


def link_words(gpu: Gpu) -> str | None:
    """`x16 PCIe 5.0` (`, can do x16` when it runs narrower than it can); None when
    nvidia-smi could not say (some read `[N/A]` under WSL)."""
    if gpu.lanes is None:
        return None
    gen = f" PCIe {gpu.pcie_gen}.0" if gpu.pcie_gen else ""
    more = f", can do x{gpu.lanes_max}" if gpu.lanes_max and gpu.lanes_max > gpu.lanes else ""
    return f"x{gpu.lanes}{gen}{more}"
