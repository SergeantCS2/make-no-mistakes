"""Each engine's image per backend and architecture: the rows
`recipes/registry.py` holds, what `EngineSpec.image_for` answers, and the refusal for a box an
engine publishes nothing for (`NoImage`), worded per engine.

Every tag here was read with `docker buildx imagetools inspect` on 2026-10-27 (no pull)
except where a row says "planned": vLLM's ROCm image is published but never run by mnm.
A pulled tag never moves by itself; `mnm engines update` moves one.

Two choices beyond the matrix:

- **A CUDA 13 box keeps the CUDA 12 tags on amd64** where the engine's CUDA 12 build runs there:
  `server-cuda13` asks for CUDA 13.4 (`NVIDIA_REQUIRE_CUDA=cuda>=13.4`, outside the 580, 590, 595
  and 610 branches it names), which a GeForce driver of the 580 or 590 branch does not satisfy, and
  `server-cuda` carries native code for every card a 580 driver runs on amd64 (8.6/8.9/12.0 SASS,
  the rest by PTX: `hardware/cards.py`). The CUDA 13 tags are for arm64 (the DGX Spark's sm_121,
  which the CUDA 12 image runs by PTX at a quarter of the speed).
- **An AMD box on ROCm runs sd.cpp, whisper.cpp and audio.cpp from their Vulkan images**: none
  publishes a ROCm one (`master-rocm`, `main-rocm`, `full-rocm` answer 404), and the card runs
  Vulkan too. The instance's device arguments follow its image's backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from makenomistakes.backend import Backend, label
from makenomistakes.hub.facts import Engine

AMD64 = "amd64"
ARM64 = "arm64"
_ARCH = {"x86_64": AMD64, "amd64": AMD64, "aarch64": ARM64, "arm64": ARM64}
ENGINE_NAMES = {
    Engine.LLAMACPP: "llama.cpp",
    Engine.SDCPP: "stable-diffusion.cpp",
    Engine.WHISPERCPP: "whisper.cpp",
    Engine.AUDIOCPP: "audio.cpp",
    Engine.VLLM: "vLLM",
}
KINDS = {  # the subject and its verb: "on this box text-to-speech has no engine"
    Engine.LLAMACPP: "chat models have",
    Engine.SDCPP: "pictures and clips have",
    Engine.WHISPERCPP: "speech-to-text has",
    Engine.AUDIOCPP: "text-to-speech has",
    Engine.VLLM: "vLLM's models have",
}
CHECKED = "imagetools 2026-10-27"
# The day every engine's and app's arm64 platforms were read again with `imagetools`
ARCH_READ = "2026-10-28"
PLANNED = "planned, unverified"


def image_arch(machine: str) -> str:
    """`uname -m` as an image platform's architecture: `x86_64` → `amd64`."""
    return _ARCH.get(machine.lower(), machine.lower())


@dataclass(frozen=True)
class ImageRow:
    tag: str
    arches: frozenset[str] = frozenset({AMD64})
    checked: str = CHECKED  # `imagetools` on a date, or PLANNED


@dataclass(frozen=True)
class NoImage:
    """An engine with no image for this box: why, in a sentence per engine."""

    engine: Engine
    backend: Backend
    arch: str
    why: str

    @property
    def words(self) -> str:
        """`audio.cpp publishes no ROCm image: on this box text-to-speech has no engine.`"""
        return f"{self.why}: on this box {KINDS[self.engine]} no engine."


Rows = Mapping[Backend, tuple[ImageRow, ...]]


@dataclass(frozen=True)
class EngineImages:
    """One engine's rows: per backend the tags to try in order (the first carrying the box's
    arch), and for a backend it publishes nothing for, another backend whose image the cards
    run (ROCm boxes use the Vulkan images) or the sentence why not."""

    engine: Engine
    # the default tag: a plan made for another machine (`--vram`) and a box mnm has not
    # read keep it
    default: str
    rows: Rows
    instead: Mapping[Backend, Backend] = field(default_factory=dict)
    missing: Mapping[Backend, str] = field(default_factory=dict)

    def image_for(self, backend: Backend, arch: str) -> str | NoImage:
        wanted = image_arch(arch)
        used = self.instead.get(backend, backend)
        rows = self.rows.get(used, ())
        for row in rows:
            if wanted in row.arches:
                return row.tag
        name = ENGINE_NAMES[self.engine]
        if not rows:
            why = self.missing.get(backend) or f"{name} publishes no {label(backend)} image"
            return NoImage(self.engine, backend, wanted, why)
        built = sorted({a for row in rows for a in row.arches})
        elsewhere = self.arch_backends(wanted)
        other = (
            f"; its {wanted} builds are {' and '.join(label(b) for b in elsewhere)}"
            if elsewhere
            else ""
        )
        return NoImage(
            self.engine,
            backend,
            wanted,
            f"{name} has no {wanted} {label(used)} image yet (the registry, read {ARCH_READ}: "
            f"it builds {', '.join(built)}{other})",
        )

    def arch_backends(self, arch: str) -> list[Backend]:
        """The backends this engine publishes an `arch` image for (whisper.cpp's arm64 builds
        are CPU and Vulkan only)."""
        return [b for b, rows in self.rows.items() if any(arch in r.arches for r in rows)]

    def backend_of(self, tag: str) -> Backend | None:
        """The backend an image of this engine is built for; None for a tag mnm does not
        list (an image the user named by hand)."""
        for backend, rows in self.rows.items():
            if any(row.tag == tag for row in rows):
                return backend
        return None

    def runs_as(self, backend: Backend, arch: str) -> Backend:
        """The backend the image a `backend` box runs on `arch` is listed under, as
        `backend_of` names it: what a reading taken on that image says (`measurements.
        built_for`), so what a plan compares this box's readings by. A CUDA 13 box on amd64
        runs the CUDA 12 builds (`server-cuda`, `master-cuda`, `full-cuda12`: their readings
        say `cuda12`), a ROCm box audio.cpp's `full-vulkan` (`vulkan`). `backend` itself
        where it runs no image of this engine (the plan is refused there)."""
        found = self.image_for(backend, arch)
        named = self.backend_of(found) if isinstance(found, str) else None
        return named if named is not None else backend

    def tags(self) -> list[str]:
        return [row.tag for rows in self.rows.values() for row in rows]


def _row(tag: str, *arches: str, checked: str = CHECKED) -> ImageRow:
    return ImageRow(tag=tag, arches=frozenset(arches or (AMD64,)), checked=checked)


LLAMA = "ghcr.io/ggml-org/llama.cpp"
SDCPP = "ghcr.io/leejet/stable-diffusion.cpp"
WHISPER = "ghcr.io/ggml-org/whisper.cpp"
AUDIO = "ghcr.io/0xshug0/audio.cpp"
ROCM_TO_VULKAN = {Backend.ROCM: Backend.VULKAN}

IMAGES: dict[Engine, EngineImages] = {
    Engine.LLAMACPP: EngineImages(
        Engine.LLAMACPP,
        f"{LLAMA}:server-cuda",
        {
            Backend.CUDA12: (_row(f"{LLAMA}:server-cuda", AMD64, ARM64),),
            Backend.CUDA13: (
                _row(f"{LLAMA}:server-cuda13", ARM64),
                _row(f"{LLAMA}:server-cuda", AMD64),
            ),
            Backend.ROCM: (_row(f"{LLAMA}:server-rocm"),),
            Backend.VULKAN: (_row(f"{LLAMA}:server-vulkan", AMD64, ARM64),),
            Backend.CPU: (_row(f"{LLAMA}:server", AMD64, ARM64),),
        },
    ),
    Engine.SDCPP: EngineImages(
        Engine.SDCPP,
        f"{SDCPP}:master-cuda",
        {
            Backend.CUDA12: (_row(f"{SDCPP}:master-cuda"),),
            Backend.CUDA13: (
                _row(f"{SDCPP}:master-cuda"),
                _row(f"{SDCPP}:master-cuda-spark", ARM64),
            ),
            Backend.VULKAN: (_row(f"{SDCPP}:master-vulkan"),),
        },
        instead=ROCM_TO_VULKAN,
        missing={Backend.CPU: "stable-diffusion.cpp publishes no CPU image"},
    ),
    Engine.WHISPERCPP: EngineImages(
        Engine.WHISPERCPP,
        f"{WHISPER}:main-cuda",
        {
            Backend.CUDA13: (_row(f"{WHISPER}:main-cuda"),),
            Backend.VULKAN: (
                _row(f"{WHISPER}:main-vulkan"),
                _row(f"{WHISPER}:main-vulkan-arm64", ARM64),
            ),
            Backend.CPU: (_row(f"{WHISPER}:main"), _row(f"{WHISPER}:main-arm64", ARM64)),
        },
        instead=ROCM_TO_VULKAN,
        missing={
            Backend.CUDA12: "whisper.cpp's CUDA image is CUDA 13 (`main-cuda`): it needs the "
            "580 driver and a Turing or newer card",
        },
    ),
    Engine.AUDIOCPP: EngineImages(
        Engine.AUDIOCPP,
        f"{AUDIO}:full-cuda12",
        {
            Backend.CUDA12: (_row(f"{AUDIO}:full-cuda12", AMD64, ARM64),),
            Backend.CUDA13: (_row(f"{AUDIO}:full-cuda12", AMD64, ARM64),),
            Backend.VULKAN: (_row(f"{AUDIO}:full-vulkan", AMD64, ARM64),),
            Backend.CPU: (_row(f"{AUDIO}:full-cpu", AMD64, ARM64),),
        },
        instead=ROCM_TO_VULKAN,
    ),
    Engine.VLLM: EngineImages(
        Engine.VLLM,
        "vllm/vllm-openai:latest",
        {
            Backend.CUDA12: (_row("vllm/vllm-openai:latest", AMD64, ARM64),),
            Backend.CUDA13: (_row("vllm/vllm-openai:latest", AMD64, ARM64),),
            Backend.ROCM: (_row("vllm/vllm-openai-rocm:latest", checked=PLANNED),),
        },
        missing={
            Backend.CPU: "vLLM publishes no CPU image mnm runs (its models run on llama.cpp's "
            "GGUF builds here)",
            Backend.VULKAN: "vLLM has no Vulkan build (on an AMD card: `MNM_BACKEND=rocm`)",
        },
    ),
}

# llama.cpp's `full` images carry `llama-perplexity` beside `llama-server`
# (`/app/tools.sh --perplexity`), for the quality tune; the same backend and arch rules as the
# server rows (a CUDA 13 box on amd64 keeps the CUDA 12 build), labelled with the same build
# revision as the `server` tags of a release. Read with `imagetools` on 2026-11-14:
# a `full-cuda` pull beside
# `server-cuda` downloads about 1 GB, the CUDA layers being shared.
FULL_CHECKED = "imagetools 2026-11-14"
QUALITY_IMAGES = EngineImages(
    Engine.LLAMACPP,
    f"{LLAMA}:full-cuda",
    {
        Backend.CUDA12: (_row(f"{LLAMA}:full-cuda", AMD64, ARM64, checked=FULL_CHECKED),),
        Backend.CUDA13: (
            _row(f"{LLAMA}:full-cuda13", ARM64, checked=FULL_CHECKED),
            _row(f"{LLAMA}:full-cuda", AMD64, checked=FULL_CHECKED),
        ),
        Backend.ROCM: (_row(f"{LLAMA}:full-rocm", checked=FULL_CHECKED),),
        Backend.VULKAN: (_row(f"{LLAMA}:full-vulkan", AMD64, ARM64, checked=FULL_CHECKED),),
        Backend.CPU: (_row(f"{LLAMA}:full", AMD64, ARM64, checked=FULL_CHECKED),),
    },
)


def all_tags() -> set[str]:
    """Every image tag mnm may pull: each engine's per backend and llama.cpp's `full` images
    (what the uninstall's `MNM_TAGS` lists)."""
    return {tag for row in IMAGES.values() for tag in row.tags()} | set(QUALITY_IMAGES.tags())
