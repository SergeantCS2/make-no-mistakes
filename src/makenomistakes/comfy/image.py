"""The ComfyUI image mnm builds and the ledger that records it: what it is built
from (the pin, the Dockerfile, the constraints), its tag, the proxy options for a build behind
one, and `<state>/comfyui/engine.json` with the image before it and the packs layer over it.

The ledger carries a schema version. A file this version cannot read, or one a newer mnm wrote,
is a ComfyError naming it, never "not installed": reading it as absent would hide a running
container from the router (a model would load over ComfyUI's VRAM) and drop the image to roll
back to at the next install.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes import comfyui
from makenomistakes.comfy.models import MODEL_FOLDERS
from makenomistakes.comfy.packs import VETTED

DOCKER_DIR = Path(__file__).parent / "docker"
COMFYUI_VERSION = "v0.42.1"  # the release mnm's workflow reading was checked against
COMFYUI_COMMIT = "63be2f9b82ad0dac88a7ceaa28a9cf3154125a18"
IMAGE_REPO = "mnm/comfyui"
# the built image, roughly (torch is most of it; the ROCm wheels are the largest)
IMAGE_GIB = {"cu130": 9.0, "cu126": 9.0, "rocm7.2": 14.0, "cpu": 2.0}
LEDGER_SCHEMA = 1
TORCH_PACKAGES = ("torch", "torchvision", "torchaudio", "numpy", "comfyui-frontend-package")


class ComfyError(RuntimeError):
    """A ComfyUI engine problem worded for the user."""


class ComfyBusyError(ComfyError):
    """ComfyUI is still running a workflow: a model loaded now would share the card with it,
    run out of memory and have auto-fix save smaller settings for good, so it waits."""


class Variant(StrEnum):
    CUDA = "cu130"  # PyTorch's CUDA 13 wheels: driver 580 or newer, Turing or newer
    # PyTorch's CUDA 12.6 wheels for Pascal and Volta (P100, P40, V100) and a driver
    # below 580: the same torch 2.14.1 the constraints pin, whose cu126 x86_64 build carries
    # 5.0, 6.0, 7.0, 7.5, 8.0, 8.6 and 9.0 code (PyTorch's `.ci/manywheel/build_env_setup.py`
    # at v2.14.1, `TORCH_CUDA_ARCH_LIST_TABLE["12.6"]`; cu130's starts at 7.5), started with
    # `--disable-cuda-malloc` (cudaMallocAsync is not there for them: ComfyUI's startup flags,
    # docs.comfy.org; its hardware page sends GTX 10-series cards to cu126)
    CUDA_LEGACY = "cu126"
    ROCM = "rocm7.2"  # PyTorch's ROCm 7.2 wheels (gfx1100-1201, gfx1150/1151), planned
    CPU = "cpu"  # no NVIDIA card (tests, a box without one): slow, but every node loads


CUDA_VARIANTS = frozenset({Variant.CUDA, Variant.CUDA_LEGACY})
TORCH_INDEX = {
    Variant.CUDA: "https://download.pytorch.org/whl/cu130",
    Variant.CUDA_LEGACY: "https://download.pytorch.org/whl/cu126",
    Variant.ROCM: "https://download.pytorch.org/whl/rocm7.2",
    Variant.CPU: "https://download.pytorch.org/whl/cpu",
}
# What each variant is for, as the doctor and `mnm comfyui status` say it.
VARIANT_WORDS = {
    Variant.CUDA: "CUDA 13",
    Variant.CUDA_LEGACY: "CUDA 12.6 (cards older than Turing, or a driver below 580)",
    Variant.ROCM: "ROCm",
    Variant.CPU: "the CPU",
}


class Ledger(BaseModel):
    """`<state>/comfyui/engine.json`: the image in use, the one before it, and where the
    running container answers."""

    schema_version: int = LEDGER_SCHEMA
    image: str
    variant: Variant
    previous: str | None = None
    built_at: datetime
    address: str | None = None
    packs_image: str | None = None  # the approved node packs over `image`
    packs: list[str] = Field(default_factory=list)  # `pack@version` in that layer
    packs_previous: str | None = None
    previous_packs: list[str] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)  # what the build resolved (torch …)

    @property
    def running_image(self) -> str:
        return self.packs_image or self.image


def read_ledger(path: Path) -> Ledger | None:
    """The ledger; None when there is none (not installed). Unreadable or newer: ComfyError."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ComfyError(f"{path} cannot be read: {exc}") from exc
    try:
        ledger = Ledger.model_validate_json(text)
    except ValidationError as exc:
        raise ComfyError(
            f"{path} cannot be read ({exc.errors()[0]['msg']}): `mnm comfyui install` "
            "builds the image again and writes it anew."
        ) from None
    if ledger.schema_version > LEDGER_SCHEMA:
        raise ComfyError(f"{path} is from a newer mnm: run `mnm update`.")
    return ledger


def recipe_hash() -> str:
    """What the image is built from: the Dockerfile, the constraints and the pin."""
    digest = hashlib.sha256(COMFYUI_COMMIT.encode())
    for name in ("Dockerfile", "constraints.txt"):
        digest.update((DOCKER_DIR / name).read_bytes())
    return digest.hexdigest()[:10]


def image_tag(variant: Variant) -> str:
    return f"{IMAGE_REPO}:{COMFYUI_VERSION}-{variant.value}-{recipe_hash()}"


def build_options(ca: Path | None = None) -> list[str]:
    """A box behind a proxy builds through it (and trusts its certificate, as a secret)."""
    argv: list[str] = []
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        if os.environ.get(name):
            argv += ["--network", "host"] if "--network" not in argv else []
            argv += ["--build-arg", f"{name}={os.environ[name]}"]
    if ca is not None:
        argv += ["--secret", f"id=ca,src={ca}"]
    return argv


def resolved_versions(freeze: str) -> dict[str, str]:
    """`pip freeze` of a built image, cut to what decides how it runs (torch and friends)."""
    found = {}
    for line in freeze.splitlines():
        name, sep, version = line.strip().partition("==")
        if sep and name.lower() in TORCH_PACKAGES:
            found[name.lower()] = version
    return found


def extra_kinds(layout: comfyui.Layout) -> tuple[str, ...]:
    """Model folders beyond `comfyui.KINDS` that ComfyUI must be told about: the ones workflows'
    files were linked into (`vae_approx`) and the vetted packs' own (`mmaudio`)."""
    kinds = {kind for kind in layout.links if "/" not in kind and kind in MODEL_FOLDERS}
    kinds |= {folder for row in VETTED for folder in row.model_folders}
    return tuple(sorted(kinds - set(comfyui.KINDS)))
