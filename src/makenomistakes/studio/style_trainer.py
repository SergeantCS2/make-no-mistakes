"""The studio's style trainer: `mnm/studio-style-trainer`, ACE-Step 1.5's own LoRA
trainer (`acestep.training_v2`, MIT) at a pinned commit with PyTorch's wheels for this box
(`cpu`, or `cu130` for an NVIDIA card with driver 580+), built on the box as the voice trainer
is (`built_image.BuiltImage`) and recorded in `<state>/studio/style-trainer.json`.

A training runs as the voice trainer's do (`trainer.TrainingContainer`): `mnm-train-<job>`,
`--network none --read-only`, the user's uid, no capabilities, over its own job folder, with
the models folder's Hugging Face cache (`<models>/hf`, never the rest) mounted read-only for the
ACE-Step checkpoint mnm downloaded (`style_weights.py`, `train_kinds.StyleKind.mounts`). Unlike
Applio's, no weights are baked into the image.

The settings follow the card (`settings_for`): upstream's own presets by its VRAM, the "Tight"
one on a card under 16 GB (rank 32, the 8-bit AdamW, the text encoder and VAE moved to the
processor, gradient checkpointing); the first epoch of every training is the probe that
measures the card's real peak (`train_runner`), since nobody has published one.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.comfy import build_stream
from makenomistakes.comfy.image import build_options
from makenomistakes.config import Settings
from makenomistakes.engine_args_music import LIBRARY_NAME
from makenomistakes.hardware.probe import LineRunner
from makenomistakes.studio.built_image import BuiltImage, ImageSpec, docker_arch
from makenomistakes.studio.trainer import (
    TrainerError,
    TrainerMissingError,
    Variant,
    install_variant,
    variant_of,
)

DOCKER_DIR = Path(__file__).parent / "style_docker"
IMAGE_REPO = "mnm/studio-style-trainer"
ACESTEP_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"  # main, 2026-08-29 (v0.1.8 ignored
# `--precision` on CUDA); read 2026-11-02
BUILD_SECONDS = 5400.0
BUILT_FROM = (
    "Dockerfile",
    "requirements-cpu-amd64.txt",
    "requirements-cu130-amd64.txt",
    "requirements-cu130-arm64.txt",
    "mnm_style.py",
)
LOCKED = {(Variant.CPU, "amd64"), (Variant.CUDA, "amd64"), (Variant.CUDA, "arm64")}
CARRIES = (
    "ACE-Step 1.5's own trainer (ace-step, MIT; Side-Step 0.8 built in): LoRA adapters that "
    "teach a song model a style",
    "PEFT, LyCORIS, Lightning and bitsandbytes (Apache-2.0, MIT): the adapter, the loop and "
    "the 8-bit optimizer",
)
IMAGE_GIB = {Variant.CPU: 2.9, Variant.CUDA: 9.0}  # the CPU build measured; CUDA estimated
STYLE = "style"
BUILD = "build"  # a style run's ACE-Step build (base or turbo); `variant` is the image's
RESULT = "result/result.json"  # the driver's answer in the job folder, written last
TRAIN_STAGE = "train"  # `mnm_style.py`'s stage once the songs are preprocessed


class StyleTrainerError(TrainerError):
    """The style trainer could not build, start or finish, worded for the user."""


class StyleTrainerMissingError(StyleTrainerError, TrainerMissingError):
    """The style trainer's image is not on this box."""


INSTALL = "mnm studio trainer install --for styles"
NOT_INSTALLED = (
    f"The style trainer is not installed: `{INSTALL}` builds it (about "
    f"{IMAGE_GIB[Variant.CPU]:.0f} GB on the CPU, {IMAGE_GIB[Variant.CUDA]:.0f} GB for an "
    "NVIDIA card; 10-30 minutes), or System → Trainer."
)
SPEC = ImageSpec(
    repo=IMAGE_REPO,
    ledger="style-trainer.json",
    lock=".style-trainer.lock",
    label="style trainer",
    possessive="The style trainer's",
    install=INSTALL,
    uninstall="mnm studio trainer uninstall --for styles",
    not_installed=NOT_INSTALLED,
    gib=IMAGE_GIB[Variant.CUDA],
    minutes="10-30",
    build_seconds=BUILD_SECONDS,
    error=StyleTrainerError,
    missing=StyleTrainerMissingError,
)


@dataclass(frozen=True)
class Preset:
    """One of upstream's presets (`acestep/training_v2/presets/vram_*.json`) as mnm runs it."""

    name: str  # upstream's file name, said in the plan
    words: str
    rank: int
    alpha: int
    grad_accum: int
    optimizer: Literal["adamw", "adamw8bit"]
    offload_encoder: bool


TIGHT = Preset("vram_12gb", "tight (10-16 GB)", 32, 64, 4, "adamw8bit", True)
PRESETS = (  # (cards from this many GiB, preset), largest first; a card reads a little under
    # its label (a 16 GB card is 15.99 GiB to nvidia-smi)
    (23.0, Preset("vram_24gb_plus", "24 GB and up", 128, 256, 2, "adamw", False)),
    (15.0, Preset("vram_16gb", "16 GB", 64, 128, 4, "adamw", False)),
    (9.5, TIGHT),
    (0.0, Preset("vram_8gb", "8 GB", 16, 32, 8, "adamw8bit", True)),
)
CPU_PRESET = Preset("cpu", "the processor", 16, 32, 4, "adamw", False)


def preset_for(card_gib: float | None) -> Preset:
    """Upstream's preset for a card of `card_gib` GiB in all; the CPU's for none."""
    if card_gib is None:
        return CPU_PRESET
    return next(p for floor, p in PRESETS if card_gib >= floor)


class StyleJob(BaseModel):
    """`/job/job.json` for `mnm_style.py`."""

    name: str = Field(pattern=LIBRARY_NAME)
    variant: Literal["base", "turbo"]
    dit_dir: str  # the variant's checkpoint folder as mounted (`/models/hf/…`)
    vae_dir: str
    text_encoder_dir: str
    epochs: int = Field(ge=1, le=10_000)
    rank: int = Field(ge=1, le=512)
    alpha: int = Field(ge=1, le=1024)
    grad_accum: int = Field(ge=1, le=64)
    optimizer: Literal["adamw", "adamw8bit"]
    offload_encoder: bool
    checkpointing: bool = True
    lr: float = 1e-4
    max_seconds: float = Field(ge=10, le=600)
    device: Literal["cuda", "cpu"]
    precision: Literal["bf16", "fp32"] = "bf16"
    seed: int = 42


class StyleTrained(BaseModel):
    """`/job/result` and the driver's last line: what it made."""

    tool: Literal["trained"]
    epochs: int
    steps: int | None = None
    loss: float | None = None
    rank: int | None = None
    alpha: int | None = None
    variant: Literal["base", "turbo"]
    seconds: float


def style_variant(cpu: bool, arch: str | None = None) -> Variant:
    """The build for this box (the voice trainer's rule), refusing the one ACE-Step's audio
    reader has no wheel for: the CPU on arm64 (torchcodec 0.10 publishes none)."""
    variant = install_variant(cpu)
    arch = arch or docker_arch()
    if (variant, arch) not in LOCKED:
        raise StyleTrainerError(
            f"The style trainer has no {variant.value} build for {arch}: torchcodec, which "
            "ACE-Step's trainer reads songs with, publishes no wheel for it. An NVIDIA card "
            "with driver 580+ trains styles on this box."
        )
    return variant


def recipe_hash(variant: Variant) -> str:
    digest = hashlib.sha256(f"{ACESTEP_COMMIT}\n{variant.value}".encode())
    for name in BUILT_FROM:
        digest.update((DOCKER_DIR / name).read_bytes())
    return digest.hexdigest()[:12]


def image_tag(variant: Variant) -> str:
    return f"{IMAGE_REPO}:{variant.value}-{recipe_hash(variant)}"


def read_trained(folder: Path) -> StyleTrained:
    try:
        return StyleTrained.model_validate_json((folder / RESULT).read_text("utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise StyleTrainerError(f"The training wrote no result mnm can read ({exc}).") from exc


class StyleTrainer(BuiltImage):
    """The style trainer's image: its ledger, build, rollback and removal (`BuiltImage`)."""

    spec = SPEC

    def __init__(
        self,
        settings: Settings,
        variant: Variant = Variant.CPU,
        runner: LineRunner | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._variant = variant
        super().__init__(settings, runner, clock)

    def variant(self) -> str | None:
        return self._variant.value

    def variant_of(self, image: str) -> str | None:
        found = variant_of(image)
        return found.value if found is not None else None

    def image_tag(self) -> str:
        return image_tag(self._variant)

    def gib(self) -> float:
        return IMAGE_GIB[self._variant]

    def build_argv(self, tag: str, ca: Path | None = None) -> list[str]:
        argv = [
            "docker", "build", build_stream.PLAIN, "--tag", tag,
            "--build-arg", f"VARIANT={self._variant.value}",
            "--build-arg", f"ACESTEP_COMMIT={ACESTEP_COMMIT}",
            "--label", "mnm.engine=studio-style-trainer",
        ]  # fmt: skip
        return [*argv, *build_options(ca), str(DOCKER_DIR)]


def box_style_trainer(settings: Settings, variant: Variant | None = None) -> StyleTrainer:
    """This box's style trainer: built for `variant`, else the build its ledger's image is."""
    if variant is None:
        try:
            ledger = StyleTrainer(settings).ledger()
        except StyleTrainerError:
            ledger = None
        variant = (variant_of(ledger.image) if ledger else None) or Variant.CPU
    return StyleTrainer(settings, variant)
