"""The studio's trainer image: `mnm/studio-trainer`, Applio at a pinned release with PyTorch's
wheels for this box (`cpu`, or `cu130` for an NVIDIA card with driver 580+), built on the box as
the studio tools are (`built_image.BuiltImage`) and recorded in `<state>/studio/trainer.json`.

A training runs as one container named `mnm-train-<job>` over one job folder (the dataset
mnm copied in, Applio's work, the result): `--network none --read-only`, the user's uid, no
capabilities, `--init` so a `docker stop` reaches Applio's scripts, the card by `MNM_GPU` on a
CUDA build. It is started detached and waited for, so the router can stop it the moment a
request needs the card (`train_runner.py`) and start it again later: Applio resumes from its
latest checkpoint in the job folder.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.comfy import build_stream
from makenomistakes.comfy.image import build_options
from makenomistakes.config import Settings
from makenomistakes.engine_args_music import LIBRARY_NAME
from makenomistakes.hardware.probe import CommandRunner, LineRunner
from makenomistakes.studio.built_image import (
    BuiltImage,
    ImageError,
    ImageMissingError,
    ImageSpec,
    sandbox,
)

DOCKER_DIR = Path(__file__).parent / "trainer_docker"
IMAGE_REPO = "mnm/studio-trainer"
CONTAINER_PREFIX = "mnm-train-"
APPLIO_REF = "3.7.1"
APPLIO_COMMIT = "6f69726b140c615c8cc6831a7db8b7c5e48ba8af"  # read 2026-11-01 (git ls-remote)
HUB_REVISION = "3ff370b7023086f57e200ef86a9a251ddf4da472"  # IAHispano/Applio, 2026-10-26
BUILD_SECONDS = 5400.0
TMP_SIZE = "4g"  # numba's and matplotlib's caches, the scripts' scratch; the root is read-only
SHM_SIZE = "2g"  # Applio's data loader's workers share batches through /dev/shm
STOP_SECONDS = 30  # what `docker stop` gives Applio before it is killed
WAIT_SECONDS = 7 * 24 * 3600.0  # a training on the CPU may take days; a cut ends it sooner
BUILT_FROM = (
    "Dockerfile",
    "requirements-cpu-amd64.txt",
    "requirements-cpu-arm64.txt",
    "requirements-cu130-amd64.txt",
    "requirements-cu130-arm64.txt",
    "weights.sha256",
    "mnm_train.py",
)
# What the image carries, as the user reads it (`mnm studio trainer status`, System → Studio)
CARRIES = (
    "Applio 3.7.1 (IAHispano, MIT): RVC voices trained from your recordings",
    "Applio's pretrained HiFi-GAN generators and discriminators at 32, 40 and 48 kHz (MIT)",
    "RMVPE (MIT): the recordings' pitch",
    "ContentVec (MIT): what the recordings say, the features audio.cpp's RVC reads too",
)


class Variant(StrEnum):
    CPU = "cpu"  # no NVIDIA card, or asked (`--cpu`): hours to days for a voice
    CUDA = "cu130"  # PyTorch's CUDA 13 wheels: driver 580 or newer


IMAGE_GIB = {Variant.CPU: 5.2, Variant.CUDA: 9.0}  # the CPU build measured; CUDA estimated
VARIANT_WORDS = {Variant.CPU: "the CPU", Variant.CUDA: "CUDA 13"}


class TrainerError(ImageError):
    """The trainer could not build, start or finish, worded for the user."""


class TrainerMissingError(TrainerError, ImageMissingError):
    """The trainer's image is not on this box (`mnm studio trainer install` builds it)."""


NOT_INSTALLED = (
    "The trainer is not installed: `mnm studio trainer install` builds it (about "
    f"{IMAGE_GIB[Variant.CPU]:.0f} GB on the CPU, {IMAGE_GIB[Variant.CUDA]:.0f} GB for an "
    "NVIDIA card; 10-30 minutes), or System → Studio."
)
SPEC = ImageSpec(
    repo=IMAGE_REPO,
    ledger="trainer.json",
    lock=".trainer.lock",
    label="trainer",
    possessive="The trainer's",
    install="mnm studio trainer install",
    uninstall="mnm studio trainer uninstall",
    not_installed=NOT_INSTALLED,
    gib=IMAGE_GIB[Variant.CUDA],
    minutes="10-30",
    build_seconds=BUILD_SECONDS,
    error=TrainerError,
    missing=TrainerMissingError,
)


class Job(BaseModel):
    """`/job/job.json`: what the trainer's driver is asked to train (`mnm_train.py`)."""

    name: str = Field(pattern=LIBRARY_NAME)
    sample_rate: int
    epochs: int = Field(ge=1, le=10_000)
    batch_size: int = Field(ge=1, le=64)
    save_every: int = Field(ge=1)
    cpu_cores: int = Field(ge=1)
    gpu: str  # "-" on the CPU, else the card's index inside the container ("0")


class Trained(BaseModel):
    """`/job/out/result.json`: what the driver made."""

    tool: str
    epochs: int
    steps: int
    sample_rate: int
    index: str  # ivf (kept), flat (too few recordings for an IVF index: left out), none
    dataset_seconds: float | None = None
    seconds: float


class Progress(BaseModel):
    """`/job/progress.json`, rewritten by the driver as it goes."""

    stage: str
    epoch: int | None = None  # the epoch under way
    epochs: int | None = None
    step: int | None = None
    saved: int | None = None  # the epoch the latest checkpoint was saved after
    loss: float | None = None  # a style's: the last step's
    epoch_seconds: float | None = None  # a style's: how long its last whole epoch took
    at: float

    @property
    def resumes(self) -> str:
        """Where a cut training goes on from, in words."""
        if self.saved:
            return f"from its checkpoint after epoch {self.saved}"
        return "from the beginning (no checkpoint yet)"


@dataclass(frozen=True)
class Stop:
    """A kind's word, while its container runs, that the training must stop (a style's
    probe): for good (`pause` False: the run fails with `why`), or paused (it stays queued and
    starts again `until` what it waits for holds)."""

    why: str
    pause: bool = False
    until: str | None = None


def install_variant(cpu: bool) -> Variant:
    """What `mnm studio trainer install` builds: the CPU when asked or when the box runs on
    its processor; CUDA 13 for an NVIDIA card on driver 580+; anything else refused with the
    way to train on the CPU (no ROCm, Vulkan or older-CUDA build exists)."""
    from makenomistakes.backend import Backend, BackendError  # noqa: PLC0415
    from makenomistakes.backend_now import box_backend  # noqa: PLC0415

    if cpu:
        return Variant.CPU
    try:
        backend = box_backend()
    except BackendError as exc:
        raise TrainerError(f"{exc} (`--cpu` builds the trainer for the processor).") from exc
    if backend is Backend.CUDA13:
        return Variant.CUDA
    if backend is Backend.CPU:
        return Variant.CPU
    raise TrainerError(
        f"This box runs {backend.value}: the trainer is built for CUDA 13 (an NVIDIA card on "
        "driver 580 or newer) and for the processor. `mnm studio trainer install --cpu` "
        "builds the processor's (a voice takes hours to days there)."
    )


def recipe_hash(variant: Variant) -> str:
    """What the image is built from: its files, the pins and the variant."""
    digest = hashlib.sha256(f"{APPLIO_COMMIT}\n{HUB_REVISION}\n{variant.value}".encode())
    for name in BUILT_FROM:
        digest.update((DOCKER_DIR / name).read_bytes())
    return digest.hexdigest()[:12]


def image_tag(variant: Variant) -> str:
    return f"{IMAGE_REPO}:{variant.value}-{recipe_hash(variant)}"


def container_name(job: str) -> str:
    return f"{CONTAINER_PREFIX}{job}"


def read_progress(folder: Path) -> Progress | None:
    """The job's progress as its driver last wrote it; None before it wrote any."""
    try:
        return Progress.model_validate_json((folder / "progress.json").read_text("utf-8"))
    except (OSError, ValidationError, ValueError):
        return None


def read_trained(folder: Path) -> Trained:
    try:
        return Trained.model_validate_json((folder / "out" / "result.json").read_text("utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise TrainerError(f"The training wrote no result mnm can read ({exc}).") from exc


class StudioTrainer(BuiltImage):
    """The trainer's image: its ledger, build, rollback and removal (`BuiltImage`)."""

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
            "--build-arg", f"APPLIO_REF={APPLIO_REF}",
            "--build-arg", f"APPLIO_COMMIT={APPLIO_COMMIT}",
            "--build-arg", f"HUB_REVISION={HUB_REVISION}",
            "--label", "mnm.engine=studio-trainer",
        ]  # fmt: skip
        return [*argv, *build_options(ca), str(DOCKER_DIR)]


def variant_of(image: str) -> Variant | None:
    """The build an image is, from its tag (`mnm/studio-trainer:cu130-<hash>`): what a rollback
    or a reinstall leaves in use, never a ledger field that may lag it."""
    tag = image.rpartition(":")[2]
    return next((v for v in Variant if tag.startswith(f"{v.value}-")), None)


class TrainingContainer:
    """A training's container: started detached, so `docker stop` from the router cuts it,
    waited for, and removed by `finished`."""

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    def _docker(self, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        return self.runner.run(["docker", *args], timeout=timeout)

    def run_argv(
        self, image: str, job: str, folder: Path, gpu: list[str], mounts: Sequence[str] = ()
    ) -> list[str]:
        seen = [part for mount in mounts for part in ("-v", mount)]  # read-only
        return [
            "docker", "run", "--detach", "--init", "--name", container_name(job),
            *sandbox(TMP_SIZE), "--shm-size", SHM_SIZE, "--label", "mnm.engine=studio-trainer",
            "--label", "mnm.instance=studio-trainer",  # uninstall.sh removes it with the rest
            *gpu, *seen, "-v", f"{folder}:/job:rw", image, "train",
        ]  # fmt: skip

    def start(
        self, image: str, job: str, folder: Path, gpu: list[str], mounts: Sequence[str] = ()
    ) -> None:
        """Start the training's container (an earlier one of this job removed first)."""
        self._docker(["rm", "--force", container_name(job)])
        argv = self.run_argv(image, job, folder, gpu, mounts)
        code, out, err = self.runner.run(argv, timeout=120)
        if code != 0:  # a `docker run` that timed out may have started it: never left training
            self._docker(["rm", "--force", container_name(job)])  # beside a run marked failed
            raise TrainerError(f"Docker did not start the training: {(err or out).strip()[:300]}")

    def wait(self, job: str) -> int:
        """Block until the container ends; its exit code (143 after a `docker stop`). A
        `docker wait` that outlasts `WAIT_SECONDS` (exit 124, nothing said) is the client's
        timeout, never the training's end: asked again (a CPU training may run past a week)."""
        while True:
            code, out, err = self._docker(["wait", container_name(job)], timeout=WAIT_SECONDS)
            if code != 124 or out.strip():
                break
        if code != 0 or not out.strip().lstrip("-").isdigit():
            raise TrainerError(f"Docker lost the training: {(err or out).strip()[:300]}")
        return int(out.strip())

    def stop(self, job: str) -> None:
        """Cut a running training (its checkpoints stay for the resume)."""
        self._docker(["stop", "--time", str(STOP_SECONDS), container_name(job)], timeout=90)

    def finished(self, job: str) -> str:
        """The container's last words (its JSON line), then the container removed."""
        _code, out, err = self._docker(["logs", "--tail", "40", container_name(job)])
        self._docker(["rm", "--force", container_name(job)])
        return (out or err).strip()

    def leftovers(self) -> list[str]:
        """`mnm-train-*` containers (a router stopped while one trained); TrainerError when
        Docker does not answer, so the router asks again rather than take that for none."""
        code, out, err = self._docker(
            ["ps", "--all", "--filter", f"name=^{CONTAINER_PREFIX}", "--format", "{{.Names}}"]
        )
        if code != 0:
            raise TrainerError(f"Docker did not list the trainings' containers: {err.strip()}")
        return [n for n in out.split() if n.startswith(CONTAINER_PREFIX)]


def box_trainer(settings: Settings, variant: Variant | None = None) -> StudioTrainer:
    """This box's trainer: built for `variant`, else the build its ledger's image is (an
    unreadable ledger reads as the CPU's: every use says it is unreadable)."""
    if variant is None:
        try:
            ledger = StudioTrainer(settings).ledger()
        except TrainerError:
            ledger = None
        variant = (variant_of(ledger.image) if ledger else None) or Variant.CPU
    return StudioTrainer(settings, variant)


def write_job(folder: Path, job: BaseModel) -> None:
    """`job.json`: a voice's `Job` or a style's `StyleJob`, whole."""
    (folder / "job.json").write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_job(folder: Path) -> Job:
    try:
        return Job.model_validate_json((folder / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise TrainerError(f"The training's job.json cannot be read ({exc}).") from exc
