"""What `mnm comfyui install` checks before a build of several GB that takes 10-30 minutes,
on any machine: an NVIDIA build needs driver 580 or newer and a Turing or newer
card (PyTorch's CUDA 13 wheels carry kernels for compute capability 7.5 and up only), and
every build needs Docker answering, BuildKit and room where Docker keeps its images. A
check that cannot be read warns; only a definite "no" refuses, with what to do instead. The
doctor reads the same checks once ComfyUI is installed, so a driver or card changed later
is said too."""

from __future__ import annotations

from makenomistakes.comfy.engine import IMAGE_GIB, Variant
from makenomistakes.docker_disk import FREE_SOME, SPARE_GIB, image_room
from makenomistakes.doctor_types import Check, Status
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import CommandRunner, Gpu, ProbeError, chosen_card
from makenomistakes.setup import current_user
from makenomistakes.setup_docker import NOT_INSTALLED, Docker

CUDA13_DRIVER_MAJOR = 580
MIN_COMPUTE = (7, 5)  # Turing (GTX 16xx, RTX 20xx) and newer
CPU_INSTEAD = "`mnm comfyui install --cpu` builds it for the CPU instead (slow, but it runs)."


def preflight(variant: Variant, gpus: list[Gpu] | ProbeError, runner: CommandRunner) -> list[Check]:
    """Every check for a build of `variant` on this machine. Docker that does not answer
    refuses first (else the build would begin, and fail on Docker's own error); its disk
    cannot be read then, so it is not asked."""
    daemon = daemon_check(runner)
    room = [disk_check(variant, runner)] if daemon.status is Status.OK else []
    checks = [daemon, buildkit_check(runner), *room]
    if variant is Variant.CUDA:
        checks = [driver_check(gpus), compute_check(runner), *checks]
    return checks


def refusals(checks: list[Check]) -> list[str]:
    """The definite "no"s, each with its remedy, worded for a refusal."""
    return [f"{c.name}: {c.detail}. {c.remedy}".strip() for c in checks if c.status is Status.FAIL]


def warnings(checks: list[Check]) -> list[str]:
    """What could not be read, said before the build (a "no" is said once, in the refusal)."""
    return [f"{c.name}: {c.detail}" for c in checks if c.status is Status.WARN]


def driver_check(gpus: list[Gpu] | ProbeError) -> Check:
    name = "Driver for ComfyUI"
    if isinstance(gpus, ProbeError) or not gpus:
        return Check(
            name=name, status=Status.WARN, detail="no NVIDIA card is visible", remedy=CPU_INSTEAD
        )
    version = gpus[0].driver_version or ""
    head = version.split(".", 1)[0]
    if head.isdigit() and int(head) < CUDA13_DRIVER_MAJOR:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"driver {version}",
            remedy=f"ComfyUI's PyTorch needs driver {CUDA13_DRIVER_MAJOR} or newer. "
            f"{platforms.current().driver_update}",
        )
    return Check(name=name, status=Status.OK, detail=f"driver {version or '?'}")


def compute_check(runner: CommandRunner) -> Check:
    name = "Card generation for ComfyUI"
    code, out, _err = runner.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], timeout=15
    )
    lines = out.strip().splitlines() if code == 0 else []
    try:
        card: int | None = chosen_card()  # MNM_GPU: the card ComfyUI is given
    except ValueError:
        card = None  # not a card number: nothing to read (the doctor's GPU line says why)
    first = lines[card].strip() if card is not None and card < len(lines) else ""
    major, _dot, minor = first.partition(".")
    if not (major.isdigit() and minor.isdigit()):
        return Check(
            name=name,
            status=Status.WARN,
            detail="nvidia-smi did not say the card's compute capability",
            remedy="ComfyUI's PyTorch needs a Turing (RTX 20xx, GTX 16xx) or newer card.",
        )
    if (int(major), int(minor)) < MIN_COMPUTE:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"compute capability {first}: older than Turing",
            remedy=f"PyTorch's CUDA 13 wheels have no kernels for it. {CPU_INSTEAD}",
        )
    return Check(name=name, status=Status.OK, detail=f"compute capability {first}")


def daemon_check(runner: CommandRunner, docker: Docker | None = None) -> Check:
    """The build and every start talk to Docker's daemon as this user. Read through setup's
    own check (`Docker.daemon`), so its cases are worded the same: not installed, not
    running, or running but not yet usable by this shell (a group added, a re-login
    pending), each a definite "no" with its own remedy."""
    name = "Docker"
    docker = docker or Docker(runner, lambda _p: "", current_user())
    up, detail = docker.daemon()
    if up and not docker.relogin_pending():
        return Check(name=name, status=Status.OK, detail=detail)
    if up:  # running, and this shell may not use it yet: the build would be refused
        member, _ = docker.group_membership()
        remedy = (
            "The build talks to Docker as you: once `docker info` works without sudo, "
            "install again."
            if member
            else "`mnm setup` adds you to the docker group; then install again."
        )
        said = detail.removesuffix(" (next step)")  # setup's own step order, not ours
        return Check(name=name, status=Status.FAIL, detail=said, remedy=remedy)
    if detail == NOT_INSTALLED:
        return Check(
            name=name, status=Status.FAIL, detail=detail, remedy="`mnm setup` installs it."
        )
    return Check(
        name=name,
        status=Status.FAIL,
        detail=f"does not answer ({detail})",
        remedy="Start Docker (`sudo systemctl start docker`), then install again.",
    )


def buildkit_check(runner: CommandRunner) -> Check:
    """The Dockerfile mounts a proxy's certificate as a build secret, which needs BuildKit
    (`docker buildx`): Docker's own repository installs it; Debian's `docker.io` alone may not."""
    name = "Docker BuildKit"
    code, out, _err = runner.run(["docker", "buildx", "version"], timeout=30)
    if code == 0:
        return Check(name=name, status=Status.OK, detail=out.strip().split("\n")[0][:80])
    return Check(
        name=name,
        status=Status.FAIL,
        detail="`docker buildx` is not installed",
        remedy="Install the `docker-buildx-plugin` package (`mnm setup` adds Docker's own "
        "repository), then install again.",
    )


def disk_check(variant: Variant, runner: CommandRunner) -> Check:
    name = "Disk for the ComfyUI image"
    room = image_room(runner)
    need = IMAGE_GIB[variant.value] + SPARE_GIB
    if room.free_gib is None:
        return Check(
            name=name,
            status=Status.WARN,
            detail="could not read the free space where Docker keeps its images",
            remedy=f"The build needs about {need:.0f} GB there.",
        )
    if room.short_of(IMAGE_GIB[variant.value]) is not None:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"{room.free_gib:.1f} GiB free in {room.root}, the build needs about {need:.0f}",
            remedy=f"{FREE_SOME}, then install again.",
        )
    return Check(name=name, status=Status.OK, detail=f"{room.free_gib:.0f} GiB free in {room.root}")
