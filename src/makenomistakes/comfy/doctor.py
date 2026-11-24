"""The doctor's ComfyUI lines: only once it is installed, so a box without it
reads as before. The image is there, the driver is new enough for its CUDA 13 PyTorch, the
memory is enough for the big video models, its network has no way out, and what on this
machine its container could still open."""

from __future__ import annotations

from makenomistakes.backend import Backend, BackendError
from makenomistakes.backend import words as backend_words
from makenomistakes.backend_now import box_backend, box_now
from makenomistakes.comfy import reach
from makenomistakes.comfy.engine import NETWORK, ComfyEngine
from makenomistakes.comfy.image import (
    COMFYUI_VERSION,
    CUDA_VARIANTS,
    VARIANT_WORDS,
    ComfyError,
    Ledger,
    Variant,
    image_tag,
)
from makenomistakes.comfy.preflight import compute_check, driver_check
from makenomistakes.comfy.variants import variant_for
from makenomistakes.doctor_types import Check, Status
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import CommandRunner, HardwareReport

GIB = 2**30
# What Wan 2.2 14B workflows want: they stream about 38 GB of files through RAM (and swap).
# `memory=24GB` in .wslconfig reads a little under 24 GiB: a box set so read 24036 MB
# (23.47 GiB), which a bar of 23.5 would have warned.
VIDEO_RAM_GIB = 23.0
STREAMED_GB = 38  # the files a Wan 2.2 14B workflow streams through RAM


def comfy_checks(engine: ComfyEngine, report: HardwareReport, runner: CommandRunner) -> list[Check]:
    try:
        ledger = engine.ledger()
    except ComfyError as exc:
        return [Check(name="ComfyUI", status=Status.FAIL, detail=str(exc))]
    if ledger is None:
        return []
    checks = [
        _image(ledger.image, runner),
        _pin(ledger),
        _variant(ledger, report),
        _network(runner),
        reach.check(engine.settings, runner),
        _memory(report),
    ]
    if ledger.variant in CUDA_VARIANTS:  # a driver or card changed since the build is said
        checks += [driver_check(report.gpus), compute_check(runner)]
    return checks


def _pin(ledger: Ledger) -> Check:
    """After `mnm update` moved ComfyUI's pin (or a rollback), the image is not this mnm's."""
    wanted = image_tag(ledger.variant)
    if ledger.image == wanted:
        return Check(name="ComfyUI version", status=Status.OK, detail=COMFYUI_VERSION)
    return Check(
        name="ComfyUI version",
        status=Status.WARN,
        detail=f"running {ledger.image}; this mnm builds {wanted}",
        remedy="`mnm comfyui install` builds it (the image before stays for rollback), then "
        "`mnm comfyui packs install` builds the approved packs over it.",
    )


def _variant(ledger: Ledger, report: HardwareReport) -> Check:
    """A card added after a CPU build, or a CUDA build on a box whose card is gone; or
    a build for another backend than the box runs now."""
    card = isinstance(report.gpus, list) and bool(report.gpus)
    name = "ComfyUI build"
    moved = _backend_moved(ledger)
    if moved is not None:
        return moved
    if ledger.variant is Variant.CPU and card:
        return Check(
            name=name,
            status=Status.WARN,
            detail="built for the CPU, and an NVIDIA card is visible now",
            remedy="`mnm comfyui install` builds it for the card.",
        )
    if ledger.variant is Variant.CUDA and not card:
        return Check(
            name=name,
            status=Status.WARN,
            detail="built for an NVIDIA card, and none is visible",
            remedy="`mnm comfyui install --cpu` builds it for the CPU.",
        )
    return Check(name=name, status=Status.OK, detail=ledger.variant.value)


def _backend_moved(ledger: Ledger) -> Check | None:
    """`built for CUDA 13, this box runs Vulkan`: the build's variant is not the one the box's
    backend chooses now; None when it is, or before any box was read."""
    box = box_now()
    if box is None:
        return None
    try:
        backend = box_backend()
    except BackendError:
        return None  # the Machine line says why MNM_BACKEND is refused
    wanted = variant_for(backend, box)
    if wanted is ledger.variant or (ledger.variant is Variant.CPU and backend is Backend.CPU):
        return None
    return Check(
        name="ComfyUI build",
        status=Status.WARN,
        detail=f"built for {VARIANT_WORDS[ledger.variant]}, this box runs "
        f"{backend_words(backend, box)}",
        remedy="`mnm comfyui install` builds it for this box (the image before stays for "
        "rollback).",
    )


def _image(image: str, runner: CommandRunner) -> Check:
    code, _out, _err = runner.run(["docker", "image", "inspect", image], timeout=30)
    if code == 0:
        return Check(name="ComfyUI image", status=Status.OK, detail=image)
    return Check(
        name="ComfyUI image",
        status=Status.WARN,
        detail=f"{image} is not in Docker",
        remedy="`mnm comfyui install` builds it again (`mnm comfyui uninstall` if you are done).",
    )


def _network(runner: CommandRunner) -> Check:
    name = "ComfyUI network"
    code, out, _err = runner.run(
        ["docker", "network", "inspect", "--format", "{{.Internal}}", NETWORK], timeout=30
    )
    if code != 0:
        return Check(name=name, status=Status.OK, detail="made at its first start")
    if out.strip() == "true":
        return Check(name=name, status=Status.OK, detail=f"{NETWORK}: no internet while it runs")
    return Check(
        name=name,
        status=Status.WARN,
        detail=f"{NETWORK} is not an internal network: node packs could reach the internet",
        remedy=f"`mnm comfyui stop`, `docker network rm {NETWORK}`, then `mnm comfyui start`.",
    )


def _memory(report: HardwareReport) -> Check:
    name = "RAM for video workflows"
    if report.memory is None:
        return Check(name=name, status=Status.WARN, detail="could not read /proc/meminfo")
    total = report.memory.total_bytes / GIB
    want = f"{VIDEO_RAM_GIB:.0f} GiB is what Wan 2.2 14B workflows want"
    if total >= VIDEO_RAM_GIB:
        return Check(name=name, status=Status.OK, detail=f"{total:.1f} GiB; {want}")
    detail = (
        f"{total:.1f} GiB, short of it: {want} (they stream about {STREAMED_GB} GB of model "
        "files through RAM beside ComfyUI)"
    )
    if platforms.current().is_wsl:
        remedy = wsl_remedy(report.host_ram_bytes)
    else:
        remedy = "Add RAM or a swap file for Wan 2.2 14B workflows; 5B and pictures run as it is."
    return Check(name=name, status=Status.WARN, detail=detail, remedy=remedy)


def wsl_remedy(host: int | None) -> str:
    """`.wslconfig`'s lines for the video workflows: `memory=` at most 24GB and never more
    than this PC has (a 16 GB PC is never told 24GB), swap for the rest of the
    files they stream."""
    then = "under [wsl2] in %UserProfile%\\.wslconfig, then `wsl --shutdown`."
    rest = " Wan 2.2 5B and picture workflows run as it is."
    if host is None:  # Windows not asked (interop off): the limit said in words
        return (
            f"In Windows, set `memory=24GB` (or all this PC has, if it has less) and "
            f"`swap=16GB` {then}{rest}"
        )
    memory = min(24, host // GIB)
    swap = max(16, STREAMED_GB + 2 - memory)
    whole = " (all this PC has)" if memory < 24 else ""
    return f"In Windows, set `memory={memory}GB`{whole} and `swap={swap}GB` {then}{rest}"
