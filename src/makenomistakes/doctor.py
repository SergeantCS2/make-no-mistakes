"""Turn a HardwareReport into human-readable checks with remedies."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from makenomistakes.config import is_system_folder
from makenomistakes.doctor_box import box_checks
from makenomistakes.doctor_cards import card_checks
from makenomistakes.doctor_docker import docker_check, in_docker_group, relogin_pending
from makenomistakes.doctor_lanes import lane_checks
from makenomistakes.doctor_network import (
    api_key_check,
    firewall_check,
    networking_check,
    on_the_lan,
    router_host_check,
)
from makenomistakes.doctor_types import GPU_ROW, RUNTIME_ROW, SMOKE_ROW, VRAM_ROW, Check, Status
from makenomistakes.fit.budget import BUSY_CARD_BYTES
from makenomistakes.fit.devices import device_for
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.box_facts import BoxFacts
from makenomistakes.hardware.probe import (
    GPU_SMOKE_IMAGE,
    DockerInfo,
    Gpu,
    HardwareReport,
    ProbeError,
    SmokeResult,
    chosen_cards,
)

__all__ = ["Check", "Status", "run_checks", "worst"]  # Check and Status live in doctor_types

GIB = 1024**3
MIN_WSL_RAM_GIB = 16
# `memory=16GB` in .wslconfig shows as about 15.6 GiB MemTotal (the kernel keeps some).
RAM_WARN_BELOW_GIB = 15.0
MIN_FREE_DISK_GIB = 50
DESKTOP_VRAM_WARN_GIB = BUSY_CARD_BYTES / GIB  # the line above which a plan says "busy"


def run_checks(  # noqa: PLR0913, PLR0917 - what mnm knows, beside the report
    report: HardwareReport,
    hf_token_present: bool,
    instances: dict[str, int] | None = None,
    router: str | None = None,
    key_set: bool = False,
    key_differs: bool = False,
    router_host: str = "0.0.0.0",
    router_port: int = 7801,
    ufw_rule_noted: bool = False,
    router_host_running: str | None = None,
    in_group: Callable[[], bool] = in_docker_group,
    box: BoxFacts | None = None,
    gpu_set: bool = True,
    health: str | None = None,
) -> list[Check]:
    """`instances`: running instance → its estimated VRAM, so their share is not blamed on
    Windows (`InstanceService.running_vram`). `router`: None when the router answers, else
    what to do about it (`router_client.not_answering_note`). `key_set`: `MNM_API_KEY` is in
    the secrets file (what the router's service reads); `key_differs`: this shell's own
    differs from it. `router_host`: where it listens (`MNM_ROUTER_HOST`, as the secrets file
    says); `router_host_running`: as the installed router unit says, which is what runs.
    `in_group`: the login user is in the docker group by the group file. `box`:
    the CPU and RAM slots (`hardware.box_facts`); `gpu_set`, `MNM_GPU` is set (a box of
    several cards without it is told to set it). `health`: what
    the router's `/__mnm/health` says, in words."""
    host = router_host_running or router_host
    checks = [
        _gpu_check(report),
        *card_checks(report),
        *lane_checks(report, gpu_set),
        _vram_headroom_check(report, instances or {}),
        _memory_check(report),
        *box_checks(box),
        networking_check(report, host, router_port),
        docker_check(report.docker, in_group),
        _docker_gpu_runtime_check(report.docker, relogin_pending(report.docker, in_group)),
        _disk_check(report),
        _hf_token_check(hf_token_present),
        _router_check(router, health),
        api_key_check(key_set, on_the_lan(report, host), key_differs),
    ]
    if (drift := router_host_check(router_host_running, router_host)) is not None:
        checks.append(drift)
    if (firewall := firewall_check(report, router_port, ufw_rule_noted)) is not None:
        checks.append(firewall)
    if report.docker_gpu_smoke is not None:
        checks.append(_docker_gpu_smoke_check(report.docker_gpu_smoke))
    return checks


def worst(checks: list[Check]) -> Status:
    if any(c.status is Status.FAIL for c in checks):
        return Status.FAIL
    if any(c.status is Status.WARN for c in checks):
        return Status.WARN
    return Status.OK


# whisper.cpp's main-cuda image is built on CUDA 13.0, which needs an R580+ driver; the
# llama.cpp (12.8.1) and stable-diffusion.cpp (12.6.3) images run on older ones.
CUDA13_DRIVER_MAJOR = 580


def _router_check(problem: str | None, health: str | None = None) -> Check:
    """Every client URL goes through the router: down, no model answers."""
    if problem is None:
        detail = f"answers; {health}" if health else "answers"
        return Check(name="Router", status=Status.OK, detail=detail)
    return Check(name="Router", status=Status.WARN, detail="does not answer", remedy=problem)


def _gpu_check(report: HardwareReport) -> Check:
    name = GPU_ROW
    if isinstance(report.gpus, ProbeError) or not report.gpus:
        detail = report.gpus.message if isinstance(report.gpus, ProbeError) else "no GPU listed"
        return Check(
            name=name,
            status=Status.FAIL,
            detail=detail,
            remedy=platforms.current().no_driver,
        )
    gpu = report.gpus[0]
    detail = (
        f"{gpu.name}, {gpu.vram_total_bytes / GIB:.1f} GiB VRAM, driver {gpu.driver_version or '?'}"
    )
    if gpu.unified:  # A DGX Spark's GB10 answers [N/A] for its memory
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{gpu.name}, unified memory (read as RAM), driver {gpu.driver_version or '?'}",
            remedy="Plans count the pool once (MemAvailable less 8 GiB for the system), "
            "from vendor documentation until a box like this has measured itself.",
        )
    if len(report.gpus) > 1:
        detail += f" (+{len(report.gpus) - 1} more)"
    detail += f", priced: {device_for(gpu).source_word}"  # Its device row's basis
    major = _driver_major(gpu.driver_version)
    if major is not None and major < CUDA13_DRIVER_MAJOR:
        return Check(
            name=name,
            status=Status.WARN,
            detail=detail,
            remedy=(
                f"Speech-to-text needs NVIDIA driver {CUDA13_DRIVER_MAJOR} or newer (whisper.cpp's "
                "image is built on CUDA 13); chat, embeddings and images (CUDA 12) work as they "
                f"are. {platforms.current().driver_update}"
            ),
        )
    return Check(name=name, status=Status.OK, detail=detail)


def _driver_major(version: str | None) -> int | None:
    head = (version or "").split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _vram_headroom_check(report: HardwareReport, instances: dict[str, int]) -> Check:
    name = VRAM_ROW
    if isinstance(report.gpus, ProbeError) or not report.gpus:
        return Check(name=name, status=Status.WARN, detail="skipped: no GPU")
    cards = chosen_cards(report.gpus, report.chosen)  # every chosen card
    used = sum(c.vram_used_bytes for c in cards)
    each = _per_card(cards)
    ours = sum(instances.values())
    names = ", ".join(sorted(instances))
    if instances and ours >= used:
        # Estimates are peak figures (an idle image server holds less), so they can cover
        # everything nvidia-smi reports: Windows' share is then unknown, not zero.
        return Check(
            name=name,
            status=Status.OK,
            detail=(
                f"{used / GIB:.1f} GiB in use; the running instances ({names}) are estimated "
                f"at {ours / GIB:.1f} GiB at their peak, so what "
                f"{platforms.current().others} use is unknown{each}"
            ),
        )
    others_gib = (used - ours) / GIB
    if instances:
        detail = (
            f"{used / GIB:.1f} GiB in use: up to {ours / GIB:.1f} GiB by running instances "
            f"({names}, estimated at their peak), so at least {others_gib:.1f} GiB by "
            f"{platforms.current().others}"
        )
    else:
        detail = f"{others_gib:.1f} GiB already used ({platforms.current().desktop})"
    detail += each
    if others_gib > DESKTOP_VRAM_WARN_GIB:
        return Check(
            name=name,
            status=Status.WARN,
            detail=detail,
            remedy=(
                "Models started now have that much less of the card; fit estimates subtract it, so "
                "a model added meanwhile was planned around it: `mnm replan --all` when more of "
                "the card is free."
            ),
        )
    return Check(name=name, status=Status.OK, detail=detail)


def _per_card(cards: list[Gpu]) -> str:
    """` (GiB in use: 1.1 on card 0, 0.0 on card 1)` on several cards; nothing on one."""
    if len(cards) < 2:
        return ""
    each = ", ".join(f"{c.vram_used_bytes / GIB:.1f} on card {c.index}" for c in cards)
    return f" (GiB in use: {each})"


def _memory_check(report: HardwareReport) -> Check:
    name = "RAM"
    if report.memory is None:
        return Check(name=name, status=Status.WARN, detail="could not read /proc/meminfo")
    total_gib = report.memory.total_bytes / GIB
    available = report.memory.available_bytes
    detail = f"{total_gib:.1f} GiB total"
    if available is not None:
        detail += f", {available / GIB:.1f} GiB available"
    if report.cgroup_limit_bytes:
        detail += f", cgroup limit {report.cgroup_limit_bytes / GIB:.1f} GiB"
    if report.is_wsl:  # What the VM has is what .wslconfig gave it
        detail += wsl_ram_words(report)
    if total_gib < RAM_WARN_BELOW_GIB:
        return Check(
            name=name,
            status=Status.WARN,
            detail=detail,
            remedy=platforms.current().more_ram(MIN_WSL_RAM_GIB),
        )
    return Check(name=name, status=Status.OK, detail=detail)


WINDOWS_KEEPS_GIB = 8  # what the doctor leaves Windows when it says what more RAM would buy


def wsl_ram_words(report: HardwareReport) -> str:
    """`; 24 GB by .wslconfig (memory=, or WSL's default of half the PC's RAM) of the PC's
    32 GB`, and the large models more would hold (`ladder`), when the PC's RAM is known."""
    from makenomistakes.ladder import more_ram_buys  # noqa: PLC0415 - only under WSL

    assert report.memory is not None  # the caller read it
    given = math.ceil(report.memory.total_bytes / GIB - 0.01)
    words = f"; {given} GB by .wslconfig (`memory=`, or WSL's default of half the PC's RAM)"
    host = report.host_ram_bytes
    if host is None:
        return words
    words += f" of the PC's {host / GIB:.0f} GB"
    cards = chosen_cards(report.gpus, report.chosen)
    vram = sum(c.vram_total_bytes for c in cards)
    most = (host // GIB) - WINDOWS_KEEPS_GIB  # what `memory=` could say, in GB
    kept = given * GIB - report.memory.total_bytes  # what the VM's kernel keeps of it
    more = more_ram_buys(report.memory.total_bytes, vram, most * GIB - kept)
    if more and most > given:
        names = ", ".join(r.label for r in more)
        words += f"; memory={most}GB would hold {names}"
    return words


def _docker_gpu_runtime_check(docker: DockerInfo | ProbeError, relogin: bool) -> Check:
    name = RUNTIME_ROW
    if isinstance(docker, ProbeError):
        why = "after the re-login (above)" if relogin else "Docker unavailable"
        return Check(name=name, status=Status.WARN, detail=f"skipped: {why}")
    if docker.has_nvidia_runtime:
        return Check(name=name, status=Status.OK, detail="nvidia runtime registered")
    return Check(
        name=name,
        status=Status.FAIL,
        detail=f"runtimes: {', '.join(docker.runtimes) or 'none'}",
        remedy=(
            "Install the NVIDIA Container Toolkit, then `sudo nvidia-ctk runtime configure "
            "--runtime=docker && sudo systemctl restart docker`."
        ),
    )


def _docker_gpu_smoke_check(result: SmokeResult) -> Check:
    name = SMOKE_ROW
    if result.ok:
        return Check(name=name, status=Status.OK, detail=result.output)
    return Check(
        name=name,
        status=Status.FAIL,
        detail=result.output,
        remedy=(
            f"`docker run --rm --gpus all {GPU_SMOKE_IMAGE} nvidia-smi` must work before any "
            "model can start."
        ),
    )


def _disk_check(report: HardwareReport) -> Check:
    name = "Model storage"
    disk = report.disk
    if not disk.exists:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"{disk.path} does not exist",
            remedy=(
                "set MNM_MODELS_DIR to a folder of your own"
                if is_system_folder(Path(disk.path))
                else f"`sudo mkdir -p {disk.path} && sudo chown $USER {disk.path}`, "
                "or set MNM_MODELS_DIR."
            ),
        )
    if not disk.writable:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"{disk.path} is not writable",
            remedy=(
                "set MNM_MODELS_DIR to a folder of your own (a system folder is never made "
                "writable for one user)"
                if is_system_folder(Path(disk.path))
                else f"`sudo chown $USER {disk.path}`"
            ),
        )
    free_gib = disk.free_bytes / GIB
    detail = f"{disk.path}: {free_gib:.0f} GiB free of {disk.total_bytes / GIB:.0f} GiB"
    if free_gib < MIN_FREE_DISK_GIB:
        return Check(
            name=name,
            status=Status.WARN,
            detail=detail,
            remedy="Free space or move MNM_MODELS_DIR to a larger disk.",
        )
    return Check(name=name, status=Status.OK, detail=detail)


def _hf_token_check(present: bool) -> Check:
    name = "Hugging Face token"
    if present:
        return Check(name=name, status=Status.OK, detail="set (gated repos available)")
    return Check(
        name=name,
        status=Status.WARN,
        detail="not set; public repos only",
        remedy=(
            "Add HF_TOKEN=... to ~/.config/makenomistakes/secrets.env (chmod 600) when you "
            "need gated models."
        ),
    )
