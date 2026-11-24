"""The doctor per backend: the backend in the Machine line, and the card and Docker
rows of a box that does not run CUDA. On CUDA nothing changes. On the CPU backend (no card) the
"NVIDIA GPU" and "Docker NVIDIA runtime" FAILs become one OK each ("CPU backend: no card") and
the VRAM line goes; on Vulkan and ROCm the runtime row says the image reaches the card through
its device files, and `--deep`'s smoke test runs in the backend's own image:

- Vulkan: `llama-server --list-devices` in llama.cpp's Vulkan image with `/dev/dri` mapped
  (the image ships no `vulkaninfo`, and the engine's own list is the
  device check that needs only Docker; on lavapipe it printed `Vulkan0: llvmpipe (LLVM …)`),
  a WARN on Mesa's software device;
- ROCm: `rocminfo` in llama.cpp's ROCm image (its base is ROCm's `-complete`) with `/dev/kfd`
  and the render nodes mapped, a FAIL that names the groups when it is refused.

A `MNM_BACKEND` the cards cannot run is one FAIL with the reason."""

from __future__ import annotations

import re

from makenomistakes.backend import CUDA, Backend, device_argv, natural, software
from makenomistakes.backend import (
    words as backend_words,
)
from makenomistakes.backend_now import host_devices
from makenomistakes.config import Settings
from makenomistakes.doctor_amd import AmdHost, amd_checks, card_detail
from makenomistakes.doctor_docker import login_user
from makenomistakes.doctor_types import GPU_ROW, RUNTIME_ROW, SMOKE_ROW, VRAM_ROW, Check, Status
from makenomistakes.hardware import amd_probe
from makenomistakes.hardware.fingerprint import MachineFingerprint
from makenomistakes.hardware.probe import CommandRunner, Gpu, HardwareReport
from makenomistakes.hub.facts import Engine
from makenomistakes.machine import backend_read
from makenomistakes.recipes import spec

NAME = "Backend"
DOCKER_GPU = "Docker GPU"
_DEVICE = re.compile(r"^\s*((?:Vulkan|ROCm|CUDA)\d+):\s*(.+?)\s*$", re.MULTILINE)
SMOKE_SECONDS = 300.0  # a first run pulls the image (0.3 GB for Vulkan)


def backend_checks(
    settings: Settings,
    report: HardwareReport,
    checks: list[Check],
    runner: CommandRunner | None = None,
    deep: bool = False,
) -> list[Check]:
    """`checks` as this box's backend reads them: unchanged on CUDA (and on a box that
    could not be read), its own rows elsewhere."""
    read = backend_read(settings, report.box_cards, report.memory)
    box, backend, refused = read.box, read.backend, read.refused
    if refused is not None and box is not None:
        fail = Check(
            name=NAME,
            status=Status.FAIL,
            detail=refused,
            remedy="Fix `MNM_BACKEND` in the secrets file, or remove the line: mnm then "
            "decides from the cards.",
        )
        checks = _after(checks, "Machine", fail)
        backend = natural(box)  # the other rows as the cards alone would read them
    if box is None or backend is None:
        return checks
    # The AMD lines, read under the root the probe read the cards from
    amd = amd_checks(
        report, backend, AmdHost(root=amd_probe.ROOT, runner=runner, user=login_user())
    )
    if backend in CUDA:
        return _after(checks, DOCKER_GPU if _has(checks, DOCKER_GPU) else RUNTIME_ROW, *amd)
    rows: list[Check] = []
    for check in checks:
        if check.name == GPU_ROW:
            rows.append(_card_row(box, backend, _listed(report)))
        elif check.name == VRAM_ROW and not box.cards:
            continue  # no card: nothing holds VRAM
        elif check.name == RUNTIME_ROW:
            rows.append(_runtime_row(backend, box))
        elif check.name == SMOKE_ROW:
            continue  # the CUDA smoke test: replaced below on `--deep`
        else:
            rows.append(check)
    rows = _after(rows, DOCKER_GPU, *amd)
    if deep and runner is not None and backend is not Backend.CPU:
        rows = _after(rows, DOCKER_GPU, smoke_check(backend, box, runner))
    return rows


def _listed(report: HardwareReport) -> list[Gpu]:
    return report.gpus if isinstance(report.gpus, list) else []


def _after(checks: list[Check], name: str, *added: Check) -> list[Check]:
    at = next((i + 1 for i, c in enumerate(checks) if c.name == name), len(checks))
    return [*checks[:at], *added, *checks[at:]]


def _has(checks: list[Check], name: str) -> bool:
    return any(c.name == name for c in checks)


def _card_row(box: MachineFingerprint, backend: Backend, report_cards: list[Gpu]) -> Check:
    if not box.cards and backend is Backend.CPU:
        return Check(
            name="Card",
            status=Status.OK,
            detail="none: the CPU backend runs every model in RAM (the engines' CPU images)",
        )
    if software(box, backend):
        return Check(
            name="Card",
            status=Status.WARN,
            detail="none: MNM_BACKEND=vulkan runs on Mesa's software device, slower than the CPU",
            remedy="Remove MNM_BACKEND from the secrets file: the CPU backend is faster here.",
        )
    amd = [g for g in report_cards if g.is_amd]
    if amd:  # Name, gfx, VRAM or GTT, the backend
        return Check(
            name="Card", status=Status.OK, detail="; ".join(card_detail(g, backend) for g in amd)
        )
    names = ", ".join(c.short for c in box.cards)
    return Check(name="Card", status=Status.OK, detail=f"{names} ({backend_words(backend, box)})")


def _runtime_row(backend: Backend, box: MachineFingerprint) -> Check:
    if backend is Backend.CPU:
        return Check(name=DOCKER_GPU, status=Status.OK, detail="CPU backend: no card")
    if software(box, backend):
        return Check(
            name=DOCKER_GPU,
            status=Status.WARN,
            detail="Vulkan on Mesa's software device: slow (`--deep` runs the Vulkan image)",
        )
    files = "/dev/kfd and the render nodes" if backend is Backend.ROCM else "/dev/dri"
    return Check(
        name=DOCKER_GPU,
        status=Status.OK,
        detail=f"{backend_words(backend, box)}: the images reach the card through {files} "
        "(no NVIDIA runtime needed); `mnm doctor --deep` runs one",
    )


def smoke_argv(backend: Backend, box: MachineFingerprint) -> list[str]:
    """`docker run` of the backend's own device check (llama.cpp's image for the backend)."""
    image = spec(Engine.LLAMACPP).image_for(backend, box.arch)
    tag = image if isinstance(image, str) else ""
    cards = tuple(range(len(box.cards))) if box.cards else (0,)
    devices = device_argv(backend, cards, host_devices(backend))
    if backend is Backend.ROCM:
        return ["docker", "run", "--rm", *devices, "--entrypoint", "rocminfo", tag]
    return ["docker", "run", "--rm", *devices, tag, "--list-devices"]


def smoke_check(backend: Backend, box: MachineFingerprint, runner: CommandRunner) -> Check:
    """The backend's device check in its own image: each device it lists, a WARN on Mesa's
    software device, a FAIL with the diagnosis when it lists none."""
    code, out, err = runner.run(smoke_argv(backend, box), timeout=SMOKE_SECONDS)
    text = f"{out}\n{err}"
    listed = [f"{name}: {what}" for name, what in _DEVICE.findall(text)]
    if backend is Backend.ROCM:
        listed = re.findall(r"Marketing Name:\s*(.+)", text) if code == 0 else []
    if code == 0 and listed:
        lavapipe = any("llvmpipe" in line or "lavapipe" in line for line in listed)
        if lavapipe:
            return Check(
                name=SMOKE_ROW,
                status=Status.WARN,
                detail=f"{'; '.join(listed)}: Mesa's software device, slow",
                remedy="A card's Vulkan driver is missing (`mesa-vulkan-drivers`) or /dev/dri "
                "is not reachable; on a box without a card the CPU backend is faster.",
            )
        return Check(name=SMOKE_ROW, status=Status.OK, detail="; ".join(listed))
    from makenomistakes.diagnose import diagnose  # noqa: PLC0415 - the doctor's only use

    found = diagnose(text, Engine.LLAMACPP, backend)
    last = next((line for line in reversed(text.splitlines()) if line.strip()), "no output")
    return Check(
        name=SMOKE_ROW,
        status=Status.FAIL,
        detail=found[0].cause if found else last.strip(),
        remedy=found[0].remedy if found else "`mnm doctor` after the card's driver is loaded.",
    )
