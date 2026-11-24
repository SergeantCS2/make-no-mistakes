"""The doctor on a box with AMD cards: what a box needs for
llama.cpp's ROCm and Vulkan images, read without ROCm on the host and without sudo. Every
line is worded; nothing here runs on the card (`mnm doctor --deep` runs the engine's own
`--list-devices` in the Vulkan image: `doctor_backend.smoke_check`).

- **Driver:** `amdgpu` loaded (`/sys/module/amdgpu`); the kernel against the card's floor
  (RDNA 4 needs 6.12 or newer, Phoronix's launch review; Strix Halo's GTT above 15.5 GB needs
  6.16.9); the firmware: a kernel log line "failed to load firmware" for
  amdgpu is a FAIL (the log is read through the runner: `dmesg`, else `journalctl -k`; neither
  readable without sudo on Ubuntu is a plain note, never a guess).
- **Access:** `/dev/kfd` (ROCm only) and each card's render node must exist and open read-write
  for this user; in `render`/`video` by `/etc/group` but not in this login's groups is the
  re-login trap the docker group has (`doctor_docker`'s words).
- **Resizable BAR:** a discrete card whose visible VRAM is smaller than its VRAM has Resizable
  BAR or Above 4G Decoding off (ROCm's install page requires Above 4G): a WARN naming both.
- **An APU's GTT:** below half the RAM is a WARN naming `amdgpu.gttsize` and `ttm.pages_limit`
  with the values for this RAM (what `mnm setup` offers to write); its 512 MB of VRAM never is.
- **Several cards:** each card's slot link (from the bridge, `hardware.amd_probe`); a card
  below x8 is on chipset lanes, which throttle peer to peer (alexkmiller.com, 2026-09-07);
  `CONFIG_HSA_AMD_P2P` and `CONFIG_PCI_P2PDMA` in `/boot/config-<kernel>` for a tensor split;
  never rebind amdgpu to vfio (Puget, Level1Techs: kernel panics)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from makenomistakes.backend import Backend
from makenomistakes.doctor_types import Check, Status
from makenomistakes.fit.ram_floor import OS_RAM_MARGIN
from makenomistakes.hardware.amd_cards import generation_words, rocm_runs
from makenomistakes.hardware.devices import groups
from makenomistakes.hardware.probe import CommandRunner, Gpu, HardwareReport, read_text_or_empty

GIB = 2**30
MIB = 2**20
PAGE = 4096
RDNA4_KERNEL = (6, 12)
APU_KERNEL = (6, 16)  # 6.16.9: early 6.15 kernels capped ROCm at about 15.5 GB
CPU_LANES = 8  # below this a card is on the chipset's lanes (AM5, Z790: x4)
P2P_OPTIONS = ("CONFIG_HSA_AMD_P2P", "CONFIG_PCI_P2PDMA")
LOG_COMMANDS = (["dmesg"], ["journalctl", "-k", "-b", "-q", "--no-pager"])
RELOGIN = (
    "Reboot (a log out is not enough: the user services keep the groups they started with), "
    "so your login and mnm's services can open the card (`newgrp render` only fixes the "
    "current shell)."
)


@dataclass(frozen=True)
class AmdHost:
    """What the AMD lines read of the host: its files under `root`, the kernel log through
    `runner`, the login user and this login's groups (`os.getgroups`)."""

    root: Path = Path("/")
    runner: CommandRunner | None = None
    user: str = ""
    groups_now: Callable[[], set[int]] = field(default=lambda: set(os.getgroups()))
    access: Callable[[Path], bool] = field(default=lambda p: os.access(p, os.R_OK | os.W_OK))

    def log(self, argv: list[str]) -> str | None:
        """A kernel log command's output; None when it cannot be read (no runner, refused)."""
        if self.runner is None:
            return None
        code, out, _ = self.runner.run(list(argv), timeout=10.0)
        return out if code == 0 and out.strip() else None

    def text(self, relative: str) -> str:
        return read_text_or_empty(self.root / relative)


def amd_checks(report: HardwareReport, backend: Backend, host: AmdHost) -> list[Check]:
    """The AMD lines, or none on a box without an AMD card."""
    cards = [g for g in report.gpus if g.is_amd] if isinstance(report.gpus, list) else []
    if not cards:
        return []
    found = [
        driver_check(host),
        kernel_check(cards, host),
        firmware_check(host),
        access_check(cards, backend, host),
        *[c for c in (bar_check(card) for card in cards) if c is not None],
        *[c for c in (gtt_check(card, report) for card in cards) if c is not None],
    ]
    if len(cards) > 1:
        found.append(links_check(cards, host))
    return found


def card_detail(card: Gpu, backend: Backend) -> str:
    """`Radeon AI PRO R9700, gfx1201 (RDNA 4), 32.0 GiB VRAM, Vulkan` or an APU's GTT; a gfx
    target the ROCm image lacks says "Vulkan only"."""
    size = card.vram_total_bytes / GIB
    memory = f"{size:.1f} GiB of GTT (an APU)" if card.carve_out else f"{size:.1f} GiB VRAM"
    only = "" if rocm_runs(card.gfx_target) else ", Vulkan only"
    used = "ROCm" if backend is Backend.ROCM else "Vulkan"
    return f"{card.name}, {generation_words(card.gfx_target)}, {memory}, {used}{only}"


def driver_check(host: AmdHost) -> Check:
    if (host.root / "sys/module/amdgpu").exists():
        return Check(name="AMD driver", status=Status.OK, detail="amdgpu loaded (the kernel's own)")
    return Check(
        name="AMD driver",
        status=Status.FAIL,
        detail="amdgpu is not loaded",
        remedy="Install the current firmware (`sudo apt install linux-firmware`) and load it "
        "(`sudo modprobe amdgpu`), or reboot; when it still does not load, `mnm setup` offers "
        "AMD's packaged driver (asked first, a reboot after).",
    )


def kernel_version(text: str) -> tuple[int, int] | None:
    """`6.17.0-14-generic` as (6, 17); None when it is not one."""
    head = text.strip().split("-")[0].split(".")
    return (int(head[0]), int(head[1])) if len(head) >= 2 and head[1].isdigit() else None


def kernel_check(cards: list[Gpu], host: AmdHost) -> Check:
    release = host.text("proc/sys/kernel/osrelease").strip()
    found = kernel_version(release)
    rdna4 = any((c.gfx_target or "").startswith("gfx120") for c in cards)
    apu = any(c.carve_out for c in cards)
    if found is None:
        return Check(name="Kernel", status=Status.WARN, detail="the kernel's version was not read")
    if rdna4 and found < RDNA4_KERNEL:
        return Check(
            name="Kernel",
            status=Status.FAIL,
            detail=f"{release}: RDNA 4 cards need 6.12 or newer",
            remedy="On Ubuntu 24.04 install the HWE kernel (`sudo apt install "
            "linux-generic-hwe-24.04`) and reboot; `mnm setup` offers it.",
        )
    if apu and found < APU_KERNEL:
        return Check(
            name="Kernel",
            status=Status.WARN,
            detail=f"{release}: kernels before 6.16.9 capped an APU's GPU memory at about 15.5 GB",
            remedy="Install a newer kernel (Ubuntu 24.04: AMD names the OEM kernel, "
            "`sudo apt install linux-oem-24.04d`) and reboot.",
        )
    return Check(name="Kernel", status=Status.OK, detail=release)


def firmware_check(host: AmdHost) -> Check:
    """The kernel log's amdgpu firmware lines; not readable without sudo is a note."""
    text = next((out for out in map(host.log, LOG_COMMANDS) if out), None)
    if text is None:
        return Check(
            name="AMD firmware",
            status=Status.OK,
            detail="the kernel log is not readable without sudo; `sudo dmesg | grep -i "
            "'amdgpu.*firmware'` shows a firmware the card lacks",
        )
    failed = [
        line.strip()
        for line in text.splitlines()
        if "amdgpu" in line and "failed to load firmware" in line.lower()
    ]
    if failed:
        return Check(
            name="AMD firmware",
            status=Status.FAIL,
            detail=failed[0],
            remedy="Update the firmware (`sudo apt install linux-firmware`, on Ubuntu 24.04 the "
            "HWE kernel brings a newer one) and reboot.",
        )
    return Check(name="AMD firmware", status=Status.OK, detail="no firmware load failed")


def access_check(cards: list[Gpu], backend: Backend, host: AmdHost) -> Check:
    """`/dev/kfd` (ROCm) and the render nodes exist and open; the groups explain why not."""
    name = "AMD device files"
    wanted = [n for n in (c.render_node for c in cards) if n]
    if backend is Backend.ROCM:
        wanted = ["/dev/kfd", *wanted]
    missing = [p for p in wanted if not (host.root / p.lstrip("/")).exists()]
    if missing:
        kfd = "/dev/kfd" in missing
        what = "the amdgpu driver (KFD) is not loaded" if kfd else "no render node"
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"{', '.join(missing)} missing: {what}",
            remedy="Check `lsmod | grep amdgpu` and reboot after a kernel or firmware update; "
            "`mnm doctor` again.",
        )
    closed = [p for p in wanted if not host.access(host.root / p.lstrip("/"))]
    if not closed:
        return Check(name=name, status=Status.OK, detail=f"{', '.join(wanted)} open")
    found = groups(host.text("etc/group"))
    member = {n for n, (_gid, users) in found.items() if host.user in users}
    listed = bool(found) and member == set(found)
    if listed and not {gid for gid, _users in found.values()} <= host.groups_now():
        return Check(
            name=name,
            status=Status.FAIL,
            detail=f"{host.user} is in the render and video groups, but this login started "
            f"before they were added ({', '.join(closed)} refused)",
            remedy=RELOGIN,
        )
    return Check(
        name=name,
        status=Status.FAIL,
        detail=f"{', '.join(closed)} refused: {host.user or 'this user'} is not in the render "
        "and video groups",
        remedy=f"`sudo usermod -aG render,video {host.user or '$USER'}` (or `mnm setup`), then "
        "log in again.",
    )


def bar_check(card: Gpu) -> Check | None:
    """A discrete card whose CPU-visible VRAM is smaller than its VRAM: Resizable BAR is off."""
    visible = card.vis_vram_total_bytes
    if card.carve_out or visible is None or visible >= card.vram_total_bytes:
        return None
    return Check(
        name=f"Card {card.index} BAR",
        status=Status.WARN,
        detail=f"{card.name}: the CPU sees {visible / MIB:,.0f} MiB of its "
        f"{card.vram_total_bytes / GIB:.1f} GiB: Resizable BAR is off",
        remedy="Turn on Above 4G Decoding and Resizable BAR (Re-Size BAR Support) in the "
        "board's BIOS; ROCm's install page requires Above 4G Decoding, and loading a model "
        "is slower without it.",
    )


def gtt_settings(ram_bytes: int) -> tuple[int, int]:
    """`amdgpu.gttsize` (MiB) and `ttm.pages_limit` (4 KiB pages) giving an APU the RAM less
    the system's margin (`OS_RAM_MARGIN`)."""
    pool = max(ram_bytes - OS_RAM_MARGIN, 0)
    return pool // MIB, pool // PAGE


def gtt_check(card: Gpu, report: HardwareReport) -> Check | None:
    """An APU's GTT against the RAM: below half is a WARN with the kernel parameters."""
    if not card.carve_out or report.memory is None or not card.gtt_total_bytes:
        return None
    ram = report.memory.total_bytes
    gtt = card.gtt_total_bytes
    detail = f"{card.name}: {gtt / GIB:.1f} GiB of GTT of {ram / GIB:.1f} GiB RAM"
    if gtt >= ram / 2:
        return Check(name="APU memory", status=Status.OK, detail=detail)
    size, pages = gtt_settings(ram)
    return Check(
        name="APU memory",
        status=Status.WARN,
        detail=f"{detail}: models larger than that do not fit",
        remedy=f"Give the card more of the RAM with the kernel parameters `amdgpu.gttsize={size} "
        f"ttm.pages_limit={pages}` (`mnm setup` offers to write them to GRUB), keep the BIOS "
        "UMA frame buffer at its lowest (512 MB), reboot, then `mnm replan --all`.",
    )


def links_check(cards: list[Gpu], host: AmdHost) -> Check:
    """Several AMD cards: each slot's link, chipset lanes, the P2P options for a tensor split."""
    words = []
    slow = []
    for card in cards:
        if card.lanes is None:
            words.append(f"{card.index}: link unknown")
            continue
        gen = f" PCIe {card.pcie_gen}.0" if card.pcie_gen else ""
        words.append(f"{card.index}: x{card.lanes}{gen}")
        if card.lanes < CPU_LANES:
            slow.append(card.index)
    release = host.text("proc/sys/kernel/osrelease").strip()
    config = host.text(f"boot/config-{release}") if release else ""
    missing = [o for o in P2P_OPTIONS if f"{o}=y" not in config]
    problems = []
    if slow:
        problems.append(
            f"card {', '.join(str(i) for i in slow)} below x8 is on the chipset's lanes, which "
            "throttle peer to peer: put both cards in the CPU's slots"
        )
    if config and missing:
        problems.append(f"the kernel lacks {', '.join(missing)}: `split=tensor` needs peer to peer")
    detail = "; ".join(words) + ("; " + "; ".join(problems) if problems else "")
    return Check(
        name="AMD cards",
        status=Status.WARN if problems else Status.OK,
        detail=detail,
        remedy="Never rebind a card from amdgpu to vfio while mnm runs it (a kernel panic on "
        "R9700 boxes)."
        if problems
        else None,
    )
