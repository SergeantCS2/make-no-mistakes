"""`mnm setup` on a box with AMD cards: the engines
carry ROCm and Vulkan in their images, so the box needs only the kernel's own `amdgpu` driver
with current firmware, the `render` and `video` groups, and Docker without NVIDIA's toolkit.
Every ROCm line is planned, unverified until an AMD box runs it.

The steps, each checked first and its commands printed:
- **AMD driver:** the in-tree amdgpu; a card on the PCI bus that no driver runs gets
  `linux-firmware` (and a reboot); AMD's packaged (DKMS) driver only as a printed note, after
  the in-tree one failed (`mnm doctor` says so): its route (amdgpu-install from
  repo.radeon.com) is not one mnm runs unread.
- **Kernel:** Ubuntu 24.04's HWE kernel (6.17) when the card needs a newer one than runs (RDNA
  4: 6.12; an APU's GTT above 15.5 GB: 6.16.9), asked, a reboot after; AMD's own docs name
  `linux-oem-24.04d` for APUs, said in the note.
- **Groups:** `usermod -aG render,video` (the re-login the docker group needs).
- **Vulkan tools:** `vulkan-tools` for `vulkaninfo`, the Vulkan device order mnm maps by.
- **amd-smi (optional, asked):** ROCm's apt repository only for `amd-smi` (the card's name and
  its link where sysfs is silent); the engines never need it.
- **An APU's GTT:** the kernel parameters that give the card the RAM less the system's margin
  (`amdgpu.gttsize`, `ttm.pages_limit`), written to GRUB after asking, a reboot after; the
  BIOS's UMA frame buffer at its lowest is a printed note.
- **AMD under WSL** is refused with the reason (one card, a technical preview through
  librocdxg, outside the official images' path).

The engine images: llama.cpp's for the box's backend (Vulkan by default; `server-rocm`, 7.7
GB, only when the user chose ROCm in section 1), and `server-vulkan` beside a ROCm one (the
doctor's device check and the Vulkan-only engines' neighbour)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.backend import Backend
from makenomistakes.doctor_amd import (
    APU_KERNEL,
    RDNA4_KERNEL,
    RELOGIN,
    gtt_settings,
    kernel_version,
)
from makenomistakes.hardware.devices import GROUPS, groups
from makenomistakes.hardware.platform import Family, Platform
from makenomistakes.hardware.probe import (
    CommandRunner,
    FileReader,
    Gpu,
    HardwareReport,
)
from makenomistakes.setup_commands import APT_GET, Command
from makenomistakes.setup_machine import gpu_step
from makenomistakes.setup_types import Step

GIB = 2**30
OSRELEASE = Path("/proc/sys/kernel/osrelease")
GROUP = Path("/etc/group")
GTT_FILE = "/etc/default/grub.d/90-mnm-amd-gtt.cfg"
ROCM_KEY = "https://stable.repo.amd.com/rocm/gpg/packages.gpg"
ROCM_REPO = "https://stable.repo.amd.com/rocm/core/packages"
AMD_SMI_PACKAGE = "amd-smi-lib"  # the CLI's package in ROCm's repository (planned, unverified)
WSL_REFUSAL = (
    "An AMD card under WSL is not supported: ROCm reaches it there through librocdxg, one "
    "card, a technical preview outside the engines' official images (AMD's WSL docs). Run "
    "mnm on native Ubuntu or Debian for this card."
)
DKMS_NOTE = (
    "If amdgpu still does not load after a reboot with current firmware, AMD's packaged driver "
    "(DKMS) is the next step: rocm.docs.amd.com, Radeon, `amdgpu-install --usecase=graphics` "
    "(mnm does not run it; then `mnm setup` again)."
)


@dataclass(frozen=True)
class AmdBox:
    """What setup reads of an AMD box: the cards the probe read, whether one sits on the PCI
    bus without a driver, the kernel's version and the RAM."""

    cards: tuple[Gpu, ...]
    unread_on_bus: bool  # an AMD display device sysfs shows no driver for
    kernel: str
    ram_bytes: int

    @property
    def apu(self) -> Gpu | None:
        return next((c for c in self.cards if c.carve_out), None)

    @property
    def rdna4(self) -> bool:
        return any((c.gfx_target or "").startswith("gfx120") for c in self.cards)


def amd_box(
    report: HardwareReport, read_text: FileReader, on_bus: Callable[[], list[str]]
) -> AmdBox | None:
    """The AMD box setup is on (an AMD card read, or one on the PCI bus while no card is
    read), or None: a box of NVIDIA cards, or none."""
    listed = report.gpus if isinstance(report.gpus, list) else []
    cards = tuple(g for g in listed if g.is_amd)
    unread = not listed and bool(on_bus())
    if not cards and not unread:
        return None
    kernel = read_text(OSRELEASE).strip()
    ram = report.memory.total_bytes if report.memory is not None else 0
    return AmdBox(cards=cards, unread_on_bus=unread, kernel=kernel, ram_bytes=ram)


def wsl_amd(runner: CommandRunner) -> bool:
    """Under WSL, Windows names an AMD card and no NVIDIA one (WSL shows no PCI bus)."""
    argv = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_VideoController).Name",
    ]
    try:
        code, out, _ = runner.run(argv, timeout=20.0)
    except OSError:
        return False
    said = out.lower()
    return code == 0 and ("radeon" in said or "amd" in said) and "nvidia" not in said


def amd_steps(
    found: AmdBox,
    where: Platform,
    runner: CommandRunner,
    user: str,
    group_text: Callable[[], str],
) -> list[Step]:
    """The AMD box's steps in place of the NVIDIA driver's."""
    if where.is_wsl:
        return [Step("AMD card", check=lambda: (False, "under WSL"), manual=WSL_REFUSAL)]
    steps = [driver_step(found, runner)]
    if (kernel := kernel_step(found, where)) is not None:
        steps.append(kernel)
    steps += [groups_step(user, group_text), vulkan_tools_step(runner)]
    silent = [c for c in found.cards if not c.carve_out and c.lanes is None]
    if silent:  # sysfs did not say the link: amd-smi might (asked, optional)
        steps.append(amd_smi_step(where, runner))
    if (gtt := gtt_step(found)) is not None:
        steps.append(gtt)
    return steps


def driver_step(found: AmdBox, runner: CommandRunner) -> Step:
    name = "AMD driver"
    if found.cards:
        names = ", ".join(f"{c.name} ({c.gfx_target or 'gfx not read'})" for c in found.cards)
        return Step(name, check=lambda: (True, f"amdgpu runs {names}"))
    if runner.run(["modinfo", "amdgpu"])[0] == 0:
        return Step(
            name,
            check=lambda: (False, "an AMD card is on the PCI bus; amdgpu is installed, not loaded"),
            fix=[[*APT_GET, "update"], [*APT_GET, "install", "-y", "linux-firmware"]],
            after=f"Reboot now, then run `mnm setup` again. {DKMS_NOTE}",
            confirm="Update the firmware the card needs (linux-firmware)? A reboot follows.",
            reboot=True,
            settles_at_relogin=lambda: True,
        )
    return Step(
        name,
        check=lambda: (False, "an AMD card is on the PCI bus, and no amdgpu module is installed"),
        manual="This kernel was built without amdgpu: install your distro's generic kernel "
        f"(Ubuntu: `sudo apt install linux-generic`), reboot, then `mnm setup` again. {DKMS_NOTE}",
    )


def kernel_step(found: AmdBox, where: Platform) -> Step | None:
    """Ubuntu 24.04's HWE kernel when the card needs a newer kernel than runs; None when the
    kernel is new enough (or not read)."""
    now = kernel_version(found.kernel)
    floor = RDNA4_KERNEL if found.rdna4 else APU_KERNEL if found.apu is not None else None
    if now is None or floor is None or now >= floor:
        return None
    need = ".".join(str(n) for n in floor)
    detail = f"{found.kernel}: this card needs {need} or newer"
    if where.distro.family is not Family.UBUNTU or where.distro.cuda_repo() != "ubuntu2404":
        return Step(
            "Kernel",
            check=lambda: (False, detail),
            manual=f"Install a kernel {need} or newer the way {where.distro.label} documents, "
            "reboot, then `mnm setup` again.",
        )
    oem = (
        " AMD's docs name the OEM kernel for APUs on 24.04 (`sudo apt install "
        "linux-oem-24.04d`); the HWE kernel is newer."
        if found.apu is not None
        else ""
    )
    return Step(
        "Kernel",
        check=lambda: (False, detail),
        fix=[[*APT_GET, "update"], [*APT_GET, "install", "-y", "linux-generic-hwe-24.04"]],
        after=f"Reboot now, then run `mnm setup` again.{oem}",
        confirm="Install Ubuntu 24.04's HWE kernel? It needs a reboot afterwards.",
        reboot=True,
        settles_at_relogin=lambda: True,
    )


def groups_step(user: str, group_text: Callable[[], str]) -> Step:
    """The render and video groups: the device files' owners (`/dev/kfd`, `renderD*`)."""

    def check() -> tuple[bool, str]:
        if user == "root":
            return True, "root opens the device files without them"
        found = groups(group_text())
        missing = [g for g in GROUPS if user not in found.get(g, (0, ()))[1]]
        return not missing, f"{user} not in {', '.join(missing)}" if missing else "render, video"

    return Step(
        "Render and video groups",
        check=check,
        fix=[["sudo", "usermod", "-aG", ",".join(GROUPS), user]],
        after=RELOGIN,
        settles_at_relogin=lambda: True,
    )


def vulkan_tools_step(runner: CommandRunner) -> Step:
    def check() -> tuple[bool, str]:
        code, out, _ = runner.run(["dpkg-query", "-W", "-f=${Status}", "vulkan-tools"])
        ok = code == 0 and "install ok installed" in out
        return ok, "installed" if ok else "missing: the doctor reads the Vulkan device order"

    return Step(
        "Vulkan tools",
        check=check,
        fix=[[*APT_GET, "update"], [*APT_GET, "install", "-y", "vulkan-tools"]],
    )


def amd_smi_commands(release: str) -> list[Command]:
    """ROCm's apt repository (ROCm 10.1's install page) and amd-smi alone."""
    source = (
        f"Types: deb\nURIs: {ROCM_REPO}/{release}/\nSuites: stable\nComponents: main\n"
        "Signed-By: /etc/apt/keyrings/amdrocm.gpg\n"
    )
    return [
        ["sudo", "install", "-m", "0755", "-d", "/etc/apt/keyrings"],
        [
            "sudo",
            "sh",
            "-c",
            f"curl -fsSL {ROCM_KEY} | gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg",
        ],
        ["sudo", "sh", "-c", f"printf '%s' '{source}' > /etc/apt/sources.list.d/amdrocm.sources"],
        [*APT_GET, "update"],
        [*APT_GET, "install", "-y", AMD_SMI_PACKAGE],
    ]


def amd_smi_step(where: Platform, runner: CommandRunner) -> Step:
    def check() -> tuple[bool, str]:
        code, _out, _err = runner.run(["amd-smi", "version"])
        return code == 0, "installed" if code == 0 else "not installed (optional)"

    release = where.distro.cuda_repo()
    if release is None:
        return Step(
            "amd-smi",
            check=check,
            manual=f"ROCm's repository has no folder for {where.distro.label}; mnm reads the "
            "cards from sysfs without amd-smi.",
        )
    return Step(
        "amd-smi",
        check=check,
        fix=amd_smi_commands(release),
        confirm="Add ROCm's apt repository for amd-smi (the card's name and link where sysfs is "
        "silent; the engines never need it)?",
    )


def gtt_commands(ram_bytes: int) -> list[Command]:
    size, pages = gtt_settings(ram_bytes)
    line = (
        'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT '
        f'amdgpu.gttsize={size} ttm.pages_limit={pages}"'
    )
    return [
        ["sudo", "install", "-m", "0755", "-d", "/etc/default/grub.d"],
        ["sudo", "sh", "-c", f"printf '%s\\n' '{line}' > {GTT_FILE}"],
        ["sudo", "update-grub"],
    ]


def gtt_step(found: AmdBox) -> Step | None:
    """An APU's GTT below half the RAM: the kernel parameters, asked, a reboot after."""
    apu = found.apu
    if apu is None or not found.ram_bytes:
        return None
    gtt = apu.gtt_total_bytes or 0
    size, pages = gtt_settings(found.ram_bytes)
    detail = f"{gtt / GIB:.1f} GiB of GTT of {found.ram_bytes / GIB:.1f} GiB RAM"
    if gtt >= found.ram_bytes / 2:
        return Step("APU memory (GTT)", check=lambda: (True, detail))
    return Step(
        "APU memory (GTT)",
        check=lambda: (False, detail),
        fix=gtt_commands(found.ram_bytes),
        after="Set the BIOS's UMA frame buffer to its lowest (512 MB), reboot, then "
        "`mnm replan --all` (plans made before were sized to the smaller pool).",
        confirm=f"Give the card {size / 1024:.0f} GiB of the RAM (amdgpu.gttsize={size} "
        f"ttm.pages_limit={pages} in GRUB)? It needs a reboot afterwards.",
        reboot=True,
        settles_at_relogin=lambda: True,
    )


CARD_STEP = "Graphics card"
NO_CARD = "none: the CPU backend runs every model in RAM (`mnm doctor` says the same)"


def card_steps(  # noqa: PLR0913 - the probe's seams, and the box's backend
    report: HardwareReport,
    where: Platform,
    runner: CommandRunner,
    user: str,
    read_text: FileReader,
    pci_devices: Callable[[], list[str]],
    amd_devices: Callable[[], list[str]],
    *,
    backend: Backend | None = None,
) -> list[Step]:
    """The NVIDIA driver's step (`setup_machine.gpu_step`), an AMD box's in its place, or
    both on a box with an NVIDIA and an AMD card; on a native box with no card at all and
    the CPU backend (a supported box, which the doctor calls OK), one OK row."""
    found = None if where.is_wsl else amd_box(report, read_text, amd_devices)
    listed = report.gpus if isinstance(report.gpus, list) else []
    cardless = found is None and not listed and not where.is_wsl
    if cardless and backend is Backend.CPU and not pci_devices():
        return [Step(CARD_STEP, check=lambda: (True, NO_CARD))]
    if found is not None:
        amd = amd_steps(found, where, runner, user, lambda: _read_or_empty(read_text, GROUP))
        nvidia = any(not g.is_amd for g in listed)
        return [gpu_step(report, where, runner, pci_devices), *amd] if nvidia else amd
    if where.is_wsl and not listed and wsl_amd(runner):
        return amd_steps(AmdBox((), True, "", 0), where, runner, user, lambda: "")
    return [gpu_step(report, where, runner, pci_devices)]


def _read_or_empty(read_text: FileReader, path: Path) -> str:
    try:
        return read_text(path)
    except OSError:
        return ""
