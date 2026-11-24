"""The steps of `mnm setup` that depend on the machine: base packages on a
native box, the NVIDIA driver (Windows' under WSL; the distro's or NVIDIA's on a native box,
asked first), and Docker Engine from the repository of the base distro."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.hardware import CommandRunner, HardwareReport, ProbeError
from makenomistakes.hardware.platform import Family, Platform, nvidia_pci_devices
from makenomistakes.setup_commands import (
    BASE_PACKAGES,
    Command,
    apt_base_commands,
    apt_docker_commands,
    nvidia_repo_driver_commands,
    ubuntu_driver_commands,
)
from makenomistakes.setup_docker import Docker
from makenomistakes.setup_types import Step


def base_packages_step(runner: CommandRunner) -> Step:
    """A minimal Debian or Ubuntu (a server or netinst image) may lack curl, git or gnupg;
    the steps after this one and `mnm update` use them."""

    def missing() -> list[str]:
        return [p for p in BASE_PACKAGES if not _package_installed(runner, p)]

    def check() -> tuple[bool, str]:
        absent = missing()
        return not absent, f"missing: {', '.join(absent)}" if absent else "all present"

    return Step("Base packages", check=check, fix=apt_base_commands(missing()))


def _package_installed(runner: CommandRunner, package: str) -> bool:
    code, out, _ = runner.run(["dpkg-query", "-W", "-f=${Status}", package])
    return code == 0 and "install ok installed" in out


def gpu_step(
    report: HardwareReport,
    where: Platform,
    runner: CommandRunner,
    pci_devices: Callable[[], list[str]] = nvidia_pci_devices,
) -> Step:
    """The NVIDIA driver: Windows' under WSL (only the user can install it), the distro's or
    NVIDIA's packaged one on a native box (asked first; it takes effect at the next boot)."""
    gpu_ok = not isinstance(report.gpus, ProbeError) and bool(report.gpus)
    detail = gpu_detail(report)
    if where.is_wsl:
        return Step(
            "NVIDIA GPU visible in WSL",
            check=lambda: (gpu_ok, detail),
            manual=(
                "Install the NVIDIA Windows driver (it provides nvidia-smi inside WSL; GPU "
                "passthrough needs no .wslconfig key), run `wsl --shutdown` from PowerShell, "
                "then run `mnm setup` again."
            ),
        )
    name = "NVIDIA driver"
    if gpu_ok:
        return Step(name, check=lambda: (True, detail))
    if not pci_devices():
        return Step(
            name,
            check=lambda: (False, "no NVIDIA card on the PCI bus"),
            manual="mnm needs an NVIDIA GPU. If the card is in, check that it is seated and "
            "powered and that the BIOS enables it, then run `mnm setup` again.",
        )
    if runner.run(["modinfo", "nvidia"])[0] == 0:
        return Step(
            name,
            check=lambda: (False, f"installed but not loaded ({detail})"),
            manual="The driver is installed but not running: reboot, then run `mnm setup` "
            "again. If it still does not load and Secure Boot is on, enroll the driver's key "
            "(`sudo mokutil --import /var/lib/dkms/mok.pub`, reboot and pick Enroll MOK) or "
            "turn Secure Boot off.",
            reboot=True,
        )
    fix = driver_commands(where, runner)
    if fix is None:
        return Step(
            name,
            check=lambda: (False, detail),
            manual=f"mnm has no driver recipe for {where.distro.label}: install the NVIDIA "
            "driver (580 or newer) the way your distro documents, reboot, then run "
            "`mnm setup` again.",
        )
    note = "Reboot now, then run `mnm setup` again (the driver loads at boot)."
    if _secure_boot_on(runner):
        if where.distro.family is Family.DEBIAN:  # Debian's DKMS signs but never enrolls
            note += (
                " Secure Boot is on: before the reboot run `sudo mokutil --import "
                "/var/lib/dkms/mok.pub` and choose a password, then pick Enroll MOK at the blue "
                "screen after the reboot and type it."
            )
        else:
            note += (
                " Secure Boot is on: if the install asked you for a password, pick Enroll MOK "
                "at the blue screen after the reboot and type it."
            )
    return Step(
        name,
        check=lambda: (False, detail),
        fix=fix,
        after=note,
        confirm="Install the NVIDIA driver now? It needs a reboot afterwards.",
        reboot=True,
        settles_at_relogin=lambda: True,  # the commands ran; the driver loads at the reboot
    )


def driver_commands(where: Platform, runner: CommandRunner) -> list[Command] | None:
    """Ubuntu's own signed driver; on Debian NVIDIA's repository (amd64 only)."""
    distro = where.distro
    if distro.family is Family.UBUNTU:
        return ubuntu_driver_commands(pop_os=distro.id == "pop")
    if distro.family is Family.DEBIAN and dpkg_arch(runner) == "amd64":
        repo = distro.cuda_repo()
        if repo is not None:
            code, out, _ = runner.run(["uname", "-r"])
            if code == 0 and out.strip():
                return nvidia_repo_driver_commands(repo, out.strip())
    return None


def _secure_boot_on(runner: CommandRunner) -> bool:
    code, out, _ = runner.run(["mokutil", "--sb-state"])
    return code == 0 and "secureboot enabled" in out.lower()


def docker_step(where: Platform, runner: CommandRunner, user: str, docker: Docker) -> Step:
    """Docker Engine from Docker's repository for the base distro (a derivative uses its
    base's: Mint's packages are Ubuntu's)."""
    repo = where.distro.docker_repo()
    if repo is None:
        return Step(
            "Docker Engine",
            check=docker.installed,
            manual=f"mnm installs Docker on Debian and Ubuntu only; {where.distro.label} is "
            "neither. Install Docker Engine (docs.docker.com/engine/install), add yourself to "
            "the docker group, then run `mnm setup` again.",
        )
    return Step(
        "Docker Engine",
        check=docker.installed,
        fix=apt_docker_commands(repo[0], repo[1], dpkg_arch(runner), user),
        after=f"Then {where.relogin}.",
    )


def dpkg_arch(runner: CommandRunner) -> str:
    code, out, _ = runner.run(["dpkg", "--print-architecture"])
    return out.strip() if code == 0 and out.strip() else "amd64"


def gpu_detail(report: HardwareReport) -> str:
    if isinstance(report.gpus, ProbeError):
        return report.gpus.message
    return report.gpus[0].name if report.gpus else "no GPU listed"
