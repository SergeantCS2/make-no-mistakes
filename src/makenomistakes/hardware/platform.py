"""Where mnm runs: WSL2 on Windows, or a native Debian/Ubuntu install, and
which distro, so setup picks the right apt repositories and every message says the right
thing (`wsl --shutdown` means nothing on a native box; a reboot means nothing to Windows).

`current()` reads /proc/version and /etc/os-release once per process; `use()` sets
it for the rest of the process.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from makenomistakes.hardware.probe import (
    PROC_VERSION,
    FileReader,
    is_wsl_kernel,
    read_text_or_empty,
)

OS_RELEASE = Path("/etc/os-release")
PCI_DEVICES = Path("/sys/bus/pci/devices")
NVIDIA_VENDOR = "0x10de"
AMD_VENDOR = "0x1002"


class Kind(StrEnum):
    WSL = "wsl"
    NATIVE = "native"


class Family(StrEnum):
    UBUNTU = "ubuntu"  # Ubuntu and what is built on it (Mint, Pop!_OS, Zorin, neon …)
    DEBIAN = "debian"  # Debian and what is built on it with Debian's codenames
    OTHER = "other"  # setup prints the steps instead of running them


# Docker's and NVIDIA's apt repositories name Debian releases by codename and version.
DEBIAN_RELEASES = {"bullseye": "11", "bookworm": "12", "trixie": "13"}
# NVIDIA's CUDA repository folders that carry the driver (`nvidia-open`).
CUDA_REPOS = {
    ("ubuntu", "22.04"): "ubuntu2204",
    ("ubuntu", "24.04"): "ubuntu2404",
    ("debian", "12"): "debian12",
    ("debian", "13"): "debian13",
}
UBUNTU_VERSIONS = {"jammy": "22.04", "noble": "24.04", "focal": "20.04"}


class Distro(BaseModel):
    """/etc/os-release, typed. A derivative keeps its own id and names its base in
    `id_like`; Ubuntu derivatives also carry `UBUNTU_CODENAME`."""

    id: str = ""
    name: str = ""
    version_id: str = ""
    codename: str = ""
    ubuntu_codename: str = ""
    id_like: tuple[str, ...] = ()

    @property
    def family(self) -> Family:
        bases = {self.id, *self.id_like}
        if self.id == "ubuntu" or self.ubuntu_codename or "ubuntu" in bases:
            return Family.UBUNTU
        if "debian" in bases and self.codename in DEBIAN_RELEASES:
            return Family.DEBIAN
        return Family.OTHER

    @property
    def label(self) -> str:
        return self.name or self.id or "an unknown Linux"

    def docker_repo(self) -> tuple[str, str] | None:
        """(repository, suite) at download.docker.com: a derivative uses its base's."""
        match self.family:
            case Family.UBUNTU:
                suite = self.ubuntu_codename or self.codename
                return ("ubuntu", suite) if suite else None
            case Family.DEBIAN:
                return "debian", self.codename
            case Family.OTHER:
                return None

    def cuda_repo(self) -> str | None:
        """NVIDIA's CUDA repository folder for this release (x86_64), when there is one."""
        match self.family:
            case Family.UBUNTU:
                version = UBUNTU_VERSIONS.get(self.ubuntu_codename or self.codename, "")
                return CUDA_REPOS.get(("ubuntu", version))
            case Family.DEBIAN:
                return CUDA_REPOS.get(("debian", DEBIAN_RELEASES.get(self.codename, "")))
            case Family.OTHER:
                return None


class Platform(BaseModel):
    kind: Kind
    distro: Distro

    @property
    def is_wsl(self) -> bool:
        return self.kind is Kind.WSL

    @property
    def label(self) -> str:
        where = "WSL2 on Windows" if self.is_wsl else "native Linux"
        return f"{self.distro.label} ({where})"

    # --- wording: one place for what differs between the two -------------------------

    @property
    def relogin(self) -> str:
        """What makes a new group (docker) reach the shell and the user's services."""
        if self.is_wsl:
            return (
                "run `wsl --shutdown` from PowerShell and open Ubuntu again, so your user and "
                "the web UI service can use Docker without sudo (`newgrp docker` only fixes "
                "the current shell)"
            )
        return (
            "reboot (a log out is not enough: the user services keep running with the old "
            "groups), so your user and the web UI service can use Docker without sudo "
            "(`newgrp docker` only fixes the current shell)"
        )

    @property
    def no_driver(self) -> str:
        """What to do when no NVIDIA driver answers."""
        if self.is_wsl:
            return (
                "Install the NVIDIA Windows driver (it provides /usr/lib/wsl/lib/nvidia-smi "
                "inside WSL; GPU passthrough needs no .wslconfig key), then run "
                "`wsl --shutdown` from PowerShell."
            )
        return "Run `mnm setup`: it installs the NVIDIA driver (it asks first), then reboot."

    def more_ram(self, minimum_gib: int) -> str:
        if self.is_wsl:
            return (
                f"Raise memory= in %UserProfile%\\.wslconfig to at least {minimum_gib}GB "
                "and run `wsl --shutdown`."
            )
        return (
            f"mnm plans for {minimum_gib} GiB or more: models that stream weights from RAM "
            "(video, the MoE) may not fit while other programs hold RAM."
        )

    @property
    def services_relogin(self) -> str:
        """How the user services pick up a new group (the user manager keeps its old ones)."""
        if self.is_wsl:
            return (
                "Run `wsl --shutdown` from PowerShell, open Ubuntu again, then "
                "`mnm service install`."
            )
        return (
            "Reboot (with linger the user services outlive a log out), then `mnm service install`."
        )

    @property
    def starts_with(self) -> str:
        return "the WSL distro" if self.is_wsl else "the machine"

    @property
    def others(self) -> str:
        """Who holds the VRAM mnm's instances do not."""
        return "Windows and other apps" if self.is_wsl else "the desktop and other apps"

    @property
    def editor_label(self) -> str:
        """Continue's card heading: VS Code runs on Windows under WSL."""
        return "VS Code on Windows" if self.is_wsl else "VS Code"

    @property
    def localhost_note(self) -> str:
        return " (Windows sees WSL as localhost)." if self.is_wsl else "."

    @property
    def driver_update(self) -> str:
        if self.is_wsl:
            return "Update the NVIDIA Windows driver, then `wsl --shutdown`."
        if self.distro.family is Family.UBUNTU:
            return "Update the driver (`sudo ubuntu-drivers install nvidia:580`), then reboot."
        return (
            "Update the driver (`sudo apt install nvidia-open` from NVIDIA's repository), "
            "then reboot."
        )

    @property
    def apps_label(self) -> str:
        """The card heading for clients outside Docker on this machine."""
        return "Windows or WSL apps" if self.is_wsl else "Apps on this machine"

    @property
    def here(self) -> str:
        return "on Windows or WSL" if self.is_wsl else "on this machine"

    @property
    def terminal(self) -> str:
        return "a WSL terminal" if self.is_wsl else "a terminal"

    @property
    def shell_label(self) -> str:
        return "a bash shell (WSL)" if self.is_wsl else "a bash shell"

    @property
    def desktop(self) -> str:
        """Who else holds VRAM besides mnm."""
        return "Windows desktop, browsers, other apps" if self.is_wsl else "the desktop, other apps"


def parse_os_release(text: str) -> Distro:
    pairs = (line.partition("=") for line in text.splitlines() if "=" in line)
    values = {k.strip(): v.strip().strip('"').strip("'") for k, _, v in pairs}
    return Distro(
        id=values.get("ID", "").lower(),
        name=values.get("PRETTY_NAME", values.get("NAME", "")),
        version_id=values.get("VERSION_ID", ""),
        codename=values.get("VERSION_CODENAME", "").lower(),
        ubuntu_codename=values.get("UBUNTU_CODENAME", "").lower(),
        id_like=tuple(values.get("ID_LIKE", "").lower().split()),
    )


def detect(read_text: FileReader = read_text_or_empty) -> Platform:
    kind = Kind.WSL if is_wsl_kernel(read_text(PROC_VERSION)) else Kind.NATIVE
    return Platform(kind=kind, distro=parse_os_release(read_text(OS_RELEASE)))


_known: dict[str, Platform] = {}


def current() -> Platform:
    if "here" not in _known:
        _known["here"] = detect()
    return _known["here"]


def use(platform: Platform | None) -> None:
    """Pin the platform (tests); None reads the machine again at the next `current()`."""
    _known.pop("here", None)
    if platform is not None:
        _known["here"] = platform


def nvidia_pci_devices(root: Path = PCI_DEVICES) -> list[str]:
    """NVIDIA display controllers on the PCI bus (vendor 0x10de, class 0x03…): is there a
    card at all, before a driver is in place. WSL shows none (its GPU is paravirtual)."""
    return display_devices(root, NVIDIA_VENDOR)


def amd_pci_devices(root: Path = PCI_DEVICES) -> list[str]:
    """AMD display controllers on the PCI bus (vendor 0x1002, class 0x03…): "an AMD
    card without its driver" told from "no card" (an APU's iGPU is one too)."""
    return display_devices(root, AMD_VENDOR)


def display_devices(root: Path, vendor: str) -> list[str]:
    """The PCI addresses of `vendor`'s display controllers (class 0x03…), in bus order."""
    found: list[str] = []
    try:
        devices = sorted(root.iterdir())
    except OSError:
        return found
    for device in devices:
        try:
            said = (device / "vendor").read_text(encoding="ascii").strip()
            klass = (device / "class").read_text(encoding="ascii").strip()
        except OSError:
            continue
        if said == vendor and klass.startswith("0x03"):
            found.append(device.name)
    return found
