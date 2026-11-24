"""The vendors' install commands `mnm setup` runs (and prints first): Docker Engine from
Docker's apt repository, the NVIDIA Container Toolkit from NVIDIA's, and on a native box the
base packages and the NVIDIA driver. Pure lists of argv."""

from __future__ import annotations

import shlex

Command = list[str]


BASE_PACKAGES = ("ca-certificates", "curl", "git", "gnupg", "openssl")
# apt with debconf's noninteractive frontend: without a terminal the default one tries to
# re-open /dev/tty and prints "dpkg-preconfigure: unable to re-open stdin". These packages
# ask nothing anyway. The driver's commands keep the default frontend: a DKMS build under
# Secure Boot asks for a MOK password through it.
APT_GET = ["sudo", "env", "DEBIAN_FRONTEND=noninteractive", "apt-get"]


def apt_base_commands(missing: list[str]) -> list[Command]:
    """What a minimal Debian or Ubuntu lacks that the steps after this one use."""
    return [[*APT_GET, "update"], [*APT_GET, "install", "-y", *missing]]


def apt_docker_commands(repo: str, suite: str, arch: str, user: str) -> list[Command]:
    """Docker Engine from Docker's apt repository (docs.docker.com/engine/install/ubuntu and
    …/debian: the same steps, the repository named after the base distro)."""
    source = (
        f"Types: deb\nURIs: https://download.docker.com/linux/{repo}\n"
        f"Suites: {suite}\nComponents: stable\nArchitectures: {arch}\n"
        "Signed-By: /etc/apt/keyrings/docker.asc\n"
    )
    return [
        [*APT_GET, "update"],
        [*APT_GET, "install", "-y", "ca-certificates", "curl"],
        ["sudo", "install", "-m", "0755", "-d", "/etc/apt/keyrings"],
        [
            "sudo",
            "curl",
            "-fsSL",
            f"https://download.docker.com/linux/{repo}/gpg",
            "-o",
            "/etc/apt/keyrings/docker.asc",
        ],
        ["sudo", "chmod", "a+r", "/etc/apt/keyrings/docker.asc"],
        ["sudo", "sh", "-c", f"printf '%s' '{source}' > /etc/apt/sources.list.d/docker.sources"],
        [*APT_GET, "update"],
        [
            *APT_GET,
            "install",
            "-y",
            "docker-ce",
            "docker-ce-cli",
            "containerd.io",
            "docker-buildx-plugin",
            "docker-compose-plugin",
        ],
        start_docker_command(),
        ["sudo", "usermod", "-aG", "docker", user],
    ]


def start_docker_command() -> Command:
    """Works with and without systemd (WSL without `systemd=true` has only `service`)."""
    return ["sudo", "sh", "-c", "systemctl enable --now docker 2>/dev/null || service docker start"]


def apt_nvidia_toolkit_commands() -> list[Command]:
    """NVIDIA Container Toolkit (docs.nvidia.com/datacenter/cloud-native/container-toolkit)."""
    keyring = "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    return [
        [
            *APT_GET,
            "install",
            "-y",
            "--no-install-recommends",
            "ca-certificates",
            "curl",
            "gnupg2",
        ],
        [
            "sudo",
            "sh",
            "-c",
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey "
            f"| gpg --batch --yes --dearmor -o {keyring}",
        ],
        [
            "sudo",
            "sh",
            "-c",
            # -f: a 404 page must never become the apt source (the pipeline's status was sed's)
            "set -e; curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/"
            "nvidia-container-toolkit.list -o /tmp/nvidia-container-toolkit.list "
            f"&& sed 's#deb https://#deb [signed-by={keyring}] https://#g' "
            "/tmp/nvidia-container-toolkit.list "
            "> /etc/apt/sources.list.d/nvidia-container-toolkit.list",
        ],
        [*APT_GET, "update"],
        [*APT_GET, "install", "-y", "nvidia-container-toolkit"],
        ["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"],
        ["sudo", "sh", "-c", "systemctl restart docker 2>/dev/null || service docker restart"],
    ]


def ubuntu_driver_commands(pop_os: bool = False) -> list[Command]:
    """The NVIDIA driver on Ubuntu: Ubuntu's own signed build, which Secure Boot accepts
    (`ubuntu-drivers` installs it for the card); Pop!_OS packages its own."""
    if pop_os:
        return [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "system76-driver-nvidia"],
        ]
    # R580 first: whisper.cpp's CUDA 13 image needs it, and the recommended one may be 570
    # (both are in 22.04's and 24.04's archives); the recommended one when 580 is not there.
    return [
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "ubuntu-drivers-common"],
        ["sudo", "sh", "-c", "ubuntu-drivers install nvidia:580 || ubuntu-drivers install"],
    ]


CUDA_KEYRING = "https://developer.download.nvidia.com/compute/cuda/repos/{repo}/x86_64/cuda-keyring_1.1-1_all.deb"


def nvidia_repo_driver_commands(repo: str, kernel: str) -> list[Command]:
    """The NVIDIA driver from NVIDIA's own repository (Debian: its packaged driver is older
    than whisper.cpp's CUDA 13 needs). `nvidia-open`: the open kernel modules, built by DKMS
    against this kernel's headers (Turing and newer; NVIDIA's default since 560)."""
    # Debian drops an old kernel's headers once a newer kernel ships (a box not upgraded in a
    # while): then the current kernel and its headers come in, and the reboot boots into it.
    headers = (
        f"apt-get install -y {shlex.quote('linux-headers-' + kernel)} "
        "|| apt-get install -y linux-image-amd64 linux-headers-amd64"
    )
    # Root downloads the keyring into a file only it can write, installs it, removes it.
    keyring = (
        'set -e; f="$(mktemp --suffix=.deb)"; '
        f'curl -fsSL -o "$f" {shlex.quote(CUDA_KEYRING.format(repo=repo))}; '
        'dpkg -i "$f"; rm -f "$f"'
    )
    return [
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "dkms", "curl"],
        ["sudo", "sh", "-c", headers],
        ["sudo", "sh", "-c", keyring],
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "nvidia-open"],
    ]
