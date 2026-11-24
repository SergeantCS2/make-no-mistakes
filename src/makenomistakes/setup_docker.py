"""Live Docker checks for `mnm setup`, through the runner: installed, daemon up, the user's
group, the NVIDIA runtime, the engine image. The shell's group membership decides sudo."""

from __future__ import annotations

import grp
import shlex
from collections.abc import Callable
from pathlib import Path

from makenomistakes.hardware import CommandRunner, FileReader
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.platform import Platform
from makenomistakes.recipes import LLAMACPP_IMAGE
from makenomistakes.setup_commands import Command

DOCKER_DAEMON_JSON = Path("/etc/docker/daemon.json")
_PERMISSION_DENIED = "permission denied"
NOT_INSTALLED = "Docker is not installed yet"


def docker_group_members(group: str) -> list[str]:
    """Who the group file lists (what a re-login will grant), or nobody when it is missing."""
    try:
        return list(grp.getgrnam(group).gr_mem)
    except KeyError:
        return []


class Docker:
    """Live Docker checks through the runner; the shell's group membership decides sudo."""

    def __init__(
        self,
        runner: CommandRunner,
        read_text: FileReader,
        user: str = "",
        group_members: Callable[[str], list[str]] | None = None,
        platform: Platform | None = None,
        engine_image: str = LLAMACPP_IMAGE,
    ) -> None:
        self._runner = runner
        self.engine_image = engine_image  # llama.cpp's tag for the box's backend
        self._relogin = (platform or platforms.current()).relogin
        self._read_text = read_text
        self._user = user
        self._group_members = group_members or docker_group_members
        code, _, _ = runner.run(["docker", "info", "--format", "{{.ServerVersion}}"])
        self._reachable = code == 0

    def group_membership(self) -> tuple[bool, str]:
        """The user is in the docker group (by the group file, not this shell's stale list), or
        Docker answers without it (rootless, or the user's own arrangement)."""
        if self._reachable:
            return True, "docker answers this shell"
        members = self._group_members("docker")
        if self._user in members:
            return True, f"{self._user} is in the docker group; {self._relogin}"
        return False, f"{self._user} is not in the docker group"

    def pull_command(self, image: str) -> Command:
        """`docker pull`, or before a re-login a sudo shell that checks first: the plain
        check was refused, so the image may well be there already."""
        if self._reachable:
            return ["docker", "pull", image]
        quoted = shlex.quote(image)
        script = f"docker image inspect {quoted} >/dev/null 2>&1 || docker pull {quoted}"
        return ["sudo", "sh", "-c", script]

    def installed(self) -> tuple[bool, str]:
        code, out, err = self._runner.run(["docker", "--version"])
        if code != 0:
            return False, "not installed"
        return True, out.strip() or err.strip() or "installed"

    def daemon(self) -> tuple[bool, str]:
        code, out, err = self._runner.run(["docker", "info", "--format", "{{.ServerVersion}}"])
        if code == 0:
            return True, f"running (engine {out.strip() or '?'})"
        if _PERMISSION_DENIED in err.lower():
            if self._user and self._user not in self._group_members("docker"):
                return True, f"running, but {self._user} is not in the docker group (next step)"
            return True, f"running; {self._relogin}"
        if code == 127 or "not found" in err:
            return False, NOT_INSTALLED
        return False, err.strip().splitlines()[0] if err.strip() else f"docker info exited {code}"

    def nvidia_runtime(self) -> tuple[bool, str]:
        code, out, err = self._runner.run(["docker", "info", "--format", "{{json .Runtimes}}"])
        if code == 0:
            present = '"nvidia"' in out
            return present, "nvidia runtime registered" if present else "nvidia runtime missing"
        if _PERMISSION_DENIED in err.lower():
            # The daemon is up but this shell cannot ask it; its config file knows.
            try:
                configured = '"nvidia"' in self._read_text(DOCKER_DAEMON_JSON)
            except OSError:
                configured = False
            return configured, (
                "nvidia runtime configured in daemon.json" if configured else "not configured"
            )
        return False, "Docker not available yet"

    def relogin_pending(self) -> bool:
        """Docker answers this shell only after a re-login (asked now, not at build time)."""
        code, _, err = self._runner.run(["docker", "info", "--format", "{{.ServerVersion}}"])
        return code != 0 and _PERMISSION_DENIED in err.lower()

    def image_present(self) -> tuple[bool, str]:
        return self.tag_present(self.engine_image)

    def tag_present(self, image: str) -> tuple[bool, str]:
        """`image` is pulled (a ROCm box pulls llama.cpp's Vulkan image too)."""
        code, _, err = self._runner.run(["docker", "image", "inspect", image])
        if code == 0:
            return True, image
        if _PERMISSION_DENIED in err.lower():
            # Before the re-login, ask as the pull does (under sudo), but `-n`: a check never
            # prompts, so a dry run stays silent. Else the row reads PLANNED, then FIXED by the
            # pull's own guard, on every run until the re-login.
            code, _, _ = self._runner.run(["sudo", "-n", "docker", "image", "inspect", image])
            if code == 0:
                return True, f"{image} (checked with sudo until you re-login)"
            return False, f"{image}: plain `docker` can check it after you re-login"
        return False, f"{image} not pulled yet"
