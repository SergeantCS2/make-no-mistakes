"""The doctor's Docker line. Right after `mnm setup` installs Docker, this login is not in the
`docker` group yet: `docker info` answers "permission denied" on the socket while the daemon
runs and the group file already lists the user. That is the state setup calls "daemon
running, re-login pending" (`setup_docker.Docker.daemon`), and the doctor says the same
remedy (never to start Docker or join the group: both are done)."""

from __future__ import annotations

import getpass
import os
from collections.abc import Callable

from makenomistakes.doctor_types import Check, Status
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import DockerInfo, ProbeError
from makenomistakes.setup_docker import docker_group_members

NAME = "Docker"
_DENIED = "permission denied"


def login_user() -> str:
    """The login user, even under sudo (setup's `current_user` rule)."""
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and sudo_user != "root":
        return sudo_user
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "root"


def in_docker_group() -> bool:
    """The group file lists the login user (what a re-login grants), not this shell's list."""
    return login_user() in docker_group_members("docker")


def relogin_pending(docker: DockerInfo | ProbeError, in_group: Callable[[], bool]) -> bool:
    """The socket refused this shell, and a new login would be let in."""
    return isinstance(docker, ProbeError) and _DENIED in docker.message.lower() and in_group()


def docker_check(docker: DockerInfo | ProbeError, in_group: Callable[[], bool]) -> Check:
    if isinstance(docker, DockerInfo):
        return Check(name=NAME, status=Status.OK, detail=f"engine {docker.server_version or '?'}")
    if relogin_pending(docker, in_group):
        relogin = platforms.current().relogin
        return Check(
            name=NAME,
            status=Status.FAIL,
            detail=(
                f"the daemon runs and {login_user()} is in the docker group, but this login "
                "started before the group was added (permission denied on the socket)"
            ),
            remedy=relogin[:1].upper() + relogin[1:] + ".",
        )
    if _DENIED in docker.message.lower():
        return Check(
            name=NAME,
            status=Status.FAIL,
            detail=docker.message,
            remedy=(
                f"Add your user to the docker group (`sudo usermod -aG docker {login_user()}`, "
                "or `mnm setup`), then log in again."
            ),
        )
    return Check(
        name=NAME,
        status=Status.FAIL,
        detail=docker.message,
        remedy=(
            "Start Docker (`sudo systemctl start docker`) and make sure your user is in "
            "the docker group."
        ),
    )
