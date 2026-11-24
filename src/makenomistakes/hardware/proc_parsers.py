"""Small parsers of what the probe reads (`probe` re-exports them):
the cgroup's memory limit, `docker info`, WSL's networking mode, ufw's switch, the kernel."""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field


class DockerInfo(BaseModel):
    server_version: str | None = None
    runtimes: list[str] = Field(default_factory=list)

    @property
    def has_nvidia_runtime(self) -> bool:
        return "nvidia" in self.runtimes


class NetworkingMode(StrEnum):
    NAT = "nat"
    MIRRORED = "mirrored"
    UNKNOWN = "unknown"


def parse_cgroup_limit(text: str) -> int | None:
    value = text.strip()
    return int(value) if value.isdigit() else None


def parse_docker_info(text: str) -> DockerInfo:
    data = json.loads(text)
    runtimes = data.get("Runtimes") or {}
    return DockerInfo(
        server_version=data.get("ServerVersion"),
        runtimes=sorted(runtimes) if isinstance(runtimes, dict) else [],
    )


def parse_networking_mode(text: str) -> NetworkingMode:
    value = text.strip().lower()
    return NetworkingMode(value) if value in {"nat", "mirrored"} else NetworkingMode.UNKNOWN


def parse_ufw_enabled(text: str) -> bool:
    """ufw's own switch (readable without root, unlike `ufw status`)."""
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ENABLED":
            return value.strip().strip('"').lower() == "yes"
    return False


def is_wsl_kernel(proc_version: str) -> bool:
    return "microsoft" in proc_version.lower()
