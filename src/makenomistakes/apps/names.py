"""The workspace apps by name, and where each one sits: what decides the URL its file names
and where `mnm doctor --app` asks from."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path


class App(StrEnum):
    OPENCODE = "opencode"
    LIBRECHAT = "librechat"
    OPEN_WEBUI = "open-webui"
    OPENHANDS = "openhands"
    GOOSE = "goose"
    ODYSSEUS = "odysseus"
    HERMES = "hermes"  # a card only: its own files are not written yet


class Vantage(StrEnum):
    HOST = "host"  # a program on this machine (or a Windows app through WSL's forwarding)
    DOCKER = "docker"  # a container on this machine: `host.docker.internal`


LABELS: dict[App, str] = {
    App.OPENCODE: "OpenCode",
    App.LIBRECHAT: "LibreChat",
    App.OPEN_WEBUI: "Open WebUI",
    App.OPENHANDS: "OpenHands",
    App.GOOSE: "Goose",
    App.ODYSSEUS: "Odysseus",
    App.HERMES: "Hermes Agent",
}

VANTAGE: dict[App, Vantage] = {
    App.OPENCODE: Vantage.HOST,  # native: its tools act on the user's repos
    App.LIBRECHAT: Vantage.DOCKER,
    App.OPEN_WEBUI: Vantage.DOCKER,
    App.OPENHANDS: Vantage.DOCKER,  # Agent Canvas in Docker, without the socket
    App.GOOSE: Vantage.HOST,  # the CLI here, or the Windows desktop app through localhost
    App.ODYSSEUS: Vantage.DOCKER,
    App.HERMES: Vantage.HOST,  # its TUI and gateway on this machine
}

# The container `mnm doctor --app` asks from, as `mnm apps install` names it: the
# compose project `mnm-librechat`'s `api` service, the `mnm-openhands` and `mnm-open-webui`
# containers. Odysseus's
# own compose names its container as the user set it up: `--container` names it.
CONTAINERS: dict[App, str] = {
    App.LIBRECHAT: "mnm-librechat-api-1",
    App.OPENHANDS: "mnm-openhands",
    App.OPEN_WEBUI: "mnm-open-webui",
}


def config_home() -> Path:
    """`$XDG_CONFIG_HOME`, else `~/.config`: where OpenCode and Goose read their files."""
    found = os.environ.get("XDG_CONFIG_HOME")
    return Path(found) if found else Path.home() / ".config"
