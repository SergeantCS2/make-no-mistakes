"""OpenCode's systemd user unit: `mnm-opencode`, `opencode serve` on 127.0.0.1:4096 in
the projects folder, as `service.py` writes the web UI's.

The password is never in the unit file (which any process of the user's can read with
`systemctl --user cat`): `EnvironmentFile=` names a 0600 file under the state folder, written
from the secrets file at every install and start, holding `OPENCODE_SERVER_PASSWORD` (HTTP
basic auth for the server and its web UI, user `opencode`; opencode.ai/docs/server, read
2026-10-22) and, when the router asks for one, `MNM_API_KEY`, which OpenCode's
`opencode.json` reads as `{env:MNM_API_KEY}`. No other secret goes in: OpenCode's tools run
shell commands in that environment."""

from __future__ import annotations

import os
from pathlib import Path

from makenomistakes import service
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.service import exec_word, specifiers

UNIT = "mnm-opencode"
ENV_NAME = "unit.env"


def unit_path(unit_dir: Path | None = None) -> Path:
    return (unit_dir or service.UNIT_DIR) / f"{UNIT}.service"  # read now: tests move it


def unit_text(
    binary: Path, port: int, projects: Path, env_file: Path, host: str = "127.0.0.1"
) -> str:
    """`host`: 0.0.0.0 once `mnm apps expose opencode` opened it, else loopback."""
    return (
        "[Unit]\n"
        "Description=OpenCode server for makenomistakes (mnm apps)\n"
        "\n"
        "[Service]\n"
        f"WorkingDirectory={specifiers(str(projects))}\n"
        f"EnvironmentFile={specifiers(str(env_file))}\n"
        f"ExecStart={exec_word(str(binary))} serve --port {port} --hostname {host}\n"
        "Restart=on-failure\n"
        "RestartSec=3\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def env_text(password: str, api_key: str | None) -> str:
    lines = [f"OPENCODE_SERVER_PASSWORD={password}"]
    if api_key:
        lines.append(f"MNM_API_KEY={api_key}")
    return "".join(f"{line}\n" for line in lines)


def write_private(path: Path, text: str) -> None:
    """A 0600 file, born 0600 (a secret is never readable by others, not even briefly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    tmp.replace(path)


def systemd_answers(runner: CommandRunner) -> bool:
    code, out, err = runner.run(["systemctl", "--user", "is-system-running"])
    answer = (out or err).strip()
    return code in (0, 1) and answer not in ("", "offline")


def systemctl(runner: CommandRunner, *args: str) -> tuple[int, str]:
    code, out, err = runner.run(["systemctl", "--user", *args], timeout=60.0)
    return code, (err or out).strip()


def active(runner: CommandRunner) -> bool | None:
    """Whether the unit runs (None: systemd's user manager does not answer)."""
    if not systemd_answers(runner):
        return None
    return systemctl(runner, "is-active", UNIT)[0] == 0
