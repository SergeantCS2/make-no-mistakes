"""The last run's engine log, kept when `mnm stop` removes its container, so a crashed
engine's cause can still be read after the stop. `mnm logs` falls back to it once Docker says
the container is gone; `mnm rm` deletes it. A new `start` replaces a dead container without
keeping its log: the start's own error report (or auto-fix's retry, which read it) has already
quoted it."""

from __future__ import annotations

import contextlib
from pathlib import Path

from makenomistakes.supervisor import DockerSupervisor, SupervisorError

LOGS_DIR = "logs"
KEPT_NOTE = "(the last run's log, kept when its container was removed)\n"


def log_path(state_dir: Path, name: str) -> Path:
    return state_dir / LOGS_DIR / f"{name}.log"


def keep_log(supervisor: DockerSupervisor, state_dir: Path, name: str) -> None:
    """Best effort: a container Docker cannot hand the log of keeps nothing (and stops all
    the same)."""
    try:
        text = supervisor.logs(name, 0)
    except SupervisorError:
        return
    if text.strip():
        path = log_path(state_dir, name)
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def kept_log(state_dir: Path, name: str, tail: int) -> str | None:
    """The kept log's last `tail` lines (0: all of it), or None when there is none."""
    try:
        lines = log_path(state_dir, name).read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return None
    return KEPT_NOTE + "".join(lines[-tail:] if tail > 0 else lines)


def forget_log(state_dir: Path, name: str) -> None:
    log_path(state_dir, name).unlink(missing_ok=True)


def no_log_words(name: str, off: bool) -> str:
    """What `mnm logs` says for an instance with no container and no kept log, in place of
    Docker's own "No such container: mnm-tiny", which `mnm report` would count an error."""
    state = "stopped (off)" if off else "parked"
    return (
        f"{name} is {state} (its container is gone) and no log is kept: `mnm clean logs` "
        f"removed it, or it has not run since it was added. `mnm start {name}` writes a new one."
    )
