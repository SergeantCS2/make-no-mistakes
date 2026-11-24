"""Whether the process that made a run still runs (`process_started`, which `runs`
re-exports): its pid answers and, when its start was recorded, it
started then, since after a reboot the number can belong to another process."""

from __future__ import annotations

import os
from pathlib import Path


def alive(pid: int, started: int | None = None) -> bool:
    """The process that made the run still runs: its pid answers and, when its start was
    recorded, it started then (after a reboot the number can belong to another process)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's process holds that number now
        return False
    return started is None or process_started(pid) in (None, started)


def process_started(pid: int) -> int | None:
    """When a process started, in clock ticks since boot (`/proc/<pid>/stat`, field 22);
    None where /proc cannot say."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[-1].split()  # the name in brackets may hold spaces
    return int(fields[19]) if len(fields) > 19 and fields[19].isdigit() else None
