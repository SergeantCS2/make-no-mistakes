"""Who runs a tune, or downloads a quality plan's quants, now: one rule for every reader.

A lease is a process (its pid and its identity beyond the pid, `run_token`: a pid a SIGTERM'd
router left, or one reused after a boot, holds nothing) and a token naming the run or the
approval in that process. It is live while that process runs and, in this very process, while
its run or approval is still going (`going`, a process-local registry): so a run this process
left behind (an end that failed, a lease never let go) holds nothing, and a run in this very
process (the router's own, the web UI's) is told apart from another one by its token, never
by the pid alone (else a leftover lease of the router's own pid would refuse every later
cancel and plan until the router restarted)."""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from pydantic import BaseModel

_GOING: set[str] = set()  # this process's runs and approvals going now, by their tokens
_GOING_LOCK = threading.Lock()


@contextmanager
def going() -> Iterator[str]:
    """A run or an approval goes on in this process until the block ends: its token, live
    for every reader here while it goes (a `finally` lets go of it, whatever ends it)."""
    token = uuid.uuid4().hex
    with _GOING_LOCK:
        _GOING.add(token)
    try:
        yield token
    finally:
        with _GOING_LOCK:
            _GOING.discard(token)


def _goes(token: str | None) -> bool:
    with _GOING_LOCK:
        return token is not None and token in _GOING


class Lease(BaseModel):
    pid: int
    since: str | None = None  # `run_token` of that process when it took the lease
    token: str | None = None  # the run or approval in that process (`going`)

    @classmethod
    def here(cls, token: str) -> Self:
        return cls(pid=os.getpid(), since=run_token(), token=token)

    @property
    def alive(self) -> bool:
        """Its run or approval goes on: here while its token is going, elsewhere while its
        process runs and is the one that took it."""
        if self.pid == os.getpid():
            return _goes(self.token)
        return alive(self.pid, self.since)

    def same(self, other: Lease | None) -> bool:
        """`other` is this very lease (its process, that process's identity, its token)."""
        return other is not None and (other.pid, other.since, other.token) == (
            self.pid,
            self.since,
            self.token,
        )


def run_token(pid: int | None = None) -> str | None:
    """A process's identity beyond its pid: this boot and the process's start time, from
    /proc (a router stopped by SIGTERM leaves its pid in the state, and after a reboot that pid
    can be another process's); None where /proc does not say."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        stat = Path(f"/proc/{pid or os.getpid()}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2 :].split()  # after the command, which may hold spaces
    return f"{boot}:{fields[19]}" if len(fields) > 19 else None  # field 22: its start time


def alive(pid: int | None, since: str | None = None) -> bool:
    """`pid` runs, and is the process that took the lease when its identity was recorded."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return since is None or run_token(pid) in (since, None)
