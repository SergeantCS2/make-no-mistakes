"""A plain-file download's progress and its stall (Ollama, ModelScope, Civitai): a Hub
download shows a bar, and so does this one (tqdm, as huggingface_hub draws it; on a terminal
only), and a download that gets no byte for a minute stops (httpx's read timeout), kept to
resume."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import httpx
from tqdm import tqdm

STALL_SECONDS = 60.0  # no byte for this long: stalled (each client's read timeout)
MIB = 2**20


@dataclass(frozen=True)
class Transfer:
    """One file coming in: where from (in messages), its listed size (0: unknown) and the
    User-Agent the request sends."""

    source: str
    size: int = 0
    agent: str = ""


class FileLine(str):
    """A progress line naming a file a download is about to fetch or check: its own bar
    (tqdm, on stderr) may follow, so a spinner steps aside for it (`cli._app.spinner_progress`
    keys on this type, never on the words)."""

    __slots__ = ()


class StalledError(OSError):
    """No byte for STALL_SECONDS: the `.part` stays, the next run resumes it."""


class _Sink(Protocol):
    def write(self, data: bytes, /) -> int: ...


def stalled(transfer: Transfer, got: int) -> StalledError:
    of = f" of {transfer.size / MIB:,.0f} MiB" if transfer.size else ""
    return StalledError(
        f"{transfer.source}'s download stalled: no byte for {STALL_SECONDS:.0f} s, after "
        f"{got / MIB:,.0f} MiB{of}. Run the same command again to resume (what came is kept)."
    )


class _Bar(Protocol):
    def update(self, n: float | None = 1) -> bool | None: ...


class PlainProgress:
    """Without a terminal (a piped install, a session with no tty, the services): a line at
    each tenth with the rate, where tqdm draws nothing."""

    def __init__(
        self,
        name: str,
        size: int | None,
        have: int,
        say: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name, self._size, self._got, self._say, self._clock = name, size, have, say, clock
        self._began, self._from = clock(), have
        self._tenth = self._tenths()

    def _tenths(self) -> int:
        return self._got * 10 // self._size if self._size else 0

    def update(self, n: float | None = 1) -> bool | None:
        self._got += int(n or 0)
        tenth = self._tenths()
        if not self._size or tenth <= self._tenth:
            return None
        self._tenth = tenth
        seconds = max(self._clock() - self._began, 1e-6)
        rate = (self._got - self._from) / seconds / 1e6
        self._say(
            f"{self._name}: {min(tenth, 10) * 10}% ({self._got / MIB:,.0f} of "
            f"{self._size / MIB:,.0f} MiB, {rate:.1f} MB/s)"
        )
        return None


def _stderr_line(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


@contextmanager
def progress_bar(name: str, size: int | None, have: int) -> Iterator[_Bar]:
    """The Hub's bar for one file: bytes, %, MiB/s; drawn on a terminal only, a line a tenth
    with the rate otherwise (`PlainProgress`)."""
    if not sys.stderr.isatty():
        yield PlainProgress(name, size, have, _stderr_line)
        return
    with tqdm(
        total=size or None,
        initial=have,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=name,
        disable=None,  # off when stderr is not a terminal (the web's jobs, the services)
        leave=True,
    ) as bar:
        yield bar


def copy_chunks(
    chunks: Iterable[bytes],
    out: _Sink,
    bar: _Bar,
    seen: Callable[[bytes], None] = lambda _chunk: None,
) -> None:
    """Write every chunk as it comes, feed `seen` (a digest), move the bar. A stall is the
    client's read timeout (`worded_timeout`): every byte that came is written first."""
    for chunk in chunks:
        out.write(chunk)
        seen(chunk)
        bar.update(len(chunk))


def worded_timeout(exc: httpx.HTTPError, transfer: Transfer, got: int) -> OSError:
    """A read that waited a minute for a byte is a stall; anything else stays itself."""
    if isinstance(exc, httpx.ReadTimeout):
        return stalled(transfer, got)
    return OSError(f"{transfer.source} download stopped: {exc} (start again to resume)")
