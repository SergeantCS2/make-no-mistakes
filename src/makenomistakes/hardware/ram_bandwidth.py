"""This box's RAM bandwidth (`mnm doctor --deep`, on every physical core), without a
dependency. One copy is a `bytearray` copied into a second one allocated and written once
before timing (slice assignment: a memcpy), read and written once: allocating new `bytes`
for every timed copy would time page faults instead (3.0 GB/s on an RTX 4070 machine, 2.0 on
a 4-vCPU machine where the copy into warm pages reads 11).

One thread cannot read a box's RAM: four vCPUs read 20.4 GB/s on one and 74.9 summed over
four processes, and one Genoa core reads 37 GB/s against its socket's 375. So `measure_all`
runs one process per physical core, up to
`MAX_WORKERS`, each its own `WORKER_SIZE` pair written once, all copying together after a
start signal; a run's figure is every worker's bytes over the slowest worker's time, the
best of `RUNS` runs. The one-thread figure (`measure`) is kept beside it. Unverified against
STREAM; the RAM-held experts' own rate is the CPU's (`fit.cpu_rates`), well under either."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

from makenomistakes.tune.cpus import read_layout

SIZE = 1 << 30
WORKER_SIZE = 256 << 20
RUNS = 3
MAX_WORKERS = 16
PROBE = 2  # 1: one thread (`ram_gbps` of an older file); 2: every core

Clock = Callable[[], float]
# Every worker's seconds for each run: `runs` lists of `workers` times (swappable)
Runner = Callable[[int, int, int], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class RamReading:
    """What the doctor saves: the aggregate over the workers, and one thread's."""

    gbps: float  # read + write, every worker together
    one_thread_gbps: float
    workers: int


def measure(size: int = SIZE, runs: int = RUNS, clock: Clock = time.perf_counter) -> float:
    """Read + write GB/s (10^9 bytes a second) on one thread, the best of `runs` copies of
    `size` bytes."""
    source = bytearray(size)
    dest = bytearray(size)
    dest[:] = source  # every page of both touched once: the timed runs never fault one in
    best = 0.0
    for _ in range(runs):
        start = clock()
        dest[:] = source
        seconds = clock() - start
        if seconds > 0:
            best = max(best, 2 * size / seconds / 1e9)
    return best


def physical_cores(root: Path = Path("/")) -> int:
    """Physical cores from `/proc/cpuinfo` (the tuner's reading), else the logical count."""
    return read_layout(root, wsl=False).cores or os.cpu_count() or 1  # cores, never WSL's vCPUs


def measure_all(
    workers: int | None = None,
    size: int = WORKER_SIZE,
    runs: int = RUNS,
    runner: Runner | None = None,
    one_thread: Callable[[], float] = measure,
) -> RamReading:
    """Every worker copying at once (see the module's text), and one thread's figure."""
    count = max(1, min(workers or physical_cores(), MAX_WORKERS))
    times = (runner or _processes)(count, size, runs)
    best = 0.0
    for run in times:
        slowest = max(run, default=0.0)
        if slowest > 0:
            best = max(best, 2 * size * len(run) / slowest / 1e9)
    return RamReading(gbps=best, one_thread_gbps=one_thread(), workers=count)


class RamProbeError(Exception):
    """A copying process stopped before it answered (the system's out-of-memory killer, a
    MemoryError in the worker): the reading is not had, never a traceback."""


def _worker(conn: Connection, size: int, runs: int) -> None:
    source = bytearray(size)
    dest = bytearray(size)
    dest[:] = source  # warm: no page fault inside a timed run
    conn.send("ready")
    for _ in range(runs):
        conn.recv()  # the start signal, sent to every worker together
        start = time.perf_counter()
        dest[:] = source
        conn.send(time.perf_counter() - start)
    conn.close()


def _processes(count: int, size: int, runs: int) -> list[list[float]]:
    """`count` processes, each its own pair, started together `runs` times."""
    # spawned, never forked: the web UI and the router run threads (a fork copies their locks)
    context = multiprocessing.get_context("spawn")
    pipes: list[Connection] = []
    procs = []
    try:
        for _ in range(count):
            parent, child = context.Pipe()
            proc = context.Process(target=_worker, args=(child, size, runs), daemon=True)
            proc.start()
            child.close()  # the worker's end is its own: its death reads as EOF, never a wait
            pipes.append(parent)
            procs.append(proc)
        for pipe in pipes:
            pipe.recv()
        found: list[list[float]] = []
        for _ in range(runs):
            for pipe in pipes:
                pipe.send("go")
            found.append([float(pipe.recv()) for pipe in pipes])
        return found
    except (EOFError, OSError) as exc:  # BrokenPipeError is an OSError
        raise RamProbeError(
            "a copying process stopped before it answered (most likely the memory for its "
            "copy could not be had)"
        ) from exc
    finally:
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
