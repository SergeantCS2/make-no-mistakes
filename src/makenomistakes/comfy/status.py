"""ComfyUI on the Status page and in `mnm ls`: whether it is installed and running, the VRAM it
holds (its own `system_stats`), the run it is making or how long it has been idle and when the
router's sweep frees its card, in one place."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from makenomistakes.comfy.engine import ComfyDockerError, ComfyEngine, State
from makenomistakes.comfy.idle import FREED, IDLE_SECONDS, LAST_RUN
from makenomistakes.comfy.image import ComfyError
from makenomistakes.local_time import local
from makenomistakes.runs import RunError, RunKind, RunRecord, RunStore

GIB = 2**30
BAR_SECONDS = 0.5  # the status bar asks every 10 s: a ComfyUI that does not answer is skipped


@dataclass(frozen=True)
class ComfyStatus:
    state: State
    error: str | None = None  # its ledger cannot be read (the doctor says the same)
    docker: str | None = None  # Docker could not be asked: the state is unknown, and why
    held_bytes: int | None = None  # what it holds in all (`held_estimate`); None: no answer
    held_method: str = ""  # how that figure was reached (the panel's hover)
    run: RunRecord | None = None  # a workflow run going now
    step: str = ""  # that run's last progress line, when the web started it ("Node 12: …")
    # ComfyUI's queue has work mnm did not start (or not yet recorded); None: it did not
    # answer, unknown (never "idle")
    queued: bool | None = False
    idle_since: datetime | None = None  # the last run's end (aware)
    frees_at: datetime | None = None  # when the router's sweep frees the card; None: freed
    freed_at: datetime | None = None

    @property
    def installed(self) -> bool:
        return self.state is not State.NOT_INSTALLED or self.error is not None or self.unknown

    @property
    def unknown(self) -> bool:
        """Docker did not answer: running or not cannot be told (never "stopped")."""
        return self.docker is not None

    @property
    def running(self) -> bool:
        return self.state is State.RUNNING

    @property
    def holds(self) -> bool:
        """It holds card memory worth a word (more than its CUDA context's ~0.1 GiB)."""
        return self.running and (self.held_bytes or 0) >= GIB // 4

    @property
    def busy(self) -> bool:
        return self.run is not None or self.queued is True

    def doing(self) -> str:
        """What it is doing, in a few words: the run, or how long it has been quiet."""
        if self.run is not None:
            since = f"{local(self.run.created_at):%H:%M}"
            return f"a run of {self.run.target} in progress since {since}" + (
                f" ({self.step})" if self.step else ""
            )
        if self.queued is not False:
            return "a job in its queue" if self.queued else "not answering whether it runs a job"
        idle = (
            f"idle since {local(self.idle_since):%H:%M}"
            if self.idle_since is not None
            else "idle, no run since it started"
        )
        if self.frees_at is not None and self.frees_at > datetime.now(UTC):
            return f"{idle}; the router frees its card at {local(self.frees_at):%H:%M}"
        if self.frees_at is not None:  # past due: the router's sweep has not run (is it up?)
            return (
                f"{idle}; the router's next sweep frees its card (due {local(self.frees_at):%H:%M})"
            )
        if self.freed_at is not None:  # what the free left, said where it is read
            now = (
                f" (holding about {self.held_bytes / GIB:.1f} GiB now: its CUDA context and "
                "PyTorch's pool stay while it runs)"
                if self.held_bytes is not None
                else ""
            )
            return f"{idle}; its card freed at {local(self.freed_at):%H:%M}{now}"
        return idle

    def words(self) -> str:
        """`running, holding 9.8 GiB (a run of wan5b in progress since 04:20)`."""
        if self.error is not None:
            return f"its state cannot be read: {self.error}"
        if self.docker is not None:
            return f"unknown (Docker unavailable: {self.docker})"
        if not self.running:
            return self.state.value
        held = (
            f"holding about {self.held_bytes / GIB:.1f} GiB"
            if self.held_bytes is not None
            else "not answering"
        )
        return f"running, {held} ({self.doing()})"


@dataclass(frozen=True)
class ComfyFeed:
    """What the web's Status page and status bar ask about ComfyUI: its whole status, and the
    bar's quick look at what it holds."""

    status: Callable[[], ComfyStatus] = lambda: ComfyStatus(State.NOT_INSTALLED)
    held: Callable[[], int | None] = lambda: None


def comfy_status(
    engine: ComfyEngine,
    runs: RunStore | None = None,
    step_of: Callable[[RunRecord], str] = lambda _run: "",
) -> ComfyStatus:
    """What the Status page and `mnm ls` say about ComfyUI (Docker, the ledger, its HTTP)."""
    try:
        ledger = engine.ledger()
    except ComfyError as exc:
        return ComfyStatus(State.NOT_INSTALLED, error=str(exc))
    if ledger is None:
        return ComfyStatus(State.NOT_INSTALLED)
    try:
        state = engine.observe()  # one `docker inspect` for the whole status
    except ComfyDockerError as exc:
        return ComfyStatus(State.UNKNOWN, docker=str(exc))
    address = engine.address(state)
    if state is not State.RUNNING or address is None:
        return ComfyStatus(state)
    stats = engine.stats(address)
    going = _going(runs)
    last, freed = _stamp(engine.base / LAST_RUN), _stamp(engine.base / FREED)
    pending = last is not None and (freed is None or freed < last)
    ran = ran_since(engine) or going is not None
    return ComfyStatus(
        state,
        held_bytes=stats.held_estimate(ran) if stats is not None else None,
        held_method=stats.held_method(ran) if stats is not None else "",
        run=going,
        step=step_of(going) if going is not None else "",
        queued=engine.queue_busy(address) if going is None else False,
        idle_since=last,
        frees_at=last + timedelta(seconds=IDLE_SECONDS) if last and pending else None,
        freed_at=freed if freed and not pending else None,
    )


def held_for_bar(engine: ComfyEngine) -> int | None:
    """The VRAM ComfyUI holds, for the status bar: its ledger and one quick question (no
    Docker); None when it is not installed, not started, or does not answer quickly."""
    try:
        ledger = engine.ledger()
    except ComfyError:
        return None
    if ledger is None or ledger.address is None:
        return None
    stats = engine.stats(timeout=BAR_SECONDS)
    return stats.held_estimate(ran_since(engine)) if stats is not None else None


def ran_since(engine: ComfyEngine) -> bool:
    """A run has ended since ComfyUI's start (its ledger was last written then) and since
    its card was last freed: its dynamic-VRAM buffers are held (`api.held_estimate`)."""
    last = _stamp(engine.base / LAST_RUN)
    others = [_stamp(engine.base / FREED), _stamp(engine.ledger_path)]
    return last is not None and all(t is None or t <= last for t in others)


def _going(runs: RunStore | None) -> RunRecord | None:
    if runs is None:
        return None
    try:
        found = [r for r in runs.all() if r.kind is RunKind.WORKFLOW and r.going]
    except RunError:
        return None
    return max(found, key=lambda r: r.created_at) if found else None


def _stamp(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return None
