"""Background jobs for the web UI: downloads and starts run in threads, pages poll status.

A job is identified by (kind, instance); at most one job runs per instance at a time.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Collection
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field

from makenomistakes.diagnose import Diagnosis
from makenomistakes.instance_types import CHANGED, InstanceError

Report = Callable[[str], None]
Work = Callable[[Report], list[str]]  # returns its notes, each worded whole (`changed`)
Spawn = Callable[[Callable[[], None]], None]
RECENT = timedelta(minutes=30)  # how long an instance page keeps showing a finished job


def changed(what: str) -> str:
    """A job note for what auto-fix changed and saved (the CLI prints the same label)."""
    return f"{CHANGED}: {what}"


BUILD_LINES = 40  # what a build's job keeps of its output for the page


class JobFailed(Exception):  # noqa: N818 - read as "the job failed", like the state
    """A job that ends failed with notes of its own (an engine update's lines)."""

    def __init__(self, message: str, notes: list[str]) -> None:
        super().__init__(message)
        self.notes = notes


class JobKind(StrEnum):
    DOWNLOAD = "download"
    START = "start"
    TRY = "try"  # the web's Test it panel (`mnm try`)
    PIPELINE = "pipeline"  # a pipeline run's steps, keyed `pipeline:<run id>`
    RUN = "run"  # the web's Run tab: the user's prompt, kept as a run (`mnm run`)
    ENGINES = "engines"  # an engine update or self-test (`mnm engines update|test`)
    CHECK = "check"  # the doctor's GPU container test (`mnm doctor --deep`)
    APPS = "apps"  # a workspace app's install, start, stop, update or rollback
    BACKUP = "backup"  # the web's Back up now (`mnm backup --auto`)
    TUNE = "tune"  # the Tuning tab's Optimise for speed or quality and Cancel (`mnm tune`)
    NIGHTLY = "nightly"  # System → Nightly's Run now (`mnm nightly run --now`)
    ADOPT = "adopt"  # System → Machine's Adopt and Dry run (`mnm machine adopt`)


class JobState(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    kind: JobKind
    instance: str
    state: JobState = JobState.RUNNING
    message: str = ""
    error: str | None = None
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # what to say when it ends, worded whole
    lines: list[str] = Field(default_factory=list)  # the last progress lines, when kept (a build)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    label: str = ""  # a title of its own, for a job not about one instance
    # How long it took, from the monotonic clock: WSL steps the wall clock back now and then,
    # so `finished_at - started_at` can come out short or negative. The stamps above
    # are for the user to read; `seconds` is the duration.
    began: float = Field(default_factory=time.monotonic, exclude=True)
    seconds: float | None = None

    @property
    def title(self) -> str:
        return self.label or f"{self.kind.value} {self.instance}"

    @property
    def running(self) -> bool:
        return self.state is JobState.RUNNING

    def ago(self, now: datetime | None = None) -> str:
        """ "started 5 s ago" while running, "finished 3 min ago" after."""
        verb = "finished" if self.finished_at else "started"
        return f"{verb} {ago(self.finished_at or self.started_at, now)}"


def ago(moment: datetime, now: datetime | None = None) -> str:
    """`5 s ago`, `3 min ago`, `2 h ago`, `4 days ago`: the one wording, jobs and pages alike."""
    seconds = int(((now or datetime.now(UTC)) - moment).total_seconds())
    if seconds < 60:
        return f"{max(seconds, 0)} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} days ago"


def _thread(target: Callable[[], None]) -> None:
    threading.Thread(target=target, daemon=True).start()


class JobRegistry:
    """Thread-backed jobs with a status line. `spawn` is injectable so tests run inline."""

    def __init__(
        self,
        spawn: Spawn = _thread,
        keep: int = 50,
        on_finish: Callable[[Job], None] | None = None,  # a notice when it ends
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._spawn = spawn
        self._keep = keep
        self._on_finish = on_finish

    def start(
        self,
        kind: JobKind,
        key: str,
        work: Work,
        *,
        busy: str,
        label: str = "",
        keep_lines: int = 0,
    ) -> Job:
        """`start_unless_active` for a per-instance action: a job of
        another kind holding the key is refused in words (`busy`: what to do once it ends,
        "tune it"), never swallowed; the same kind's running job is returned (a double
        click). The one check every per-instance action shares."""
        with self._lock:
            running = self._active_locked(key)
            if running is not None and running.kind is not kind:
                raise InstanceError(f"{key} is busy ({running.title}); {busy} when that is done.")
        return self.start_unless_active(kind, key, work, label, keep_lines)

    def refuse_if_busy(self, key: str, what: str) -> None:
        """A second job for an instance would be swallowed by the first: say so instead."""
        busy = self.active_for(key)
        if busy is not None:
            raise InstanceError(f"{key} is busy ({busy.title}); {what} when that is done.")

    def start_unless_active(
        self, kind: JobKind, instance: str, work: Work, label: str = "", keep_lines: int = 0
    ) -> Job:
        """Start a job for the instance, or return the one already running for it. With
        `keep_lines`, the last that many progress lines stay on the job (a build's output,
        which the page shows as it comes)."""
        with self._lock:
            running = self._active_locked(instance)
            if running is not None:
                return running.model_copy()
            job = self._new_locked(kind, instance, label)
        return self._launch(job, work, keep_lines)

    def start_alone(  # noqa: PLR0913 - the job, then what it must not run beside
        self,
        kind: JobKind,
        key: str,
        work: Work,
        *,
        among: Collection[str],
        then: str,
        check: Callable[[], None] = lambda: None,
        label: str = "",
        keep_lines: int = 0,
    ) -> Job:
        """A job under `key` unless one runs under any key of `among` (`key`'s own too):
        refused in words naming it (`then`: what to do once it ends), never swallowed.
        `check` raises InstanceError when another process holds what the job needs. Checked
        and started under one lock, so two clicks never both start (ComfyUI's engine and
        packs jobs)."""
        with self._lock:
            for running in self._jobs.values():
                if running.running and running.instance in among:
                    raise InstanceError(f"'{running.title}' is under way; {then} once that ends.")
            check()
            job = self._new_locked(kind, key, label)
        return self._launch(job, work, keep_lines)

    def _new_locked(self, kind: JobKind, key: str, label: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:8], kind=kind, instance=key, label=label)
        self._jobs[job.id] = job
        self._trim_locked()
        return job

    def _launch(self, job: Job, work: Work, keep_lines: int) -> Job:
        """Run `work` for the job on a thread of its own; it ends done or failed."""

        def report(message: str) -> None:
            with self._lock:
                job.message = message
                if keep_lines:  # `[-0:]` would keep every line at 1
                    job.lines = [*job.lines, message][-keep_lines:]

        def run() -> None:
            try:
                notes = work(report)
            except InstanceError as exc:
                notes = [changed(what) for what in exc.fixes]
                self._finish(job, JobState.FAILED, str(exc), exc.diagnoses, notes)
            except JobFailed as exc:
                self._finish(job, JobState.FAILED, str(exc), [], exc.notes)
            except Exception as exc:  # the UI must show any failure, whatever its type
                self._finish(job, JobState.FAILED, f"{type(exc).__name__}: {exc}", [])
            else:
                self._finish(job, JobState.DONE, None, [], notes)

        self._spawn(run)
        return job.model_copy()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())[::-1]  # by insertion: the clock may step back
            return [j.model_copy() for j in jobs[:limit]]

    def latest_for(
        self, instance: str, within: timedelta = RECENT, now: datetime | None = None
    ) -> Job | None:
        """The running job for the instance, else the newest one that finished within
        `within`: a page keeps showing how the last start ended (and what auto-fix saved)."""
        with self._lock:
            running = self._active_locked(instance)
            if running is not None:
                return running.model_copy()
            cutoff = (now or datetime.now(UTC)) - within
            for job in reversed(self._jobs.values()):  # insertion order is start order
                if job.instance == instance and job.finished_at and job.finished_at >= cutoff:
                    return job.model_copy()
            return None

    def forget(self, instance: str) -> None:
        """Drop the jobs of an instance or workflow that was removed (its removal is refused
        while one runs), so one added again under the same name starts without them."""
        with self._lock:
            for job_id in [j.id for j in self._jobs.values() if j.instance == instance]:
                del self._jobs[job_id]

    def running_jobs(self) -> dict[str, Job]:
        """Every running job by the instance (or key) it is for."""
        with self._lock:
            return {j.instance: j.model_copy() for j in self._jobs.values() if j.running}

    def active_for(self, instance: str) -> Job | None:
        with self._lock:
            job = self._active_locked(instance)
            return job.model_copy() if job else None

    def _active_locked(self, instance: str) -> Job | None:
        for job in self._jobs.values():
            if job.state is JobState.RUNNING and job.instance == instance:
                return job
        return None

    def _finish(
        self,
        job: Job,
        state: JobState,
        error: str | None,
        diagnoses: list[Diagnosis],
        notes: list[str] | None = None,
    ) -> None:
        with self._lock:
            job.state = state
            job.error = error
            job.diagnoses = diagnoses
            job.notes = notes or []
            job.finished_at = datetime.now(UTC)
            job.seconds = max(time.monotonic() - job.began, 0.0)
            ended = job.model_copy()
        if self._on_finish is not None:
            self._on_finish(ended)

    def _trim_locked(self) -> None:
        finished = [j for j in self._jobs.values() if j.state is not JobState.RUNNING]
        for old in finished[: max(len(finished) - self._keep, 0)]:  # oldest first, as added
            del self._jobs[old.id]
