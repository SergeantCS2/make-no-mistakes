"""The stage-by-stage loop every pipeline shares: one item at a time, the run record written
after each, so a failure or a restart loses at most the item in flight.

`advance` runs every item not done yet, stage by stage, and stops at a review point the user
has not passed (unless the run goes straight through), at the end, or at a failure. The user
then continues, or redoes one item, which resets the items made from it (each pipeline says
which). The loop asks nothing of the router between stages: a stage's first request loads
its model, and the router parks what must go by its own rules.

A pipeline fills in the hooks: its stages and labels, what a run's settings mean
(`_context`), what one item makes (`_make`), where it waits (`_waits`) and what a redo makes
stale (`_stale`).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.progress_words import Whole
from makenomistakes.runs import (
    RunError,
    RunKind,
    RunRecord,
    RunState,
    RunStore,
    StepRecord,
    StepState,
    process_started,
)

Report = Callable[[str], None]


def quiet(_message: str) -> None:
    return None


def refuse_if_going(run: RunRecord, what: str) -> None:
    """The one refusal of a move while the run runs in a live process."""
    if run.going:
        raise PipelineError(f"{run.id} is running: wait for it, then {what}.")


class StageRunner:
    """The loop; a pipeline subclasses it (`FilmRunner`, `SongRunner`)."""

    name: str = ""  # the run's target (`short-film`, `song`)
    stages: tuple[str, ...] = ()
    labels: dict[str, str] = {}  # noqa: RUF012 - each pipeline sets its own

    def __init__(
        self,
        store: RunStore,
        http: Callable[[], httpx.Client],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.http = http
        self.clock = clock  # a step's seconds: never the wall clock, which WSL steps back

    # -- the hooks -------------------------------------------------------------------------

    def _fresh(self) -> None:
        """Forget what a pass cached (a workflow that may have changed since)."""

    def _context(self, run: RunRecord, report: Report) -> Any:
        """What the run's settings mean for this pass (the film's inputs, the song's)."""
        raise NotImplementedError

    def _skip(self, run: RunRecord, stage: str, context: Any, report: Report) -> bool:
        """True to pass over a whole stage (no voice: no lines)."""
        return False

    def _waits(self, run: RunRecord, stage: str, context: Any) -> bool:
        """True to stop for the user after `stage`."""
        return False

    def _make(self, run: RunRecord, step: StepRecord, context: Any, http: httpx.Client) -> None:
        raise NotImplementedError

    def _which(self, run: RunRecord, step: StepRecord, context: Any) -> tuple[str, str]:
        """The item's words (`Stills 2 of 3`) and what follows them while it runs."""
        label = self.labels.get(step.stage, step.stage)
        count = len(self.items(run, step.stage))
        return (f"{label} {step.index + 1} of {count}" if count > 1 else label), ""

    def _says_done(self, step: StepRecord) -> bool:
        """Whether a finished item gets its own line with its seconds."""
        return True

    def _finish(self, run: RunRecord) -> RunRecord:
        raise NotImplementedError

    def _stale(self, run: RunRecord, step: StepRecord) -> list[StepRecord]:
        """The items a redo of `step` makes stale, itself included; may change the run."""
        return [step]

    def _ours(self, run: RunRecord) -> bool:
        """Whether this runner runs `run` (a pipeline run of its template)."""
        return run.target == self.name

    # -- running ---------------------------------------------------------------------------

    def advance(self, run_id: str, report: Report = quiet) -> RunRecord:
        """Every item not done yet, until a review point, the end, or a failure."""
        self._fresh()
        with self.store.locked():  # read, check, take: one step against another process
            run = self._get(run_id)
            if (
                run.state is RunState.RUNNING
                and run.pid not in (None, os.getpid())
                and not run.interrupted
            ):
                raise PipelineError(f"{run.id} is running already (process {run.pid}).")
            run.state, run.pid, run.waiting_at, run.error = RunState.RUNNING, os.getpid(), None, ""
            run.pid_started = process_started(os.getpid())  # a pid reused after a reboot differs
            self.store.save(run)
        context = self._context(run, report)
        try:
            with self.http() as http:
                for stage in self.stages:
                    if self._skip(run, stage, context, report):
                        continue
                    for step in self.items(run, stage):
                        if step.state not in (StepState.DONE, StepState.SKIPPED):
                            self._run_step(run, step, context, http, report)
                    if self._waits(run, stage, context):
                        run.state, run.waiting_at, run.pid = RunState.WAITING, stage, None
                        self.store.save(run)
                        return run
        except (StepError, PipelineError) as exc:
            run.pid = None
            return self.store.fail(run, str(exc))
        except BaseException as exc:  # never left running in a live process: Resume works
            run.pid = None
            self.store.fail(run, str(exc) or f"Interrupted ({type(exc).__name__}).")
            raise
        return self._finish(run)

    def items(self, run: RunRecord, stage: str) -> list[StepRecord]:
        return sorted((s for s in run.steps if s.stage == stage), key=lambda s: s.index)

    def _run_step(
        self,
        run: RunRecord,
        step: StepRecord,
        context: Any,
        http: httpx.Client,
        report: Report,
    ) -> None:
        label = self.labels.get(step.stage, step.stage)
        which, shape = self._which(run, step, context)
        report(f"{which}{shape}")  # a step under way: the CLI adds its "…"
        step.state, step.error = StepState.RUNNING, ""
        self.store.save(run)
        started = self.clock()
        try:
            self._make(run, step, context, http)
        except (StepError, PipelineError, OSError) as exc:
            step.state, step.error = StepState.FAILED, str(exc)
            self.store.save(run)
            raise StepError(f"{label} {step.index + 1}: {exc}") from exc
        if step.state is StepState.RUNNING:
            step.state = StepState.DONE
        step.seconds = max(self.clock() - started, 0.0)
        self.store.save(run)
        if step.state is StepState.DONE and self._says_done(step):
            report(Whole(f"{which}: {step.seconds:.1f} s."))

    # -- the user's moves ------------------------------------------------------------------

    def proceed(self, run_id: str, report: Report = quiet) -> RunRecord:
        """Past the review point the run waits at, then on."""
        run = self._get(run_id)
        if run.state is not RunState.WAITING or run.waiting_at is None:
            raise PipelineError(f"{run.id} is not waiting for you.")
        self._passing(run, run.waiting_at)
        run.reviewed.append(run.waiting_at)
        self.store.save(run)
        return self.advance(run.id, report)

    def _passing(self, run: RunRecord, stage: str) -> None:
        """A check before the user passes a review point (a song needs its pick)."""

    def redo(self, run_id: str, stage: str, index: int, report: Report = quiet) -> RunRecord:
        """Make one item again, and what was made from it; the run then goes on as far as it
        may (an item redone at its review point stays there)."""
        run = self._get(run_id)
        refuse_if_going(run, "redo")
        self.make_again(run, stage, index)
        return self.advance(run.id, report)

    def make_again(self, run: RunRecord, stage: str, index: int = 0) -> None:
        """One item and what was made from it reset for the next pass, saved (a redo's, or a
        move's inside its claim)."""
        step = run.step(stage, index)
        if step is None:
            raise PipelineError(f"{run.id} has no {stage} {index + 1}.")
        for item in self._stale(run, step):
            item.state, item.error = StepState.PENDING, ""
        self.store.save(run)

    @contextmanager
    def claim(self, run_id: str, what: str, *, hand_back: bool = False) -> Iterator[RunRecord]:
        """The run held for a move of the user's that makes something before the run goes
        on (an edit, a new take): refused while it runs, running under this process while
        the move works, and put back exactly as it was (state, review point, process, error)
        if the move fails. The pass that follows (`advance`) takes it from this process; a
        move no pass follows (an export again) asks `hand_back`, and the run is put back as
        it was after it too, what the move saved kept."""
        with self.store.locked():
            run = self._get(run_id)
            refuse_if_going(run, what)
            before = (run.state, run.waiting_at, run.pid, run.pid_started, run.error)
            run.state, run.pid = RunState.RUNNING, os.getpid()
            run.pid_started = process_started(os.getpid())
            self.store.save(run)
        try:
            yield run
        except BaseException:
            self._put_back(run_id, before)
            raise
        if hand_back:
            self._put_back(run_id, before)

    def _put_back(self, run_id: str, before: tuple[Any, ...]) -> None:
        with self.store.locked():
            back = self._get(run_id)
            back.state, back.waiting_at, back.pid, back.pid_started, back.error = before
            self.store.save(back)

    def _get(self, run_id: str) -> RunRecord:
        try:
            run = self.store.get(run_id)
        except RunError as exc:
            raise PipelineError(str(exc)) from exc
        if run.kind is not RunKind.PIPELINE:
            raise PipelineError(f"{run.id} is not a pipeline run.")
        if not self._ours(run):
            raise PipelineError(f"{run.id} is a {run.target} run, not a {self.name} run.")
        return run
