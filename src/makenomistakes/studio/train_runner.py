"""The trainer's loop inside the router process: `card_runner.CardJobRunner`, the
loop the tuner runs too, over the trainings queued as runs (`train_jobs`), on the card gate
they share.

A training starts as any card job does (the router idle a minute, nothing held, no tune on the
card, the nightly holding nothing) once its image is on the box. Under the box's start lock it
makes room as the router would for a load that needs the whole card (ComfyUI's models freed,
every model the router may park parked), checking the cut before and after, and starts the
training's container, built for what the installed image is (its tag says CPU or CUDA 13).

**Anything that wants the card cuts it**: a request (through `CardGate.arrived`), the
nightly's hold, and, while the container runs, someone else's start or stop on the box (the
user's `mnm start`, a set, `try --all`, a self-test) or a ComfyUI run (a watcher that looks
every few seconds). A cut set before the container exists stops it as soon as it does. The
training stays queued and resumes from its checkpoint after the next idle minute; one that
finished as it was cut is finished. The end is one notice: the voice in the library, or the
reason it failed with its folder and log kept (its container stopped and removed first).
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ValidationError

from makenomistakes import box_events
from makenomistakes.card_jobs import CardGate
from makenomistakes.card_runner import CardJobRunner
from makenomistakes.comfy.engine import release_for
from makenomistakes.comfy.image import ComfyBusyError
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Note
from makenomistakes.notify import Notice
from makenomistakes.parkable import parkable
from makenomistakes.router_client import Traffic
from makenomistakes.runs import RunError, RunKind, RunRecord, RunStore, StepState, process_started
from makenomistakes.studio import train_jobs
from makenomistakes.studio.built_image import EXIT_WORDS, tool_error
from makenomistakes.studio.train_kinds import TrainKind, kinds_for
from makenomistakes.studio.trainer import (
    CONTAINER_PREFIX,
    Stop,
    TrainerError,
    TrainingContainer,
    read_progress,
)
from makenomistakes.swap import SwapService, start_turn
from makenomistakes.tune.trial_types import Cut

QUEUE_EVERY = 60.0  # how often the queue is read from the run records (a POST reads it at once)
WATCH_SECONDS = 5.0  # how often a running training looks for a start by hand or a ComfyUI run
JOIN_SECONDS = 60.0  # the watcher's last look (a probe reading the card) once the container ended


class TrainAsk(BaseModel):
    action: Literal["start", "cancel"] = "start"
    now: bool = False  # skip the idle minute (the rest still holds)


class TrainRunner(CardJobRunner[RunRecord, str]):
    noun = "training"
    name = "trainer"

    def __init__(  # noqa: PLR0913 - the router's parts the loop reads
        self,
        service: InstanceService,
        gate: CardGate,
        traffic: Callable[[], Traffic],
        kinds: Mapping[str, TrainKind],
        container: TrainingContainer,
        busy: Callable[[str], int],
        *,
        comfy_busy: Callable[[], str | None] = lambda: None,
        note: Note | None = None,
        notify: Callable[[Notice], None] = lambda _n: None,
        clock: Callable[[], float] = time.monotonic,
        watch_every: float = WATCH_SECONDS,
    ) -> None:
        super().__init__(service, gate, traffic, comfy_busy, note, clock)
        self._kinds = kinds
        self.container = container
        self._busy = busy
        self._notify = notify
        self._watch_every = watch_every
        self.store = RunStore.for_settings(service.settings)
        self._queue: list[str] = []
        self._scanned_at: float | None = None
        self._now: set[str] = set()
        self._cleared = False

    # --- the queue, as the shared loop asks for it -----------------------------------------

    def entries(self) -> Sequence[RunRecord]:
        """The queued trainings, oldest first; the run records are read in full only every
        `QUEUE_EVERY` (a song makes many runs), each queued one again every tick."""
        if self._scanned_at is None or self._clock() - self._scanned_at >= QUEUE_EVERY:
            self._scanned_at = self._clock()
            self._queue = [run.id for run in train_jobs.queued(self.store)]
        found = []
        for run_id in list(self._queue):
            try:
                run: RunRecord | None = self.store.get(run_id)
            except RunError:
                run = None
            if run is None or not train_jobs.waiting(run) or train_jobs.held_elsewhere(run):
                self._queue.remove(run_id)
                continue
            found.append(run)
        return found

    def key(self, entry: RunRecord) -> str:
        return entry.id

    def target(self, entry: RunRecord) -> str:
        return entry.target

    def asked_now(self, entry: RunRecord) -> bool:
        return entry.id in self._now

    def own_waits(self, entry: RunRecord) -> str | None:
        """Its kind's image on the box and what the kind waits for (a style's measured room),
        or why the training waits (never failed for it: its checkpoints stay for the image
        that comes back)."""
        try:
            kind = self.kind(entry)
            return kind.waits(entry, kind.image().installed().image)
        except TrainerError as exc:
            return str(exc)

    def kind(self, run: RunRecord) -> TrainKind:
        """What the run trains (`train_jobs.kind_of`); a kind this mnm does not know is a
        training a newer mnm queued, refused in words."""
        found = self._kinds.get(train_jobs.kind_of(run))
        if found is None:
            raise TrainerError(
                f"The training {run.id} is of a kind this mnm does not run "
                f"({train_jobs.kind_of(run)}): run `mnm update`."
            )
        return found

    def before_tick(self) -> None:
        if not self._cleared:  # asked again every tick until Docker answers (the router may
            with contextlib.suppress(TrainerError):  # start before it)
                self.clear_leftovers()
                self._cleared = True

    def stop_job(self, key: str) -> None:
        self.container.stop(key)

    def work(self, entry: RunRecord, cut: Cut) -> str:
        started: list[bool] = [False]
        try:
            return self._train(entry, cut, started)
        except Exception as exc:  # anything else fails the run, said, never a retry loop
            return self._failed(entry, f"The trainer failed: {exc}", started=started[0])

    def after(self, entry: RunRecord, result: str, cancelled: bool) -> None:
        self._now.discard(entry.id)
        if not cancelled:
            return
        try:
            fresh = self.store.get(entry.id)
        except RunError:
            return
        if train_jobs.waiting(fresh) or fresh.interrupted:  # else it ended as the cancel came
            self.store.fail(fresh, train_jobs.CANCELLED)
            self.say(f"The training of {entry.target} is cancelled.")

    # --- the router's side ----------------------------------------------------------------

    def ask(self, run_id: str, body: bytes) -> tuple[int, str]:
        """`POST /__mnm/train/<run>`: wake the loop for a queued training (`now` skips the
        idle minute) or cancel one; the status and the words for the CLI."""
        try:
            asked = TrainAsk.model_validate_json(body or b"{}")
        except ValidationError as exc:
            return 400, f"not a training request: {exc.errors()[0]['msg']}"
        try:
            run = self.store.get(run_id)
        except RunError as exc:
            return 404, str(exc)
        if run.kind is not RunKind.TRAINING:
            return 404, f"{run_id} is not a training (`mnm studio train ls`)."
        if asked.action == "cancel":
            return 200, self.cancel(run)
        if not train_jobs.waiting(run) and not run.interrupted:
            return 409, f"The training {run_id} has ended ({run.state.value})."
        self._scanned_at = None  # read the queue at the next tick
        if asked.now:
            self._now.add(run.id)
        when = "now" if asked.now else "once the router has been idle a minute"
        return 202, (
            f"Queued the training of {run.target}: the router's trainer starts it {when} "
            "(`mnm studio train ls`)."
        )

    def cancel(self, run: RunRecord) -> str:
        ended: list[str] = []

        def not_running() -> None:
            fresh = self.store.get(run.id)
            if train_jobs.waiting(fresh) or fresh.interrupted:
                self.store.fail(fresh, train_jobs.CANCELLED)
            else:
                ended.append(fresh.state.value)

        if self.cancel_running(run.id, not_running):
            return f"Cancelling the training of {run.target}: its container is being stopped."
        if ended:
            return f"The training {run.id} has ended ({ended[0]}): nothing to cancel."
        return f"The training of {run.target} is cancelled; its folder stays until `mnm clean`."

    def line(self) -> str | None:
        """`training alto (epoch 12 of 200)` while one runs."""
        run_id, words = self.active, self.running()
        if run_id is None or words is None:
            return None
        target = words.removeprefix("the training of ")
        try:
            progress = read_progress(self.store.folder(self.store.get(run_id)))
        except RunError:
            progress = None
        if progress is None or progress.epoch is None:
            return f"training {target} ({progress.stage if progress else 'starting'})"
        return f"training {target} (epoch {progress.epoch} of {progress.epochs})"

    def clear_leftovers(self) -> None:
        """A training's container left by a router that stopped: removed (the training is
        queued still, and resumes from its checkpoint)."""
        for name in self.container.leftovers():
            self.container.finished(name.removeprefix(CONTAINER_PREFIX))
            self.say(f"Removed {name}, a training's container left by a router that stopped.")

    # --- one training ---------------------------------------------------------------------

    def _train(self, run: RunRecord, cut: Cut, started: list[bool]) -> str:
        fresh = self._claim(run)
        if fresh is None:
            return "gone"  # cancelled between the queue's read and now
        kind = self.kind(fresh)
        if (stopped := self._start(fresh, cut, kind)) is not None:
            return stopped
        started[0] = True
        if cut.is_set():  # cut before the container existed: its `docker stop` found none
            self.container.stop(fresh.id)
        code, stop = self._waited(fresh, cut, kind)
        said = self.container.finished(fresh.id)
        started[0] = False
        if code == 0 and (self.store.folder(fresh) / kind.result).is_file():
            return self._finished(fresh, kind)  # done, even if a cut came as it ended
        if stop is not None and not stop.pause:
            return self._failed(fresh, stop.why, started=False)
        if cut.is_set():
            return self._paused(fresh, cut, stop.until if stop is not None else None)
        why = tool_error(said) or EXIT_WORDS.get(code, f"exit {code}")
        return self._failed(fresh, f"The training stopped: {why}", started=False)

    def _start(self, run: RunRecord, cut: Cut, kind: TrainKind) -> str | None:
        """Room made and the container started, under the box's start lock, the cut checked
        before and after the room (the tuner's trial does the same); None when it started,
        else what it came to."""
        with start_turn(self.service.settings, f"the training of {run.target}"):
            if cut.is_set():
                return self._paused(run, cut)
            try:
                parked = self._room(cut)
            except TrainerError as exc:
                self._said(run, str(exc))
                return "waits"
            if parked:
                them = "them" if len(parked) > 1 else "it"
                self.say(
                    f"Parked {', '.join(parked)} to make room for the training of "
                    f"{run.target} (the next request loads {them} again)."
                )
            if cut.is_set():  # a request came while room was made: the card is its
                return self._paused(run, cut)
            image = kind.image().installed().image
            gpu = kind.prepare(self.store, run, image)
            progress = read_progress(self.store.folder(run))
            again = f" again {progress.resumes}" if progress is not None and progress.epoch else ""
            self.say(f"The training of {run.target} starts{again} ({run.id}).")
            self.container.start(image, run.id, self.store.folder(run), gpu, kind.mounts())
        return None

    def _claim(self, run: RunRecord) -> RunRecord | None:
        """The run as it is now (a cancel may have ended it since the queue was read),
        adopted by this router: its pid, its step running."""
        with self.locked():
            try:
                fresh = self.store.get(run.id)
            except RunError:
                return None
            if (not train_jobs.waiting(fresh) and not fresh.interrupted) or (
                train_jobs.held_elsewhere(fresh)
            ):
                return None
            fresh.pid, fresh.pid_started = os.getpid(), process_started(os.getpid())
            step = fresh.step(train_jobs.STAGE)
            if step is not None:
                step.state = StepState.RUNNING
            self.store.save(fresh)
        return fresh

    def _waited(self, run: RunRecord, cut: Cut, kind: TrainKind) -> tuple[int, Stop | None]:
        """`docker wait`, while a watcher cuts it for someone else's start or a ComfyUI run,
        and stops it when this start's watch says so (a style's probe); the exit code and
        that word. The watcher is joined before the word is read: a cut that lands while the
        probe judges waits for its verdict."""
        done = threading.Event()
        watch = kind.watch(self.store, run)  # the container has just started
        stops: list[Stop] = []

        def watching() -> None:
            while not done.wait(self._watch_every):
                if cut.is_set():
                    return
                why = None
                if (stop := watch.check()) is not None:
                    stops.append(stop)
                    why = stop.why
                elif box_events.others_changing(self.service.settings):
                    why = "a start or stop by hand"
                else:
                    why = self._comfy_busy() or None  # running a workflow, or not saying
                if why is not None:
                    self.cut_running(why)  # never `arrived`: it would wait for this job
                    return

        watcher = threading.Thread(target=watching, daemon=True)
        watcher.start()
        try:
            code = self.container.wait(run.id)
        finally:
            done.set()
            watcher.join(timeout=JOIN_SECONDS)
            watch.close()
        return code, stops[0] if stops else None

    def _room(self, cut: Cut) -> list[str]:
        """ComfyUI's models freed, then every running model the router may park, parked,
        until a request cuts the training: the router counts a request busy only after it
        told the trainer (`arrived`), so its own model would be parked under it."""
        try:
            release_for(self.service.settings)(None)
        except ComfyBusyError as exc:
            raise TrainerError(str(exc)) from exc
        members = parkable(self.service, self._workflows, "", self._busy)
        if members is None:
            raise TrainerError("the sets cannot be read, so nothing is known to be parkable")
        swap = SwapService(self.service, busy=lambda r: self._busy(r.name))
        parked = []
        for recipe in members:
            if cut.is_set():
                break
            if self.service.status(recipe).running:
                swap.park(recipe)
                parked.append(recipe.name)
        return parked

    def _paused(self, run: RunRecord, cut: Cut, until: str | None = None) -> str:
        if self.was_cancelled(run.id):  # never the cut's last word: a request may follow
            return "cancelled"  # `after` ends the run
        progress = read_progress(self.store.folder(run))
        where = (
            f" in epoch {progress.epoch} of {progress.epochs}"
            if progress is not None and progress.epoch
            else ""
        )
        resumes = progress.resumes if progress is not None else "from the beginning"
        when = (
            f"{until} and the router has been idle a minute"
            if until is not None
            else "once the hold ends and the router has been idle a minute"
            if "holds the card" in cut.why
            else "once the router has been idle a minute"
        )
        self.say(
            f"The training of {run.target} was paused{where} ({cut.why}); it resumes "
            f"{resumes} {when}."
        )
        return "cut"

    def _finished(self, run: RunRecord, kind: TrainKind) -> str:
        try:
            done = kind.finish(self.store, run, self._seconds(run))
        except TrainerError as exc:
            return self._failed(run, str(exc), started=False)
        self.say(f"The training of {run.target} is done: {done.answer}.")
        self._notify(Notice(f"mnm studio train {run.target}", f"done: {done.answer}"))
        return "done"

    def _seconds(self, run: RunRecord) -> float | None:
        progress = read_progress(self.store.folder(run))
        return progress.at - run.created_at.timestamp() if progress is not None else None

    def _failed(self, run: RunRecord, why: str, *, started: bool) -> str:
        if started:  # its container stopped and removed: a tune must never start beside it
            self._stop_quietly(run.id)
            self.container.finished(run.id)
        fresh = self.store.get(run.id)
        step = fresh.step(train_jobs.STAGE)
        if step is not None:
            step.state, step.error = StepState.FAILED, why
        log = self.store.folder(fresh) / train_jobs.LOG
        if log.is_file():
            self.store.keep(fresh, log)
        self.store.fail(fresh, f"{why} (its folder and log: {self.store.folder(fresh)})")
        self.say(f"The training of {run.target} failed: {why}")
        self._notify(Notice(f"mnm studio train {run.target}", why, failed=True))
        return "failed"


def for_router(
    service: InstanceService,
    gate: CardGate,
    busy: Callable[[str], int],
    traffic: Callable[[], Traffic],
    note: Note,
    notify: Callable[[Notice], None],
) -> TrainRunner:
    """The loop `mnm router` runs on a thread of its own, sharing the tuner's card gate."""
    from makenomistakes.hardware.probe import gpu_device  # noqa: PLC0415
    from makenomistakes.hardware.runner import SubprocessRunner  # noqa: PLC0415
    from makenomistakes.studio.style_trainer import box_style_trainer  # noqa: PLC0415
    from makenomistakes.studio.trainer import box_trainer  # noqa: PLC0415
    from makenomistakes.tune.wiring import comfy_busy_for  # noqa: PLC0415

    settings = service.settings
    kinds = kinds_for(
        service,
        lambda: box_trainer(settings),
        lambda: box_style_trainer(settings),
        lambda: ["--gpus", gpu_device()],
    )
    return TrainRunner(
        service,
        gate,
        traffic,
        kinds,
        TrainingContainer(SubprocessRunner()),
        busy,
        comfy_busy=comfy_busy_for(service),
        note=note,
        notify=notify,
    )
