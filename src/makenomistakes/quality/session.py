"""A quality tune on the card: the approved plan's steps, each a line in the
tune's trials log (`TrialResult`, `Candidate.quant` and its stage), so a cut tune resumes by
replaying them as a speed tune does.

The steps: the reference's base run (`reference`: it reads the text and saves its
log-probabilities in the tune's work folder), then each candidate, the most faithful first,
its speed trial on a copy of its serving recipe (`speed`, the tuner's own `TrialRunner`) and,
but for the reference, its KL-divergence run against the base (`quality`). A cut or held step
runs again; a crashed one is kept (that quant drops out); a base file that went missing runs
the reference again. The perplexity container takes the copy's name, starts under the box's
start lock after the room a trial makes, and is cut as a trial is. The finish is
`quality.apply`'s.

Whether it was cancelled is its queue entry's mark (`cancelled`), read at every look and when
it stops (`_stop`, under the tuning folder's lock), never the cut's last word. Before its
finish it marks the entry `saving` under the same lock: a
cancel from then on is too late, from any process."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from makenomistakes import box_events
from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.engine_images import NoImage
from makenomistakes.instances import InstanceService
from makenomistakes.quality.apply import finish
from makenomistakes.quality.entries import Asker, Bystander, OwnRun, end_plan, end_saved
from makenomistakes.quality.figures import QualityFigures
from makenomistakes.quality.output import QualityOutputError, base_figures, kld_figures
from makenomistakes.quality.perplexity import (
    BASE_FILE,
    TEXT_FILE,
    DockerUnreachableError,
    PerplexityError,
    PerplexityRunner,
    Ran,
    RunKind,
    quality_image,
    run_argv,
)
from makenomistakes.quality.plan import QualityCandidate, QualityPlan
from makenomistakes.quality.steps import (
    Ended,
    Status,
    Step,
    last_of,
    next_step,
    plan_moved,
    steps_of,
    work_folder,
)
from makenomistakes.quality.texts import TEXTS, TextError, checked
from makenomistakes.readings_filter import here_of
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.tuned import ScoreMode
from makenomistakes.swap import start_turn
from makenomistakes.tune.ends import End, EndOutcome, left_words
from makenomistakes.tune.lease import going
from makenomistakes.tune.orphans import Settled, settle
from makenomistakes.tune.room import NoRoomError
from makenomistakes.tune.store import (
    QualityQueued,
    RunElsewhereError,
    TuneState,
    TuneStore,
    TuneStoreError,
)
from makenomistakes.tune.trial import TrialRunner, copy_name
from makenomistakes.tune.trial_types import Candidate, Cut, Outcome, TrialResult
from makenomistakes.tune.words import capital, of, what

Room = Callable[[Recipe], list[str]]
Progress = Callable[[str], None]


class QualityTuner:
    def __init__(
        self,
        service: InstanceService,
        store: TuneStore,
        trials: TrialRunner,
        perplexity: PerplexityRunner,
        room: Room | None = None,
        progress: Progress | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.service = service
        self.store = store
        self._trials = trials
        self._perplexity = perplexity
        self._room = room or (lambda _copy: [])
        self._progress = progress
        self._clock = clock  # each run's wall clock (`TrialResult.seconds`)
        self.state: TuneState | None = None

    def state_for(self, entry: QualityQueued) -> TuneState:
        """The tune in progress when it is this very entry's; else a new one (another kind's,
        another plan's: its steps were measured on another plan's recipes, or the same plan
        approved again after the first one ended)."""
        plan = entry.quality
        try:
            state = self.store.state(entry.name)
        except TuneStoreError:
            state = None
        mine = state is not None and state.kind == "quality" and state.holder == entry.holder
        if state is None or not mine or state.plan != plan.fingerprint:
            mode: ScoreMode = "agents" if plan.use == "agents" else "chat"
            state = TuneState(
                name=entry.name,
                mode=mode,
                minutes=entry.minutes,
                settings=plan.settings,
                machine=here_of(self.service.settings.expanded_state_dir()).id,
                kind="quality",
                max_trials=len(steps_of(plan)),
                plan=plan.fingerprint,
                holder=entry.holder,
            )
            self.store.save_state(state)
        return state

    def tune(self, entry: QualityQueued, cut: Cut) -> Ended:
        """Run `entry`'s quality tune until it ends or is cut; the queue and state follow. The
        run's token goes for the whole tune (`going`): its lease is live exactly that long,
        and let go whatever ended it (Ctrl-C, a failure too), so no process reads it running."""
        with going() as token:
            try:
                return self._tune(entry, cut, token)
            finally:
                self.store.let_go_run(entry.name, token)
                self.state = None

    def _tune(self, entry: QualityQueued, cut: Cut, token: str) -> Ended:
        begun = self._begin(entry, token)
        if isinstance(begun, Ended):
            return begun
        entry, state = begun  # the entry as queued now
        name, plan = entry.name, entry.quality
        steps = steps_of(plan)
        while True:
            found = self.store.current(entry)  # each look before the finish too
            if found is None or found.cancelled or cut.is_set():
                return self._stop(entry, state, token, "cut", cut.why)
            if self.service.find(name) is None:  # `mnm rm` meanwhile
                return self._dropped(entry, None, OwnRun(token))
            done = self.store.trials(name, state.session)
            base_here = (work_folder(self.service, name) / BASE_FILE).exists()
            step = next_step(steps, done, base_here)
            reference = last_of(done, steps[0])
            unscored = reference is not None and reference.outcome is not Outcome.MEASURED
            if step is None or unscored:  # nothing scores the others without it
                return self._finish(entry, state, done, token)
            result = self.run_step(step, plan, state, done, cut)
            if result.outcome in {Outcome.CUT, Outcome.HELD}:
                status: Status = "cut" if result.outcome is Outcome.CUT else "held"
                return self._stop(entry, state, token, status, result.why or status)

    def _begin(self, entry: QualityQueued, token: str) -> tuple[QualityQueued, TuneState] | Ended:
        """The entry as queued now and its state, held by this run (`token`) for the whole
        tune; else how it ended (said). Under the store's lock, in this order: the entry still
        queued (a cancel since it was read wins: a cancelled plan's copy never runs and
        saves), its quants here, no other run of it, a mark its last run
        outlived settled (`tune.orphans`: a save that landed ends it, one that did not runs
        on, one it cannot judge or end is not run), the instance's recipe still the plan's,
        then the lease taken. What the settle said is said once the lock is let go; a plan
        whose instance went or moved ends after it too (`end_plan` judges it again)."""
        recipe: Recipe | None = None
        settled: Settled | None = None
        held: TuneState | None = None
        try:
            with self.store.locked():
                now = self.store.current(entry)
                if now is not None and now.ready:
                    self.store.refuse_running(now.name)
                    settled = settle(self.service, self.store, now)
                    now = self.store.current(entry) if settled.runs else None
                if now is not None and now.ready:
                    recipe = self.service.find(now.name)
                    if recipe is not None and not plan_moved(recipe, now.quality):
                        held = self.store.hold_run(self.state_for(now), token)
        except RunElsewhereError as exc:  # another run of it goes on: it goes on there
            return self._told(Ended("held", str(exc)))
        for line in settled.told if settled is not None else ():  # the lock let go
            self._say(line)
        if settled is not None and not settled.runs:  # ended or left marked: said above
            return Ended(settled.status, settled.line or settled.why)
        if now is not None and held is not None:
            return now, held
        if now is None:
            line = f"The quality tune of {entry.name} was cancelled or ended: not run."
            return self._told(Ended("skipped", line))
        if not now.ready:  # the router never takes one: its approval downloads still
            return self._told(Ended("held", f"{now.words} is still downloading its quants."))
        return self._dropped(now, recipe, Bystander())  # not its run yet: it holds no lease

    def _ended(self, end: End, name: str) -> None:
        """What the plan's end left, said (a run elsewhere, the folder unwritable)."""
        if (left := left_words(end, name)) is not None:
            self._say(left)

    def _dropped(self, entry: QualityQueued, recipe: Recipe | None, asker: Asker) -> Ended:
        """The instance gone (`mnm rm`) or moved since the plan: nothing more is measured and
        the quants the tune downloaded go (the instance's own and every other plan's stay)."""
        name = entry.name
        end = end_plan(self.service, entry, self._say, asker)
        if end.outcome is not EndOutcome.ENDED:  # a run elsewhere ends it, or it ended meanwhile
            line = left_words(end, name) or f"The quality tune of {name} was ended meanwhile."
            status: Status = "held" if end.outcome is EndOutcome.RUNS else "skipped"
            return self._told(Ended(status, line))
        if recipe is None:
            self.store.forget(name)  # its trials too: `mnm rm` cleared them, a step wrote again
            line = f"{name} was removed: its quality tune ends, nothing saved."
            self._say(line)
            return Ended("skipped", line)
        line = (
            f"{name} changed since its quality tune was planned: nothing more was measured "
            f"(`mnm tune {name} --quality` plans it again)."
        )
        self._say(line)
        return Ended("failed", line)

    def _finish(
        self, entry: QualityQueued, state: TuneState, done: list[TrialResult], token: str
    ) -> Ended:
        """The choice saved and the plan ended, once its entry is marked `saving` under the
        store's lock (a cancel marked first ends it here, nothing saved; one from then on is
        too late, from any process)."""
        name, plan = entry.name, entry.quality
        if not self.store.mark_saving(entry):  # never saved after its plan ended or a cancel
            return self._stop(entry, state, token, "cut", "cancelled")
        saved = None
        try:
            kept, line, saved = finish(self.service, plan, done, entry.queued_at)
        finally:  # whatever the finish did, the plan ends here (with its speed tune if moved)
            asker = OwnRun(token)
            if saved is None:
                end = end_plan(self.service, entry, self._say, asker)
            else:
                moved_from = plan.current if saved.quant != plan.current else None
                end = end_saved(self.service, entry, self._say, asker, moved_from)
            self._ended(end, name)
        self._say(line)
        return Ended("tuned" if kept else "failed", line)

    def _stop(
        self, entry: QualityQueued, state: TuneState, token: str, status: Status, why: str
    ) -> Ended:
        """The tune stops before its finish, judged under the store's lock: its entry marked
        cancelled ends through `end_plan` (its holds released, what it downloaded deleted);
        gone from under it (a file removed by hand: every end waits for a live run), it stops,
        nothing saved; else it pauses, keeping its place and its state, its lease let go."""
        name = entry.name
        found = self.store.stop_run(entry, state, token)
        if found is None:
            line = f"The quality tune of {name} was ended meanwhile: it stops, nothing saved."
            return self._told(Ended("skipped", line))
        if found.cancelled:  # its step's container stopped by now: the plan ends
            self._ended(end_plan(self.service, found, self._say, OwnRun(token)), name)
            line = f"{capital(what(found))} is cancelled; {of(found).nothing}."
            return self._told(Ended("cancelled", line))
        line = f"The quality tune of {name} paused ({why}); it resumes when the card is idle."
        return self._told(Ended(status, line))

    def run_step(
        self, step: Step, plan: QualityPlan, state: TuneState, done: list[TrialResult], cut: Cut
    ) -> TrialResult:
        name = state.name
        number = max((r.number for r in done), default=0) + 1
        state.trial = sum(1 for s in steps_of(plan) if last_of(done, s) is not None) + 1
        state.step = step.words
        self.state = state
        self.store.save_state(state)
        self._say(f"Tuning {name} for quality: {step.words}")
        candidate = Candidate(stage=step.stage, quant=step.row.quant)
        if not self.service.files_present(step.row.recipe):  # deleted to free disk, say
            why = f"{step.row.quant}'s files are no longer on disk"
            result = TrialResult(
                session=state.session,
                instance=name,
                number=number,
                candidate=candidate,
                outcome=Outcome.REFUSED,
                why=why,
            )
        elif step.stage == "speed":
            result = self._trials.run(
                step.row.recipe, candidate, number, state.mode, state.session, cut
            )
        else:
            result = self._measure(step, plan, candidate, number, state.session, cut)
        self.store.append(result)
        state.step = None
        self.store.save_state(state)
        self._say(f"Tuning {name} for quality: {result.line}")
        return result

    def _measure(
        self,
        step: Step,
        plan: QualityPlan,
        candidate: Candidate,
        number: int,
        session: str,
        cut: Cut,
    ) -> TrialResult:
        """One `llama-perplexity` run, timed: the base run for the reference, else a KLD run."""
        started = self._clock()
        base = TrialResult(
            session=session,
            instance=step.row.recipe.name,
            number=number,
            candidate=candidate,
            outcome=Outcome.MEASURED,
        )
        result = self._run(step, plan, base, cut)
        return result.model_copy(update={"seconds": self._clock() - started})

    def _run(  # noqa: PLR0911 - one outcome per way a run can end
        self, step: Step, plan: QualityPlan, base: TrialResult, cut: Cut
    ) -> TrialResult:
        recipe = step.row.recipe
        image = quality_image(recipe)
        if isinstance(image, NoImage):
            return base.model_copy(update={"outcome": Outcome.REFUSED, "why": image.words})
        kind = RunKind.BASE if step.stage == "reference" else RunKind.KLD
        work = work_folder(self.service, recipe.name)
        try:
            if kind is RunKind.BASE:
                work.mkdir(parents=True, exist_ok=True)
                (work / TEXT_FILE).write_bytes(checked(TEXTS[plan.text]))
                (work / BASE_FILE).unlink(missing_ok=True)
        except (OSError, TextError) as exc:
            return base.model_copy(update={"outcome": Outcome.REFUSED, "why": str(exc)})
        copy = copy_name(recipe.name)
        start = self._start(recipe, copy, work, image, kind, plan, cut)
        if isinstance(start, Failed):  # Docker refused the run: a crash, never a pause
            return base.model_copy(update={"outcome": Outcome.CRASHED, "why": start.why})
        if isinstance(start, Held):
            outcome = Outcome.CUT if cut.is_set() else Outcome.HELD
            why = cut.why if cut.is_set() else start.why
            return base.model_copy(update={"outcome": outcome, "why": why})
        try:
            ran: Ran | PerplexityError = self._perplexity.finish(
                copy, recipe_backend_or_none(recipe)
            )
        except PerplexityError as exc:
            ran = exc
        # A cut stops and removes the container: its code and its output, whatever they are,
        # are never the run's (a cut base run's empty log is no crash).
        if cut.is_set():
            return base.model_copy(update={"outcome": Outcome.CUT, "why": cut.why})
        if isinstance(ran, PerplexityError):
            return base.model_copy(update={"outcome": Outcome.CRASHED, "why": str(ran)})
        base = base.model_copy(update={"build": self.service.build_of(image)})
        if ran.code != 0:
            update = {"outcome": Outcome.CRASHED, "cause": ran.cause, "why": ran.why}
            return base.model_copy(update=update)
        try:
            figures = _figures(ran.log, kind, step.row, plan)
        except (QualityOutputError, ValueError) as exc:
            return base.model_copy(update={"outcome": Outcome.CRASHED, "why": str(exc)})
        return base.model_copy(update={"quality": figures})

    def _start(
        self,
        recipe: Recipe,
        copy: str,
        work: Path,
        image: str,
        kind: RunKind,
        plan: QualityPlan,
        cut: Cut,
    ) -> Held | Failed | None:
        """Room made and the run started under the box's start lock; None once it runs. A cut
        that came while Docker started it (the router's stop found no container yet) stops
        it here."""
        settings = self.service.settings
        room_for = recipe.model_copy(update={"name": copy})
        with box_events.changing(settings), start_turn(settings, copy, self._progress):
            try:
                parked = self._room(room_for)
            except NoRoomError as exc:
                return Held(str(exc))
            if parked and self._progress:
                self._progress(f"Parked {', '.join(parked)} for the quality run.")
            if cut.is_set():
                return Held("cut")
            argv = run_argv(recipe, self.service.models_dir, work, image, kind, copy, plan.chunks)
            try:
                self._perplexity.start(argv, copy)
            except DockerUnreachableError as exc:  # Docker itself silent: it runs later
                return Held(str(exc))
            except PerplexityError as exc:
                return Failed(str(exc))
        if cut.is_set():
            self._perplexity.stop(copy)
            return Held("cut")
        return None

    def _told(self, ended: Ended) -> Ended:
        """`ended`, its line said (a terminal's `--now` prints it)."""
        self._say(ended.line)
        return ended

    def _say(self, line: str) -> None:
        if self._progress is not None:
            self._progress(line)


@dataclass(frozen=True)
class Held:
    """No room for the run now (or cut before it ran): it runs again later."""

    why: str


@dataclass(frozen=True)
class Failed:
    """Docker refused the run: its step crashed."""

    why: str


def _figures(log: str, kind: RunKind, row: QualityCandidate, plan: QualityPlan) -> QualityFigures:
    text = TEXTS[plan.text]
    common = {
        "repo_id": plan.repo_id,
        "quant": row.quant,
        "file": PurePosixPath(row.recipe.files[0]).name,
        "reference": plan.reference.quant,
        "text": text.name.value,
        "text_sha256": text.short,
        "chunks": plan.chunks,
    }
    if kind is RunKind.BASE:
        found = base_figures(log)
        return QualityFigures(**common, ppl=found.ppl, ppl_err=found.ppl_err)
    kld = kld_figures(log)
    return QualityFigures(
        **common,
        ppl=kld.ppl,
        ppl_err=kld.ppl_err,
        kld=kld.kld,
        kld_err=kld.kld_err,
        kld_p99=kld.kld_p99,
        same_top=kld.same_top,
        same_top_err=kld.same_top_err,
        rms_dp=kld.rms_dp,
        ppl_ratio=kld.ppl_ratio,
    )
