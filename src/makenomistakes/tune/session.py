"""One tune of one instance: the search's next trial, run, recorded, until
the search finishes, the budget is spent, or the card is wanted back.

Resumable by design: the state (`TuneState`) names the session and the settings it started
from; the trials done are the session's lines in `<name>.jsonl`, which the search replays
(`Search.next`). A cut or held trial is recorded and not counted, and the tune stops there;
the runner resumes it later. A tune whose recipe changed since it started (the user's
`mnm set`) starts over from the new settings. The result is saved through `save_tune`.

Whether it was cancelled is its queue entry's mark (`cancelled`), read at every look and when
it stops (`_stop`, under the tuning folder's lock): never the cut's last word, which the next
request or a trial held for room would overwrite. Before it saves it marks the entry
`saving` under the same lock, so a cancel from then on is told it is too late, from any
process."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from makenomistakes.hub.facts import CHAT_LIKE, Engine
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.measurements import settings_key
from makenomistakes.readings_filter import here_of
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.tuned import Tuned
from makenomistakes.start_fixes import StartRefusedError
from makenomistakes.tune.apply import TuneChangedError, save_tune
from makenomistakes.tune.baseline import baseline
from makenomistakes.tune.key import key_now
from makenomistakes.tune.lease import going
from makenomistakes.tune.orphans import Refused, queue_speed, settle
from makenomistakes.tune.score import (
    DECODE_PROMPTS_IN_RAM,
    default_mode,
    judged_spread,
    spread_limit,
)
from makenomistakes.tune.search import Budget, Finished, Search, SearchFacts, axes_for
from makenomistakes.tune.search_facts import search_facts
from makenomistakes.tune.store import (
    Entry,
    GaveUp,
    QualityQueued,
    Queued,
    RunElsewhereError,
    TuneState,
    TuneStore,
    TuneStoreError,
)
from makenomistakes.tune.trial import TrialRunner
from makenomistakes.tune.trial_types import Candidate, Cut, Outcome, TrialResult
from makenomistakes.tune.words import capital, of, replaced_words, what

if TYPE_CHECKING:
    from makenomistakes.quality.session import QualityTuner

VLLM_NOTE = "vLLM instances are planned, not tuned"
NOTHING = "nothing to tune: the tuner tunes llama.cpp chat, coding and vision instances"

Status = Literal["tuned", "cut", "held", "failed", "skipped", "cancelled"]
Progress = Callable[[str], None]


@dataclass(frozen=True)
class TuneEnd:
    name: str
    status: Status
    line: str
    tuned: Tuned | None = None


@dataclass(frozen=True)
class Run:
    """A speed tune running: its entry, its state and the run's token (its lease's)."""

    entry: Queued
    state: TuneState
    token: str


def untunable(recipe: Recipe) -> str | None:
    """Why the tuner leaves it alone; None when it tunes it."""
    if recipe.engine is Engine.VLLM:
        return VLLM_NOTE
    if recipe.engine is not Engine.LLAMACPP or recipe.modality not in CHAT_LIKE:
        return NOTHING
    if not recipe.downloaded:
        return f"{recipe.name} is not downloaded: `mnm start {recipe.name}` downloads it"
    return None


class Tuner:
    def __init__(
        self,
        service: InstanceService,
        store: TuneStore,
        runner: TrialRunner,
        progress: Progress | None = None,
        facts: Callable[[InstanceService, Recipe], SearchFacts] = search_facts,
        quality: QualityTuner | None = None,
    ) -> None:
        self.service = service
        self.store = store
        self._runner = runner
        self._progress = progress
        self._facts = facts
        self.quality = quality  # A tune of kind quality runs there
        self._running: TuneState | None = None  # the speed tune running now

    @property
    def state(self) -> TuneState | None:
        """The tune running now, either kind, for the status line."""
        return self._running or (self.quality.state if self.quality is not None else None)

    def tune(self, entry: Entry, cut: Cut | None = None) -> TuneEnd:
        """Run `entry`'s tune until it ends or is cut; the queue and state follow."""
        cut = cut or Cut()
        name = entry.name
        if isinstance(entry, QualityQueued):
            if self.quality is None:
                return TuneEnd(name, "skipped", f"{name}: this tuner measures no quality.")
            ended = self.quality.tune(entry, cut)
            return TuneEnd(name, ended.status, ended.line)
        recipe = self.service.find(name)
        if recipe is None:
            self.store.forget(name)
            return TuneEnd(name, "skipped", f"{name} is gone: its tune ends.")
        if (why := untunable(recipe)) is not None:
            self._drop_own(entry)
            return self._told(TuneEnd(name, "skipped", f"{name}: {why}."))
        with going() as token:  # the run's lease is live exactly as long as this
            try:
                begun = self._begin(entry, recipe, token)
                if isinstance(begun, TuneEnd):
                    return begun
                state, search = begun
                return self._trials(Run(entry, state, token), recipe, search, cut)
            finally:  # whatever ended it (Ctrl-C, a failure too): no process reads it running
                self.store.let_go_run(name, token)
                self._running = None

    def _trials(self, run: Run, recipe: Recipe, search: Search, cut: Cut) -> TuneEnd:
        """Trial after trial until the search finishes, the tune is cut or cancelled, or its
        entry went; each look before the finish too, so a cancel after the last trial ends it
        before it saves."""
        name, state = run.entry.name, run.state
        while True:
            found = self.store.current(run.entry)
            if found is None or found.cancelled or cut.is_set():
                return self._stop(run, "cut", cut.why)
            if self.service.find(name) is None:  # `mnm rm` meanwhile: the tune ends
                self.store.forget(name)
                return TuneEnd(name, "skipped", f"{name} was removed: its tune ends.")
            done = self.store.trials(name, state.session)
            step = search.next(done)
            if isinstance(step, Finished):
                return self._finish(run, recipe, step, done)
            result = self._trial(recipe, state, step.candidate, done, cut)
            if result.outcome in {Outcome.CUT, Outcome.HELD}:
                status: Status = "cut" if result.outcome is Outcome.CUT else "held"
                return self._stop(run, status, result.why or status)

    def _begin(
        self, entry: Queued, recipe: Recipe, token: str
    ) -> tuple[TuneState, Search] | TuneEnd:
        """Its state, held by this run (`token`) for the whole tune, and its search; else how
        it ended (said). The search's reads first (a failure holds nothing), then in one hold
        of the store's lock: still queued (a cancel since the queue was read wins, never lost
        while the search read the files), no other run of it, a
        mark its last run outlived settled (`tune.orphans`: one it ends or cannot settle is not
        run), the state, the lease. What the settle said is said once the lock is let go."""
        name = entry.name
        facts = self._facts(self.service, recipe)
        cards = self.service.memory().cards
        another_box, settled, replaced = False, None, None
        try:
            with self.store.locked():
                now = self.store.current(entry)
                if now is not None:
                    self.store.refuse_running(name)
                    settled = settle(self.service, self.store, now)
                    now = self.store.current(entry) if settled.runs else None
                if now is not None:
                    state, another_box = self._state(now, recipe)
                    self.store.hold_run(state, token)
                elif settled is None:  # gone: cancelled, or replaced by another kind of tune
                    replaced = self.store.queued(name)
        except RunElsewhereError as exc:  # another run of it goes on: it goes on there
            return self._told(TuneEnd(name, "held", str(exc)))
        for line in settled.told if settled is not None else ():  # the lock let go
            self._say(line)
        if settled is not None and not settled.runs:  # ended or left marked: said above
            return TuneEnd(name, settled.status, settled.line or settled.why)
        if now is None:  # most often the user's cancel, between the queue's read and here
            line = replaced_words(entry, replaced)
            status: Status = "cancelled" if replaced is None else "skipped"
            return self._told(TuneEnd(name, status, line))
        if another_box:  # Its trials were another box's
            self._say(f"The tune of {name} dropped: another box. It starts over here.")
        return state, Search(facts, axes_for(facts, cards), Budget(seconds=state.minutes * 60.0))

    def _state(self, entry: Queued, recipe: Recipe) -> tuple[TuneState, bool]:
        """The tune in progress, unless the recipe moved since it began (then a new one), and
        whether one of another box's was dropped for it."""
        try:
            state = self.store.state(recipe.name)
        except TuneStoreError:
            state = None
        settings = settings_key(recipe)
        box = here_of(self.service.settings.expanded_state_dir()).id
        another_box = state is not None and _another_box(state.machine, box)
        if another_box:  # nothing of this box: it starts over, under a new session
            state = None
        if state is None or state.settings != settings or state.kind != "speed":
            state = TuneState(
                name=recipe.name,
                mode=entry.mode or default_mode(recipe),
                minutes=entry.minutes,
                settings=settings,
                machine=box,
            )
            self.store.save_state(state)
        return state, another_box

    def _drop_own(self, entry: Queued) -> None:
        """Its own entry and progress go (one queued in its place since stays), unless a live
        run holds it."""
        left = None
        with self.store.locked():
            now = self.store.current(entry)
            if now is not None and self.store.holder(now) is None:
                left = self.store.end_speed(now)
        if left is not None:  # the lock let go
            self._say(left)

    def _stop(self, run: Run, status: Status, why: str) -> TuneEnd:
        """The tune stops before its finish, judged under the store's lock (`stop_run`): its
        entry marked cancelled ends here with its progress (its trials stay in the log); gone
        from under it (a file removed by hand: every end waits for a live run), it stops,
        nothing saved; else it pauses, keeping its place and its state, its lease let go."""
        name = run.entry.name
        found = self.store.stop_run(run.entry, run.state, run.token)
        if found is None:
            line = f"The tune of {name} was ended meanwhile: it stops, nothing saved."
            return self._told(TuneEnd(name, "skipped", line))
        if found.cancelled:
            if (left := self.store.end_speed(found, run.token)) is not None:
                self._say(left)
            line = f"{capital(what(found))} is cancelled; {of(found).nothing}."
            return self._told(TuneEnd(name, "cancelled", line))
        line = f"The tune of {name} paused ({why}); it resumes when the card is idle."
        return self._told(TuneEnd(name, status, line))

    def _trial(
        self,
        recipe: Recipe,
        state: TuneState,
        candidate: Candidate,
        done: list[TrialResult],
        cut: Cut,
    ) -> TrialResult:
        counted = sum(1 for r in done if r.counted)
        spent = sum(r.seconds for r in done if r.counted)  # a cut or held trial costs none
        state.trial = counted + 1
        state.seconds_left = max(state.minutes * 60.0 - spent, 0.0)
        self._running = state
        self.store.save_state(state)
        number = max((r.number for r in done), default=0) + 1  # a cut one keeps its number
        self._say(f"Tuning {recipe.name}: trial {number} ({candidate.label})")
        try:
            result = self._runner.run(recipe, candidate, number, state.mode, state.session, cut)
        except StartRefusedError as exc:  # a start the box refuses for every candidate
            result = TrialResult(
                session=state.session,
                instance=recipe.name,
                number=number,
                candidate=candidate,
                outcome=Outcome.REFUSED,
                why=str(exc),
            )
        self.store.append(result)
        self.store.save_state(state)
        self._say(f"Tuning {recipe.name}: {result.line}")
        seed = not candidate.changes and candidate.draft is None
        if seed and result.outcome is Outcome.MEASURED and result.decode:
            self._compare(recipe, statistics.median(result.decode))  # The seed
        return result

    def _compare(self, recipe: Recipe, measured: float) -> None:
        """The seed measured beside what its card's row priced it at."""
        try:
            found = baseline(self.service, recipe)
        except Exception:
            return
        if found is not None:
            self._say(f"Tuning {recipe.name}: {found.words(measured)}.")

    def _finish(self, run: Run, recipe: Recipe, step: Finished, done: list[TrialResult]) -> TuneEnd:
        """The result saved once its entry is marked `saving` under the store's lock (a
        cancel marked first ends it here, nothing saved; one from then on is too late), then
        its own entry unqueued and its state dropped (one queued in its place stays)."""
        if not self.store.mark_saving(run.entry):  # never saved after a cancel took
            return self._stop(run, "cut", "cancelled")
        try:
            return self._save(run, recipe, step, done)
        finally:
            if (left := self.store.end_speed(run.entry, run.token)) is not None:
                self._say(left)

    def _save(self, run: Run, recipe: Recipe, step: Finished, done: list[TrialResult]) -> TuneEnd:
        name, state = recipe.name, run.state
        counted = sum(1 for r in done if r.counted)
        best, seed = step.best, step.seed
        if best is None or best.score is None or seed is None or seed.score is None:
            why = gave_up_words(seed, step.reason)
            self.store.give_up(GaveUp(name=name, why=why, settings=state.settings, trials=counted))
            line = (
                f"The tune of {name} gave up: {why}. Nothing changed; the nightly does not queue "
                f"it again for this (`mnm tune {name} --now` tries again)."
            )
            self._drop_if_stale(recipe)
            self._say(line)
            return TuneEnd(name, "failed", line)
        tuned = Tuned(
            changes=dict(best.candidate.changes),
            draft=best.candidate.draft,
            score=best.score,
            seed=seed.score,
            trials=max(counted, 1),
            key=key_now(self.service, recipe),
            made_for=run.entry.queued_at,  # how a death mid-save is told landed
        )
        try:
            save_tune(self.service, name, state.settings, tuned)
        except TuneChangedError as exc:
            self._say(str(exc))
            return TuneEnd(name, "failed", str(exc))
        except InstanceError as exc:
            line = f"The tune of {name} could not be saved: {exc}"
            self._say(line)
            return TuneEnd(name, "failed", line)
        self.store.clear_gave_up(name)
        line = (
            f"Tuned {name}: {tuned.summary} ({tuned.gain:+.0%} over mnm's plan, score "
            f"{tuned.score.value:.1f} against {tuned.seed.value:.1f}; {tuned.trials} trials, "
            f"{step.reason})."
        )
        self._say(line)
        return TuneEnd(name, "tuned", line, tuned)

    def _drop_if_stale(self, recipe: Recipe) -> None:
        """A tune that could not be redone leaves no stale block behind (else it would be
        queued again every few minutes): the plan's settings stay as the tune left them."""
        if recipe.tuned is None or recipe.tuned.key.differs(key_now(self.service, recipe)) is None:
            return
        try:
            self.service.change(recipe.name, lambda r: r.model_copy(update={"tuned": None}))
        except InstanceError:
            return

    def _told(self, end: TuneEnd) -> TuneEnd:
        """`end`, its line said (a terminal's `--now` prints it)."""
        self._say(end.line)
        return end

    def _say(self, line: str) -> None:
        if self._progress is not None:
            self._progress(line)


def gave_up_words(seed: TrialResult | None, reason: str) -> str:
    """Why a tune ended without a result, in `mnm tune --status`'s words: mnm's plan
    never settling is `decode readings spread 55% over 3 (the rule: 10%)`."""
    if seed is not None and seed.outcome is Outcome.NOISY and len(seed.decode) >= 2:
        spread, count = judged_spread(seed.decode), len(seed.decode)
        over = f"the middle {count - 2} of {count}" if count >= DECODE_PROMPTS_IN_RAM else count
        return f"decode readings spread {spread:.0%} over {over} (the rule: {spread_limit():.0%})"
    if seed is not None and seed.outcome is Outcome.NOISY:
        return f"mnm's plan was never measured alone: {seed.why}"
    return reason


def _another_box(before: str | None, now: str | None) -> bool:
    return before is not None and now is not None and before != now


def ensure_queued(
    service: InstanceService, recipe: Recipe, reason: str, say: Progress, now: bool = False
) -> bool:
    """Queue a tune of a llama.cpp chat, coding or vision instance, the way every tune is
    queued (`tune.orphans.queue_speed`: a mark left by a run that stopped settled first);
    False (nothing queued) for any other (vLLM is planned, not tuned; pictures, speech:
    nothing to tune) and while a quality tune of it waits (never replaced)."""
    if recipe.engine is not Engine.LLAMACPP or recipe.modality not in CHAT_LIKE:
        return False
    store = TuneStore(service.settings.expanded_state_dir())
    entry = Queued(name=recipe.name, reason=reason, now=now)
    queued = queue_speed(service, store, entry, say)
    if isinstance(queued, Refused):  # why, said (else `mnm add` says nothing)
        say(queued.line(recipe.name))
        return False
    return True
