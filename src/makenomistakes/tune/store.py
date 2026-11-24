"""What may be written to the tuning folder when: `TuneStore`, on the folder's
files (`tune.folder.TuneFolder`). Every reader of the folder imports the records from here.

- Who holds a queued tune now and ends it itself (`holder`): a live approval downloading a
  quality plan's quants, else the live run holding its tune's lease (`run_beside`), the asker
  left out. The one answer for every reader.
- A run's lease (`hold_run`, `let_go_run`, `refuse_running`): one tune never runs twice.
- A queued tune's marks (`closing`), every transition under the lock: the user's cancel
  (`mark_cancelled`), a run saving its result (`mark_saving`), a save a stopped run never
  landed (`unmark`), a run stopping before its finish (`stop_run`), a speed tune's end
  (`end_speed`).
- The queue's rules: a speed tune queued (`queue`), refused in words (`blocking`); a new
  entry dated by its write (`put_new`); the queue's order (`waiting`).
- What an entry is (`current`: its kind and when it was queued) and whose progress is whose
  (`drop_own_state`, `drop_run`), and `mnm rm`'s `forget`.

The rules reach the files only through `TuneFolder`'s methods, never its folders. Its raw
writers (`put`, `unqueue`, `save_state`, `drop_state`) are still public: a quality approval's
own steps write through them (`quality.entries`), under this lock."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import overload

from makenomistakes.json_folder import BadNameError
from makenomistakes.tune.ends import runs_words
from makenomistakes.tune.folder import TuneFolder, TuneStoreError
from makenomistakes.tune.lease import Lease
from makenomistakes.tune.records import (
    AFTER_QUALITY,
    DEFAULT_MINUTES,
    Approval,
    Closing,
    Entry,
    GaveUp,
    Holds,
    Marked,
    QualityQueued,
    Queued,
    TuneAsk,
    TuneKind,
    TuneState,
    same_entry,
    same_plan,
)
from makenomistakes.tune.words import blocking_words, capital, what

__all__ = [  # the records are `tune.records'`, the files `tune.folder`'s: imported here
    "AFTER_QUALITY",
    "DEFAULT_MINUTES",
    "Approval",
    "Closing",
    "Entry",
    "GaveUp",
    "Holds",
    "Marked",
    "QualityQueued",
    "Queued",
    "RunElsewhereError",
    "TuneAsk",
    "TuneKind",
    "TuneState",
    "TuneStore",
    "TuneStoreError",
]


class RunElsewhereError(RuntimeError):
    """Another live process runs this tune now (`TuneStore.hold_run`), in words."""


class TuneStore(TuneFolder):
    # --- who holds a queued tune, and a run's lease --------------------------------------

    def run_beside(self, name: str, token: str | None = None) -> int | None:
        """The pid of the live run of `name`'s tune unless it is the caller's own run (`token`):
        what a runner, an end and a cancel ask under the store's lock. A run in this very
        process is another run too, and one a process left behind is none (`tune.lease`)."""
        try:
            state = self.state(name)
        except TuneStoreError:
            return None
        return state.run_beside(token) if state is not None else None

    def holder(
        self, entry: Entry, token: str | None = None, lease: Approval | None = None
    ) -> Holds | None:
        """Who holds `entry` now and ends it itself: the live approval downloading a quality
        plan's quants (it holds no run lease), else the live run holding its tune's lease;
        None when nothing does. The asker never holds it against itself: its own run
        (`token`), its own approval (`lease`). The one answer for every reader (the rule for
        a mark its run outlived, the store's refusal, the words, the router's waits, a plan's
        end: a refusal counting the run alone would read a stopping approval as a stopped
        run)."""
        approver = entry.approver if isinstance(entry, QualityQueued) else None
        if approver is not None and approver.alive and not approver.same(lease):
            return Holds("approval", approver.pid)
        pid = self.run_beside(entry.name, token)
        return Holds("run", pid) if pid is not None else None

    def refuse_running(self, name: str, token: str | None = None) -> None:
        """RunElsewhereError while another live run holds `name`'s tune: checked under the
        store's lock before a runner touches the state, so its state is never written over."""
        if (pid := self.run_beside(name, token)) is not None:
            raise RunElsewhereError(f"{runs_words(name, pid)}: it goes on there")

    def hold_run(self, state: TuneState, token: str) -> TuneState:
        """This run (`token`, going in this process) holds `state`'s tune now, for the whole
        tune; refused (`RunElsewhereError`) while another live run does: one tune never runs
        twice."""
        with self.locked():
            self.refuse_running(state.name, token)
            here = Lease.here(token)
            state.running_pid, state.running_since = here.pid, here.since
            state.running_token = token
            self.save_state(state)
        return state

    def let_go_run(self, name: str, token: str) -> None:
        """A run lets go of its lease whatever ended it (a session's `finally`: its end, a
        pause, Ctrl-C, a failure), so no process reads a run that stopped as running; nothing
        when the state is gone or another run's."""
        with contextlib.suppress(TuneStoreError, OSError), self.locked():
            state = self.state(name)
            if state is not None and state.running_token == token:
                gone = {"running_pid": None, "running_since": None, "running_token": None}
                self.save_state(state.model_copy(update=gone))

    # --- a queued tune's mark (`closing`), every transition under the lock -----------------

    def mark_cancelled(self, entry: Entry) -> Marked:
        """The user's cancel of `entry`, which a live run or approval holds (its caller judged
        that under the lock): the entry as queued now marked for that run or approval to end
        it at its next look; too late when its run saves its result already; gone when
        nothing of it is queued any more."""
        with self.locked():
            found = self.current(entry)
            if found is None:
                return Marked.GONE
            if found.saving:
                return Marked.TOO_LATE
            if not found.cancelled:
                self.put(found.model_copy(update={"closing": "cancelled"}))
            return Marked.MARKED

    def mark_saving(self, entry: Entry) -> bool:
        """The run of `entry` saves its result now: marked, so a cancel from any process is
        told it is too late; False when the entry is gone or was cancelled first (it saves
        nothing). The save stamps its record with the entry (`Tuned.made_for`), which is how
        `tune.orphans` tells a save that landed before its run died."""
        with self.locked():
            found = self.current(entry)
            if found is None or found.cancelled:
                return False
            self.put(found.model_copy(update={"closing": "saving"}))
            return True

    def unmark(self, entry: Entry) -> None:
        """A `saving` mark whose run stopped before its save landed (`tune.orphans`): the entry
        waits again, and its next run saves the result."""
        with self.locked():
            found = self.current(entry)
            if found is not None and found.saving:
                self.put(found.model_copy(update={"closing": None}))

    @overload
    def stop_run(self, entry: Queued, state: TuneState, token: str) -> Queued | None: ...
    @overload
    def stop_run(
        self, entry: QualityQueued, state: TuneState, token: str
    ) -> QualityQueued | None: ...
    def stop_run(self, entry: Entry, state: TuneState, token: str) -> Entry | None:
        """The run of `entry` (`token`) stops before its finish, judged under the lock: None
        when its entry went from under it (its progress dropped); the entry marked cancelled,
        for the run to end (`end_speed`, or a quality plan's `end_plan`); else the entry as it
        is and the run paused, its lease let go and the cut counted on `state`."""
        with self.locked():
            found = self.current(entry)
            if found is None:
                self.drop_run(entry.name, token)
            elif not found.cancelled:
                state.running_pid = state.running_since = state.running_token = None
                state.cuts += 1
                self.save_state(state)
            return found

    def running(self) -> list[TuneState]:
        """The tunes a live process is running a trial of now, for the words that show them
        (`mnm ls`, the web's bar); a tuning folder the system will not list shows none, and
        nothing is run, ended or let go by what it lacks: who holds an entry is read by name
        (`holder`), and `waiting` and the sweep refuse an unlistable queue."""
        try:
            states = self.states()
        except TuneStoreError:
            return []
        return [state for state in states if state.run_beside(None) is not None]

    def end_speed(self, entry: Queued, token: str | None = None) -> str | None:
        """Unqueue this speed entry (one queued in its place since stays) and drop its
        progress: the run's own when it ends itself (`token`), else the entry's. Raises only
        when the unqueue itself fails: once the entry is out, a progress that could not be
        dropped comes back in words for the caller to say (an end that took must never read
        as one that did not)."""
        with self.locked():
            self.unqueue(entry.name, entry)
            try:
                if token is not None:
                    self.drop_run(entry.name, token)
                else:
                    self.drop_own_state(entry)
            except (TuneStoreError, OSError) as exc:
                return (
                    f"{capital(what(entry))} ended, but its progress could not be dropped "
                    f"({exc}): `mnm tune {entry.name} --cancel` drops it."
                )
        return None

    # --- the queue's rules -------------------------------------------------------------

    def queue(self, entry: Queued) -> Queued | None:
        """Queue a speed tune; one already queued keeps its place and takes `now` if asked.
        Refused (None, `blocking` says why) over a queued quality plan, and over an entry that
        carries a mark: a live run's (cancelled, or saving its result) or one a stopped run
        left. Every way into the queue settles such a mark first (`tune.orphans.queue_speed`),
        so nothing merges into an entry whose run's save may have landed: the merged entry
        would keep that run's `queued_at` and read its save as its own.
        Queued again, a cancelled tune is no longer declined. A quality plan is queued by its
        approval alone (`quality.entries.admit`): handed one, this raises."""
        if not isinstance(entry, Queued):  # a quality plan written here would lose its holds
            raise TypeError(f"{entry.name}: a quality plan is queued by its approval alone")
        with self.locked():
            if self.blocking(entry) is not None:
                return None
            self.undecline(entry.name)
            before = self.queued(entry.name)
            if not isinstance(before, Queued):  # none (`blocking` refused a quality plan)
                return self.put_new(entry)
            kept = entry.model_copy(  # unmarked (`blocking`): it keeps its place
                update={"queued_at": before.queued_at, "now": entry.now or before.now}
            )
            self.put(kept)
            return kept

    @overload
    def put_new(self, entry: Queued) -> Queued: ...
    @overload
    def put_new(self, entry: QualityQueued) -> QualityQueued: ...
    def put_new(self, entry: Entry) -> Entry:
        """Write `entry` as a new entry, dated by this write under the lock: after whatever a
        settle in the same hold decided, so it never reads a save made before it as its own.
        Every new entry is written here (`queue`, a quality approval's
        `admit`); one that keeps its place keeps its date."""
        with self.locked():
            fresh = entry.model_copy(update={"queued_at": datetime.now(UTC)})
            self.put(fresh)
        return fresh

    def mark_now(self, entry: Entry) -> bool:
        """The entry runs without the idle minute; False when it is no longer queued, or a
        run is ending it (cancelled, or saving its result)."""
        with self.locked():
            found = self.current(entry)
            if found is None or found.closing is not None:
                return False
            self.put(found.model_copy(update={"now": True}))
        return True

    def blocking(self, entry: Entry) -> str | None:
        """Why `entry` cannot be queued, in words; None when it can. A quality plan waits for a
        live run beside no entry or beside a speed tune's (a speed run, or a quality run whose
        entry went); a state mnm cannot read is no run here, as every runner reads it (it
        writes a new state over it, never refusing the approval)."""
        name, before = entry.name, self.queued(entry.name)
        if before is not None and before.closing is not None:  # its holder's, or a stopped run's
            return blocking_words(before, self.holder(before))
        if isinstance(before, QualityQueued) and not same_plan(before, entry):
            return f"{before.words} is queued: `mnm tune {name} --cancel` first"
        no_plan = before is None or isinstance(before, Queued)
        if isinstance(entry, QualityQueued) and no_plan and self.run_beside(name) is not None:
            return f"a tune of {name} is running: wait for it, or `mnm tune {name} --cancel`"
        return None

    def wait_again(self, entry: Entry) -> None:
        """A cut tune waits for the next idle minute again (`now` dropped), unless another
        entry took its place meanwhile."""
        with self.locked():
            found = self.current(entry)
            if found is not None:  # as it is now, `now` dropped
                self.put(found.model_copy(update={"now": False}))

    # --- what an entry is, and whose its progress is ------------------------------------

    @overload
    def current(self, entry: Queued) -> Queued | None: ...
    @overload
    def current(self, entry: QualityQueued) -> QualityQueued | None: ...
    def current(self, entry: Entry) -> Entry | None:
        """`entry` as it is queued now, if it is still that entry (its kind and when it was
        queued); None when it was cancelled, ended or replaced since it was read. Every runner,
        cancel and end asks this, under the store's lock, before acting on an entry it read."""
        with self.locked():
            found = self.queued(entry.name)
            if found is None or not same_entry(found, entry):
                return None
            return found

    def drop_own_state(self, entry: Entry) -> None:
        """Drop the tune in progress if it is `entry`'s (a quality plan's by its holder, a
        speed tune's by its kind): an end never drops another entry's state."""
        with self.locked():
            try:
                state = self.state(entry.name)
            except TuneStoreError:
                return
            if state is not None and entry.owns(state):
                self.drop_state(entry.name)

    def drop_run(self, name: str, token: str) -> None:
        """Drop the tune in progress if this run (`token`) still holds it: a run ending its own."""
        with self.locked():
            try:
                state = self.state(name)
            except TuneStoreError:
                return
            if state is not None and state.running_token == token:
                self.drop_state(name)

    def queued_holders(self) -> set[str] | None:
        """Every queued quality plan's holder on the downloads ledger; None when the queue
        cannot be listed or an entry read (a hold of theirs would look orphaned: an
        unlistable queue read as empty would let a ready plan's downloads be deleted)."""
        with self.locked():
            holders = set()
            try:
                names = self.queued_names()
            except TuneStoreError:
                return None
            for name in names:
                try:
                    found = self.queued(name)
                except TuneStoreError:
                    return None
                if isinstance(found, QualityQueued):
                    holders.add(found.holder)
            return holders

    def waiting(self) -> list[Entry]:
        """The queue, `--now` first, then oldest first; an unreadable entry is skipped. A
        queue folder the system will not list raises TuneStoreError: nothing may be run or
        ended on a queue mnm could not read (the router's tuner says it once, `TuneRunner`)."""
        found = []
        for name in self.queued_names():
            with contextlib.suppress(TuneStoreError):
                if (entry := self.queued(name)) is not None:
                    found.append(entry)
        return sorted(found, key=lambda e: (not e.now, e.queued_at))

    def forget(self, name: str) -> None:
        """`mnm rm`: the queue entry, the state, the trials, a cancel's key and a give-up go. A
        queued quality tune stays for the router's tuner to end: it finds the instance gone and
        deletes the quants the tune downloaded (the store alone cannot tell which are listed)."""
        self.clear_gave_up(name)
        self.undecline(name)
        with contextlib.suppress(TuneStoreError):
            if not isinstance(self.queued(name), QualityQueued):
                self.unqueue(name)
        self.drop_state(name)
        with contextlib.suppress(OSError, BadNameError):
            self.path(name).unlink(missing_ok=True)
