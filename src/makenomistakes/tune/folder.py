"""The tuning folder's files: `<state>/tuning/`, read and written whole behind the
folder's lock (`JsonFolder`, `FolderLock`). What may be written when (who holds a queued tune,
its marks, the queue's rules) is `tune.store.TuneStore`'s, built on this.

- `<name>.jsonl`: every trial, appended (`TrialResult`, shape-versioned like readings; a line
  of another shape is skipped).
- `<name>.state.json`: the tune in progress (`TuneState`): its session, the score's mode, its
  budget and the settings it started from. A tune cut by an outside request resumes from it
  and the session's trials; the process running it says so (`running_pid`), which is what
  `mnm ls` and `mnm tune --status` read.
- `queue/<name>.json`: a tune waiting for an idle card (`Queued`, `QualityQueued`).
- `<name>.declined.json`: the key a tune was cancelled on (the stale sweep leaves it).
- `<name>.gaveup.json`: the last tune that gave up without a result, and why (`GaveUp`, so
  `mnm tune --status` says why rather than nothing); the nightly does not queue a tune again
  for the same settings, the user's own `mnm tune` does."""

from __future__ import annotations

import contextlib
from pathlib import Path

from pydantic import ValidationError

from makenomistakes.json_folder import BadNameError, JsonFolder, unlisted
from makenomistakes.recipes.tuned import TuneKey
from makenomistakes.tune.records import (
    ENTRY,
    Entry,
    GaveUp,
    TuneState,
    same_entry,
)
from makenomistakes.tune.trial_types import TrialResult

FOLDER = "tuning"
QUEUE = "queue"
STATE = ".state"  # `<name>.state.json`: the stem carries it, so names() lists instances only
DECLINED = ".declined"  # `<name>.declined.json`: the key a tune was cancelled on
GAVE_UP = ".gaveup"  # `<name>.gaveup.json`: a tune that ended without a result
KEEP = 400  # trial lines kept per instance; the oldest go once a file holds twice as many


class TuneStoreError(RuntimeError):
    """A tuning file mnm cannot read (a newer mnm wrote it, or it is damaged)."""


class TuneFolder:
    def __init__(self, state_dir: Path) -> None:
        self.folder = state_dir / FOLDER
        self._states = JsonFolder(self.folder)
        self._queue = JsonFolder(self.folder / QUEUE)
        self.locked = self._states.locked

    # --- the trials -------------------------------------------------------------------

    def path(self, name: str) -> Path:
        self._states.path(name)  # refuses a name that would leave the folder
        return self.folder / f"{name}.jsonl"

    def append(self, result: TrialResult) -> None:
        path = self.path(result.instance)
        with self.locked():
            self.folder.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                out.write(result.model_dump_json() + "\n")
            self._trim(path)

    def trials(self, name: str, session: str | None = None) -> list[TrialResult]:
        """Every trial readable, oldest first; only `session`'s when given."""
        try:
            text = self.path(name).read_text(encoding="utf-8")
        except (OSError, BadNameError):
            return []
        found = []
        for line in text.splitlines():
            try:
                result = TrialResult.model_validate_json(line)
            except ValidationError:
                continue
            if session is None or result.session == session:
                found.append(result)
        return found

    def _trim(self, path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 2 * KEEP:
            path.write_text("\n".join(lines[-KEEP:]) + "\n", encoding="utf-8")

    # --- the tune in progress ---------------------------------------------------------

    def state(self, name: str) -> TuneState | None:
        try:
            text = self._states.read(name + STATE)
        except (FileNotFoundError, BadNameError):
            return None
        except OSError as exc:
            raise TuneStoreError(f"{self.folder}/{name}{STATE}.json: {exc}") from exc
        try:
            return TuneState.model_validate_json(text)
        except ValidationError as exc:
            raise TuneStoreError(
                f"{self.folder}/{name}{STATE}.json is not a tune mnm wrote (a newer mnm?): "
                f"`mnm update`, or `mnm tune {name} --cancel` starts over"
            ) from exc

    def save_state(self, state: TuneState) -> None:
        with self.locked():
            self._states.write(state.name + STATE, state.model_dump_json(indent=2))

    def drop_state(self, name: str) -> None:
        with self.locked(), contextlib.suppress(FileNotFoundError, BadNameError):
            self._states.unlink(name + STATE)

    def states(self) -> list[TuneState]:
        """Every tune in progress mnm can read (a `.declined` or `.gaveup` file is no tune); a
        folder the system will not list is a TuneStoreError naming it, never none."""
        try:
            stems = self._states.names()
        except OSError as exc:
            raise TuneStoreError(unlisted("tuning folder", self.folder, exc)) from exc
        found = []
        for stem in stems:
            if stem.endswith(STATE):
                with contextlib.suppress(TuneStoreError):
                    if (state := self.state(stem.removesuffix(STATE))) is not None:
                        found.append(state)
        return found

    # --- the queue's files -----------------------------------------------------------

    def put(self, entry: Entry) -> None:
        """Write `entry` as it is (the quality entries' own steps, `quality.entries`)."""
        with self.locked():
            self._queue.write(entry.name, entry.model_dump_json(indent=2))

    def queued_names(self) -> list[str]:
        """The names with a queue file, readable or not; a queue folder the system will not
        list is a TuneStoreError naming it, never an empty queue (read as empty, the sweep
        would let go of a ready plan's holds and delete its downloads)."""
        try:
            return self._queue.names()
        except OSError as exc:
            raise TuneStoreError(unlisted("tuning queue", self.folder / QUEUE, exc)) from exc

    def queued(self, name: str) -> Entry | None:
        try:
            text = self._queue.read(name)
        except (FileNotFoundError, BadNameError):
            return None
        except OSError as exc:
            raise TuneStoreError(f"{self.folder}/{QUEUE}/{name}.json: {exc}") from exc
        try:
            return ENTRY.validate_json(text)
        except ValidationError as exc:
            raise TuneStoreError(
                f"{self.folder}/{QUEUE}/{name}.json is not a queue entry this mnm reads (a newer "
                f"mnm's, or an earlier shape): `mnm update`, or `mnm tune {name} --cancel` "
                "removes it"
            ) from exc

    def unqueue(self, name: str, entry: Entry | None = None) -> bool:
        """Unqueue the instance's entry, or only `entry` itself when given (an entry queued
        in its place since stays)."""
        with self.locked():
            if entry is not None and not same_entry(self.queued(name), entry):
                return False
            try:
                self._queue.unlink(name)
            except (FileNotFoundError, BadNameError):
                return False
            return True

    # --- a cancel ---------------------------------------------------------------------

    def decline(self, name: str, key: TuneKey) -> None:
        """The user cancelled its tune while the box read `key`: the stale sweep leaves it
        until the key changes (or the user queues it again)."""
        with self.locked():
            self._states.write(name + DECLINED, key.model_dump_json())

    def undecline(self, name: str) -> None:
        """Queued again by the user: no cancel's key is remembered any more."""
        with self.locked(), contextlib.suppress(FileNotFoundError, BadNameError):
            self._states.unlink(name + DECLINED)

    def declined(self, name: str, key: TuneKey) -> bool:
        """Whether its tune was cancelled on this very key."""
        try:
            text = self._states.read(name + DECLINED)
        except (OSError, BadNameError):
            return False
        try:
            return TuneKey.model_validate_json(text) == key
        except ValidationError:
            return False

    # --- a tune that gave up -----------------------------------------------------------

    def give_up(self, ended: GaveUp) -> None:
        with self.locked():
            self._states.write(ended.name + GAVE_UP, ended.model_dump_json(indent=2))

    def gave_up(self, name: str) -> GaveUp | None:
        try:
            return GaveUp.model_validate_json(self._states.read(name + GAVE_UP))
        except (OSError, BadNameError, ValidationError):
            return None

    def clear_gave_up(self, name: str) -> None:
        with self.locked(), contextlib.suppress(FileNotFoundError, BadNameError):
            self._states.unlink(name + GAVE_UP)
