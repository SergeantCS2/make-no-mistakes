"""The take a song uses: the picked candidate's, or an edit made from it. One strict
rule every reader goes through (the steps, the edits, the page, the CLI, the export), so an
edit that is gone, unfinished, cleaned, or made from another pick's take is said once with
the way back, and never read as the candidate's.

An edit is a child run of the song (role `edit`) that records the take it was made from
(`source`: the candidate or an earlier edit) and the picked candidate its takes go back to
(`root`). A take is usable while it is a finished edit of the song with its files, from the
candidate picked now: no walk through the edits between, so removing an older edit never
locks the later ones out. Its folder holds the new take (`song.wav`) and the lyrics it was
sung with (`lyrics.json`), so a re-sing carries on to the edits after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from makenomistakes.pipeline import song
from makenomistakes.pipeline.song import Lyrics, SongError
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.runs import RunError, RunRecord, RunState, RunStore

EDIT: Final = "edit"  # an edit's role
SOURCE = "source"  # an edit's setting: the take it was made from (a run id)
ROOT = "root"  # an edit's setting: the picked candidate its takes go back to (a run id)
MAX_HISTORY = 64  # edits followed back for the provenance (a loop in hand-edited records)


class TakeError(PipelineError):
    """The take in use, or one asked for, cannot be used; worded for the user."""


@dataclass(frozen=True)
class Take:
    """The take the song uses: the picked candidate it goes back to, and the edits that made
    it from there, the first one first (those still on record; `gaps` names the others)."""

    candidate: RunRecord
    edits: tuple[RunRecord, ...] = ()
    gaps: tuple[str, ...] = ()

    @property
    def run(self) -> RunRecord:
        """The run holding the take's `song.wav`."""
        return self.edits[-1] if self.edits else self.candidate

    @property
    def edit(self) -> RunRecord | None:
        return self.edits[-1] if self.edits else None


def candidate(store: RunStore, run: RunRecord, index: int) -> RunRecord | None:
    """Candidate `index`'s child run, or None (not made yet, or gone)."""
    step = run.step(song.SONG, index)
    if step is None or not step.text:
        return None
    try:
        return store.get(step.text)
    except RunError:
        return None


def edits(store: RunStore, run: RunRecord) -> list[RunRecord]:
    """The song's edits, oldest first (any take's, finished or not)."""
    return [c for c in store.children(run.id) if c.role == EDIT]


def picked(store: RunStore, run: RunRecord) -> RunRecord:
    """The picked candidate's run; refused when nothing is picked, or saying how to make it
    again when its run is gone."""
    index = song.picked_index(run.settings)
    if index is None:
        raise TakeError(
            f"{run.id} has no picked take: pick a candidate first (`mnm song pick {run.id} <n>`)."
        )
    child = candidate(store, run, index)
    if child is None:
        raise TakeError(
            f"{run.id}'s picked {song.candidate_label(index)} is gone: `mnm song redo {run.id} "
            f"song {index + 1}` makes it again, or `mnm song pick {run.id} <n>` picks another."
        )
    return child


def refusal(store: RunStore, run: RunRecord, edit_id: str) -> str | None:
    """Why the song cannot use this edit's take (None when it can): it is no finished edit
    of the song with its files, or it goes back to another candidate than the picked one."""
    root = picked(store, run)
    if edit_id == root.id:
        return None  # the picked candidate itself (`use_take` reads it as `candidate`)
    try:
        edit = store.get(edit_id)
    except RunError:
        return f"{edit_id} is gone"
    checks = (
        (edit.parent != run.id or edit.role != EDIT, f"{edit_id} is no edit of {run.id}"),
        (edit.state is not RunState.DONE, f"{edit_id} is not a finished edit ({edit.state.value})"),
        (edit.files_removed, f"{edit_id} had its files cleaned"),
        (
            ROOT not in edit.settings,  # made before edits kept it
            f"{edit_id} does not record which candidate's take it goes back to",
        ),
        (
            edit.settings.get(ROOT) != root.id,
            f"{edit_id} was made from another pick's take, not the picked candidate's",
        ),
    )
    return next((why for failed, why in checks if failed), None)


def usable(store: RunStore, run: RunRecord) -> dict[str, str | None]:
    """Each of the song's edits by id, oldest first, with why it cannot be used (None when it
    can)."""
    return {e.id: refusal(store, run, e.id) for e in edits(store, run)}


def take(store: RunStore, run: RunRecord) -> Take:
    """The take in use: the edit the song names (`take_run`) when it can be used, else the
    picked candidate's when it names none; refused, saying how to go back, otherwise."""
    root = picked(store, run)
    chosen = run.settings.get(song.TAKE_RUN)
    if not chosen or chosen == root.id:
        return Take(root)
    why = refusal(store, run, chosen)
    if why is not None:
        raise TakeError(
            f"The take in use, the edit {chosen}, cannot be used: {why}. `mnm song take "
            f"{run.id} candidate` goes back to the picked candidate's."
        )
    return _history(store, run, store.get(chosen), root)


def _history(store: RunStore, run: RunRecord, last: RunRecord, root: RunRecord) -> Take:
    """The edits that made `last`'s take, back to the candidate as far as the records go."""
    found: list[RunRecord] = [last]
    gaps: list[str] = []
    at = last.settings.get(SOURCE, "")
    while at and at != root.id and len(found) < MAX_HISTORY:
        try:
            edit = store.get(at)
        except RunError:
            gaps.append(at)  # removed (`mnm outputs rm`): the take itself is whole
            break
        if edit.parent != run.id or edit.role != EDIT:
            break
        found.append(edit)
        at = edit.settings.get(SOURCE, "")
    return Take(root, tuple(found[::-1]), tuple(gaps))


def take_file(store: RunStore, run: RunRecord, found: Take | None = None) -> Path:
    """The take in use's `song.wav` (of `found` when the caller has the take already)."""
    found = found or take(store, run)
    try:
        return store.file(found.run, CANDIDATE_FILE)
    except RunError as exc:
        how = (
            f"`mnm song take {run.id} candidate` goes back to the picked candidate's."
            if found.edit is not None
            else "Redo that candidate, or pick another."
        )
        raise TakeError(f"{exc} {how}") from exc


def lyrics(store: RunStore, run: RunRecord, found: Take) -> Lyrics:
    """The lyrics the take was sung with: the last edit's (a re-sing changes them), else the
    song's."""
    folder = store.folder(found.edit) if found.edit is not None else store.folder(run)
    try:
        return song.read_lyrics(folder)
    except SongError as exc:
        which = f"the edit {found.edit.id}'s" if found.edit is not None else f"{run.id}'s"
        raise TakeError(f"The lyrics, {which}, are missing or unreadable ({exc}).") from exc


def words(run: RunRecord, found: Take) -> str:
    """`candidate 2` or `the edit <id>`: the take, as a line names it."""
    if found.edit is not None:
        return f"the edit {found.edit.id}"
    index = song.picked_index(run.settings)
    return song.candidate_label(index) if index is not None else "the picked candidate"


@dataclass(frozen=True)
class Shown:
    """What the Song page's Edits panel shows: the take in use, why the one the song names
    cannot be used, the lyrics it was sung with, and each edit's refusal (None: usable)."""

    take: Take | None = None
    problem: str | None = None
    lyrics: Lyrics | None = None
    refused: dict[str, str | None] = field(default_factory=dict)


def shown(store: RunStore, run: RunRecord) -> Shown:
    """The Edits panel's state; nothing before a pick (no edits to offer yet)."""
    try:
        picked(store, run)
    except TakeError:
        return Shown()
    refused = usable(store, run)
    try:
        found = take(store, run)
    except TakeError as exc:
        return Shown(problem=str(exc), refused=refused)
    try:
        sung = lyrics(store, run, found)
    except TakeError:
        sung = None  # the re-sing form says it needs the lyrics
    return Shown(take=found, lyrics=sung, refused=refused)
