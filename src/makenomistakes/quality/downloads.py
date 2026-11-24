"""The quality tunes' downloads: one ledger for every process (an approval in
the CLI or the web, a tune in the router or a terminal), under the tuning folder's lock, of
which quants each queued quality plan holds and which ones a quality tune downloaded.

- An approval writes its queue entry and holds every quant its plan measures in one step under
  the lock (`quality.entries.admit`), before anything downloads. A holder is an entry
  (`QualityQueued.holder`), never an instance's name: two plans of one instance, or of one
  repo, never let go of each other's quants.
- Each download is claimed as it starts (`claim`), under the same lock that checks the entry
  is still that approval's and the files are not here: a quant is a tune's own exactly when a
  tune downloaded it (never the instance's current quant, which stays the user's).
- Every end of a plan releases its holds (`quality.entries.end_plan`). A quant nobody holds
  that a tune downloaded, or that a save moved its instance away from, is pending: its files
  are deleted where no saved recipe or ComfyUI link lists them, and a pending quant whose files
  could not be deleted stays pending until a later end deletes them. `--keep` makes every quant
  that plan holds the user's (recorded when it holds, so an end cut short keeps them too). A
  holder whose entry is gone (an end that failed after its unqueue) is released by the next
  end or the router's sweep (`release_orphans`)."""

from __future__ import annotations

import os
from collections.abc import Collection
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.quality.plan import QualityCandidate, QualityPlan, SavedRecipe
from makenomistakes.tune.store import TuneStore, TuneStoreError

FILE = "quality-downloads.json"


class Hold(BaseModel):
    repo_id: str
    quant: str
    # The recipe the first plan measures it with: where its files are (filled from the cache
    # when it is deleted, so a download cut short or never recorded on a plan goes too).
    recipe: SavedRecipe
    fetched: bool = False  # a quality tune downloaded it (or a save moved away from it)
    holders: list[str] = Field(default_factory=list)  # the entries whose plan measures it

    @property
    def pending(self) -> bool:
        """Nobody holds it and a tune made it its own: its files are to be deleted."""
        return self.fetched and not self.holders


class Holds(BaseModel):
    schema_version: Literal[1] = 1
    holds: list[Hold] = Field(default_factory=list)
    # The holders whose plan said `--keep`, recorded when they hold: an end cut short still
    # keeps what they hold when the sweep releases them.
    keepers: list[str] = Field(default_factory=list)


class Downloads:
    def __init__(self, store: TuneStore) -> None:
        self.store = store
        self.path = store.folder / FILE

    def hold(self, holder: str, plan: QualityPlan) -> None:
        """`holder`'s plan holds each quant it measures (again: nothing changes); a `--keep`
        plan's holder is recorded as keeping them."""
        with self.store.locked():
            book = self._read()
            if plan.keep and holder not in book.keepers:
                book.keepers.append(holder)
            for row in plan.measured:
                found = _find(book, plan.repo_id, row.quant)
                if found is None:
                    found = Hold(repo_id=plan.repo_id, quant=row.quant, recipe=row.recipe)
                    book.holds.append(found)
                if holder not in found.holders:
                    found.holders.append(holder)
            self._write(book)

    def claim(self, holder: str, repo_id: str, row: QualityCandidate) -> None:
        """`holder`'s approval downloads `row` now: from here on a tune's own, deleted once
        nobody holds it (unless a `--keep` plan holding it ends)."""
        with self.store.locked():
            book = self._read()
            found = _find(book, repo_id, row.quant)
            if found is None:  # held at its admission: only a ledger edited meanwhile
                found = Hold(repo_id=repo_id, quant=row.quant, recipe=row.recipe)
                book.holds.append(found)
            if holder not in found.holders:
                found.holders.append(holder)
            found.fetched = True
            self._write(book)

    def end(self, holder: str, moved_from: str | None = None) -> set[tuple[str, str]]:
        """`holder`'s plan ends and holds nothing now, in one write: a keeper's quants are the
        user's (`--keep`: whoever downloaded them); else `moved_from`, the quant a save moved
        the instance away from, is a tune's own from now on. A quant nobody holds that is no
        tune's leaves the ledger; one that is a tune's waits as pending to be deleted. The
        (repo, quant) pairs this end left pending: its own to delete."""
        with self.store.locked():
            book = self._read()
            left = _let_go(book, holder, moved_from)
            self._write(book)
            return left

    def release_orphans(self, queued: Collection[str]) -> None:
        """End every holder that names no queued plan (an end cut short after its entry was
        unqueued: a full disk, Ctrl-C), a keeper's quants kept."""
        with self.store.locked():
            book = self._read()
            gone = sorted({h for found in book.holds for h in found.holders} - set(queued))
            for holder in gone:
                _let_go(book, holder, None)
            if gone:
                self._write(book)

    def pending(self) -> list[Hold]:
        """The quants whose files are to be deleted (this end's and any an earlier end could
        not delete)."""
        with self.store.locked():
            return [h for h in self._read().holds if h.pending]

    def forget(self, done: Collection[tuple[str, str]]) -> None:
        """The pending quants whose files were deleted (or that a recipe lists now) leave the
        ledger; one held again meanwhile stays."""
        with self.store.locked():
            book = self._read()
            book.holds = [h for h in book.holds if not (h.pending and (h.repo_id, h.quant) in done)]
            self._write(book)

    def _read(self) -> Holds:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Holds()
        except OSError as exc:
            raise TuneStoreError(f"{self.path}: {exc}") from exc
        try:
            return Holds.model_validate_json(text)
        except ValidationError as exc:
            raise TuneStoreError(
                f"{self.path} is not a ledger mnm wrote (a newer mnm?): `mnm update`"
            ) from exc

    def _write(self, book: Holds) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_name(f".{FILE}.{os.getpid()}")
        part.write_text(book.model_dump_json(indent=2), encoding="utf-8")
        part.replace(self.path)


def _find(book: Holds, repo_id: str, quant: str) -> Hold | None:
    return next((h for h in book.holds if (h.repo_id, h.quant) == (repo_id, quant)), None)


def _let_go(book: Holds, holder: str, moved_from: str | None) -> set[tuple[str, str]]:
    """`holder` released from every quant it holds; the ones that it leaves pending."""
    keeping, left = holder in book.keepers, set()
    for found in book.holds:
        if holder not in found.holders:
            continue
        found.holders.remove(holder)
        if keeping:
            found.fetched = False
        elif found.quant == moved_from:
            found.fetched = True
        if found.pending:
            left.add((found.repo_id, found.quant))
    book.keepers = [k for k in book.keepers if k != holder]
    book.holds = [h for h in book.holds if h.holders or h.fetched]
    return left
