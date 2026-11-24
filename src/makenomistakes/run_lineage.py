"""A run's place in its take's tree: the children a step made under one parent (four
candidates of a song) and the line back to the first run, read over the records the store
gives, so the gallery's one parse serves it too."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from makenomistakes.runs import RunRecord, RunsRead


def children(records: Iterable[RunRecord], run_id: str) -> list[RunRecord]:
    """The runs made under `run_id`, oldest first (candidate 1 before candidate 2)."""
    return sorted((r for r in records if r.parent == run_id), key=lambda r: r.created_at)


def lineage(records: Iterable[RunRecord], run_id: str) -> list[RunRecord]:
    """The run and every parent above it, the run first; a parent that is gone (deleted)
    ends the line where it is, and a loop (a hand-edited file) is never followed twice."""
    by_id = {r.id: r for r in records}
    line: list[RunRecord] = []
    seen: set[str] = set()
    current = by_id.get(run_id)
    while current is not None and current.id not in seen:
        line.append(current)
        seen.add(current.id)
        current = by_id.get(current.parent) if current.parent else None
    return line


def nested(records: list[RunRecord]) -> list[tuple[RunRecord, list[RunRecord]]]:
    """The gallery's rows: each run whose parent is not listed, with its children under it
    (newest parent first, as the records come)."""
    listed = {r.id for r in records}
    under: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.parent in listed:
            under.setdefault(record.parent, []).append(record)
    return [
        (record, sorted(under.get(record.id, []), key=lambda r: r.created_at))
        for record in records
        if record.parent not in listed
    ]


def kept(read: RunsRead) -> set[str]:
    """The runs `mnm clean` and a prune never touch: each one kept for the user (going, or
    a pipeline waiting or interrupted) and every run under it (a song waiting at the pick
    keeps its candidates), and every run under a record that cannot be read
    (`RunsRead.unread`: it may be a waiting song's, whose candidates prune would otherwise
    delete while `mnm clean runs` kept them). The whole read is
    taken, so no caller can leave that half out; a child whose parent was simply deleted is
    no one's to keep."""
    records, unread = read.records, set(read.unread)
    by_parent: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.parent:
            by_parent.setdefault(record.parent, []).append(record)
    found: set[str] = set()
    todo = [r for r in records if r.kept_for_owner or (r.parent is not None and r.parent in unread)]
    while todo:
        record = todo.pop()
        if record.id in found:
            continue
        found.add(record.id)
        todo += by_parent.get(record.id, [])
    return found
