"""Which runs a prune deletes (`mnm outputs prune`, the Runs gallery's Prune): out of
`runs.py`, which sits at the size line."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from makenomistakes import run_lineage
from makenomistakes.runs import RunRecord, RunsRead


def select(
    read: RunsRead,
    older_than: timedelta | None = None,
    until_bytes: int | None = None,
    now: datetime | None = None,
) -> list[RunRecord]:
    """The runs a prune deletes: older than the age, then the oldest until the rest fit under
    `until_bytes`. A run kept for the user (going, or a film waiting at a review or
    interrupted, `RunRecord.kept_for_owner`) is never picked, and never counted as one that
    could go, nor is a run under a record that cannot be read (`run_lineage.kept`'s rule,
    the one `mnm clean runs` reads)."""
    held = run_lineage.kept(read)  # The runs under a kept one, or an unread one
    runs = [r for r in read.records if r.id not in held]
    picked: list[RunRecord] = []
    if older_than is not None:
        cut = (now or datetime.now(UTC)) - older_than
        picked = [r for r in runs if r.created_at < cut]
    if until_bytes is not None:
        left = [r for r in runs if r not in picked]
        total = sum(r.size for r in left)
        for record in sorted(left, key=lambda r: r.created_at):
            if total <= until_bytes:
                break
            if not record.size:
                continue  # frees nothing: its files are gone already
            picked.append(record)
            total -= record.size
    return picked
