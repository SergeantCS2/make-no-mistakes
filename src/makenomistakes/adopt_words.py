"""`mnm machine adopt`'s counts and the one sentence they make: the CLI's last line,
the notice, `machine.json`'s `adopted.summary`, and the router's journal line, which the
router words itself from the counts (`POST /__mnm/adopted` carries numbers, never text)."""

from __future__ import annotations

from pydantic import BaseModel, Field

ADOPT_BODY_LIMIT = 1024


class AdoptCounts(BaseModel):
    refitted: int = Field(default=0, ge=0)  # plans saved anew
    downloads: int = Field(default=0, ge=0)  # files fetched (or, unanswered, to fetch)
    tunes: int = Field(default=0, ge=0)  # tunes queued
    kept: int = Field(default=0, ge=0)  # not for this box (or not saved): kept as they were
    declined: int = Field(default=0, ge=0)  # files the user said no to: the next start's


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def summary(counts: AdoptCounts) -> str:
    """`3 plans re-fitted, 2 downloads, 1 tune queued` (+ what was kept or left to fetch)."""
    parts = [
        _n(counts.refitted, "plan re-fitted", "plans re-fitted"),
        _n(counts.downloads, "download", "downloads"),
        _n(counts.tunes, "tune queued", "tunes queued"),
    ]
    if counts.declined:
        parts.append(
            _n(counts.declined, "file left for the next start", "files left for the next start")
        )
    if counts.kept:
        parts.append(_n(counts.kept, "instance kept as it was", "instances kept as they were"))
    return ", ".join(parts)


def journal_line(counts: AdoptCounts, box: str | None) -> str:
    """The router's line: `Adopted RTX 3090 24 GB, …, WSL: 3 plans re-fitted, …`."""
    return f"Adopted {box or 'this box'}: {summary(counts)}."
