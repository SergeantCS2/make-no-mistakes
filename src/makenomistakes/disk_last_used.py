"""When an instance was last used, for the disk policy's order.

The newest of: its container running now; its last start, by the router or by hand (every
ready start leaves a `load` reading, `measurements.py`, and an exited container keeps Docker's
StartedAt); its last request (the `request` readings, which the router writes only while a
recipe has fewer than a few or the newest is stale, and each window of the
router's queue, dated by its last request's end, written a few minutes after it);
and, as the floor of floors, when its recipe file was last written (when it was added, and by
every file a download finishes), its `created_at` only when the file cannot be read. The
router's idle clock, which knows each reply's time, lives in the router's memory only and is
never saved, so it is not a source; a model the router keeps loaded is running, and the
policy never evicts a running one."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from makenomistakes.measurements import Kind, Reading
from makenomistakes.recipes import Recipe
from makenomistakes.supervisor import ContainerState, ContainerStatus

RECENT = timedelta(hours=24)  # used since: the policy never evicts it


@dataclass(frozen=True)
class LastUse:
    at: datetime | None
    source: str  # running now, a start, a request, saved, added

    def recent(self, now: datetime) -> bool:
        return self.at is not None and now - self.at < RECENT


def last_use(
    recipe: Recipe,
    readings: list[Reading],
    recipe_file: Path | None,
    container: ContainerStatus | None,
    now: datetime,
) -> LastUse:
    if container is not None and container.state is ContainerState.RUNNING:
        return LastUse(now, "running now")
    found: list[tuple[datetime, str]] = []
    loads = [r.at for r in readings if r.kind is Kind.LOAD]
    # a request reading, and a window of the router's queue, dated by its last request
    requests = [r.at for r in readings if r.kind in (Kind.REQUEST, Kind.QUEUE)]
    if loads:
        found.append((max(loads), "a start"))
    if requests:
        found.append((max(requests), "a request"))
    if container is not None and (started := docker_time(container.started_at)) is not None:
        found.append((started, "a start"))
    saved = None
    if recipe_file is not None:
        with contextlib.suppress(OSError):
            saved = datetime.fromtimestamp(recipe_file.stat().st_mtime, UTC)
    # The file's time is the floor (written when added, and by every download step); the
    # creation time inside it only when the file cannot say
    if recipe.evicted is not None:  # the eviction wrote the file: its own record says
        found.append((recipe.evicted.last_used or recipe.created_at, "before its eviction"))
    elif saved is not None:
        found.append((saved, "saved"))
    else:
        found.append((recipe.created_at, "added"))
    at, source = max(found, key=lambda pair: _aware(pair[0]))
    return LastUse(_aware(at), source)


def docker_time(value: str | None) -> datetime | None:
    """Docker's StartedAt (`2026-10-26T03:00:01.123456789Z`); None for none or the zero time
    a container that never started reports."""
    if not value or value.startswith("0001-"):
        return None
    head, _, fraction = value.removesuffix("Z").partition(".")
    try:
        return datetime.fromisoformat(f"{head}.{(fraction + '000000')[:6]}+00:00")
    except ValueError:
        return None


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
