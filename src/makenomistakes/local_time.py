"""The box's own clock for every name and time the user reads (a run's id and its day
folder, a calibration export's name, a reading's "When"): records keep an aware UTC time, and
what is shown is that time on this machine's clock."""

from __future__ import annotations

from datetime import UTC, datetime


def local(when: datetime | None = None) -> datetime:
    """`when` (now when None) on this machine's clock; a naive time counts as UTC."""
    moment = when or datetime.now(UTC)
    return (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).astimezone()
