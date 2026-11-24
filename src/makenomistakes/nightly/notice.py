"""The night's one notice (ntfy and the toast, both as configured): sent only when at least
one verdict is a regression or a failure the user must hear of; silence
on a quiet night, and the first night's "baseline recorded" is a journal line, not a notice.
At most `LINES` lines of figures, then "and N more: `mnm nightly show`"."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.nightly.report import NightlyReport
from makenomistakes.notify import Notice

LINES = 10
QUIET = "no notice: nothing regressed"


def compose(report: NightlyReport) -> Notice | None:
    told = report.told
    if not told:
        return None
    count = len(told)
    regressed = len(report.regressions)
    words = (
        [f"{regressed} {'regression' if regressed == 1 else 'regressions'}"] if regressed else []
    )
    if count > regressed:  # told, never a regression (a model first not tried for room)
        words.append(f"{count - regressed} to read")
    title = f"mnm nightly: {', '.join(words)}"
    lines = []
    for verdict in told[:LINES]:
        action = verdict.action
        done = f" ({action.line.rstrip('.')})" if action is not None and action.automatic else ""
        lines.append(f"{verdict.line}{done}")
    if count > LINES:
        lines.append(f"and {count - LINES} more: `mnm nightly show`")
    return Notice(title, "\n".join(lines), failed=regressed > 0)


def send(report: NightlyReport, channels: Callable[[Notice], list[str]]) -> list[str]:
    """Each channel's outcome in words; `QUIET` when there is nothing to tell. A channel that
    cannot be reached is a line, never an error (`Notifier.send` never raises)."""
    notice = compose(report)
    if notice is None:
        return [QUIET]
    said = channels(notice)
    return said or ["no notice sent: no channel is on (`mnm notify --ntfy URL` turns ntfy on)"]
