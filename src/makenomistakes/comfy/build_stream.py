"""`docker build` for ComfyUI's image and its packs layer, told as it happens.

A build takes 10-30 minutes, and one that printed "Building …" and then nothing looked
hung. BuildKit's plain progress is read line by line through the runner's `stream` seam, and
a quiet stretch (a large wheel downloading) says how long the build has been going at least
once a minute.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from makenomistakes.hardware.probe import LineRunner

QUIET_SECONDS = 60.0  # the longest the user waits without a line
TAIL_LINES = 20  # what a failure quotes
PLAIN = "--progress=plain"  # BuildKit's line-per-step output (the default draws a TTY panel)

Say = Callable[[str], None]


class BuildLine(str):
    """A line of the build's own output, as `progress` hears it beside mnm's status lines:
    the CLI prints it as it came (dim, never as markup); the web keeps it like any line."""

    __slots__ = ()


def build(
    runner: LineRunner,
    argv: list[str],
    say: Say,
    clock: Callable[[], float],
    timeout: float,
) -> tuple[int, str]:
    """Run a `docker build`, telling each line (a `BuildLine`); the exit code and the last
    lines."""
    started = clock()
    tail: deque[str] = deque(maxlen=TAIL_LINES)

    def on_line(line: str | None) -> None:
        if line is None:
            minutes = int((clock() - started) // 60)
            say(BuildLine(f"{minutes} min elapsed, still building"))
            return
        tail.append(line)
        say(BuildLine(line))

    code = runner.stream(argv, on_line, timeout=timeout, quiet=QUIET_SECONDS)
    return code, "\n".join(tail)
