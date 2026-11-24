"""The probe's command runner: a command's exit code and output, or its lines as they
come."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Callable


class SubprocessRunner:
    def run(self, argv: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",  # an engine log with a stray byte must not crash `mnm logs`
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return 127, "", f"{argv[0]}: not found"
        except subprocess.TimeoutExpired:
            return 124, "", f"{argv[0]}: timed out after {timeout:.0f}s"
        return proc.returncode, proc.stdout, proc.stderr

    def stream(
        self,
        argv: list[str],
        on_line: Callable[[str | None], None],
        timeout: float,
        quiet: float = 60.0,
    ) -> int:
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError:
            on_line(f"{argv[0]}: not found")
            return 127
        lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=_pump, args=(proc, lines), daemon=True).start()
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                proc.kill()
                proc.wait()
                on_line(f"{argv[0]}: timed out after {timeout:.0f}s")
                return 124
            try:
                line = lines.get(timeout=min(quiet, left))
            except queue.Empty:
                on_line(None)  # still running, nothing printed: the caller says so
                continue
            if line is None:  # the output closed: the process is ending
                return proc.wait()
            on_line(line)


def _pump(proc: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
    """Read the output line by line into the queue; None once it closes."""
    assert proc.stdout is not None  # opened with stdout=PIPE
    for line in proc.stdout:
        lines.put(line.rstrip("\r\n"))
    lines.put(None)
