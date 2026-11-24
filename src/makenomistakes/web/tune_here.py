"""A speed tune the web UI runs itself when no router answers, for the Tuning tab's
Optimise for speed. A quality tune it never runs: Optimise for quality leaves it queued for
the router."""

from __future__ import annotations

import contextlib
import threading

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import Report
from makenomistakes.supervisor import SupervisorError
from makenomistakes.tune.store import Entry
from makenomistakes.tune.trial import copy_name
from makenomistakes.tune.trial_types import CANCELLED, Cut
from makenomistakes.tune.wiring import foreground
from makenomistakes.tune.words import capital, what


class _Run(Cut):
    """One run's cut, and whether the user's Cancel was asked of it while it went."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_asked = False


class HereRuns:
    """The speed tunes this process runs when no router answers (as the CLI's `--now`): each
    one's cut, which its Cancel sets so the run looks at its entry's mark now. A run that
    pauses while a Cancel is asked of it says that and no more: what the Cancel came to is
    its own job's to say (a run's end that waited for the Cancel's verdict could time out
    into a false word, "it resumes" of a tune that ended)."""

    def __init__(self, service: InstanceService) -> None:
        self.service = service
        self.cuts: dict[str, _Run] = {}
        self._lock = threading.Lock()  # the registry, between the runs and the Cancels

    def run(self, entry: Entry, report: Report) -> list[str]:
        report("The router is not running: the tune runs in the web UI's process.")
        cut = _Run()
        with self._lock:
            self.cuts[entry.name] = cut
        try:
            end = foreground(self.service, report).tune(entry, cut)
        finally:
            with self._lock:
                self.cuts.pop(entry.name, None)
        if end.status in {"cut", "held"} and cut.cancel_asked:  # paused as the Cancel is judged
            return [
                f"{capital(what(entry))} paused as its Cancel was asked; the Cancel's job says "
                "what came of it."
            ]
        if end.status not in {"tuned", "cancelled"}:
            raise InstanceError(end.line)
        return [end.line]

    def runs(self, name: str) -> bool:
        """This process runs a tune of `name` now."""
        with self._lock:
            return name in self.cuts

    def cancel_asked(self, name: str) -> None:
        """The Tuning tab's Cancel is asked of the tune of `name` this process runs: noted on
        that run before the cancel is judged, so a pause it ends with meanwhile is worded as
        one (a run can end between a cancel and a mark made after it). Noted
        on the run itself, so no later run reads it."""
        with self._lock:
            if (run := self.cuts.get(name)) is not None:
                run.cancel_asked = True

    def cut(self, name: str) -> bool:
        """Cut the tune of `name` this process runs and stop its copy, so the run looks at its
        entry's mark now; False when none runs."""
        with self._lock:
            cut = self.cuts.get(name)
        if cut is None:
            return False
        cut.cut(CANCELLED)
        with contextlib.suppress(SupervisorError):
            self.service.supervisor.stop(copy_name(name))
        return True


def answered(asked: tuple[int, str]) -> list[str]:
    """The router's answer as a job's lines; a refusal fails the job with its words."""
    if asked[0] >= 400:
        raise InstanceError(asked[1])
    return [asked[1]]
