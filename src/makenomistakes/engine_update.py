"""`mnm engines update` and `mnm engines test` as one sequence the CLI and the web UI share:
pull the newest builds, self-test each one that changed, roll a failure back.

What happens is told line by line through `say` (a `Line` with a tone, never markup), so the
CLI colours it and the web keeps it as the job's notes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from makenomistakes import engines
from makenomistakes.engine_selftest import EngineSelfTest, Outcome, SelfTest
from makenomistakes.hub.facts import Engine
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.supervisor import DockerSupervisor, SupervisorError
from makenomistakes.tune.orphans import Refused, queue_speed
from makenomistakes.tune.store import Queued, TuneStore

FAILURES = (InstanceError, engines.EngineError, SupervisorError)


class Tone(StrEnum):
    PLAIN = "plain"
    OK = "ok"
    WARN = "warn"
    BAD = "bad"
    DIM = "dim"


@dataclass(frozen=True)
class Line:
    """One line of the story: `head` in its tone, `rest` plain; `indent` under the image."""

    head: str
    rest: str = ""
    tone: Tone = Tone.PLAIN
    indent: bool = False

    @property
    def text(self) -> str:
        return f"{'  ' if self.indent else ''}{self.head}{self.rest}"


Say = Callable[[Line], None]

CHANGE_TONE = {
    engines.Change.UPDATED: Tone.OK,
    engines.Change.PULLED: Tone.OK,
    engines.Change.CURRENT: Tone.DIM,
    engines.Change.FAILED: Tone.BAD,
}
OUTCOME_TONE = {
    Outcome.PASSED: Tone.OK,
    Outcome.FAILED: Tone.BAD,
    Outcome.INCONCLUSIVE: Tone.WARN,
    Outcome.SKIPPED: Tone.WARN,
}


def update_engines(
    service: InstanceService,
    docker: DockerSupervisor,
    state_dir: Path,
    which: set[Engine],
    say: Say,
) -> bool:
    """Pull every engine in `which` (all pulled or used ones when empty), self-test each
    build that changed. True when something failed (a pull, or a build that was rolled
    back). Raises the `FAILURES` that stop it before any pull."""
    rows = engines.engine_status(docker, service.all(), engines.load_ledger(state_dir))
    images = engines.images_to_update(rows, which)
    if not images:
        say(Line("No engine image is pulled or used yet; nothing to update."))
        return False
    say(Line("A new build is a few GB per engine; each pull is shown as it starts."))
    results = engines.update(docker, images, state_dir, progress=lambda t: say(Line(f"{t} …")))
    running = set(service.running_names())
    users = {r.image: r.instances for r in rows}
    kinds = {r.image: r.engine for r in rows}
    failed = False

    def told(line: str) -> None:  # a settle's words, under the build they came from
        say(Line(line, indent=True))

    for result in results:
        rest = f" {result.image}"
        if result.change is engines.Change.UPDATED and result.before and result.after:
            rest += f"  {result.before.short_id} → {result.after.short_id}"
        elif result.change is engines.Change.FAILED:
            rest += f"  {result.detail}"
        say(Line(result.change.value, rest, CHANGE_TONE[result.change]))
        failed |= result.change is engines.Change.FAILED
        if result.change is not engines.Change.UPDATED or result.image not in kinds:
            continue
        previous = result.before.id if result.before else None
        tested = self_test(service, kinds[result.image], result.image, state_dir, previous, say)
        if tested.outcome is Outcome.FAILED:
            failed = True
            continue  # rolled back: what runs is on the build the tag names again
        if retuned := requeue_tunes(service, result.image, told):  # A tune is per build
            say(Line(f"Re-tuning queued: {' '.join(retuned)}.", indent=True))
        stale = sorted(set(users.get(result.image, [])) & running)
        if stale:
            say(
                Line(
                    f"Still on the old build until restarted: {' '.join(stale)}. "
                    "`mnm stop <name>` then `mnm start <name>`.",
                    indent=True,
                )
            )
    return failed


def requeue_tunes(service: InstanceService, image: str, say: Callable[[str], None]) -> list[str]:
    """Queue a tune of every tuned instance on `image`: its tune was measured on the build
    before (`mnm ls` reads "tuned for another build" until it runs). Never over a quality tune
    waiting; a mark a stopped run left is settled first (`tune.orphans.queue_speed`)."""
    store = TuneStore(service.settings.expanded_state_dir())
    names = []
    for recipe in service.all():
        if recipe.image != image or recipe.tuned is None:
            continue
        entry = Queued(name=recipe.name, reason="another build")
        queued = queue_speed(service, store, entry, say)
        if isinstance(queued, Refused):  # why, said
            say(queued.line(recipe.name))
        else:
            names.append(recipe.name)
    return names


def self_test(
    service: InstanceService,
    engine: Engine,
    image: str,
    state_dir: Path,
    previous: str | None,
    say: Say,
    roll_back: bool = True,
) -> SelfTest:
    """Run the build's self-test against `previous`; a failure after an update rolls back."""
    tester = EngineSelfTest(service, progress=lambda t: say(Line(f"{t} …", indent=True)))
    try:
        result = tester.run(engine, image, previous)
    except FAILURES as exc:  # Docker itself failed: the build is untested, not proven bad
        say(Line("Not tested", f": {exc}", Tone.WARN, indent=True))
        return SelfTest(engine, image, Outcome.SKIPPED, str(exc))
    say(Line(result.reason, tone=OUTCOME_TONE[result.outcome], indent=True))
    if result.outcome is Outcome.SKIPPED:
        say(Line(f"`mnm engines test {engine.value}` runs it later.", indent=True))
    if result.outcome is not Outcome.FAILED:
        return result
    if result.log is not None:
        say(Line(f"The copy's log: {result.log}", indent=True))
    if not roll_back:
        say(
            Line(
                f"`mnm engines rollback {engine.value}` goes back to the build before it.",
                indent=True,
            )
        )
        return result
    try:
        done = engines.rollback(service.supervisor, image, state_dir)
    except FAILURES as exc:
        say(Line("Could not roll back", f": {exc}", Tone.BAD, indent=True))
        return result
    say(Line(f"Rolled back: {image} points at {done.restored.short_id} again.", indent=True))
    _started_on(service, image, result.build, say)
    return result


def _started_on(service: InstanceService, image: str, rejected: str | None, say: Say) -> None:
    """Name what started on the rejected build while it was tested (a request the router
    loaded an instance for, a start in another terminal): restarting puts it back."""
    try:
        statuses = service.statuses()
    except FAILURES:
        return
    on_it = sorted(
        s.recipe.name
        for s in statuses
        if s.running and s.recipe.image == image and rejected and s.build == rejected
    )
    if on_it:
        say(
            Line(
                f"Started on the rejected build meanwhile: {' '.join(on_it)}. "
                "`mnm stop <name>` then `mnm start <name>` puts it on the good one.",
                indent=True,
            )
        )
