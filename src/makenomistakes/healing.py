"""The self-healing router: an engine that died while the router counted it
loaded is noticed, its cause journaled, and restarted once; a second death within ten minutes
parks it until the user starts it again.

**What counts as a death.** The healer remembers each container it saw running (by its start
time), from the sweep (one `docker inspect` over every instance's container, never one per
request) and from each request the router let through. A container it saw running that is now
exited *with the same start time* died: mnm itself never leaves one exited (its stops remove
the container; a failed start's halt is a new container). One that is gone altogether died
only when nothing on the box started or stopped anything since the last sweep (`box_events`'
stamp): a stop or a park (the router's own, `mnm stop`, `--park`) moves it. While a start or a
stop is in progress anywhere the sweep judges nothing and remembers nothing new. Never judged:
what the user stopped (`off`), the tuner's and the self-test's copies (`tune-…`,
`selftest-…`; they are no instances), an instance mid-start.

**What happens.** The log is saved first (`last_logs`, what `mnm logs` falls back to), then
read (`diagnose`) for the cause beside Docker's exit code and OOMKilled (`engine_exits`); one
journal line; the router answers its requests 503 with the cause (`DeadBook`) while one
restart runs through the load path every request uses (the room rules, `InstanceService.start`'s
attempt loop and auto-fix); its outcome is one line, and a failed restart one notice. A death
of the same instance within `LOOP_SECONDS` of its last is a crash loop: no restart, the dead
container cleared (its log kept, its recipe untouched), one line and one notice, and the router
answers 503 for it until the user starts it (`mnm start`), which the next sweep sees."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from makenomistakes import box_events
from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.dead_engines import Dead, DeadBook
from makenomistakes.diagnose import diagnose
from makenomistakes.engine_exits import death_cause
from makenomistakes.healing_words import DIED, NOT_RESTARTED, RESTARTED, looped_line
from makenomistakes.instance_types import InstanceError
from makenomistakes.journal import Note
from makenomistakes.last_logs import keep_log, kept_log, log_path
from makenomistakes.notify import Notice
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import SELFTEST_PREFIX, TUNE_PREFIX
from makenomistakes.router_metrics import RouterCounts
from makenomistakes.supervisor import (
    ContainerState,
    ContainerStatus,
    SupervisorError,
)

LOOP_SECONDS = 600.0  # a second death within ten minutes is a crash loop
COPIES = (TUNE_PREFIX, SELFTEST_PREFIX)
_UNSEEN = object()

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService


class Host(Protocol):
    """What the healer asks the router (`router.Router`)."""

    dead: DeadBook
    counts: RouterCounts

    def note(self, line: str) -> None: ...

    def restart(self, recipe: Recipe) -> str | None: ...  # None once ready, else why not

    def remove_dead(self, recipe: Recipe) -> None: ...  # the dead container cleared


@dataclass(frozen=True)
class Seams:
    """The healer's outside world; `seams_for` is the box's."""

    recipes: Callable[[], list[Recipe]]
    inspect: Callable[[list[str]], dict[str, ContainerStatus]]
    save_log: Callable[[str], str | None]  # the dead container's log, kept; its text
    log_file: Callable[[str], Path]  # where the kept log is (the notice names it)
    notify: Callable[[Notice], None]
    stamp: Callable[[], int | None]  # box_events' stamp: something started or stopped
    changing: Callable[[], bool]  # a start or a stop in progress anywhere


def seams_for(service: InstanceService, notify: Callable[[Notice], None]) -> Seams:
    """The box's: the instances' recipes and Docker, logs under the state folder."""
    settings, supervisor = service.settings, service.supervisor
    state = settings.expanded_state_dir()

    def save_log(name: str) -> str | None:
        keep_log(supervisor, state, name)
        return kept_log(state, name, 0)

    return Seams(
        recipes=service.all,
        inspect=supervisor.statuses,
        save_log=save_log,
        log_file=lambda name: log_path(state, name),
        notify=notify,
        stamp=lambda: box_events.stamp(settings),
        changing=lambda: box_events.others_changing(settings),
    )


class Healer:
    """Thread-safe: the sweep thread and request threads call it; restarts run on threads of
    their own (`spawn`, inline in tests)."""

    def __init__(
        self,
        router: Host,
        seams: Seams,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._router = router
        self._seams = seams
        self._clock = clock
        self._spawn = spawn or _thread
        self._lock = threading.Lock()
        self._loaded: dict[str, str | None] = {}  # name -> the start time it was seen running
        self._handling: set[str] = set()  # a death being handled (its restart included)
        self._deaths: dict[str, float] = {}  # name -> when it last died (the loop rule)
        self._last_stamp: int | None = None
        self._docker_said: str | None = None  # Docker could not be asked: said once

    @property
    def note(self) -> Note:
        return self._router.note

    def loaded(self) -> list[str]:
        """What the router counts as loaded now (seen running, not dead)."""
        with self._lock:
            return sorted(self._loaded)

    def seen(self, name: str, started_at: str | None, restarting: bool = False) -> None:
        """A request was let through to this container (`Router.ready`), or a load finished;
        `restarting`: the container this healer's own restart started, recorded although its
        death is still being handled, so a second death before the next sweep is judged."""
        if name.startswith(COPIES):
            return
        with self._lock:
            if restarting or name not in self._handling:
                self._loaded[name] = started_at

    # --- the sweep ---------------------------------------------------------------------

    def sweep(self) -> list[str]:
        """Look at every container once; handle each death; clear a crash loop's park once
        the user started the instance again or stopped it. The names that died."""
        try:
            recipes = {r.name: r for r in self._seams.recipes() if not r.name.startswith(COPIES)}
        except InstanceError:
            return []  # the recipes cannot be read: the idle sweep's line says so
        looked = self._look(sorted(recipes))
        if looked is None:
            return []
        stamp, states = looked
        moved = stamp != self._last_stamp
        self._last_stamp = stamp
        died = []
        with self._lock:
            for name, state in states.items():
                seen = self._loaded.pop(name, _UNSEEN)
                if state.state is ContainerState.RUNNING:
                    self._loaded[name] = state.started_at
                elif _died(recipes[name], state, seen, moved) and name not in self._handling:
                    died.append(name)
        for name in died:
            self._died(recipes[name], states[name])
        for name in self._router.dead.looped():
            recipe, now = recipes.get(name), states.get(name)
            running = now is not None and now.state is ContainerState.RUNNING
            if recipe is None or recipe.off or running:
                self._router.dead.clear(name)  # started again by the user, or stopped for good
        return died

    def found_dead(self, recipe: Recipe) -> str | None:
        """A request found an instance not running that the healer saw running: judged now
        (one `docker inspect`), so a death is never restarted silently by the request's own
        load. The 503's words when it died, else None (the request loads it as usual)."""
        name = recipe.name
        with self._lock:
            seen = self._loaded.get(name, _UNSEEN)
        if seen is _UNSEEN:
            return None
        looked = self._look([name])
        if looked is None:
            return None
        stamp, states = looked
        state = states[name]
        if not _died(recipe, state, seen, stamp != self._last_stamp):
            return None
        with self._lock:
            if name in self._handling or self._loaded.get(name, _UNSEEN) is _UNSEEN:
                return self._router.dead.refusal(name)  # the sweep took it meanwhile
            self._loaded.pop(name, None)
        self._died(recipe, state)
        return self._router.dead.refusal(name)

    def _look(self, names: list[str]) -> tuple[int | None, dict[str, ContainerStatus]] | None:
        """The containers, with the stamp they were read under; None while anything on the
        box starts or stops (judged at the next sweep) or when Docker cannot be asked."""
        before = self._seams.stamp()
        if self._seams.changing():
            return None
        try:
            states = self._seams.inspect(names)
        except SupervisorError as exc:
            if self._docker_said is None:
                self.note(
                    f"The healer cannot ask Docker about the engines ({exc}); it tries again."
                )
            self._docker_said = str(exc)
            return None
        self._docker_said = None
        if self._seams.changing() or self._seams.stamp() != before:
            return None
        return before, states

    # --- a death -----------------------------------------------------------------------

    def _died(self, recipe: Recipe, state: ContainerStatus) -> None:
        name = recipe.name
        with self._lock:
            if name in self._handling:
                return
            self._handling.add(name)
            now = self._clock()
            last = self._deaths.get(name)
            looped = last is not None and now - last < LOOP_SECONDS
            self._deaths[name] = now
        try:
            text = self._seams.save_log(name) if state.state is ContainerState.EXITED else None
        except OSError:
            text = None
        cause = death_cause(
            state, diagnose(text, recipe.engine, recipe_backend_or_none(recipe)) if text else []
        )
        self._router.counts.died()
        if looped:
            self._loop(recipe, cause)
            return
        self._router.dead.mark(name, Dead(cause))
        self.note(DIED.format(name=name, cause=cause))
        self._spawn(lambda: self._restart(recipe, cause))

    def _restart(self, recipe: Recipe, cause: str) -> None:
        name = recipe.name
        try:
            why = self._router.restart(recipe)
        except Exception as exc:  # the restart thread never dies silently
            why = f"the restart failed: {exc}"
        finally:
            self._router.dead.clear(name)
            with self._lock:
                self._handling.discard(name)
        if why is None:
            self._router.counts.restarted()
            self.note(RESTARTED.format(name=name, cause=cause))
            return
        said = " ".join(why.split())
        self.note(NOT_RESTARTED.format(name=name, why=said))
        log = self._seams.log_file(name)
        self._seams.notify(
            Notice(
                f"mnm: {name} died and did not restart",
                f"{name} died: {cause}. The restart: {said} Its last log: {log} "
                f"(`mnm logs {name} --review`).",
                failed=True,
            )
        )

    def _loop(self, recipe: Recipe, cause: str) -> None:
        name = recipe.name
        self._router.dead.mark(name, Dead(cause, looped=True))
        line = looped_line(name, cause)
        self.note(line)
        try:
            self._router.remove_dead(recipe)
        except InstanceError as exc:
            self.note(f"{name}'s dead container could not be cleared: {exc}")
        finally:
            with self._lock:
                self._handling.discard(name)
                self._deaths.pop(name, None)  # after the user's start, a fresh count
        log = self._seams.log_file(name)
        self._seams.notify(Notice(f"mnm: {name} parked", f"{line} Its last log: {log}.", True))


def _died(recipe: Recipe, state: ContainerStatus, seen: object, moved: bool) -> bool:
    """Whether a container the healer saw running (`seen`: its start time) died: exited from
    that very start, or gone while nothing on the box started or stopped (`moved`)."""
    if seen is _UNSEEN or recipe.off or state.state is ContainerState.RUNNING:
        return False
    if state.state is ContainerState.ABSENT:
        return not moved
    return state.started_at == seen


def _thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True).start()
