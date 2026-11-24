"""Making room on the GPU, one start at a time: what the router and a swap workflow's
members share.

A start takes the box's lock (`start_turn`: one start at a time, so two never count the same
free VRAM), parks the running instances in its scope that the fit-now verdict names
(`with_room`, the fewest largest first, as `--stop-others` does), and starts the instance. A
park waits until the instance answers no request through the router, at most
`PARK_WAIT_SECONDS`, so a reply being written is not cut. The scope is a member's group for
a swap workflow; the router passes its own, every loaded instance it may unload. What the
scope leaves out is `mnm start`'s usual refusal to judge.

A parked instance is one with no container that the user did not stop: the router loads it
at the next request that asks for work. `mnm stop` makes it `off`: off until started again.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field

from makenomistakes import box_events
from makenomistakes.capacity import fit_now, with_room
from makenomistakes.card_room import room_view
from makenomistakes.comfy.engine import ComfyEngine, release_for
from makenomistakes.comfy.image import ComfyBusyError, ComfyError, Variant
from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError, Started
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Note, to_stderr
from makenomistakes.locking import shared_lock
from makenomistakes.progress_words import parking_for_room
from makenomistakes.ram_room import runs_on_processor
from makenomistakes.recipes.model import Recipe
from makenomistakes.router_client import Busy, router_busy
from makenomistakes.supervisor import Health

Progress = Callable[[str], None]
PARK_WAIT_SECONDS = 300.0  # a long generation may finish first; then the park goes ahead


@dataclass
class SwapStarted:
    started: Started
    parked: list[str] = field(default_factory=list)


class ParkError(InstanceError):
    """Making room failed: an instance in the way could not be stopped. Nothing is wrong with
    the instance being started."""


class HeldByComfyError(ParkError):
    """ComfyUI is mid-workflow: the start waits for it rather than load beside it (and have
    auto-fix shrink the recipe for good). Nothing is wrong with the instance."""


@contextmanager
def start_turn(settings: Settings, name: str, progress: Progress | None = None) -> Iterator[None]:
    """The box's one start lock, across processes (the router, the CLI, the web
    UI) and threads: one start at a time, so two never count the same free
    VRAM, and the router never parks a start in progress. `progress` hears when it waits."""

    def waiting() -> None:
        if progress:
            progress(f"Waiting for another start to finish before {name}…")

    # The marker: a reading taken meanwhile is not another instance's own.
    with (
        shared_lock(settings.expanded_state_dir() / "swap")(waiting),
        box_events.changing(settings),
    ):
        yield


@dataclass(frozen=True)
class Scope:
    """What a start may park to make room, and what it does when the instance has no VRAM
    estimate to price it with: a swap group makes way whole; the router's parks nothing
    then, and the start's own refusal and auto-fix judge."""

    members: Callable[[Recipe], list[Recipe]]
    makes_way_unpriced: bool


class SwapService:
    def __init__(  # noqa: PLR0913 - the router's two hooks besides the seams
        self,
        instances: InstanceService,
        busy: Busy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        scope: Scope | None = None,
        release: Callable[[Progress | None], bool] | None = None,
        *,
        told: Note = to_stderr,
        unloading: Callable[[str], AbstractContextManager[object]] = lambda _name: nullcontext(),
    ) -> None:
        self.instances = instances
        self._told = told  # a park's line (its saved prompt): the router's journal
        self._unloading = unloading  # the router's: a request for it waits the park out
        # ComfyUI, when installed and running, gives the card back first
        self._release = release or release_for(instances.settings)
        self._busy = busy or router_busy(instances.settings)
        self._sleep = sleep
        self._clock = clock
        self._scope = scope or Scope(self.group, makes_way_unpriced=True)

    def group(self, recipe: Recipe) -> list[Recipe]:
        """The other members of the instance's swap group."""
        if recipe.swap_group is None:
            return []
        return [
            r
            for r in self.instances.all()
            if r.swap_group == recipe.swap_group and r.name != recipe.name
        ]

    @contextmanager
    def turn(self, recipe: Recipe, progress: Progress | None = None) -> Iterator[None]:
        """The box's start lock (`start_turn`) for a whole start of `recipe`."""
        with start_turn(self.instances.settings, recipe.name, progress):
            yield

    def start(
        self,
        recipe: Recipe,
        progress: Progress | None = None,
        check: Callable[[], None] | None = None,
    ) -> SwapStarted:
        """Make room in the group, then start it: one start at a time in a group. `check`
        runs between the two: `mnm start`'s refusal of what still holds the card (the
        instances outside the group), which raises InstanceError."""
        with self.turn(recipe, progress):
            status = self.instances.status(recipe)
            if status.running and status.health is Health.READY:
                return SwapStarted(Started(recipe))  # started meanwhile (the turn before ours)
            try:
                if frees_comfy_for([recipe], lambda: comfy_on_processor(self.instances.settings)):
                    self._release(progress)
            except ComfyBusyError as exc:
                raise HeldByComfyError(f"{recipe.name} is not loaded: {exc}") from exc
            parked = self.make_room(recipe, progress)
            if check is not None:
                check()
            return SwapStarted(self.instances.start(recipe, progress), parked)

    def make_room(self, recipe: Recipe, progress: Progress | None = None) -> list[str]:
        """Park the instances `to_park` names, each after its replies in flight (`park`)."""
        return self.park_for(recipe, self.to_park(recipe), progress)

    def park_for(
        self, recipe: Recipe, chosen: list[Recipe], progress: Progress | None = None
    ) -> list[str]:
        """Park `chosen` to make room for `recipe`, one after another; their names."""
        for other in chosen:
            if progress:
                progress(parking_for_room(other.name, recipe.name) + "…")
            try:
                self.park(other)
            except InstanceError as exc:
                raise ParkError(f"{other.name} could not be parked to make room: {exc}") from exc
        return [r.name for r in chosen]

    def to_park(self, recipe: Recipe, pinned_too: bool = False) -> list[Recipe]:
        """The running instances in the scope that the fit-now verdict names for `recipe`
        (never a pinned one); with no estimate to go by, a swap group's every running member,
        and nothing from a caller's scope (the start's own refusal and auto-fix judge then).
        `pinned_too`: a pinned one may be named too, for a refusal that says what holds the
        room (`selftest_room`), never for parking."""
        members = [r for r in self._scope.members(recipe) if pinned_too or not r.pinned]
        statuses = {r.name: self.instances.status(r) for r in members}
        running = {n: s.recipe for n, s in statuses.items() if s.running}
        if not running:
            return []
        snapshot = self.instances.memory()
        estimate = self.instances.vram_estimate(recipe)
        holders = {n: self.instances.held_estimate(statuses[n]) for n in running}
        held = {n: e.bytes or 0 for n, e in holders.items()}
        seen = room_view(estimate, snapshot, holders)  # its cards
        verdict = with_room(
            fit_now(estimate, snapshot, False),
            seen.estimate,
            seen.snapshot,
            seen.held,
            held_on=seen.held_on,
        )
        if not verdict.warns and estimate.bytes is not None:
            return []
        if verdict.stop_first:
            names = list(verdict.stop_first)
        elif estimate.bytes is None and self._scope.makes_way_unpriced:  # a group makes way
            names = sorted(running, key=lambda n: -held[n])
        else:
            return []  # the scope cannot make room (others hold the card): the start judges
        return [running[n] for n in names]

    def park(self, recipe: Recipe) -> None:
        """Wait for its requests to finish (a while), then stop the engine: loadable again."""
        # Both bounds: the naps (a stand-in sleep can be instant) and the clock (a router slow to
        # answer must not stretch the wait past PARK_WAIT_SECONDS).
        deadline = self._clock() + PARK_WAIT_SECONDS
        naps = 0
        while naps < PARK_WAIT_SECONDS and self._clock() < deadline and self._busy(recipe) > 0:
            self._sleep(1.0)
            naps += 1
        with self._unloading(recipe.name):
            self.instances.stop(recipe, park=True, told=self._told)


def frees_comfy_for(recipes: list[Recipe], comfy_in_ram: Callable[[], bool]) -> bool:
    """Whether ComfyUI is asked for its room before `recipes` load: one of them sits on a
    card, or ComfyUI runs on the processor too (`comfy_in_ram`: a CPU build's models sit in
    the RAM). A load in the RAM beside ComfyUI on a card needs nothing of it (else
    a song on the processor's image is refused while a workflow runs)."""
    return any(not runs_on_processor(r) for r in recipes) or comfy_in_ram()


def comfy_on_processor(settings: Settings) -> bool:
    """ComfyUI is installed as its CPU build (`Variant.CPU`); False when it is not installed
    or its ledger cannot be read (the doctor says why)."""
    try:
        ledger = ComfyEngine(settings).ledger()
    except ComfyError:
        return False
    return ledger is not None and ledger.variant is Variant.CPU
