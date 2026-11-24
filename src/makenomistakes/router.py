"""The router: one port for every model.

A request names its instance (`routing.route`). A loaded instance that answers ready gets
the request at once; one still loading is waited for. A parked one (no container, not `off`,
its files on disk) is loaded first for a request that asks for work: room is made by the
same rules a swap workflow uses, over every loaded instance the router may unload, under the
box's one start lock. A stopped one (`off`: the user stopped it) is answered 409 and never
loaded, and so is one whose files are not downloaded (a request never starts a download,
but for an instance the disk policy evicted: its load downloads its files again). A
load that fails marks the instance stopped, so later requests do not retry it one by one.
Requests that only look around (a model list, a GET) never load anything, and neither do
mnm's own probes (`X-Mnm-Probe`).

What the router may unload: a loaded instance with no request in flight that is not a member
of a plain workflow in use (its members run together on purpose), and is not pinned
(`keep=always`). One idle past its `idle` limit is unloaded too (`unload_idle`).
Engines publish on 127.0.0.1 only, so every request reaches them through here, and the
router alone knows what is in flight (`/__mnm/busy` tells a start in another process). It
also rewrites what some engines refuse or obey over the instance: an image
request's `size`, a voice request's `speed`, a repaint's references. The HTTP side is
`router_http`.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from email.message import Message
from typing import TYPE_CHECKING, Protocol

from makenomistakes.aliases import Aliases
from makenomistakes.dead_engines import DeadBook
from makenomistakes.desk import read_aliases, shadowed_note
from makenomistakes.disk_settings import redownload_words
from makenomistakes.engine_args import AudioCppArgs, SdCppArgs
from makenomistakes.forward import (
    Rewrites,
    rewrite,
    with_model,
)
from makenomistakes.held_back import held_back_in
from makenomistakes.idle import IdleClock
from makenomistakes.instance_types import (
    DockerError,
    InstanceError,
    InstanceStatus,
    UnreadableStateError,
)
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Note, to_stderr
from makenomistakes.measurements import Reading, start_reading
from makenomistakes.model_meta import MetaCache
from makenomistakes.parkable import parkable
from makenomistakes.plan_budget import held_by_mnm
from makenomistakes.readings_filter import instance_readings
from makenomistakes.recipes import Recipe
from makenomistakes.request_meter import RequestMeter
from makenomistakes.router_client import Traffic
from makenomistakes.router_events import EventLog
from makenomistakes.router_metrics import RouterCounts
from makenomistakes.router_models import chat_like, entry, entry_of, listed, listing, shown
from makenomistakes.router_park import NOT_LOADED_NOW, Kept, Parking
from makenomistakes.router_queue import QueueBook, Ticket
from makenomistakes.router_refusals import (
    FAILED,
    NO_DOCKER,
    NOT_ANSWERING,
    Refusal,
    not_loadable,
    what_to_do,
)
from makenomistakes.routing import named
from makenomistakes.supervisor import Health, SupervisorError
from makenomistakes.swap import ParkError, Scope, SwapService
from makenomistakes.thinking_notes import ThinkingNotes
from makenomistakes.workflows import WorkflowStore

if TYPE_CHECKING:
    from makenomistakes.healing import Healer

__all__ = [  # `router_models`' and `router_park`'s names, found here too
    "NOT_LOADED_NOW",
    "Kept",
    "chat_like",
    "entry",
    "listed",
]

NO_ROOM = (
    "{name} does not fit beside what is loaded now, and the router may not unload those: {note}"
)
UNREADABLE = "{name} is not loaded: mnm cannot read its own files ({error}). Nothing is marked."
# A model list's statuses are reused this long: Odysseus asks
# `/v1/models` before every message, and each status is a `docker inspect`. A load or an
# unload by the router forgets them at once.
LISTING_SECONDS = 2.0


class NoRoomError(InstanceError):
    """The room a load needs is held by what the router may not unload: nothing is wrong
    with the instance, so it is not marked stopped."""


class Router(Parking):
    """What the HTTP side asks: which instances exist, and whether one may take a request
    now (loading it first when it is parked). Thread-safe: one per process."""

    def __init__(
        self,
        instances: InstanceService,
        workflows: WorkflowStore | None = None,
        swap: SwapService | None = None,
        sleep: Callable[[float], None] = time.sleep,  # a park's wait for requests in flight
        clock: Callable[[], float] = time.monotonic,  # the idle timer's
        note: Note = to_stderr,  # the journal: each load, and the parks it made
    ) -> None:
        self.instances = instances
        self.events = EventLog(out=note)  # its newest journal lines, for the web
        self.note = self.events.note
        self._sets_unreadable = False  # the sweep said so once (`unload_idle`)
        self._alias_problem: str | None = None  # why no alias is served, said once
        self._answering: dict[str, str] = {}  # alias -> the instance last said to answer it
        self._clock = clock
        self._workflows = workflows or WorkflowStore(instances.settings.expanded_state_dir())
        self._busy: dict[str, int] = {}
        self.arrivals = 0  # requests taken: a measured one counts as alone only if none came
        self._unloading: set[str] = set()  # idle unloads under way: a request waits them out
        self._count = threading.Lock()
        self.idle = IdleClock(clock)
        self.meta = MetaCache(instances.settings.expanded_models_dir)  # what files say
        self.meter: RequestMeter | None = None  # `mnm router` sets one; tests run without
        self.counts = RouterCounts()  # loads, parks and refusals for `/metrics`
        self.queue = QueueBook(self.container, self._start_reading)  # who waits
        self.tuner: TuneHook | None = None  # `mnm router` sets the tuner's loop
        self.trainer: TrainHook | None = None  # and the trainer's
        self.card_jobs: CardJobs | None = None  # their gate: every request cuts their job
        self.dead = DeadBook()  # engines that died while loaded: 503 with the cause
        self.healer: Healer | None = None  # `mnm router` sets the one that restarts them
        self.thinking = ThinkingNotes()  # answers all thinking, told once a start
        # name -> (the build its container runs, when it started), from the last status a
        # request was let through on: what a reading names, with no Docker call of its own
        self._containers: dict[str, tuple[str | None, str | None]] = {}
        self._listed: tuple[float, list[InstanceStatus]] | None = None  # `listed_statuses`
        scope = Scope(self.scope, makes_way_unpriced=False)
        self._swap = swap or SwapService(
            instances,
            busy=lambda r: self.busy(r.name),
            sleep=sleep,
            scope=scope,
            told=self.note,  # a park's line (its saved prompt) in this journal
            unloading=self.unloading,  # a request for it waits the park out
        )

    # --- requests in flight ---------------------------------------------------------

    @contextlib.contextmanager
    def request(self, recipe: Recipe, work: bool) -> Iterator[Ticket]:
        """One request for `recipe`, from before it is counted in flight (the card taken back
        from a tune or a training is a wait too) to after its reply: in the queue when it is
        work, in flight always. The count goes down whatever the queue's bookkeeping
        does (a count left up would keep the model loaded, and the card's jobs off it, for
        good)."""
        ticket = self.queue.take(recipe, work)
        counted = False
        try:
            self.serving(recipe.name, +1)  # the card's job is cut first: may wait a minute
            counted = True
            yield ticket
        finally:
            try:
                self.queue.ended(ticket)
            except Exception as exc:  # never the request's, never the count's
                self.note(f"The router's queue lost count of a request for {recipe.name}: {exc!r}")
            finally:
                if counted:
                    self.serving(recipe.name, -1)

    def _start_reading(self, name: str, started: str) -> Reading | None:
        """The load reading of one container start of `name`, this box's: the settings that
        container started with (`measurements.start_reading`)."""
        return start_reading(instance_readings(self.instances.measurements, name), started)

    def serving(self, name: str, delta: int) -> None:
        if delta > 0 and self.card_jobs is not None:  # a trial or a training gives the card
            self.card_jobs.arrived(name)  # back first
        with self._count:
            self.arrivals += max(delta, 0)
            self._busy[name] = self._busy.get(name, 0) + delta
            if delta < 0 and self._busy[name] == 0:
                self.idle.finished(name)

    def busy(self, name: str) -> int:
        with self._count:
            return self._busy.get(name, 0)

    def traffic(self) -> Traffic:
        """What a reading is judged by in this process (`measure_window`)."""
        with self._count:
            return Traffic({n: c for n, c in self._busy.items() if c > 0}, self.arrivals)

    def container(self, name: str) -> tuple[str | None, str | None]:
        """The build and start of the container a request just went to: the status `ready`
        let it through on, or (after a load) asked once."""
        with self._count:
            known = self._containers.get(name)
        if known is not None:
            return known
        recipe = self.instances.find(name)
        if recipe is None:
            return None, None
        status = self.instances.status(recipe)
        found = (status.build, status.started_at) if status.running else (None, None)
        with self._count:
            self._containers[name] = found
        return found

    def in_flight(self) -> dict[str, int]:
        """`GET /__mnm/busy`: what a park outside this process waits for (`swap.router_busy`)."""
        with self._count:
            return {name: n for name, n in self._busy.items() if n > 0}

    # --- what may be unloaded -------------------------------------------------------

    def scope(self, recipe: Recipe) -> list[Recipe]:
        """The instances a load of `recipe` may park: every other one with nothing in flight
        that the router never parks (`workflows.held_back`: pinned, or a member of a plain
        workflow in use). When the workflows cannot be read, nothing: the start's refusal
        then judges."""
        return parkable(self.instances, self._workflows, recipe.name, self.busy) or []

    # --- the questions a request asks -----------------------------------------------

    def models(self, chat_only: bool = False) -> dict[str, object]:
        """`GET /v1/models`: every loaded or parked instance by name (stopped and
        undownloaded ones are left out; `/chat/v1/models`: the chat-like ones); nothing is
        loaded to answer. InstanceError when Docker cannot be asked: a shorter list would
        read as the truth."""
        statuses = [s for s in self.listed_statuses() if not chat_only or chat_like(s)]
        broken = next((s.error for s in statuses if s.error), None)
        if broken is not None:
            raise InstanceError(NO_DOCKER.format(error=broken))
        return {"object": "list", "data": listing(statuses, self.meta, self.aliases())}

    def listed_statuses(self) -> list[InstanceStatus]:
        """Every instance's status for a model list: a container's state is read at most
        once per `LISTING_SECONDS`, its saved recipe every time (a `mnm set` applies at
        once); never kept when Docker could not say."""
        with self._count:
            kept = self._listed
        if kept is None or time.monotonic() - kept[0] >= LISTING_SECONDS:
            statuses = self.instances.statuses()
            if not any(s.error for s in statuses):
                with self._count:
                    self._listed = (time.monotonic(), statuses)
            return statuses
        known = {s.recipe.name: s for s in kept[1]}
        return [
            found
            if (found := known.get(r.name)) is not None and found.recipe == r
            else self.instances.status(r)
            for r in self.instances.all()
        ]

    def _forget_listed(self) -> None:
        with self._count:
            self._listed = None

    def model(self, name: str, chat_only: bool = False) -> dict[str, object] | None:
        """One entry (`GET /v1/models/<id>`) by an id `/v1/models` lists: an instance's name,
        or a desk role; None when it is not listed (a stopped instance is not: a request for
        it says why)."""
        found = named(name, self.instances.all(), self.aliases())
        status = self.instances.status(found.recipe) if found is not None else None
        if found is None or status is None or not shown(status):
            return None
        if chat_only and not chat_like(status):
            return None
        listing = entry_of(status, self.meta.get(status.recipe))
        return (listing.as_role(found.alias) if found.alias else listing).dumped()

    def aliases(self) -> Aliases:
        """The desk's roles as model ids now (`desk.read_aliases`); read per request, so a
        `mnm sets edit` or `sets default` applies at once. Why there are none, when a desk is
        chosen and cannot be read (or a role an instance's name shadows), is one journal line
        until it changes."""
        found = read_aliases(self._workflows, (r.name for r in self.instances.all()))
        problem = f"No desk aliases are served: {found.problem}" if found.problem else None
        problem = problem or shadowed_note(found)
        with self._count:
            said, self._alias_problem = self._alias_problem, problem
        if problem and problem != said:
            self.note(problem)
        return found

    def answered_as(self, alias: str, instance: str) -> None:
        """A request named a desk role: the journal says which instance answers it, once per
        change (a line per request would drown the journal)."""
        with self._count:
            before = self._answering.get(alias)
            self._answering[alias] = instance
        if before != instance:
            self.note(f"{alias} is {instance} (the desk's {alias}).")

    def loadable(self, recipe: Recipe) -> bool:
        return listed(self.instances.status(recipe))

    def ready(  # noqa: PLR0911 - one answer per state the instance is in
        self, recipe: Recipe, method: str, probe: bool, waited: bool = False
    ) -> Refusal | None:
        """None when the instance may take the request now (waited for while it loads,
        loaded first when it is parked and the request asks for work); else why not."""
        with self._count:
            unloading = recipe.name in self._unloading
        if unloading and not waited:  # the sweep holds the box's lock while it unloads
            return self._after_the_start_in_progress(recipe, method, probe)
        if (dead := self.dead.refusal(recipe.name)) is not None:  # Restarting, or looped
            return Refusal(503, dead)
        status = self.instances.status(recipe)
        if status.error:
            return Refusal(503, NO_DOCKER.format(error=status.error))
        if status.running and status.health is Health.READY:
            with self._count:
                self._containers[recipe.name] = (status.build, status.started_at)
            if self.healer is not None:
                self.healer.seen(recipe.name, status.started_at)
            return None
        with self._count:  # loading or loaded anew: `container` asks once it answers
            self._containers.pop(recipe.name, None)
        if status.running and probe:  # mnm's own look never waits out a start
            return Refusal(503, NOT_ANSWERING.format(name=recipe.name))
        if status.running and waited:
            return self._wait_ready(recipe)
        if status.running:
            return self._after_the_start_in_progress(recipe, method, probe)
        if self.healer is not None and (died := self.healer.found_dead(recipe)) is not None:
            return Refusal(503, died)  # never restarted silently by this request's load
        return not_loadable(status, method, probe) or self._load(recipe)

    def _after_the_start_in_progress(
        self, recipe: Recipe, method: str, probe: bool
    ) -> Refusal | None:
        """A container up but not ready: a start holds the box's lock (a load for another
        request, `mnm start`, a workflow). Wait for that start to end, without holding the
        lock while the engine finishes, then decide again: it may be ready, still loading,
        parked, stopped or dead by now."""
        with self._swap.turn(recipe):
            pass
        return self.ready(recipe, method, probe, waited=True)

    def _wait_ready(self, recipe: Recipe) -> Refusal | None:
        """Still loading after every start ended (a start that gave up waiting, or one made
        outside mnm): its health, outside the lock. A container that dies meanwhile is a 503
        that names its log, never the router's own fault."""
        try:
            health = self.instances.supervisor.wait_ready(recipe)
        except SupervisorError as exc:
            return Refusal(503, f"{recipe.name} did not come up: {exc}")
        if health is Health.READY:
            return None
        return Refusal(503, NOT_ANSWERING.format(name=recipe.name))

    def restart(self, recipe: Recipe) -> str | None:
        """A dead engine loaded again, as a request would load it (the healer's): None
        once it is ready, else why not."""
        refused = self._load(recipe, "after it died", restarting=True)
        return refused.message if refused is not None else None

    def remove_dead(self, recipe: Recipe) -> None:
        """A dead container cleared under the box's lock, its log kept (a crash loop)."""
        with self._swap.turn(recipe):
            self.instances.stop(recipe, park=True)
        self.idle.forget(recipe.name)
        self._forget_listed()

    def _load(
        self, recipe: Recipe, why: str = "for a request", restarting: bool = False
    ) -> Refusal | None:
        """A parked instance, loaded under the box's lock after the room its scope allows.
        Only a failure of its own start marks it stopped: not a room that could not be made,
        not Docker failing (a pull, the daemon), and not a start that gave up waiting while
        the engine still loads."""
        self.note(f"Loading {recipe.name} {why}.")
        self._forget_listed()
        if (fetching := redownload_words(recipe)) is not None:  # Evicted by the policy
            self.note(fetching[0].upper() + fetching[1:] + ".")
        began = self._clock()
        try:
            done = self._swap.start(recipe, check=lambda: self._refuse_if_held(recipe))
        except UnreadableStateError as exc:  # a file of mnm's, not this start: nothing marked
            return Refusal(503, UNREADABLE.format(name=recipe.name, error=exc))
        except (NoRoomError, ParkError, DockerError) as exc:
            return Refusal(503, str(exc))  # nothing known to be wrong with the instance
        except InstanceError as exc:
            after = self.instances.status(recipe)
            if after.running or after.error:  # still loading, or Docker cannot say: not its
                return Refusal(503, str(exc))  # own failure, and the next request may wait
            with contextlib.suppress(InstanceError):  # removed meanwhile: nothing to mark
                self.instances.mark_off(recipe, True)
            self._forget_listed()
            return Refusal(503, FAILED.format(error=str(exc).rstrip(), name=recipe.name))
        took = self._clock() - began
        self._forget_listed()
        if done.parked:
            self.note(f"Parked {', '.join(done.parked)}: room for {recipe.name}.")
        self.note(f"{recipe.name} ready in {took:.1f} s.")
        for line in done.started.notes:  # its saved prompt put back, its cards
            self.note(line)
        self.counts.loaded(parked=len(done.parked))
        if self.healer is not None and (now := self.instances.status(recipe)).running:
            # killed before the next sweep: dead (its own restart's container too)
            self.healer.seen(recipe.name, now.started_at, restarting=restarting)
        return None

    def _refuse_if_held(self, recipe: Recipe) -> None:
        """After the room the scope allowed: what mnm still holds (busy instances, a plain
        workflow's members) refuses the load rather than run it out of memory, which would
        have auto-fix save smaller settings for good."""
        statuses = self.instances.statuses()
        pinned = [s.recipe.name for s in statuses if s.recipe.pinned and s.running]
        kept = held_back_in(self._workflows, statuses)
        kept = kept if kept is not None else frozenset(pinned)  # the workflows unreadable
        verdict = held_by_mnm(self.instances, recipe.name, kept - {recipe.name})
        if verdict is not None:
            raise NoRoomError(
                NO_ROOM.format(name=recipe.name, note=verdict.note)
                + what_to_do(list(verdict.stop_first), list(verdict.kept), pinned)
            )


class CardJobs(Protocol):
    """The card jobs' gate (`card_jobs.CardGate`): a request cuts the job running."""

    def arrived(self, name: str) -> None: ...  # a request is about to be served


class TuneHook(Protocol):
    """What the router asks the tuner's loop (`tune.runner.TuneRunner`)."""

    def line(self) -> str | None: ...  # "tuning tiny (trial 7 of ≤24, …)" while one runs

    def ask(self, name: str, body: bytes) -> tuple[int, str]: ...  # `POST /__mnm/tune/<name>`

    def hold(self, body: bytes) -> tuple[int, str]: ...  # `POST /__mnm/tune-hold`


class TrainHook(Protocol):
    """What the router asks the trainer's loop (`studio.train_runner.TrainRunner`)."""

    def line(self) -> str | None: ...  # "training alto (epoch 12 of 200)" while one runs

    def ask(self, run_id: str, body: bytes) -> tuple[int, str]: ...  # `POST /__mnm/train/<run>`


def rewrites(recipe: Recipe, path: str, body: bytes, headers: Message) -> bytes:
    """What the engine is sent: a JSON `model` naming the instance names the engine's own id,
    an sd.cpp instance's image requests lose `size` (unless `client-size=on`), a repaint on
    an image instance whose text encoder cannot see keeps its picture from it, a voice that
    refuses `speed` loses it; anything else as it came."""
    args = recipe.args
    rules = Rewrites(
        keep_size=not isinstance(args, SdCppArgs) or args.keeps_size,
        drop_speed=isinstance(args, AudioCppArgs) and args.drops_speed,
        drop_references=recipe.drops_references,
    )
    body = with_model(body, recipe.name, recipe.served_name)
    return rewrite(path, body, rules, headers.get("Content-Type", ""))
