"""Instance lifecycle shared by the CLI and the web UI: add, download, start, stop, remove.

Every operation takes and returns typed objects and raises `InstanceError` with a message
meant for the user. Presentation (tables, pages) stays in the callers. The store is the
single writer of recipes; background downloads merge their progress through it.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from pathlib import Path

from makenomistakes import box_events, slot_files
from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.calibration import Correction
from makenomistakes.capacity import Verdict, VramEstimate
from makenomistakes.comfy.models import ModelError, Models
from makenomistakes.config import Settings
from makenomistakes.converted_models import ConversionError, ensure_converted
from makenomistakes.diagnose import LogReview, diagnose, review_log
from makenomistakes.disk_policy import fetched
from makenomistakes.download import Fetcher, download_recipe_files, wanted_files
from makenomistakes.engines import ledger_path
from makenomistakes.fit import Profile
from makenomistakes.hardware.probe import MemorySnapshot, probe_memory
from makenomistakes.hub import ModelFacts
from makenomistakes.hub.civitai import is_civitai
from makenomistakes.hub.elsewhere import elsewhere
from makenomistakes.hub.facts import Modality
from makenomistakes.instance_names import check_name, default_name
from makenomistakes.instance_start import run_start
from makenomistakes.instance_types import (
    InstanceError,
    InstanceStatus,
    NoLogError,
    Progress,
    Started,
    Stopped,
    UnreadableStateError,
    port_listening,
)
from makenomistakes.instance_vram import InstanceVram, running_vram, verdicts
from makenomistakes.journal import Note, to_stderr
from makenomistakes.last_logs import forget_log, keep_log, kept_log, no_log_words
from makenomistakes.machine import planned_here, stamped
from makenomistakes.measure_window import Card
from makenomistakes.measurements import MeasurementStore, built_for
from makenomistakes.model_cleanup import Cleanup, delete_own_files
from makenomistakes.plan import Plan
from makenomistakes.ram_room import ram_margin, with_ram
from makenomistakes.recipes import Recipe, spec_for
from makenomistakes.room_verdicts import ComfyRoom, comfy_room
from makenomistakes.router_client import router_answers, router_traffic
from makenomistakes.store import InstanceStore, NoSuchInstanceError, StoreError
from makenomistakes.supervisor import (
    ContainerState,
    DockerSupervisor,
    NoContainerError,
    SupervisorError,
)
from makenomistakes.tool_verdicts import ToolVerdicts
from makenomistakes.tune.store import TuneStore

BUILD_SECONDS = 60.0  # how long an engine build's id is trusted before Docker is asked again


class InstanceService:
    def __init__(
        self,
        settings: Settings,
        store: InstanceStore | None = None,
        supervisor: DockerSupervisor | None = None,
        fetch: Fetcher | None = None,  # huggingface_hub's, in the engine row's layout
        memory: Callable[[], MemorySnapshot] = probe_memory,
        port_in_use: Callable[[int], bool] | None = None,  # default: this machine's ports
        router_up: Callable[[Settings], bool] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or InstanceStore(settings.expanded_state_dir())
        self.supervisor = supervisor or DockerSupervisor()
        self.router_up = router_up or router_answers  # clients reach engines through it
        self._fetch = fetch
        self._memory = memory
        self._port_in_use = port_in_use or port_listening
        self.measurements = MeasurementStore(settings.expanded_state_dir())
        self.tools = ToolVerdicts(settings.expanded_state_dir())  # `mnm try --tools`'
        # The card, the router's traffic and the box's events a reading is judged by; the
        # router's own process answers its traffic itself (`mnm router`).
        self.card = Card(
            settings,
            self.used_vram,
            lambda: router_traffic(settings),
            snapshot=lambda: self.memory(),  # noqa: PLW0108 - read late: tests replace it
        )
        # image -> (when, the engine ledger's mtime then, its id)
        self._builds: dict[str, tuple[float, int | None, str | None]] = {}
        self._vram = InstanceVram(settings, self.memory, self.measurements, self.build_of)

    @property
    def models_dir(self) -> Path:
        return self.settings.expanded_models_dir()

    # --- read -----------------------------------------------------------------------

    def get(self, name: str) -> Recipe:
        try:
            return self.store.load(name)
        except NoSuchInstanceError as exc:
            raise InstanceError(str(exc)) from exc
        except StoreError as exc:
            raise UnreadableStateError(str(exc)) from exc

    def find(self, name: str) -> Recipe | None:
        """The recipe, or None when no instance has that name (a workflow member removed out
        from under it); any other store problem is still an error."""
        try:
            return self.store.load(name)
        except NoSuchInstanceError:
            return None
        except StoreError as exc:
            raise UnreadableStateError(str(exc)) from exc

    def all(self) -> list[Recipe]:
        """Every recipe (`statuses` reads through here); UnreadableStateError when a file
        cannot be read."""
        try:
            return self.store.list()
        except StoreError as exc:
            raise UnreadableStateError(str(exc)) from exc

    def status(self, recipe: Recipe) -> InstanceStatus:
        present = self.files_present(recipe)
        try:
            container = self.supervisor.status(recipe.name)
        except SupervisorError as exc:
            return InstanceStatus(
                recipe, ContainerState.ABSENT, error=str(exc), files_present=present
            )
        health = None
        if container.state is ContainerState.RUNNING:
            health = self.supervisor.health(recipe)
        return InstanceStatus(
            recipe,
            container.state,
            container.exit_code,
            health,
            files_present=present,
            started_at=container.started_at if container.state is ContainerState.RUNNING else None,
            build=container.build if container.state is ContainerState.RUNNING else None,
        )

    def files_present(self, recipe: Recipe) -> bool:
        """Every file the recipe wants is recorded and on disk: the one "downloaded" the
        CLI, the web and `show` all mean (a recorded path whose file is gone is not)."""
        return recipe.downloaded and all(
            (self.models_dir / path).exists() for path in recipe.local_paths.values()
        )

    def statuses(self) -> list[InstanceStatus]:
        return [self.status(r) for r in self.all()]

    def memory(self) -> MemorySnapshot:
        """GPU and RAM occupancy right now; the probe can be injected. The RAM is
        read as a place too (`ram_room.with_ram`): an instance on the processor's
        image holds its room there."""
        return with_ram(self._memory(), ram_margin(self.settings))

    def verdicts(
        self,
        statuses: list[InstanceStatus],
        snapshot: MemorySnapshot,
        held_back: frozenset[str] | None = None,
        kept: frozenset[str] | None = None,
        comfy: ComfyRoom | None = None,
    ) -> dict[str, Verdict]:
        """Per instance: running, fits now, near the limit, or won't fit (`instance_vram`);
        `held_back` (what the router never parks) makes a parked one *on request*; `kept`
        is never room, without that reading (the router's refusal); what ComfyUI gives back
        at a start counts free (`comfy`, read now when not given)."""
        room = comfy_room(self.settings) if comfy is None else comfy
        started = {s.recipe.name: s.started_at for s in statuses if s.running}

        def estimate(recipe: Recipe, build: str | None) -> VramEstimate:
            return self.vram_estimate(recipe, build, started.get(recipe.name))

        snapshot = with_ram(snapshot, ram_margin(self.settings))  # a caller's own read too
        return verdicts(statuses, snapshot, estimate, held_back, kept, room)

    def used_vram(self) -> int | None:
        """nvidia-smi's memory in use on the card now; None when it cannot be read."""
        gpu = self.memory().gpu
        return gpu.vram_used_bytes if gpu is not None else None

    def build_of(self, image: str) -> str | None:
        """The engine build a tag names now (its image id), which the next start runs: a
        figure is reused only for the same build. Read at most once a minute, and at
        once after `mnm engines update` or a rollback changed the ledger."""
        now = time.monotonic()
        ledger = _mtime(ledger_path(self.settings.expanded_state_dir()))
        cached = self._builds.get(image)
        if cached is not None and now - cached[0] < BUILD_SECONDS and cached[1] == ledger:
            return cached[2]
        try:
            found = self.supervisor.image(image)
        except SupervisorError:
            found = None
        build = found.short_id if found is not None else None
        self._builds[image] = (now, ledger, build)
        return build

    def engine_correction(self, recipe: Recipe) -> Correction:
        """The measured correction for the recipe's engine and type on this card."""
        return self._vram.correction(recipe.engine, recipe.modality, built_for(recipe))

    def raw_estimate(self, recipe: Recipe) -> int | None:
        """The fit model's own figure for the current settings, before any measurement."""
        return self._vram.raw(recipe).bytes

    def vram_estimate(
        self, recipe: Recipe, running_build: str | None = None, started: str | None = None
    ) -> VramEstimate:
        """What the instance puts on the GPU with its current settings (`instance_vram`);
        a running one (`started`, its container's start) holds what it started with."""
        return self._vram.estimate(recipe, running_build, started)

    def held_estimate(self, status: InstanceStatus) -> VramEstimate:
        """What a running instance holds now (the settings its container started with)."""
        return self.vram_estimate(status.recipe, status.build, status.started_at)

    def native_context(self, recipe: Recipe) -> int | None:
        """The context a downloaded model was trained for."""
        return self._vram.native_context(recipe)

    def running_vram(self, in_ram: bool | None = None) -> dict[str, int]:
        """Running instance → its estimated VRAM (`instance_vram.running_vram`); only the RAM's
        or the cards' with `in_ram`."""
        return running_vram(self, in_ram)

    def running_names(self) -> list[str]:
        """Instances whose container is up (no health probe: `fit` only needs the names).

        A Docker error counts as "not running" here on purpose: the callers only word hints
        (who holds VRAM, what keeps running after an update), never act on the answer."""
        names = []
        for recipe in self.all():
            try:
                if self.supervisor.status(recipe.name).state is ContainerState.RUNNING:
                    names.append(recipe.name)
            except SupervisorError:
                continue  # Docker could not be asked; the plan still stands on its own
        return names

    def logs(self, recipe: Recipe, tail: int = 400) -> tuple[str, LogReview]:
        """The engine log (`tail=0` for all of it) and everything in it worth a look.
        InstanceError when Docker cannot hand it over (never a review that reads as clean);
        NoLogError when there is no container and no kept log."""
        try:
            text = self.supervisor.logs(recipe.name, tail)
        except NoContainerError as exc:  # only a removed container has a kept log to show
            kept = kept_log(self.settings.expanded_state_dir(), recipe.name, tail)
            if kept is None:
                raise NoLogError(no_log_words(recipe.name, recipe.off)) from exc
            text = kept
        except SupervisorError as exc:  # Docker could not be asked: never an old log
            raise InstanceError(str(exc)) from exc
        return text, review_log(text, recipe.engine, recipe_backend_or_none(recipe))

    def review_or_none(self, recipe: Recipe) -> LogReview | None:
        """The log review when the log can be read, else None (a failure message that would
        quote the review must not claim a clean log)."""
        try:
            return self.logs(recipe)[1]
        except InstanceError:
            return None

    # --- write ----------------------------------------------------------------------

    def check_name(self, name: str) -> str:
        return check_name(self.store, name)

    def default_name(self, facts: ModelFacts) -> str:
        return default_name(facts)

    def add(self, plan: Plan, profile: Profile, name: str, upscaler: bool = False) -> Recipe:
        """Save a recipe on a free port. Does not download. `upscaler` gives a picture
        instance the ESRGAN upscaler (`/sdcpp/v1/upscale`, the engine page's hires step)."""
        self.check_name(name)
        return self.create(
            plan.facts.modality, lambda port: plan.recipe(profile, name, port, upscaler)
        )

    def create(self, modality: Modality, build: Callable[[int], Recipe]) -> Recipe:
        """Save the recipe `build` makes on a free port of the type's block (`mnm upgrade`),
        stamped with the box it is planned on (`planned_on`)."""
        here = planned_here(self.settings, self.memory())
        try:
            return self.store.create(modality, self._port_in_use, lambda p: stamped(build(p), here))
        except StoreError as exc:
            raise InstanceError(str(exc)) from exc

    def download(self, recipe: Recipe, progress: Progress | None = None) -> Recipe:
        """Fetch missing files. Progress is merged into the stored recipe file by file. A file
        removed from the cache since (the user freed disk space) is fetched again."""
        if self.files_present(recipe):
            return self._converted(recipe, progress)

        def checkpoint(partial: Recipe) -> None:
            self.store.merge_local_paths(recipe.name, partial.local_paths)

        try:
            self.fetch_files(recipe, progress, checkpoint)
        except StoreError as exc:
            if self.store.exists(recipe.name):  # still there: the store's reason is the one
                raise InstanceError(str(exc)) from exc
            raise InstanceError(f"{recipe.name} was removed while downloading.") from exc
        return self._converted(fetched(self, recipe.name), progress)  # the disk policy first

    def fetch_files(
        self,
        recipe: Recipe,
        progress: Progress | None = None,
        checkpoint: Callable[[Recipe], object] | None = None,
        after: str = "Finished files are kept; start again to resume",
    ) -> Recipe:
        """A recipe's files fetched, the recipe with `local_paths`; `checkpoint` merges each
        into the saved recipe (without one nothing is saved: a quality tune's candidate quants,
        which no recipe lists until one is chosen)."""
        from_the_hub = any(  # Civitai's and Ollama's files never ask the Hub
            not (is_civitai(repo) or elsewhere(repo)) for _k, repo, _p, _r in wanted_files(recipe)
        )
        if progress and from_the_hub and self.settings.hf_token_value() is None:
            progress(
                "No Hugging Face token set: downloading anonymously (the Hub may rate-limit, "
                "and its library prints a warning saying so). `mnm setup` can store one"
            )
        try:
            return download_recipe_files(
                recipe,
                self.models_dir,
                self.settings.hf_token_value(),
                self._fetch,
                on_file=progress,
                checkpoint=checkpoint,
                civitai_token=self.settings.civitai_token_value(),
            )
        except StoreError:
            raise
        except Exception as exc:  # huggingface_hub raises many unrelated types
            raise InstanceError(
                f"Download failed: {str(exc).rstrip('. ')}. {after}.",
                diagnose(str(exc)),
            ) from exc

    def _converted(self, recipe: Recipe, progress: Progress | None) -> Recipe:
        """A checkpoint mnm converts gets its audio.cpp folder beside it."""
        try:
            ensure_converted(recipe, self.models_dir, progress)
        except ConversionError as exc:
            raise InstanceError(str(exc)) from exc
        return recipe

    def start(
        self, recipe: Recipe, progress: Progress | None = None, auto_fix: bool = True
    ) -> Started:
        """Download if needed, run the container, wait for /health (`instance_start`)."""
        return run_start(self, recipe, progress, auto_fix)

    def stop(self, recipe: Recipe, park: bool = False, told: Note = to_stderr) -> None:
        """Stop the engine (its log kept). The user's stop
        marks it `off` (it stays off until started); `park` is mnm making room, which leaves
        it loadable on request. Its slot is saved first (`cache=on`), the
        line to `told` (the router's journal, the CLI's console)."""
        slot_files.before_stop(self, recipe, told)
        keep_log(self.supervisor, self.settings.expanded_state_dir(), recipe.name)
        try:
            with box_events.changing(self.settings):  # a reading meanwhile is not one's own
                self.supervisor.stop(recipe.name)
        except SupervisorError as exc:
            raise InstanceError(str(exc)) from exc
        self.mark_off(recipe, not park)

    def mark_off(self, recipe: Recipe, off: bool) -> Recipe:
        """Save whether the user stopped it, judged on the saved file (a caller's
        copy may be stale); an unchanged file is not rewritten."""
        current = self.get(recipe.name)
        if current.off == off:
            return current
        return self.change(recipe.name, lambda saved: saved.model_copy(update={"off": off}))

    def stop_many(
        self, names: list[str] | None, park: bool = False, told: Note = to_stderr
    ) -> list[Stopped]:
        """Stop the named instances, or with None every one that has a container (running or
        exited). Every name is looked up and Docker asked about every container before
        anything stops, so a typo or a Docker error stops nothing. `park` (`mnm stop --park`)
        leaves each loadable on request instead of off, and with None names every instance,
        a stopped one included: the one command that makes a stopped instance loadable
        without loading it. Without it, a parked one is named too: it goes off, and
        `mnm stop --all` must say so, not "Nothing is running"."""
        recipes = [self.get(n) for n in names] if names is not None else self.all()
        found: list[Stopped] = []
        for recipe in recipes:
            try:
                state = self.supervisor.status(recipe.name).state
            except SupervisorError as exc:
                raise InstanceError(str(exc)) from exc
            if names is not None or park or _changes(recipe, state):
                found.append(Stopped(recipe.name, state, recipe.downloaded))
        stopped = {item.name for item in found if item.was is not ContainerState.ABSENT}
        for name in stopped:
            self.stop(self.get(name), park=park, told=told)
        for recipe in recipes:  # a container removed behind mnm's back
            if recipe.name not in stopped:
                self.mark_off(recipe, not park)  # without --park the user's stop means off
        return found

    def remove(self, recipe: Recipe, files: bool = False) -> Cleanup | None:
        """Clears the container first: when Docker cannot be asked, the recipe is kept rather
        than leave a container (maybe still running, holding its port) nobody can name. With
        `files`, the downloads no other instance uses go too (`model_cleanup`); the other
        recipes are read first, so one that cannot be read stops the removal before anything
        is gone."""
        others = [r for r in self.all() if r.name != recipe.name] if files else []
        try:
            with box_events.changing(self.settings):  # a reading meanwhile is not one's own
                self.supervisor.stop(recipe.name)
        except SupervisorError as exc:
            raise InstanceError(
                f"{recipe.name} was kept: removing it clears its container first, and Docker "
                f"could not be asked ({exc})."
            ) from exc
        try:
            self.store.delete(recipe.name)
        except StoreError as exc:
            raise InstanceError(str(exc)) from exc
        state = self.settings.expanded_state_dir()
        forgets: list[Callable[[], object]] = [
            lambda: forget_log(state, recipe.name),
            lambda: self.measurements.forget(recipe.name),
            lambda: self.tools.forget(recipe.name),
            lambda: TuneStore(state).forget(recipe.name),
        ]
        for forget in forgets:  # each on its own: one that fails leaves the rest
            with contextlib.suppress(OSError):
                forget()
        # The links and its saved prompts go, the downloads stay in the cache; the recipe is
        # gone either way, and a folder left behind is said (a saved conversation).
        try:
            if (cleanup := spec_for(recipe).cleanup) is not None:
                cleanup(recipe, self.models_dir)
        except OSError as exc:
            raise InstanceError(
                f"{recipe.name} is removed, but its folders under the models folder were not "
                f"all cleared: {exc}"
            ) from exc
        if not files:
            return None
        try:
            workflows = Models(self.models_dir, self.settings.expanded_state_dir())
            linked = frozenset(item.target for item in workflows.ledger().links)
            return delete_own_files(recipe, others, self.models_dir, linked)
        except ModelError as exc:  # which files workflows use is unknown: none is deleted
            raise InstanceError(f"{recipe.name} is removed and its files are kept: {exc}") from exc
        except OSError as exc:
            raise InstanceError(
                f"{recipe.name} is removed, but deleting its files stopped: {exc}"
            ) from exc

    def change(self, name: str, change: Callable[[Recipe], Recipe]) -> Recipe:
        """Re-read, change and save one recipe; the store's errors come back as InstanceError,
        like every other write here (`change` raises InstanceError for its own reasons)."""
        try:
            return self.store.update(name, change)
        except StoreError as exc:
            raise InstanceError(str(exc)) from exc

    def update_settings(self, recipe: Recipe, updates: dict[str, str]) -> Recipe:
        """`mnm set` (`instance_settings`: a draft and experts across cards read files)."""
        from makenomistakes.instance_settings import update_settings  # noqa: PLC0415

        return update_settings(self, recipe, updates)


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _changes(recipe: Recipe, state: ContainerState) -> bool:
    """`mnm stop --all` acts on it: a container to stop, or a parked one it turns off."""
    return state is not ContainerState.ABSENT or not recipe.off
