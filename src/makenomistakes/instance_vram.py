"""What saved instances put on the GPU, and whether each fits now: the read side of
`InstanceService` that prices instances, kept apart from the lifecycle that runs them.

`InstanceVram` prices one recipe from the files on disk and its current settings (each
engine's own fit model); `verdicts` and `vram_holders` turn those prices into the verdicts
`mnm ls`, the web list, workflows and `mnm add` print. `InstanceService` stays the one object
callers hold: its `vram_estimate`, `verdicts` and `native_context` hand over to this module."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from makenomistakes.calibration import Correction, correction, every_reading, instance_figure
from makenomistakes.capacity import (
    COMFYUI,
    FitNow,
    Verdict,
    VramEstimate,
    fit_now,
    with_room,
)
from makenomistakes.card_room import placed, room_view
from makenomistakes.config import Settings
from makenomistakes.fit import FitLevel
from makenomistakes.fit.estimate import headroom_level
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.instance_types import InstanceError, InstanceStatus
from makenomistakes.measurements import (
    MeasurementStore,
    Reading,
    built_for,
    settings_key,
    start_reading,
)
from makenomistakes.model_files import read_model_files
from makenomistakes.plan import Plan
from makenomistakes.ram_room import runs_on_processor
from makenomistakes.readings_filter import every_here, instance_readings
from makenomistakes.recipes import Recipe
from makenomistakes.room_verdicts import (
    NOTHING,
    ComfyRoom,
    counted,
    lighter,
    on_request,
    taking_turns,
)
from makenomistakes.settings_estimates import settings_estimate
from makenomistakes.supervisor import ContainerState, SupervisorError

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

CORRECTION_SECONDS = 60.0  # the engines' corrections are read again after this long
DOCKER_UNKNOWN = "Docker could not be asked, so whether it runs (and fits) is unknown."


class InstanceVram:
    def __init__(
        self,
        settings: Settings,
        memory: Callable[[], MemorySnapshot],
        measurements: MeasurementStore | None = None,
        build_of: Callable[[str], str | None] = lambda _image: None,
    ) -> None:
        self._settings = settings
        self._memory = memory  # the service's own `memory`, so a replaced probe reaches it
        self._card: tuple[float, int] | None = None  # (when read, the card's total)
        self._measurements = measurements or MeasurementStore(settings.expanded_state_dir())
        self._build_of = build_of
        self._corrections: tuple[float, list[Reading]] | None = None

    def estimate(
        self, recipe: Recipe, running_build: str | None = None, started: str | None = None
    ) -> VramEstimate:
        """What the instance puts on the GPU, safest figure first: what it was
        measured holding on this card with this build and these settings, under the rule in
        `calibration`; else the fit model's figure times its engine's correction. The build
        is the one its container runs while it runs (`running_build`: an update leaves it on
        the old one), else the one its tag names for the next start. Placed on its cards
        when `MNM_GPU` names several (`card_room.placed`). A running container whose
        settings were changed since it started (`started`: Docker's StartedAt) holds what it
        started with, not what the next start will: `started_with`."""
        held = self.started_with(recipe, started) if started is not None else None
        return placed(recipe, held or self._estimate(recipe, running_build))

    def started_with(self, recipe: Recipe, started: str) -> VramEstimate | None:
        """What a running container holds when its saved settings changed since it started:
        its start's own load reading (measured, else the estimate of the settings then);
        None when the settings are the same, or no reading names that start. Priced from the
        saved recipe instead, a qwen3-8b still running with a draft `draft=` had removed (about
        10.8 GiB) reads as 7.1, so parking it looks useless and another model is loaded beside
        it into a full card."""
        load = start_reading(self._measurements.readings(recipe.name), started)
        if load is None or load.settings == settings_key(recipe):
            return None
        figure = load.vram_bytes or load.estimate_bytes
        if not figure:
            return None
        how = "measured" if load.vram_bytes else "estimated"
        return VramEstimate(
            bytes=figure,
            basis=f"{how} for the settings its container started with (the saved ones apply "
            "at its next start)",
        )

    def _estimate(self, recipe: Recipe, running_build: str | None) -> VramEstimate:
        raw = self.raw(recipe)
        figure = instance_figure(
            instance_readings(self._measurements, recipe.name),  # this box's
            settings_key(recipe),
            running_build or self._build_of(recipe.image),
            raw.bytes,
        )
        if figure is not None:  # the other figures (a tiled decode, streaming) stay the model's
            return raw.model_copy(update={"bytes": figure.bytes, "basis": figure.basis})
        corrected = self.correction(recipe.engine, recipe.modality, built_for(recipe))
        if raw.bytes is None or corrected.ratio == 1.0:
            return raw
        basis = (
            f"{raw.basis}, x{corrected.ratio:.2f} for {corrected.label} as measured on "
            f"this card ({corrected.readings} reading{'s' * (corrected.readings != 1)})"
        )
        return raw.model_copy(update={"bytes": corrected.apply(raw.bytes), "basis": basis})

    def correction(
        self, engine: Engine, modality: Modality, backend: str | None = None
    ) -> Correction:
        """The engine's correction for a type (on one backend) from every instance's
        readings, read again each minute."""
        now = time.monotonic()
        if self._corrections is None or now - self._corrections[0] >= CORRECTION_SECONDS:
            self._corrections = (now, every_reading(every_here(self._measurements)))
        return correction(engine, modality, self._corrections[1], backend)

    def raw(self, recipe: Recipe) -> VramEstimate:
        """The fit model's figure over the files on disk and the current settings, so it
        follows `mnm set` and fit-model fixes; the number planned at `add` stands in when
        the files cannot be read."""
        computed = self._settings_estimate(recipe)
        if computed.bytes is None and recipe.vram_estimate_bytes is not None:
            return VramEstimate(bytes=recipe.vram_estimate_bytes, basis="planned when added")
        return computed

    def _settings_estimate(self, recipe: Recipe) -> VramEstimate:
        if not recipe.downloaded:
            return VramEstimate(bytes=None, basis="the model is not downloaded yet")
        files = read_model_files(recipe, self._settings.expanded_models_dir())
        if files is None:
            return VramEstimate(bytes=None, basis="the model files on disk could not be read")
        return settings_estimate(recipe, files, self._card_bytes(recipe))

    def _card_bytes(self, recipe: Recipe) -> int | None:
        """The card's total, read again each minute (a card swapped under a long-lived
        router): vLLM's share is of it. A failed read is asked again."""
        if recipe.engine is not Engine.VLLM:
            return None
        now = time.monotonic()
        if self._card is None or now - self._card[0] >= CORRECTION_SECONDS:
            gpu = self._memory().gpu
            if gpu is None:
                return None
            self._card = (now, gpu.vram_total_bytes)
        return self._card[1]

    def native_context(self, recipe: Recipe) -> int | None:
        """The context a downloaded model was trained for: a GGUF's header, or a vLLM
        model's config.json."""
        files = read_model_files(recipe, self._settings.expanded_models_dir())
        if files and files.header:
            return files.header.context_length
        if files and files.hf_config:
            return files.hf_config.text.max_position_embeddings
        return None


def verdicts(
    statuses: list[InstanceStatus],
    snapshot: MemorySnapshot,
    estimate: Callable[[Recipe, str | None], VramEstimate],
    held_back: frozenset[str] | None = None,
    kept: frozenset[str] | None = None,
    comfy: ComfyRoom = NOTHING,
) -> dict[str, Verdict]:
    """Per instance: running, fits the free VRAM now, near the limit, or won't fit (what
    ComfyUI gives back at a start, `comfy`, counted free and said); a
    won't-fit one names the running instances to stop first (`statuses` has them all), and
    a parked one whose room the router makes reads *on request* when `held_back` is known.
    What is never stopped for room (`kept`, else `held_back`) is named apart, never as room;
    with neither (a start by hand: `mnm start` parks what its verdict names) every one is."""
    never = kept if kept is not None else (held_back or frozenset())
    estimates = {s.recipe.name: estimate(s.recipe, s.build) for s in statuses if not s.error}
    held = {
        s.recipe.name: estimates[s.recipe.name].bytes or 0
        for s in statuses
        if s.running and not s.error
    }
    groups = {s.recipe.name: s.recipe.swap_group for s in statuses}
    pinned = frozenset(s.recipe.name for s in statuses if s.recipe.pinned)
    freed = lighter(snapshot, comfy.freeable)
    found = {}
    for s in statuses:
        name = s.recipe.name
        if s.error:
            found[name] = Verdict(level=FitNow.UNKNOWN, note=DOCKER_UNKNOWN)
            continue
        verdict = fit_now(estimates[name], freed, s.running)
        others = {n: estimates[n] for n in held if n != name}
        seen = room_view(estimates[name], freed, others)  # its cards
        verdict = with_room(
            verdict, seen.estimate, seen.snapshot, seen.held, never, pinned, seen.held_on
        )
        verdict = counted(verdict, estimates[name], snapshot, comfy)
        if (group := groups[name]) is not None:
            verdict = taking_turns(verdict, group, {n for n, g in groups.items() if g == group})
        if held_back is not None and s.parked:  # still won't fit: the router makes the room
            verdict = on_request(verdict, held_back)
        found[name] = verdict
    return found


def vram_holders(plan: Plan) -> list[str]:
    """mnm's running instances whose VRAM the plan counted as free (`planning_budget`),
    when some profile would not fit comfortably beside them: those must stop before it
    starts. Empty when every profile fits beside them, and for `--vram` plans."""
    budget = plan.budget
    # Both priced alike: the free figure is the card's divided by the engine's correction.
    beside = budget.vram_free_bytes - int(budget.held_bytes / budget.correction)
    crowded = any(
        headroom_level(p.fit.gpu_bytes, beside) is not FitLevel.COMFORTABLE
        for p in plan.profiles.values()
    )
    # ComfyUI is no instance to stop: every start asks it to give the card back first
    return [name for name in budget.held_by if name != COMFYUI] if crowded else []


def running_vram(service: InstanceService, in_ram: bool | None = None) -> dict[str, int]:
    """Running instance → its estimated VRAM (instances without an estimate are left out);
    `mnm doctor` subtracts them from what nvidia-smi calls in use; `in_ram` keeps only those
    in the RAM (True) or on the cards (False). Never raises: with the store or Docker
    unreadable it is empty and the hardware checks still run."""
    try:
        recipes = service.all()
    except InstanceError:
        return {}
    found = {}
    for recipe in recipes:
        if in_ram is not None and runs_on_processor(recipe) != in_ram:
            continue
        try:
            container = service.supervisor.status(recipe.name)
        except SupervisorError:
            continue
        running = container.state is ContainerState.RUNNING
        estimate = (
            service.vram_estimate(recipe, container.build, container.started_at).bytes
            if running
            else None
        )
        if estimate is not None:
            found[recipe.name] = estimate
    return found
