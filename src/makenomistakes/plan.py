"""The planning pipeline: reference → facts → GGUF header → hardware budget → profiles.

Pure orchestration with typed errors, shared by the CLI and the web UI. When the reference
cannot be served, the error carries `Guidance`: what it is and what to use instead. What a
plan is lives in `plan_types.py`; the image branches in `plan_images.py`.
"""

from __future__ import annotations

import dataclasses

from makenomistakes.backend import Backend, BackendError, software
from makenomistakes.backend_now import decide_here
from makenomistakes.calibration import correction, every_reading, speed_corrections
from makenomistakes.config import Settings
from makenomistakes.engine_images import NoImage
from makenomistakes.fit import Budget, Profile, ProfileName
from makenomistakes.fit.audio import plan_audio_profiles
from makenomistakes.fit.budget import WSL_SPILL_MARGIN
from makenomistakes.fit.cards import CardBudget, combined_bandwidth
from makenomistakes.fit.cpu_plan import software_vulkan_budget
from makenomistakes.fit.devices import device_for
from makenomistakes.fit.music import plan_music_profiles
from makenomistakes.fit.plan_floor import RamFloorError
from makenomistakes.fit.profiles import plan_profiles
from makenomistakes.fit.ram_floor import OS_RAM_MARGIN
from makenomistakes.fit.voice import plan_voice_profiles
from makenomistakes.gguf import GgufHeader
from makenomistakes.guide import guide_ollama_tag, guide_unservable, guide_voice_image
from makenomistakes.hardware import ProbeError, probe
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.fingerprint import (
    UNIFIED_MARGIN,
    MachineFingerprint,
    MemoryKind,
    stable_id_of,
)
from makenomistakes.hardware.probe import Gpu, HardwareReport, MemoryInfo, chosen_cards
from makenomistakes.hub import (
    HubClient,
    LocalPath,
    ModelFacts,
    ModelRef,
    OllamaTagError,
    UnsupportedReferenceError,
    parse_reference,
)
from makenomistakes.hub.civitai import (
    CivitaiError,
    CivitaiRef,
    SourcesHub,
    parse_civitai,
)
from makenomistakes.hub.facts import RETYPABLE, Engine, Modality, narrow_to_version
from makenomistakes.hub.modelscope import ModelScopeError, ModelScopeRef, parse_modelscope
from makenomistakes.hub.ollama import OllamaError, OllamaRef, parse_ollama
from makenomistakes.machine import observe
from makenomistakes.machine_facts import box_priced
from makenomistakes.measurements import MeasurementStore, readings_backend
from makenomistakes.music_speed import music_rates
from makenomistakes.plan_images import checkpoint_plan, image_plan
from makenomistakes.plan_large import model_header, refusal, with_large_facts
from makenomistakes.plan_sources import facts_for, lmstudio_ref
from makenomistakes.plan_types import (
    CheckpointChoice,
    NoRoomError,
    NoSuchQuantError,
    Plan,
    PlanError,
    TypeRefusedError,
)
from makenomistakes.plan_vllm import vllm_folder, vllm_plan
from makenomistakes.ram_budget import in_ram_budget
from makenomistakes.readings_filter import every_here
from makenomistakes.recipes import spec
from makenomistakes.recipes.cards import ONE_CARD_ENGINES

__all__ = [
    "CheckpointChoice",
    "Plan",
    "PlanError",
    "budget_for",
    "explicit_budget",
    "live_budget",
    "make_plan",
    "parse",
    "resolve",
]


def parse(reference: str, hub: HubClient) -> ModelRef | CivitaiRef | OllamaRef | ModelScopeRef:
    """A Civitai version, an Ollama tag, or a Hub repo: from a Hub link or name, or from
    another source's link that stands for one (LM Studio's catalog)."""
    try:
        other = parse_ollama(reference) or parse_modelscope(reference)
    except (OllamaError, ModelScopeError) as exc:
        raise PlanError(str(exc)) from exc
    if other is not None:
        return other
    listed = lmstudio_ref(reference, hub)
    if listed is not None:
        return listed
    try:
        civitai = parse_civitai(reference)
    except CivitaiError as exc:
        raise PlanError(str(exc)) from exc
    if civitai is not None:
        return civitai
    try:
        ref = parse_reference(reference)
    except OllamaTagError as exc:
        raise PlanError(str(exc), guide_ollama_tag(exc.tag, hub)) from exc
    except UnsupportedReferenceError as exc:
        raise PlanError(str(exc)) from exc
    if isinstance(ref, LocalPath):
        raise PlanError("Local paths are not resolvable yet; paste a Hugging Face reference.")
    return ref


def open_hub(settings: Settings) -> SourcesHub:
    """The Hub client every plan uses: Hugging Face, and Civitai for `civitai/` repos."""
    return SourcesHub(settings.hf_token_value(), settings.civitai_token_value())


def resolve(reference: str, settings: Settings) -> ModelFacts:
    with open_hub(settings) as hub:
        return facts_for(hub, parse(reference, hub))[1]


def make_plan(
    reference: str,
    settings: Settings,
    budget: Budget | None = None,
    version: str | None = None,
    model_type: Modality | None = None,
    *,
    quant: str | None = None,
) -> Plan:
    """Everything `fit` and `add` need. `budget=None` probes the local hardware. `version`
    picks one of an image repo's versions (`ModelFacts.versions`); a `:Q8_0` tag on the
    reference becomes the default quant. The quant asked for (`quant`, else the tag) also
    picks the version of a multi-model GGUF repo that holds it; `choose` validates it for
    files that are not GGUF (a DiT file's as-stored or Q8_0). `model_type` retypes a chat
    model as coding (or back): the same engine, other cards and the agent profile."""
    with open_hub(settings) as hub:
        ref, facts = facts_for(hub, parse(reference, hub))
        facts = retyped(facts, model_type)
        _refuse_unrunnable(facts, hub)
        ref, facts, version = _vllm_version(hub, ref, facts, version, model_type)
        picked = version is not None or ref.path is not None
        try:
            if facts.engine is not Engine.SDCPP or facts.gguf_variants or version:
                facts = narrow_to_version(facts, version, quant or ref.quant)
        except ValueError as exc:
            raise PlanError(str(exc)) from exc
        wanted = quant or ref.quant
        if wanted is not None and facts.gguf_variants and facts.variant(wanted) is None:
            available = ", ".join(v.quant for v in facts.gguf_variants)
            raise NoSuchQuantError(
                f"{facts.repo_id} has no '{wanted}' variant (available: {available})."
            )
        budget = budget or live_budget(settings)
        if facts.engine in ONE_CARD_ENGINES:  # it runs on the first chosen card, alone
            budget = budget.one_card()
        budget = calibrated(budget, facts.engine, settings, facts.modality)
        _refuse_no_image(facts, budget)
        if facts.engine is Engine.SDCPP and not facts.gguf_variants:
            return _tagged(checkpoint_plan(facts, hub, budget, picked), ref)
        if facts.engine is Engine.SDCPP:
            return _tagged(image_plan(facts, hub, budget), ref)
        if facts.engine is Engine.WHISPERCPP:
            return _tagged(Plan(facts, None, budget, plan_audio_profiles(facts, budget)), ref)
        if facts.engine is Engine.AUDIOCPP:
            audio = plan_music_profiles if facts.music_family else plan_voice_profiles
            return _tagged(Plan(facts, None, budget, audio(facts, budget)), ref)
        if facts.engine is Engine.VLLM:
            return _tagged(vllm_plan(facts, hub, budget), ref)
        facts, header = model_header(hub, ref.revision, facts)
        facts, header = with_large_facts(hub, ref.revision, facts, header)
    if wanted is not None and (refused := refusal(facts, wanted)) is not None:
        raise PlanError(refused)
    return _tagged(Plan(facts, header, budget, _gguf_profiles(facts, header, budget)), ref)


def _gguf_profiles(
    facts: ModelFacts, header: GgufHeader, budget: Budget
) -> dict[ProfileName, Profile]:
    """The GGUF profiles, or why there are none: the box's size (`NoRoomError`, a RAM
    floor) told apart from a header the fit model cannot size."""
    try:
        profiles = plan_profiles(facts, header, budget)
    except RamFloorError as exc:  # no quant fits this machine's RAM and cards together
        raise NoRoomError(f"{facts.repo_id}: {exc}") from exc
    except ValueError as exc:  # a header the fit model cannot size (no head counts)
        raise PlanError(f"{facts.repo_id}: {exc}") from exc
    if not profiles:
        raise PlanError(f"{facts.repo_id} has no GGUF variants with a recognisable quant.")
    return profiles


def _tagged(plan: Plan, ref: ModelRef) -> Plan:
    """`hf.co/org/repo:Q8_0`: the tag is the quant `choose` uses unless told otherwise,
    checked while planning so `mnm fit` refuses what `mnm add` would."""
    if ref.quant is None:
        return plan
    plan.check_quant(ref.quant)
    return dataclasses.replace(plan, tag_quant=ref.quant)


def calibrated(
    budget: Budget, engine: Engine | None, settings: Settings, modality: Modality
) -> Budget:
    """The budget priced with the engine's measured correction for this type, for this
    card only: a `--vram` budget plans for another machine, where these readings say nothing."""
    if budget.elsewhere or engine is None:
        return budget
    music = engine is Engine.AUDIOCPP  # each music model's time on this box, RAM or card
    if not music and budget.vram_total_bytes is None:
        return budget
    store = MeasurementStore(settings.expanded_state_dir())
    readings = every_reading(every_here(store))  # this box's only
    ran = readings_backend(engine, budget.backend, budget.arch)  # as its image's readings say
    if music:  # never the processor's rate for a card's plan
        rates = music_rates(readings, ran)
        budget = budget.model_copy(update={"music_rates": {r.key: r.rate for r in rates}})
    if budget.vram_total_bytes is None:
        return budget
    found = correction(engine, modality, readings, ran)
    budget = budget.corrected(found.ratio, found.note)
    if engine is Engine.LLAMACPP:  # this box's own CPU and card figures
        return box_priced(budget, settings.expanded_state_dir())
    if engine is not Engine.SDCPP:
        return budget
    speeds = speed_corrections(readings)  # each DiT family's picture time
    return budget.model_copy(
        update={
            "speed": {c.family: c.ratio for c in speeds if c.ratio != 1.0},
            "speed_notes": {c.family: c.note for c in speeds if c.note},
        }
    )


def explicit_budget(vram_gb: float, ram_gb: float, gpu_name: str) -> Budget:
    device = device_for(gpu_name)  # another card's row: no reading of this box's
    return Budget(
        vram_free_bytes=int(vram_gb * 2**30),
        ram_free_bytes=int(ram_gb * 2**30),
        ram_total_bytes=int(ram_gb * 2**30),  # the machine `--ram` describes (the RAM floor)
        gpu_bandwidth=device.bandwidth,
        gpu_compute=device.tflops * 1e12,
        device=device,
        elsewhere=True,
    )


def live_budget(settings: Settings) -> Budget:
    """This box's budget now, for its backend: the cards' VRAM as before on CUDA;
    without a card the RAM as the one pool (`fit.cpu_plan`), every model planned in it."""
    report = probe(settings.expanded_models_dir())
    if report.memory is None or report.memory.available_bytes is None:
        raise PlanError("Could not read free RAM from /proc/meminfo; pass --ram <GiB>.")
    box, backend = box_backend(settings, report)
    on_box = {"backend": backend, "arch": box.arch if box is not None else None}
    if box is not None and backend is Backend.CPU:
        return in_ram_budget(settings, box, report.memory)
    if box is not None and backend is not None and software(box, backend):
        budget = software_vulkan_budget(
            report.memory.total_bytes or 0, report.memory.available_bytes, box.cpu
        )
        return budget.model_copy(update=on_box)
    if isinstance(report.gpus, ProbeError) and report.unchosen:  # a stale MNM_GPU
        raise PlanError(f"{report.gpus.message}: set MNM_GPU to the cards there now.")
    if isinstance(report.gpus, ProbeError) or not report.gpus:
        raise PlanError("No GPU detected; pass --vram <GiB> to plan for another machine.")
    cards = chosen_cards(report.gpus, report.chosen)
    if any(card.unified or card.carve_out for card in cards):  # one pool
        return on_backend(shared_budget(cards[0], report.memory), backend).model_copy(update=on_box)
    budget = on_backend(budget_on(cards, report.memory.available_bytes), backend)
    return budget.model_copy(update={"ram_total_bytes": report.memory.total_bytes, **on_box})


def on_backend(budget: Budget, backend: Backend | None) -> Budget:
    """The budget priced as `backend` runs its card (an R9700's compute differs on
    ROCm and Vulkan, `DeviceRow.for_backend`); unchanged where the row knows no difference."""
    device = budget.device.for_backend(backend.value if backend is not None else None)
    if device is budget.device:
        return budget
    return budget.model_copy(update={"device": device, "gpu_compute": device.tflops * 1e12})


def box_backend(
    settings: Settings, report: HardwareReport
) -> tuple[MachineFingerprint | None, Backend | None]:
    """The box read now and recorded (`machine.observe`) and its backend; (None, None) when
    the box could not be read (cards unreadable, `machine.json` unwritable): the plan then
    keeps the engines' default images. A `MNM_BACKEND` the cards cannot run is a
    PlanError with the reason."""
    observed = observe(settings, report.box_cards, report.memory)
    current = observed.file.current if observed.file is not None else None
    if current is None:
        return None, None
    try:
        return current.fingerprint, decide_here(settings, current.fingerprint)
    except BackendError as exc:
        raise PlanError(f"{exc}; fix it in the secrets file (`MNM_BACKEND=`).") from exc


def _refuse_no_image(facts: ModelFacts, budget: Budget) -> None:
    """An engine with no image for this box's backend and arch: refused while planning, so
    `mnm fit` says what `mnm add` would (`engine_images.NoImage`)."""
    if budget.backend is None or facts.engine is None:
        return
    found = spec(facts.engine).image_for(budget.backend, budget.arch or "x86_64")
    if isinstance(found, NoImage):
        raise PlanError(found.words)


def budget_on(cards: list[Gpu], ram_free_bytes: int) -> Budget:
    """The chosen cards as one Budget: one card exactly as `budget_for`, several with each
    card's own (`Budget.cards`), the sums, and the layer split's bandwidth."""
    if len(cards) < 2:
        return budget_for(cards[0], ram_free_bytes)
    each = tuple(
        CardBudget(
            index=g.index,
            name=g.name,
            free_bytes=g.vram_free_bytes,
            total_bytes=g.vram_total_bytes,
            bandwidth=device_for(g).bandwidth,
            lanes=g.lanes,
            pcie_gen=g.pcie_gen_max or g.pcie_gen,  # an idle card drops its link speed
            stable_id=stable_id_of(g.uuid, g.bus_id, g.name, g.index, g.unique_id),
        )
        for g in cards
    )
    return Budget(
        vram_free_bytes=sum(c.free_bytes for c in each),
        vram_total_bytes=sum(c.total_bytes for c in each),
        ram_free_bytes=ram_free_bytes,
        gpu_bandwidth=combined_bandwidth(list(each)),
        gpu_compute=min(device_for(g).tflops for g in cards) * 1e12,
        cards=each,
        device=device_for(cards[0]),  # the efficiencies of the first card's row
        # WSL's spill margin for each card it may spill from
        spill_margin_bytes=WSL_SPILL_MARGIN * len(cards) if platforms.current().is_wsl else 0,
    )


def unified_budget(gpu: Gpu, memory: MemoryInfo) -> Budget:
    """A card that shares the machine's RAM (a DGX Spark's GB10, whose nvidia-smi
    answers `[N/A]`): one pool, its free part MemAvailable less the firm margin
    (`UNIFIED_MARGIN`, 8 GiB: the box hangs rather than OOM-kills), no RAM beside
    it to offload to (every model planned whole), the RAM floor and the ladder counting the
    pool once, and the busy-card rule judged on what the pool holds beyond that margin
    (`Budget.busy_bytes`). The same figures the probe gives the card (`unified_pool`)."""
    device = device_for(gpu)
    if device.memory is not MemoryKind.UNIFIED:  # a card the table has no row for
        device = device.model_copy(update={"memory": MemoryKind.UNIFIED})
    total, free = memory.total_bytes or 0, memory.available_bytes or 0
    return Budget(
        vram_free_bytes=max(free - UNIFIED_MARGIN, 0),
        vram_total_bytes=max(total - UNIFIED_MARGIN, 0),
        ram_free_bytes=0,
        ram_total_bytes=total,
        gpu_bandwidth=device.bandwidth,
        gpu_compute=device.tflops * 1e12,
        device=device,
    )


def carve_out_budget(gpu: Gpu, memory: MemoryInfo) -> Budget:
    """An APU's (a Strix Halo, a Radeon 890M): its carve-out and GTT as one pool,
    capped by the RAM less the system's margin (the GTT is the RAM: the kernel may allow more
    than is free), nothing to offload to beside it, the floor counting it once."""
    device = device_for(gpu)
    if device.memory is not MemoryKind.CARVE_OUT:  # a card the table has no row for
        device = device.model_copy(update={"memory": MemoryKind.CARVE_OUT})
    total, free = memory.total_bytes or 0, memory.available_bytes or 0
    return Budget(
        vram_free_bytes=min(gpu.vram_free_bytes, max(free - OS_RAM_MARGIN, 0)),
        vram_total_bytes=min(gpu.vram_total_bytes, max(total - OS_RAM_MARGIN, 0)),
        ram_free_bytes=0,
        ram_total_bytes=total,
        gpu_bandwidth=device.bandwidth,
        gpu_compute=device.tflops * 1e12,
        device=device,
    )


def shared_budget(gpu: Gpu, memory: MemoryInfo) -> Budget:
    """A card that plans in the machine's RAM: unified (a GB10) or an APU's GTT."""
    return carve_out_budget(gpu, memory) if gpu.carve_out else unified_budget(gpu, memory)


def budget_for(gpu: Gpu, ram_free_bytes: int) -> Budget:
    """The one way a measured card becomes a Budget: `mnm add` and auto-fix retries plan
    against the same fields. A card that plans in the machine's RAM (its figures the
    pool's, `unified_pool`) has no RAM beside it: an auto-fix retry never offloads there."""
    device = device_for(gpu)  # its row in the device table
    if gpu.unified or gpu.carve_out:
        ram_free_bytes = 0
        if device.memory is MemoryKind.DISCRETE:  # a card the table has no row for
            kind = MemoryKind.CARVE_OUT if gpu.carve_out else MemoryKind.UNIFIED
            device = device.model_copy(update={"memory": kind})
    return Budget(
        vram_free_bytes=gpu.vram_free_bytes,
        vram_total_bytes=gpu.vram_total_bytes,
        ram_free_bytes=ram_free_bytes,
        gpu_bandwidth=device.bandwidth,
        gpu_compute=device.tflops * 1e12,
        device=device,
        spill_margin_bytes=WSL_SPILL_MARGIN if platforms.current().is_wsl else 0,
    )


def retyped(facts: ModelFacts, model_type: Modality | None) -> ModelFacts:
    """The facts as the type the user asked for, when the model can be that type. A vision
    model retyped runs text-only: without its projector, which neither the plan prices nor
    the recipe loads."""
    if model_type is None or model_type is facts.modality:
        return facts
    if model_type not in RETYPABLE.get(facts.modality, frozenset()):
        allowed = ", ".join(m.label for m in RETYPABLE.get(facts.modality, frozenset()))
        raise TypeRefusedError(
            f"{facts.repo_id} is {facts.modality.a_label} model and cannot be added as "
            f"{model_type.label}" + (f" (it can be: {allowed})." if allowed else "."),
            facts.modality,
        )
    text_only = {"projector_files": [], "projector_bytes": 0}
    vision = facts.modality is Modality.VISION_CHAT
    return facts.model_copy(update={"modality": model_type, **(text_only if vision else {})})


def _vllm_version(
    hub: SourcesHub,
    ref: ModelRef,
    facts: ModelFacts,
    version: str | None,
    model_type: Modality | None,
) -> tuple[ModelRef, ModelFacts, str | None]:
    """A vLLM repo's `--version` is one of its model folders: the facts are built again
    for that folder, as if it had been pasted."""
    if facts.engine is not Engine.VLLM or version is None:
        return ref, facts, version
    ref, facts = facts_for(hub, dataclasses.replace(ref, path=vllm_folder(facts, version)))
    return ref, retyped(facts, model_type), None


def _refuse_unrunnable(facts: ModelFacts, hub: SourcesHub) -> None:
    """Nothing here runs it (`guide_unservable`), or the engine's official image cannot
    (`guide_voice_image`: Kokoro): refused before the card is probed or anything downloads."""
    if not facts.servable:
        raise PlanError(
            f"{facts.repo_id} has nothing this tool can run yet.", guide_unservable(facts, hub)
        )
    unspoken = guide_voice_image(facts)
    if unspoken is not None:
        raise PlanError(f"{facts.display_name} cannot speak here yet.", unspoken)
