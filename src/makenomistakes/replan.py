"""`mnm replan <name>`: plan a saved instance again for this machine now.

New card, more RAM, a server: the profiles `mnm add` picked are sized to the hardware it
saw. A re-plan reads the model again from its source, plans against the live budget (or a
`--vram` override), takes the same profile (Quality, Balanced or Speed) and keeps what makes
the instance itself: its name, port, type, the quant it downloaded (unless asked to choose
again), its LoRAs and its swap group. Only the settings the plan sizes to the card
(`PLANNED`: context, KV precision, GPU layers and batches; picture size and offloads; vLLM's
context, share and KV) are its again; everything the user chose (seed, negative prompt,
sampler, steps, parallel slots, voice, idle sleep, extra flags) stays. An agent instance
keeps its context (chosen for its clients). Nothing restarts: the settings apply on the next start.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from makenomistakes.controls import SettingsError
from makenomistakes.download import wanted_files
from makenomistakes.download_sizes import file_sizes
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.fit import Budget, FitLevel
from makenomistakes.fit.estimate import headroom_level
from makenomistakes.fit.profiles import Profile
from makenomistakes.fit.quant_ladder import quant_bits
from makenomistakes.hub.civitai import civitai_reference, is_civitai
from makenomistakes.hub.elsewhere import elsewhere
from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.instance_settings import placed_again
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine import planned_here, stamped
from makenomistakes.plan import make_plan
from makenomistakes.plan_types import Plan, PlanError
from makenomistakes.prompt_cache import PlannedCache
from makenomistakes.recipes.model import OWNER_FIELDS, DraftModel, Recipe
from makenomistakes.recipes.quality import QualityTuned
from makenomistakes.replan_kept import kept_numa, kept_processor, processor_only
from makenomistakes.tune.apply import with_tune

# The engines whose quant names a downloaded file a re-plan keeps by default.
QUANT_PICKS = frozenset({Engine.LLAMACPP, Engine.WHISPERCPP, Engine.AUDIOCPP})
# The settings a plan sizes to the card, per engine; every other setting is the user's.
PLANNED: dict[Engine, tuple[str, ...]] = {
    Engine.LLAMACPP: (
        "ctx",
        "gpu_layers",
        "kv_type",
        "flash_attn",
        "batch",
        "ubatch",
        "cpu_moe_layers",
        "numa",  # The CPU backend's, from the box's NUMA nodes; the user's own stays
        *("tensor_split", "main_gpu", "placement"),  # the cards' layout: CARD_FIELDS
        *PlannedCache._fields,  # The prompt cache's caps, sized again like the context
    ),
    Engine.SDCPP: (
        "width",
        "height",
        "frames",
        "flash_attn",
        "offload_to_cpu",
        "vae_tiling",
        "text_encoder_on_cpu",
        "vae_on_cpu",
        "weight_type",
        "diffusion_type",
    ),
    Engine.WHISPERCPP: (),  # the size is the quant
    Engine.AUDIOCPP: (),
    Engine.VLLM: (
        "max_model_len",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "enforce_eager",
        "tensor_parallel",  # the equal cards it splits over, as the plan sees them
    ),
}
# The planned settings that lay an instance out over the cards: a `gpus=` pin governs them
# (`card_settings._fitted`), so a pinned instance keeps its own.
CARD_FIELDS = frozenset({"tensor_split", "main_gpu", "placement", "tensor_parallel"})


@dataclass(frozen=True)
class Withheld:
    """A smaller plan not taken: the saved one still fits the card now (`level`), and the
    plan now would only be smaller (`smaller` says how); `mnm replan <name> --shrink` takes it."""

    level: FitLevel
    smaller: tuple[str, ...]

    def words(self, name: str) -> str:
        """The one line the CLI prints for it."""
        return (
            f"{name} still fits the card ({self.level.value}); the plan now would be smaller "
            f"({', '.join(self.smaller)}): kept as it is. `mnm replan {name} --shrink` takes it."
        )


@dataclass(frozen=True)
class Replanned:
    old: Recipe
    new: Recipe
    changes: list[str]
    new_files: list[str]  # files the next start downloads first
    # Kept as it is (a plan that still fits): else `--all --yes` moves a context up one day
    # and halves it the next on 0.2 GiB less free.
    withheld: Withheld | None = None
    tune_note: str | None = None  # The tune kept on the new plan, or why it went
    # Each of `new_files`' size where the plan's facts know it (`mnm machine adopt`)
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    @property
    def shrinks(self) -> list[str]:
        """What gets smaller: a shorter context, fewer layers or a smaller picture on the
        GPU, another quant (said in one line before anything is saved)."""
        return _shrinks(self.old, self.new)

    @property
    def fingerprint(self) -> str:
        """Names the new settings: the web page posts it back, so the save is the plan shown."""
        shown = self.new.model_dump_json(include={"args", "quant", "files", "companions"})
        return hashlib.sha256(shown.encode()).hexdigest()[:16]


def reference(recipe: Recipe) -> str:
    """What to paste to get this recipe's files again: the repo, narrowed to the file or
    folder that picked the version, a voice package or a model folder."""
    if is_civitai(recipe.repo_id):
        return civitai_reference(recipe.repo_id, recipe.revision)
    if (where := elsewhere(recipe.repo_id)) is not None:
        return where.page(recipe.repo_id, recipe.revision)
    first = recipe.files[0]
    folder = str(PurePosixPath(first).parent)
    narrowed: str | None = None
    if recipe.engine is Engine.SDCPP and recipe.layout == "file":
        narrowed = first  # a pasted file picks that version
    elif recipe.engine is Engine.SDCPP or recipe.engine in (Engine.AUDIOCPP, Engine.VLLM):
        narrowed = None if folder == "." else _top_folder(recipe.files)
    if recipe.revision == "main":
        return f"{recipe.repo_id}/{narrowed}" if narrowed else recipe.repo_id
    section = "blob" if narrowed == first else "tree"
    tail = f"/{narrowed}" if narrowed else ""
    return f"https://huggingface.co/{recipe.repo_id}/{section}/{recipe.revision}{tail}"


def _top_folder(files: list[str]) -> str | None:
    """The folder every file sits under (a diffusers folder, a voice package), if any."""
    parts = [PurePosixPath(f).parts[:-1] for f in files]
    common: list[str] = []
    for level in zip(*parts, strict=False):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    return "/".join(common) or None


@dataclass(frozen=True)
class Pinned:
    """A quality tune's candidate: `quant` planned from `made` (one read of the Hub for
    every candidate) at the saved context, the `held` planned settings kept as they are (the
    KV cache, flash attention and batches: only the quant and where it is placed move), the
    placement sized around the kept KV cache."""

    quant: str
    made: Plan
    held: frozenset[str] = frozenset()


def plan_again(
    service: InstanceService, old: Recipe, budget: Budget | None = None, keep_quant: bool = True
) -> Plan:
    """A saved instance's plan made again from its reference, as `replan` makes it (the
    model's header read from the Hub, a coding or chat type kept, its quant asked for while
    `keep_quant`); `budget=None` probes the box. PlanError when it cannot be planned."""
    model_type = old.modality if old.modality in (Modality.CODING, Modality.CHAT) else None
    quant = old.quant if keeps_quant(old, keep_quant) else None
    return make_plan(reference(old), service.settings, budget, model_type=model_type, quant=quant)


def keeps_quant(old: Recipe, keep_quant: bool = True) -> bool:
    """Whether a re-plan keeps `old`'s quant: its own file, on the engines whose quant is a
    pick (`QUANT_PICKS`; never sd.cpp's or vLLM's). `choice` and `mnm why`'s words both read
    this rule, so the words never claim what `choice` does not do."""
    return keep_quant and old.engine in QUANT_PICKS


def keeps_context(old: Recipe) -> bool:
    """Whether a re-plan keeps `old`'s context: a coding instance's was chosen for its clients
    (`--for-agents`, `--ctx`); any other instance's is the profile's to pick again."""
    return old.modality is Modality.CODING


def choice(
    old: Recipe, plan: Plan, keep_quant: bool = True, pinned: Pinned | None = None
) -> Profile:
    """The profile a re-plan gives `old` from `plan` (`mnm why` explains this one): its own
    profile, at its own quant while `keep_quant` (its file, at the profile's sizing) or a
    quality candidate's. A coding instance's context was chosen for its clients
    (`--for-agents`, `--ctx`): it stays and the rest of the plan is sized around it; a
    quality candidate keeps the context and the KV cache it holds. PlanError: no fit."""
    quant = pinned.quant if pinned is not None else None
    if quant is None and keeps_quant(old, keep_quant):
        quant = old.quant
    held = pinned.held if pinned is not None else frozenset()
    ctx = _context(old) if keeps_context(old) or pinned is not None else None
    kv = old.args.kv_type if "kv_type" in held and isinstance(old.args, LlamaCppArgs) else None
    profile = plan.choose(old.profile, ctx=ctx, kv=kv)
    if quant is not None and profile.quant != quant:
        profile = plan.choose(old.profile, quant=quant, ctx=ctx, kv=kv)
    return profile


def replan(
    service: InstanceService,
    name: str,
    budget: Budget | None = None,
    keep_quant: bool = True,
    keep_if_fits: bool = False,
    pinned: Pinned | None = None,
    *,
    made: Plan | None = None,
) -> Replanned:
    """The instance planned again, always from scratch (not saved: `save` does that). With
    `keep_if_fits` (the CLI without `--shrink`), a plan that would only be smaller is withheld
    while the saved one still fits the budget the new plan was made from (calibration's
    correction included): a plan sits near thresholds, and a little less free VRAM must not
    move it back and forth. A plan that is not smaller, or a saved one that no longer fits,
    goes through as before. `pinned`: a quality tune's candidate (`Pinned`)."""
    old = service.get(name)
    held = pinned.held if pinned is not None else frozenset()
    try:
        if pinned is not None:
            plan = pinned.made
        elif made is not None:  # `mnm why`: the plan it explains, read from the Hub once
            plan = made
        else:
            plan = plan_again(service, old, budget, keep_quant)
        profile = choice(old, plan, keep_quant, pinned)
        fresh = plan.recipe(profile, old.name, old.port, upscaler=old.upscaler is not None)
    except PlanError as exc:
        raise InstanceError(f"{name} could not be planned again: {exc}") from exc
    if fresh.engine is not old.engine:
        raise InstanceError(
            f"{name} would move from {old.engine.value} to {fresh.engine.value}; `mnm rm` it "
            "and `mnm add` the model again for that."
        )
    if plan.image is None:  # A plan that did not read the box (`--vram`) keeps its own
        fresh = fresh.model_copy(update={"image": old.image})
    planned = {
        key: getattr(fresh.args, key)
        for key in PLANNED[old.engine]
        if (old.gpus is None or key not in CARD_FIELDS) and key not in held
    }
    planned |= kept_processor(old, fresh)
    if old.engine is Engine.LLAMACPP:
        planned |= kept_numa(old.llama_args, fresh.llama_args)
    args = type(old.args).model_validate(old.args.model_dump() | planned)
    if isinstance(args, LlamaCppArgs):
        args = _owners_experts(service, fresh, args)
    new = fresh.model_copy(
        update={
            "args": args,
            # the plan priced its own settings: with the user's kept, Fits now prices them
            "vram_estimate_bytes": fresh.vram_estimate_bytes
            if args == fresh.args and old.draft is None
            else None,
            "local_paths": _kept(old, fresh),  # kept files are not fetched again
            "loras": old.loras,
            "swap_group": old.swap_group,
            "draft": _draft_kept(old, fresh),
            "created_at": old.created_at,
            "display_name": old.display_name,
            "quality": _quality_kept(old, fresh.quant, args),
            **{key: getattr(old, key) for key in OWNER_FIELDS},  # never the plan's to change
        }
    )
    have = set(old.local_paths)
    missing = [f for f in new.own_files if f not in have] + [
        c.path for c in new.companions if c.key not in have
    ]
    smaller = _shrinks(old, new)
    # judged on the plan's own budget: the live one divided by calibration's correction
    moves = new.image != old.image  # An image for another backend is not kept
    level = (
        still_fits(service, old, plan.budget) if smaller and keep_if_fits and not moves else None
    )
    if level is not None:  # the plan's load mode and threads still reach it where unset
        kept = processor_only(old, new)
        return Replanned(
            old, kept, _changes(old, kept), [], withheld=Withheld(level, tuple(smaller))
        )
    new, note = _tune_kept(service, old, new, plan.budget)
    sizes = {f: n for f, n in file_sizes(plan.facts, new).items() if f in missing}
    return Replanned(old, new, _changes(old, new), missing, tune_note=note, sizes=sizes)


def _owners_experts(service: InstanceService, fresh: Recipe, args: LlamaCppArgs) -> LlamaCppArgs:
    """The user's `experts=ram|cards` on the new placement: placed again with that
    mode on the cards the plan placed it on. With no placement now (one card, a
    whole fit) or files that cannot be read, the plan's own placement and `experts` auto."""
    if args.experts in (None, "auto"):
        return args
    if args.placement is None:
        return args.model_copy(update={"experts": None})
    try:
        placed = placed_again(
            service, fresh.model_copy(update={"args": args}), {"experts": args.experts}
        )
    except SettingsError:
        return args.model_copy(update={"experts": None})
    return placed.llama_args


def _tune_kept(
    service: InstanceService, old: Recipe, new: Recipe, budget: Budget
) -> tuple[Recipe, str | None]:
    """A tuned recipe's tuned settings on the new plan while it still fits; else the
    plan alone, and the tune goes (the next idle card tunes it again)."""
    tuned = old.tuned
    if tuned is None:
        return new, None
    try:
        kept = with_tune(service, new, tuned)
    except (InstanceError, ValueError):
        kept = None
    if kept is not None and still_fits(service, kept, budget) is not None:
        return kept, f"{old.name}'s tune ({tuned.summary}) is kept: it still fits the new plan."
    return new, f"{old.name}'s tune ({tuned.summary}) no longer fits the new plan: dropped."


def _draft_kept(old: Recipe, fresh: Recipe) -> DraftModel | None:
    """The draft a re-plan keeps: the plan prices the model alone and Fits now adds the
    draft. A draft model always stays; an MTP head only while the new plan brings the same
    head file (the head follows the quant)."""
    draft = old.draft
    if draft is None or not draft.is_mtp or draft.file == fresh.mtp:
        return draft
    return None


def still_fits(service: InstanceService, recipe: Recipe, budget: Budget) -> FitLevel | None:
    """The saved settings' own estimate against `budget`: COMFORTABLE or TIGHT, else None (no
    fit, or no estimate: not downloaded, a file unreadable). Pass the plan's budget
    (`Plan.budget`): `make_plan` divides the free VRAM by calibration's correction there, so
    both plans are priced alike."""
    try:
        needed = service.raw_estimate(recipe)
    except (OSError, ValueError):
        return None
    if needed is None:
        return None
    level = headroom_level(needed, budget.vram_free_bytes)
    return level if level in (FitLevel.COMFORTABLE, FitLevel.TIGHT) else None


SHRINKING = ("ctx", "max_model_len", "gpu_layers", "width", "height", "frames")


def _shrinks(old: Recipe, new: Recipe) -> list[str]:
    """How the new plan is smaller: each planned size that falls (and a quant of fewer bits),
    only when none grows. A plan that trades one for another (a bigger quant at a shorter
    context, more context with fewer layers on the GPU) is not a shrink: it is taken. A plan
    for another backend's image is not one either: `-ngl 0` on the CPU is the plan."""
    if old.image != new.image:
        return []
    before, after = old.args.model_dump(), new.args.model_dump()
    falls: list[str] = []
    grows = False
    for key in SHRINKING:
        was, now = before.get(key), after.get(key)
        if isinstance(was, int) and isinstance(now, int) and was != now:
            if now < was:
                falls.append(f"{key} {was} → {now}")
            else:
                grows = True
    was_bits, now_bits = _bits(old.quant), _bits(new.quant)
    if was_bits is not None and now_bits is not None and was_bits != now_bits:
        if now_bits < was_bits:
            falls.append(f"quant {old.quant} → {new.quant}")
        else:
            grows = True
    return [] if grows else falls


def _quality_kept(old: Recipe, quant: str, args: object) -> QualityTuned | None:
    """The quality tune's record describes its quant at the KV cache it was scored
    with: a re-plan to another quant or KV cache drops it."""
    found = old.quality
    if found is None or found.quant != quant:
        return None
    return found if getattr(old.args, "kv_type", None) == getattr(args, "kv_type", None) else None


def _bits(quant: str) -> float | None:
    """The planner's own ranking of a quant label (`quant_ladder.quant_bits`), so a re-plan
    and the ladder never order two quants differently."""
    return quant_bits(quant) or None


def _context(recipe: Recipe) -> int | None:
    args = recipe.args
    if isinstance(args, LlamaCppArgs):
        return args.ctx
    return args.max_model_len if isinstance(args, VllmArgs) else None


def _kept(old: Recipe, fresh: Recipe) -> dict[str, str]:
    """The downloaded files the new recipe still runs (an old quant's file is no longer its
    own: `mnm rm --files` of another instance may then free it)."""
    wanted = {
        key
        for key, _repo, _path, _rev in wanted_files(fresh.model_copy(update={"loras": old.loras}))
    }
    return {key: path for key, path in old.local_paths.items() if key in wanted}


def _changes(old: Recipe, new: Recipe) -> list[str]:
    """What moved, one `setting: before -> after` each."""
    out: list[str] = []
    if old.image != new.image:  # The box's backend's tag, `server-cuda -> server-vulkan`
        out.append(f"image: {_tag(old.image)} -> {_tag(new.image)}")
    if old.quant != new.quant:
        out.append(f"quant: {old.quant} -> {new.quant}")
    before, after = old.args.model_dump(), new.args.model_dump()
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            out.append(f"{key}: {before.get(key)} -> {after.get(key)}")
    if [c.path for c in old.companions] != [c.path for c in new.companions]:
        out.append(f"companion files: {', '.join(c.path for c in new.companions) or 'none'}")
    if old.mtp != new.mtp:
        out.append(f"MTP head: {old.mtp or 'none'} -> {new.mtp or 'none'}")
    if old.draft is not None and new.draft is None:
        why = "another MTP head" if new.mtp else "no MTP head"
        out.append(f"draft: mtp -> none (the new quant brings {why})")
    if old.vram_estimate_bytes != new.vram_estimate_bytes and new.vram_estimate_bytes:
        out.append(
            f"planned GPU memory: {_gib(old.vram_estimate_bytes)} -> "
            f"{_gib(new.vram_estimate_bytes)}"
        )
    return out


def _tag(image: str) -> str:
    """`server-cuda` for `ghcr.io/ggml-org/llama.cpp:server-cuda`; a tagless name whole."""
    name, _, tag = image.rpartition(":")
    return tag if name and "/" not in tag else image


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 2**30:.1f} GiB"


def save(service: InstanceService, replanned: Replanned) -> Recipe:
    """Store the new recipe over the old one; the settings apply on the next start. A recipe
    changed since it was planned again (a `mnm set`, a LoRA) is refused, never overwritten;
    files a download recorded meanwhile are kept."""
    old, new = replanned.old, replanned.new
    same = {"local_paths"}
    here = planned_here(service.settings, service.memory())  # The box it is planned on

    def swap(current: Recipe) -> Recipe:
        if current.model_dump(exclude=same) != old.model_dump(exclude=same):
            raise InstanceError(
                f"{old.name} changed while it was planned again; run `mnm replan {old.name}` "
                "once more."
            )
        wanted = {key for key, _r, _p, _v in wanted_files(new)}
        paths = {k: v for k, v in current.local_paths.items() if k in wanted}
        return stamped(new.model_copy(update={"local_paths": paths}), here)

    return service.change(old.name, swap)


def planned_as_it_is(service: InstanceService, name: str) -> Recipe:
    """A re-plan that changed nothing (or kept a plan that still fits) judged it on this box:
    the recipe says so, and the doctor's Machine line stops counting it."""
    here = planned_here(service.settings, service.memory())
    if here is None:
        return service.get(name)
    return service.change(name, lambda current: stamped(current, here))
