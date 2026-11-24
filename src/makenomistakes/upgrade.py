"""`mnm upgrade <name>`: a newer family member planned with the instance's settings kept
where they apply, shown as a diff against the instance planned the same way now. Nothing is
saved here: `upgrade_apply.py` adds it beside the old one.

Kept: the type (a chat model stays chat, a coding one coding), the profile, the context and
KV precision (a context above the new model's is capped; one that does not fit falls back to
the profile's, said), the user's settings that are not the model's (`KEPT`: slots, threads,
idle, pinning, the cards it runs on, a seed, a negative prompt, a language…), LoRAs on the
same picture family, the swap group. Dropped and said: a draft (another model's vocabulary
or head: set again once it is downloaded), a tune (it was the old model's), every other
setting the user changed that the new plan decides itself."""

from __future__ import annotations

from dataclasses import dataclass, field

from makenomistakes.engine_args import KvType, LlamaCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.fit import Budget, FitLevel
from makenomistakes.fit.profiles import Profile
from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.plan import make_plan
from makenomistakes.plan_types import Plan, PlanError
from makenomistakes.recipes.model import LoraFile, Recipe
from makenomistakes.replan import PLANNED, QUANT_PICKS, reference
from makenomistakes.successors import Lookup, NoSuccessor, Successor, find

# The user's settings an upgrade carries to the new model, per engine; the rest of the
# args are the new plan's (`PLANNED` sizes them) or belong to the old model.
KEPT: dict[Engine, tuple[str, ...]] = {
    Engine.LLAMACPP: (
        "parallel",
        "threads",
        "reasoning_format",
        "jinja",
        "sleep_idle_seconds",
        "load_mode",
        "draft_max",
        "draft_p_min",
        "experts",
        "split_mode",
        "extra",
    ),
    Engine.SDCPP: ("seed", "negative_prompt", "threads", "client_size", "strength", "extra"),
    Engine.WHISPERCPP: ("language", "threads", "convert", "extra"),
    Engine.AUDIOCPP: ("seed", "threads"),
    Engine.VLLM: ("max_num_seqs", "tensor_parallel", "extra"),
}
_TOP_KEPT = ("idle", "keep", "gpus", "off", "swap_group")
_SIZED = frozenset({Modality.CHAT, Modality.CODING, Modality.VISION_CHAT})


@dataclass(frozen=True)
class Side:
    """One model as the diff shows it."""

    label: str
    quant: str
    settings: str  # `ctx 16,384, KV q8_0`
    memory: str  # `5.1 GiB`, or `… on the card + … in RAM`
    fit: str
    speed: str


@dataclass(frozen=True)
class Upgrade:
    old: Recipe
    successor: Successor
    new: Recipe  # the new instance under the old one's name (its port is the store's)
    before: Side | None  # None: the old one could not be planned again (said in `notes`)
    after: Side
    download_bytes: int
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def old_name(self) -> str:
        return f"{self.old.name}-old"


def plan_upgrade(
    service: InstanceService, name: str, budget: Budget | None, lookup: Lookup | None
) -> Upgrade | NoSuccessor:
    """The successor planned for this machine now; NoSuccessor when none is known. A plan
    the planner refuses is an InstanceError with the planner's words (the RAM floor's
    arithmetic, a busy card)."""
    old = service.get(name)
    found = find(old, lookup)
    if isinstance(found, NoSuccessor):
        return found
    model_type = old.modality if old.modality in (Modality.CODING, Modality.CHAT) else None
    notes: list[str] = []
    try:
        plan = make_plan(
            found.repo, service.settings, budget, model_type=model_type, quant=found.quant
        )
    except PlanError as exc:
        raise InstanceError(f"{found.target} could not be planned for {name}: {exc}") from exc
    if plan.facts.engine is not old.engine:
        raise InstanceError(
            f"{found.target} runs on {_engine(plan)}, {name} on {old.engine.value}: an upgrade "
            f"keeps the engine. `mnm add {found.repo}` adds it as a new instance."
        )
    profile = _profile(plan, old, found, notes)
    fresh = plan.recipe(profile, name, old.port, upscaler=old.upscaler is not None)
    new, kept, dropped = carried(old, fresh)
    return Upgrade(
        old=old,
        successor=found,
        new=new,
        before=_before(service, old, plan.budget, notes),
        after=side(found.label, plan, profile),
        download_bytes=profile.download_bytes,
        kept=kept,
        dropped=dropped,
        notes=notes,
    )


def _engine(plan: Plan) -> str:
    return plan.facts.engine.value if plan.facts.engine is not None else "no engine mnm runs"


def _context(recipe: Recipe) -> int | None:
    args = recipe.args
    if isinstance(args, LlamaCppArgs):
        return args.ctx
    return args.max_model_len if isinstance(args, VllmArgs) else None


def _kv(recipe: Recipe) -> KvType | None:
    return recipe.args.kv_type if isinstance(recipe.args, LlamaCppArgs) else None


def _profile(plan: Plan, old: Recipe, found: Successor, notes: list[str]) -> Profile:
    """The old profile, at the old context and KV where the new model takes them: capped at
    its trained context; without the KV, then at the profile's own context when not."""
    quant = found.quant if found.quant is not None else None
    if old.modality not in _SIZED or (ctx := _context(old)) is None:
        return _choose(plan, old, quant=quant)
    native = plan.header.context_length if plan.header is not None else None
    if native and ctx > native:
        notes.append(f"context {ctx:,} is above {found.label}'s {native:,}: capped at {native:,}")
        ctx = native
    kv = _kv(old)
    tries: list[tuple[int, KvType | None]] = [(ctx, kv)] if kv is not None else []
    tries.append((ctx, None))
    chosen = _choose(plan, old, quant=quant)
    for want_ctx, want_kv in tries:
        try:
            kept = plan.choose(old.profile, quant=quant, ctx=want_ctx, kv=want_kv)
        except PlanError:
            continue
        # the old context is no reason to push the new model's layers into RAM
        if kept.fit.level is not FitLevel.OFFLOAD or chosen.fit.level is FitLevel.OFFLOAD:
            return kept
    notes.append(
        f"context {ctx:,} does not fit beside {found.label} on this card: the "
        f"{old.profile.value} profile's {chosen.settings_summary}"
    )
    return chosen


def _choose(plan: Plan, old: Recipe, quant: str | None) -> Profile:
    try:
        return plan.choose(old.profile, quant=quant)
    except PlanError as exc:
        raise InstanceError(f"{plan.facts.repo_id} does not fit for {old.name}: {exc}") from exc


def side(label: str, plan: Plan, profile: Profile) -> Side:
    return Side(
        label=label,
        quant=profile.quant,
        settings=profile.settings_summary,
        memory=profile.vram_words,
        fit=profile.fit.level.value,
        speed=profile.speed_summary,
    )


def _before(service: InstanceService, old: Recipe, budget: Budget, notes: list[str]) -> Side | None:
    """The instance planned again the same way (its quant, context and KV), so both sides are
    estimates on the same budget; None when it cannot be (its repo gone, a refusal)."""
    model_type = old.modality if old.modality in (Modality.CODING, Modality.CHAT) else None
    quant = old.quant if old.engine in QUANT_PICKS else None
    try:
        plan = make_plan(
            reference(old), service.settings, budget, model_type=model_type, quant=quant
        )
        ctx = _context(old) if old.modality in _SIZED else None
        profile = plan.choose(old.profile, quant=quant, ctx=ctx, kv=_kv(old) if ctx else None)
    except (PlanError, InstanceError) as exc:
        notes.append(f"{old.name} could not be planned again for the comparison: {exc}")
        return None
    return side(old.display_name or old.name, plan, profile)


def carried(old: Recipe, fresh: Recipe) -> tuple[Recipe, list[str], list[str]]:
    """The new recipe with what carries over from the old one; the words for what was kept
    and what was dropped."""
    kept: list[str] = []
    dropped: list[str] = []
    before, after = old.args.model_dump(), fresh.args.model_dump()
    owner = {k: before[k] for k in KEPT[old.engine] if k in before and before[k] != after.get(k)}
    args = fresh.args
    if owner:
        try:
            args = type(fresh.args).model_validate(after | owner)
            kept += [f"{k} {_value(v)}" for k, v in owner.items()]
        except ValueError:
            dropped += [f"{k} {_value(v)} (the new model refuses it)" for k, v in owner.items()]
    sized = set(PLANNED[old.engine]) | set(KEPT[old.engine]) | {"engine"}
    for key in sorted(set(before) - sized):
        if before[key] != after.get(key) and before[key] not in (None, [], ""):
            dropped.append(
                f"{key} {_value(before[key])} (the new plan's: {_value(after.get(key))})"
            )
    update: dict[str, object] = {"args": args, "vram_estimate_bytes": None}
    if args == fresh.args:
        update["vram_estimate_bytes"] = fresh.vram_estimate_bytes
    for key in _TOP_KEPT:
        value = getattr(old, key)
        if value != getattr(fresh, key):
            update[key] = value
            kept.append(_top_words(key, value))
    loras, words = _loras(old, fresh)
    update["loras"] = loras  # their files are on disk already: no download for them
    update["local_paths"] = {
        lora.key: old.local_paths[lora.key] for lora in loras if lora.key in old.local_paths
    }
    kept += words[0]
    dropped += words[1]
    if old.draft is not None:
        what = "head" if old.draft.is_mtp else "vocabulary"
        dropped.append(
            f"draft={old.draft.setting} (another model's {what}: `mnm set {old.name} draft=…` "
            "again once the new model is downloaded)"
        )
    if old.tuned is not None:
        dropped.append(
            f"the tune ({old.tuned.summary}): it was the old model's; the tuner "
            "tunes the new one on an idle card"
        )
    return fresh.model_copy(update=update), kept, dropped


def _loras(old: Recipe, fresh: Recipe) -> tuple[list[LoraFile], tuple[list[str], list[str]]]:
    if not old.loras:
        return [], ([], [])
    names = ", ".join(lora.name for lora in old.loras)
    if old.image_family == fresh.image_family:
        return list(old.loras), ([f"LoRAs {names}"], [])
    return [], (
        [],
        [f"LoRAs {names} (made for {old.image_family.value}, not {fresh.image_family.value})"],
    )


def _top_words(key: str, value: object) -> str:
    match key:
        case "keep":
            return "pinned (keep=always)" if value == "always" else f"keep={value}"
        case "idle":
            return f"idle {value} s"
        case "gpus":
            return f"cards {','.join(map(str, value))}" if isinstance(value, tuple) else "cards"
        case "off":
            return "stopped (off) until started" if value else "loadable on request"
        case _:
            return f"swap group {value}"


def _value(value: object) -> str:
    if isinstance(value, list | tuple):
        return " ".join(map(str, value)) or "none"
    return str(value)
