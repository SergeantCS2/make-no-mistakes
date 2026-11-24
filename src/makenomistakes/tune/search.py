"""The search: staged coordinate descent from mnm's own plan.

The seed (mnm's plan, no changes) is trial 1. Each stage proposes candidates around the best
so far, in **directions** ordered nearest first; a crash ends its direction (an out-of-memory
start says the next step would not fit either). A stage's best replaces the best so far only
when it gains at least `MIN_GAIN` and more than the seed's readings spread; then the next
stage starts from it. llama.cpp's stages, in order:

  a. `cpu-moe-layers` ±1, ±2 (a mixture of experts with experts in RAM)
  b. `moe-cache` 0 / 10% / 25% of the VRAM the plan leaves (likewise)
  c. `ubatch`/`batch` 512, 1024, 2048, 4096 (up to the context)
  d. threads: the physical cores, cores - 2, one CCD's worth on a two-CCD CPU
  e. the KV cache f16 / q8_0 (never a step up that the fit says will not fit; never on a
     quality-tuned instance, whose quality tune measured its quant at its KV cache)
  f. the draft (an MTP head or a saved draft model) on or off; `draft-max` 3 / 8 when on
  g. `-ngl` ±1 block, only when the plan offloads dense blocks

The numa stage comes before the batches on a box of several NUMA nodes when the plan keeps
weights in RAM: `distribute`, `isolate` and `node:N` for the node with the most free memory,
one trial each; the threads stage then counts a bound node's cores. On a hybrid Intel CPU the
threads stage also tries the performance cores alone (`tune.cpus.CpuLayout.performance`).

The search is a pure function of the trials done (`Search.next`): a tune cut by an outside
request resumes by replaying them, never by a saved cursor. A per-card axis replaces
stage a for a MoE placed across several cards (`card_placement`): each card's expert blocks
±1, ±2 (`card-experts=`, the first out-of-memory start ends that card's direction; an x1
riser holds none), the rest staying in RAM; stage b then sizes the cache for those.

The backend stage comes first on an AMD card whose other llama.cpp build is pulled: one
trial on it (ROCm beside Vulkan), every later stage from the faster."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from makenomistakes.engine_args import KvType, LlamaCppArgs
from makenomistakes.hardware.probe import Gpu
from makenomistakes.recipes.tuned import Value
from makenomistakes.tune.cpus import CpuLayout
from makenomistakes.tune.score import judged_spread
from makenomistakes.tune.trial_types import Candidate, Outcome, TrialResult

MAX_TRIALS = 24
MIN_GAIN = 0.03  # a stage's best must beat the best so far by this much
UBATCHES = (512, 1024, 2048, 4096)
DRAFT_MAXES = (3, 8)
DEFAULT_UBATCH = 512  # llama.cpp's own when the recipe leaves it unset
DEFAULT_BATCH = 2048
MIB = 2**20

Direction = list[Candidate]


class StageName(StrEnum):
    BACKEND = "backend"  # llama.cpp's other build on an AMD card, when it is pulled
    EXPERTS = "cpu-moe-layers"
    CARD_EXPERTS = "card-experts"  # Expert blocks per card
    MOE_CACHE = "moe-cache"
    BATCH = "ubatch"
    THREADS = "threads"
    KV = "kv"
    DRAFT = "draft"
    LAYERS = "gpu-layers"
    NUMA = "numa"  # Several NUMA nodes, weights in RAM


@dataclass(frozen=True)
class SearchFacts:
    """What the stages read about the instance and the box, gathered once per tune."""

    args: LlamaCppArgs  # the plan's (the seed's)
    blocks: int | None = None  # the model's blocks, from its header
    moe: bool = False  # a mixture of experts (the header's expert count)
    vram_left_bytes: int | None = None  # the card less what the plan puts on it
    cpus: CpuLayout = field(default_factory=CpuLayout)
    draft: bool = False  # the plan drafts (a draft model or the MTP head)
    mtp: bool = False  # the repo's MTP head is downloaded (`Recipe.mtp`)
    fits: Callable[[Candidate], bool] = lambda _c: True  # the fit model's word on a candidate
    # The instance's backend and the others its card runs whose image is pulled here
    # (an AMD card with both llama.cpp builds): the backend stage tries each, one trial
    backend: str | None = None
    backends: tuple[str, ...] = ()
    # The box's NUMA nodes (index, physical cores) and the node `auto` would bind to
    # (the most free memory): the numa stage tries distribute, isolate and that node, and the
    # threads stage counts a bound node's cores
    nodes: tuple[tuple[int, int], ...] = ()
    roomiest: int | None = None
    in_ram: bool = False  # the plan keeps weights in RAM (experts, blocks, or the CPU backend)
    # A quality tune chose the quant at this KV cache: only it moves the cache
    kv_held: bool = False


@dataclass(frozen=True)
class Budget:
    trials: int = MAX_TRIALS
    seconds: float = 45 * 60.0


@dataclass(frozen=True)
class Next:
    candidate: Candidate


@dataclass(frozen=True)
class Finished:
    best: TrialResult | None  # None: the plan itself could not be measured
    reason: str
    seed: TrialResult | None = None


Step = Next | Finished


def axes_for(facts: SearchFacts, cards: list[Gpu]) -> list[StageName]:
    """The stages this instance is searched over, in order. One card: llama.cpp's own axes;
    several cards with a MoE placed across them: its expert blocks per card instead
    of `cpu-moe-layers`."""
    stages: list[StageName] = [StageName.BACKEND] if facts.backends else []
    placement = facts.args.placement
    if len(cards) > 1 and placement is not None and placement.in_ram:
        stages += [StageName.CARD_EXPERTS, StageName.MOE_CACHE]
    elif facts.moe and (facts.args.cpu_moe_layers or 0) > 0:
        stages += [StageName.EXPERTS, StageName.MOE_CACHE]
    if len(facts.nodes) > 1 and facts.in_ram:  # Before the threads it decides
        stages.append(StageName.NUMA)
    stages += [StageName.BATCH, StageName.THREADS]
    stages += [StageName.DRAFT] if facts.kv_held else [StageName.KV, StageName.DRAFT]
    if _dense_offload(facts):
        stages.append(StageName.LAYERS)
    return stages


def _dense_offload(facts: SearchFacts) -> bool:
    blocks = facts.blocks
    experts_in_ram = facts.moe and (facts.args.cpu_moe_layers or 0) > 0
    return blocks is not None and not experts_in_ram and facts.args.gpu_layers <= blocks


def _value(best: Candidate, facts: SearchFacts, name: str) -> Value:
    if name in best.changes:
        return best.changes[name]
    return _planned(facts, name)


def _planned(facts: SearchFacts, name: str) -> Value:
    if name == StageName.BACKEND.value:  # the image's, not an engine arg
        return facts.backend
    planned = getattr(facts.args, name)
    return planned.value if isinstance(planned, KvType) else planned


def directions(stage: StageName, best: Candidate, facts: SearchFacts) -> list[Direction]:
    """The stage's candidates around `best`, as directions nearest first."""
    return _STAGES[stage](best, facts)


def _one(best: Candidate, facts: SearchFacts, name: str, value: Value, stage: str) -> Candidate:
    return best.changed(name, value, _planned(facts, name), stage)


def _backends(best: Candidate, facts: SearchFacts) -> list[Direction]:
    """One trial on each other build (ROCm beside Vulkan: chosen per reading)."""
    now = _value(best, facts, StageName.BACKEND.value)
    stage = StageName.BACKEND.value
    return [[_one(best, facts, stage, b, stage)] for b in facts.backends if b != now]


def _experts(best: Candidate, facts: SearchFacts) -> list[Direction]:
    now = int(_value(best, facts, "cpu_moe_layers") or 0)
    top = facts.blocks or now + 2
    stage = StageName.EXPERTS.value
    fewer = [_one(best, facts, "cpu_moe_layers", now - d, stage) for d in (1, 2) if now - d >= 1]
    more = [_one(best, facts, "cpu_moe_layers", now + d, stage) for d in (1, 2) if now + d <= top]
    return [d for d in (fewer, more) if d]


def _card_experts(best: Candidate, facts: SearchFacts) -> list[Direction]:
    """Each card's expert blocks one and two more, then one and two fewer, nearest first;
    the blocks come from or go back to RAM (`fit.placement.refill`)."""
    placement = facts.args.placement
    if placement is None:
        return []
    planned = ",".join(str(n) for n in placement.counts)
    now = [int(n) for n in str(best.changes.get("placement", planned)).split(",")]
    most = sum(placement.counts) + len(placement.in_ram)
    stage = StageName.CARD_EXPERTS.value
    found: list[Direction] = []
    for k, card in enumerate(placement.cards):
        if card.lanes is not None and card.lanes <= 1:
            continue  # an x1 riser holds layers only
        for sign in (1, -1):
            direction = []
            for step in (1, 2):
                counts = list(now)
                counts[k] += sign * step
                if counts[k] < 0 or sum(counts) > most:
                    break
                shown = ",".join(str(n) for n in counts)
                direction.append(best.changed("placement", shown, planned, stage))
            if direction:
                found.append(direction)
    return found


def _moe_cache(best: Candidate, facts: SearchFacts) -> list[Direction]:
    left = facts.vram_left_bytes
    if not left or left <= 0:
        return []
    now = _value(best, facts, "moe_cache_mib") or 0
    stage = StageName.MOE_CACHE.value
    sizes = [int(left * share / MIB) for share in (0.10, 0.25)]
    grow = [_one(best, facts, "moe_cache_mib", s, stage) for s in sizes if s > 0 and s != now]
    off = [_one(best, facts, "moe_cache_mib", 0, stage)] if now else []
    return [d for d in (grow, off) if d]


def _batches(best: Candidate, facts: SearchFacts) -> list[Direction]:
    now = int(_value(best, facts, "ubatch") or DEFAULT_UBATCH)
    batch = int(_value(best, facts, "batch") or DEFAULT_BATCH)
    limit = facts.args.ctx
    stage = StageName.BATCH.value

    def with_ubatch(size: int) -> Candidate:
        moved = _one(best, facts, "ubatch", size, stage)  # the batch only when it must grow
        return _one(moved, facts, "batch", size, stage) if size > batch else moved

    up = [with_ubatch(u) for u in UBATCHES if now < u <= limit]
    down = [with_ubatch(u) for u in reversed(UBATCHES) if u < now]
    return [d for d in (up, down) if d]


def _numa(best: Candidate, facts: SearchFacts) -> list[Direction]:
    """One trial each: llama.cpp's two spreads, and the node `auto` would bind to (a node
    too small for the model is refused before a start, never tried)."""
    now = _value(best, facts, "numa")
    stage = StageName.NUMA.value
    options = ["distribute", "isolate"]
    if facts.roomiest is not None:
        options.append(f"node:{facts.roomiest}")
    return [[_one(best, facts, "numa", o, stage)] for o in options if o != now]


def _threads(best: Candidate, facts: SearchFacts) -> list[Direction]:
    now = _value(best, facts, "threads")
    stage = StageName.THREADS.value
    cpus = facts.cpus
    bound = str(_value(best, facts, "numa") or "")
    if bound.startswith("node:"):  # The bound node's cores, its CCDs shared out
        cores = dict(facts.nodes).get(int(bound.removeprefix("node:")), 0)
        groups = max(cpus.groups // max(len(facts.nodes), 1), 1)
        cpus = CpuLayout(cores, groups, cpus.performance)
    return [[_one(best, facts, "threads", n, stage)] for n in cpus.choices() if n != now]


def _kv(best: Candidate, facts: SearchFacts) -> list[Direction]:
    now = _value(best, facts, "kv_type")
    stage = StageName.KV.value
    options = [KvType.F16.value, KvType.Q8_0.value]
    if facts.args.flash_attn == "off":  # a quantised V cache needs flash attention
        options = [KvType.F16.value]
    found = []
    for kind in options:
        if kind == now:
            continue
        candidate = _one(best, facts, "kv_type", kind, stage)
        if kind == KvType.F16.value and not facts.fits(candidate):
            continue  # a step up in memory the fit says will not fit
        found.append([candidate])
    return found


def _drafts(best: Candidate, facts: SearchFacts) -> list[Direction]:
    stage = StageName.DRAFT.value
    on = (facts.draft and best.draft != "off") or best.draft == "mtp"
    if not on and not facts.mtp:
        return []
    if not on:  # no draft planned, an MTP head downloaded: try it, then its draft-max
        tried = best.drafting("mtp", stage)
        now = _value(best, facts, "draft_max")
        maxes = [_one(tried, facts, "draft_max", m, stage) for m in DRAFT_MAXES if m != now]
        return [[tried, *maxes]]
    now = _value(best, facts, "draft_max")
    found: list[Direction] = [[best.drafting("off" if facts.draft else None, stage)]]
    found += [[_one(best, facts, "draft_max", m, stage)] for m in DRAFT_MAXES if m != now]
    return found


def _layers(best: Candidate, facts: SearchFacts) -> list[Direction]:
    now = int(_value(best, facts, "gpu_layers") or 1)
    stage = StageName.LAYERS.value
    found = [[_one(best, facts, "gpu_layers", now + 1, stage)]]
    if now > 1:
        found.append([_one(best, facts, "gpu_layers", now - 1, stage)])
    return found


_STAGES: dict[StageName, Callable[[Candidate, SearchFacts], list[Direction]]] = {
    StageName.BACKEND: _backends,
    StageName.EXPERTS: _experts,
    StageName.CARD_EXPERTS: _card_experts,
    StageName.MOE_CACHE: _moe_cache,
    StageName.BATCH: _batches,
    StageName.THREADS: _threads,
    StageName.KV: _kv,
    StageName.DRAFT: _drafts,
    StageName.LAYERS: _layers,
    StageName.NUMA: _numa,
}


@dataclass
class Search:
    """The next trial, given the trials done in this tune."""

    facts: SearchFacts
    stages: list[StageName]
    budget: Budget = field(default_factory=Budget)

    def next(self, done: list[TrialResult]) -> Step:
        counted = [r for r in done if r.counted]
        spent = sum(r.seconds for r in done if r.counted)  # a cut or held trial costs none
        by_key: dict[str, list[TrialResult]] = {}
        for each in counted:
            by_key.setdefault(each.candidate.key, []).append(each)
        out_of_budget = len(counted) >= self.budget.trials or spent >= self.budget.seconds
        seed = Candidate()
        settled = _settled(by_key, seed)
        if settled is None:
            return (
                Finished(None, "out of budget before mnm's plan was measured")
                if (out_of_budget)
                else Next(seed)
            )
        if settled.outcome is not Outcome.MEASURED or settled.score is None:
            return Finished(None, f"mnm's plan itself was not measured: {settled.why}", settled)
        best: TrialResult = settled
        noise = judged_spread(settled.decode)
        for stage in self.stages:
            stage_best = best
            for direction in directions(stage, best.candidate, self.facts):
                for candidate in direction:
                    result = _settled(by_key, candidate)
                    if result is None:
                        if out_of_budget:
                            chosen = _adopted(stage_best, best, noise)
                            return Finished(
                                chosen, _budget_word(counted, spent, self.budget), settled
                            )
                        return Next(candidate)
                    if result.outcome is Outcome.CRASHED:
                        break  # the next step this way would not start either
                    if _better(result, stage_best):
                        stage_best = result
            best = _adopted(stage_best, best, noise)
        return Finished(best, "every stage searched", settled)


def _settled(by_key: dict[str, list[TrialResult]], candidate: Candidate) -> TrialResult | None:
    """The candidate's result, or None when it still needs a trial: never run, or once not
    alone (repeated once; twice not alone, it stands as unmeasured)."""
    results = by_key.get(candidate.key, [])
    if not results:
        return None
    last = results[-1]
    if last.outcome is Outcome.NOISY and len(results) < 2:
        return None
    return last


def _better(result: TrialResult, than: TrialResult) -> bool:
    if result.outcome is not Outcome.MEASURED or result.score is None or than.score is None:
        return False
    return result.score.value > than.score.value


def _adopted(stage_best: TrialResult, best: TrialResult, noise: float) -> TrialResult:
    """The stage's best when it gains enough over the best so far, else the best so far."""
    if stage_best is best or stage_best.score is None or best.score is None:
        return best
    if best.score.value <= 0:
        return stage_best
    gain = stage_best.score.value / best.score.value - 1
    return stage_best if gain >= max(MIN_GAIN, noise) else best


def _budget_word(counted: list[TrialResult], spent: float, budget: Budget) -> str:
    if len(counted) >= budget.trials:
        return f"the budget of {budget.trials} trials is spent"
    return f"the budget of {budget.seconds / 60:.0f} minutes is spent ({spent / 60:.0f} used)"
