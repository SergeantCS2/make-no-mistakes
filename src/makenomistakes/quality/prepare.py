"""Before the card: what a quality tune would measure, the downloads and the
disk it needs, and, once the user says yes, the `full` image pulled and the candidates
fetched (`quality.approve` then queues the plan and asks the router).

The planner reads the Hub once (`make_plan` on the budget `mnm add` plans against: mnm's
running models counted free, as each trial parks them), prices every quant of the repo at the
instance's own context (serves) and at `MEASURE_CTX` (can be measured), then each candidate's
serving recipe is `replan`'s, pinned to its quant, with the user's settings kept as a re-plan
keeps them. The router later runs those recipes as they are."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from makenomistakes.busy_card import held_elsewhere
from makenomistakes.engine_images import NoImage
from makenomistakes.fit.profiles import plannable
from makenomistakes.hub.facts import Modality
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.measurements import settings_key
from makenomistakes.plan import make_plan
from makenomistakes.plan_budget import budget_now
from makenomistakes.plan_types import Plan, PlanError
from makenomistakes.quality import QualityPlanError
from makenomistakes.quality.cache import found_here
from makenomistakes.quality.perplexity import quality_image
from makenomistakes.quality.plan import (
    HELD,
    MEASURE_CTX,
    QualityCandidate,
    QualityPlan,
    QuantFact,
    Use,
    named_candidates,
    pick_candidates,
    pick_reference,
)
from makenomistakes.quality.texts import TextError, checked, text_for
from makenomistakes.recipes import Recipe
from makenomistakes.replan import Pinned, reference, replan
from makenomistakes.supervisor import SupervisorError
from makenomistakes.tune.session import untunable

GIB = 2**30
ROOM_MARGIN = 2 * GIB  # left free on the models disk after the downloads and the base file
SCORED = 255  # tokens a 512-token chunk scores (its second half, less the last)
Progress = Callable[[str], None]
# Before each candidate's download: its recipe with the files found here, else None once the
# download is claimed as a tune's own (`quality.approve`).
Claim = Callable[[QualityCandidate], Recipe | None]


@dataclass(frozen=True)
class Offer:
    """What the user is asked to approve."""

    plan: QualityPlan
    facts: list[QuantFact]
    image: str
    image_here: bool
    fetch_bytes: int  # the candidates' files not on disk yet
    base_bytes: int  # the reference's log-probabilities while the tune runs
    free_bytes: int | None  # the models disk's free space now
    on_disk: frozenset[str]  # the quants whose files are here already (shown, never trusted)
    busy: str | None = None  # the card held outside mnm while it was planned, in words

    @property
    def needs_bytes(self) -> int:
        return self.fetch_bytes + self.base_bytes

    @property
    def room_short(self) -> int:
        """Bytes the disk lacks (0 when it holds the tune with `ROOM_MARGIN` to spare)."""
        if self.free_bytes is None:
            return 0
        return max(self.needs_bytes + ROOM_MARGIN - self.free_bytes, 0)

    @property
    def short_words(self) -> str:
        """The refusal when the disk cannot hold the tune (one wording, CLI, web and fetch)."""
        return (
            f"the models disk lacks {self.room_short / GIB:.1f} GiB for this tune (it needs "
            f"{self.needs_bytes / GIB:.1f} GiB and keeps {ROOM_MARGIN / GIB:.0f} GiB free): "
            "`mnm disk` shows what holds it, `--quants` measures fewer"
        )


def base_bytes(vocab: int | None, chunks: int) -> int:
    """The base run's file: the tokens, then per chunk 255 scored tokens of the vocabulary's
    log-probabilities as 2-byte numbers, rounded to an even count plus 4 (llama-perplexity's
    layout at b12359; 1.86 GB for Qwen3's 151,936 tokens at 24 chunks, measured)."""
    if not vocab:
        return 0
    per_token = (2 * ((vocab + 1) // 2) + 4) * 2
    return 8 + chunks * 512 * 4 + chunks * SCORED * per_token


def use_of(recipe: Recipe) -> Use:
    return "agents" if recipe.modality is Modality.CODING else "chat"


def floor_for(service: InstanceService, use: Use) -> float:
    settings = service.settings
    return settings.floor_agent if use == "agents" else settings.floor_chat


def offer(
    service: InstanceService,
    name: str,
    floor: float | None = None,
    quants: Sequence[str] = (),
    keep: bool = False,
) -> Offer:
    """The plan for `name`'s quality tune, nothing fetched; QualityPlanError says why not."""
    recipe = service.get(name)
    if (why := untunable(recipe)) is not None:
        raise QualityPlanError(f"{name}: {why}")
    image = quality_image(recipe)
    if isinstance(image, NoImage):
        raise QualityPlanError(f"{name} cannot be measured here: {image.words}")
    text = text_for(recipe.modality)
    try:
        checked(text)
    except TextError as exc:
        raise QualityPlanError(str(exc)) from exc
    use = use_of(recipe)
    floor = floor or floor_for(service, use)
    model_type = recipe.modality if recipe.modality in (Modality.CODING, Modality.CHAT) else None
    try:
        made = make_plan(
            reference(recipe), service.settings, budget_now(service), model_type=model_type
        )
    except PlanError as exc:
        raise QualityPlanError(f"{name} could not be planned: {exc}") from exc
    facts = quant_facts(made, recipe)
    ref = pick_reference(facts, recipe.quant)
    if ref is None:
        raise QualityPlanError(
            f"no quant of {recipe.repo_id} runs on this box for the measurement (four "
            "512-token sequences): there is nothing to score the others against"
        )
    try:
        chosen = (
            named_candidates(facts, quants, recipe.quant, ref)
            if quants
            else pick_candidates(facts, recipe.quant, ref, floor)
        )
    except ValueError as exc:
        raise QualityPlanError(str(exc)) from exc
    if not chosen:
        raise QualityPlanError(
            f"no quant of {recipe.repo_id} serves on this box at {name}'s context now "
            f"({recipe.llama_args.ctx:,} tokens)"
        )
    candidates = [_candidate(service, made, recipe, f) for f in chosen]
    reference_row = next((c for c in candidates if c.quant == ref.quant), None)
    if reference_row is None:  # measured only: its files may be here all the same
        measuring = _measuring(made, recipe, ref)
        reference_row = measuring.model_copy(
            update={"recipe": found_here(service.models_dir, measuring.recipe)}
        )
    if len({reference_row.quant, *(c.quant for c in candidates)}) < 2:
        raise QualityPlanError(
            f"{name} already runs {recipe.quant}, the reference: no quant here is more "
            "faithful, so there is nothing to move to (`--quants Q6_K,Q4_K_M` scores smaller "
            "ones beside it)"
        )
    plan = QualityPlan(
        repo_id=recipe.repo_id,
        current=recipe.quant,
        settings=settings_key(recipe),
        reference=reference_row,
        candidates=candidates,
        floor=floor,
        use=use,
        text=text.name,
        keep=keep,
    )
    measured = plan.measured
    here = frozenset(c.quant for c in measured if service.files_present(c.recipe))
    fetch = sum(c.size_bytes or 0 for c in measured if c.quant not in here)
    free = _free(service)
    vocab = made.header.vocab_size if made.header is not None else None
    return Offer(
        plan=plan,
        facts=facts,
        image=image,
        image_here=service.build_of(image) is not None,
        fetch_bytes=fetch,
        base_bytes=base_bytes(vocab, plan.chunks),
        free_bytes=free,
        busy=held_elsewhere(made.budget),
        on_disk=here,
    )


def quant_facts(made: Plan, recipe: Recipe) -> list[QuantFact]:
    """What the planner says of each quant of the repo on this budget, with the instance's
    own KV cache (`HELD`): whether it serves at the instance's context (and its tok/s there),
    and whether it can be measured. The current quant serves whatever the planner says now:
    the instance runs it."""
    ctx, kv = recipe.llama_args.ctx, recipe.llama_args.kv_type
    found = []
    for variant in made.facts.gguf_variants:
        if not plannable(variant):
            continue
        decode: float | None = None
        try:
            serving = made.choose(recipe.profile, quant=variant.quant, ctx=ctx, kv=kv)
            decode = serving.tokens_per_second or None
            serves = True
        except PlanError:
            serves = variant.quant == recipe.quant
        try:
            made.choose(recipe.profile, quant=variant.quant, ctx=min(MEASURE_CTX, ctx), kv=kv)
            measures = True
        except PlanError:
            measures = False
        found.append(
            QuantFact(
                quant=variant.quant,
                size_bytes=variant.size_bytes,
                serves=serves,
                measures=measures or serves,
                decode=decode,
            )
        )
    return found


def _candidate(
    service: InstanceService, made: Plan, recipe: Recipe, fact: QuantFact
) -> QualityCandidate:
    """The current quant: the saved recipe itself, so keeping it keeps exactly what was
    measured. Another: its serving recipe as a re-plan pinned to it saves it (the user's
    settings and `HELD` kept), with the files already on disk found."""
    if fact.quant == recipe.quant:
        return QualityCandidate(quant=fact.quant, recipe=recipe, size_bytes=fact.size_bytes)
    try:
        done = replan(service, recipe.name, pinned=Pinned(fact.quant, made, HELD))
    except InstanceError as exc:
        raise QualityPlanError(f"{fact.quant}: {exc}") from exc
    found = found_here(service.models_dir, done.new)
    return QualityCandidate(quant=fact.quant, recipe=found, size_bytes=fact.size_bytes)


def _measuring(made: Plan, recipe: Recipe, fact: QuantFact) -> QualityCandidate:
    """A reference that does not serve here: its plan for `MEASURE_CTX` tokens, only ever
    measured (`QualityCandidate.serves` False)."""
    ctx = min(MEASURE_CTX, recipe.llama_args.ctx)
    try:
        profile = made.choose(
            recipe.profile, quant=fact.quant, ctx=ctx, kv=recipe.llama_args.kv_type
        )
        built = made.recipe(profile, recipe.name, recipe.port)
    except PlanError as exc:
        raise QualityPlanError(f"the reference {fact.quant} could not be planned: {exc}") from exc
    if made.image is None:
        built = built.model_copy(update={"image": recipe.image})
    return QualityCandidate(
        quant=fact.quant, recipe=built, size_bytes=fact.size_bytes, serves=False
    )


def _free(service: InstanceService) -> int | None:
    try:
        return shutil.disk_usage(service.models_dir).free
    except OSError:
        return None


def fetch(
    service: InstanceService, offered: Offer, progress: Progress | None, claim: Claim
) -> QualityPlan:
    """The approved plan made runnable: the `full` image pulled when absent, then each
    candidate asked of `claim` (its files found here, or the download claimed) and fetched
    when it is not here; the plan with their paths."""
    if offered.room_short:
        raise QualityPlanError(offered.short_words)
    try:
        service.supervisor.ensure_image(offered.image, progress)
    except SupervisorError as exc:
        raise QualityPlanError(f"{offered.image} could not be pulled: {exc}") from exc
    plan = offered.plan
    gone = (  # the approval ends its plan on this failure (`quality.approve`)
        "What the quality tune downloaded is deleted with its plan; `mnm tune "
        f"{plan.instance} --quality` plans and downloads it again"
    )
    here: dict[str, QualityCandidate] = {}
    for row in plan.measured:
        found = claim(row)
        if found is not None:
            here[row.quant] = row.model_copy(update={"recipe": found})
            continue
        if progress:
            progress(f"Downloading {row.quant} for the quality tune")
        got = service.fetch_files(row.recipe, progress, after=gone)
        here[row.quant] = row.model_copy(update={"recipe": got})
    return plan.model_copy(
        update={
            "reference": here.get(plan.reference.quant, plan.reference),
            "candidates": [here.get(c.quant, c) for c in plan.candidates],
        }
    )
