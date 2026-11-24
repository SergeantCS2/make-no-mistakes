"""Turn model facts + hardware into three named, explained profiles.

Quality: the highest-quality quant that still fits with a useful context, f16 KV cache.
Balanced: a 4-bit quant with as much context as fits, KV cache at 8-bit.
Speed: the lightest 4-bit quant at 8k context with an f16 KV cache: the least work per
token, fully on the GPU.

Embedding and rerank models ignore the ladder: they get their native context (capped at 8k)
and an f16 cache, and the engine's batch sizes are set to the context so one long input is
processed in a single step (non-causal models cannot split a prompt across micro-batches).

Everywhere, a comfortable fit beats a tight one, and a tight one beats offloading:
the numbers are estimates, so headroom is worth more than the last few k of context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from makenomistakes.card_placement import Placement
from makenomistakes.engine_args import (
    MIN_CTX,
    AudioCppArgs,
    KvType,
    LlamaCppArgs,
    SdCppArgs,
)
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.engine_args_whisper import WhisperArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.cards import main_gpu, tensor_split
from makenomistakes.fit.estimate import (
    DEFAULT_UBATCH,
    GIB,
    Fit,
    FitLevel,
    MemoryEstimate,
    estimate_memory,
    estimate_tokens_per_second,
    every_expert_in_ram,
    fit_estimate,
    per_card,
)
from makenomistakes.fit.placement import Mode, Placed, place_experts
from makenomistakes.fit.plan_floor import check_floor, check_pick, floor_at, holds
from makenomistakes.fit.quant_ladder import (
    BALANCED_PREFERENCE,
    SPEED_PREFERENCE,
    below,
    preferred,
    quant_bits,
)
from makenomistakes.fit.ram_floor import Floor
from makenomistakes.gguf.reader import GgufHeader
from makenomistakes.hub.facts import EMBEDDING_MODALITIES, ModelFacts
from makenomistakes.hub.variants import GgufVariant

CTX_LADDER = (131072, 65536, 32768, 16384, 8192, 4096)
MIN_USEFUL_CTX = 8192
SPEED_CTX = 8192
QUALITY_MAX_CTX = 65536
_PREFERENCE = (FitLevel.COMFORTABLE, FitLevel.TIGHT, FitLevel.OFFLOAD)


class ProfileName(StrEnum):
    QUALITY = "quality"
    BALANCED = "balanced"
    SPEED = "speed"
    CUSTOM = "custom"


# Longest input per embedding/rerank profile. The whole input is one micro-batch, and a
# decoder model computes logits for every token of it (`compute_buffer_bytes`): 8192 tokens
# of Qwen3-Embedding is a 4.6 GiB buffer, 2048 tokens 1.2 GiB. Odysseus cuts each input at
# 900 characters and RAG chunks are a few hundred tokens, so balanced covers them all.
EMBEDDING_CTX = {ProfileName.QUALITY: 8192, ProfileName.BALANCED: 2048, ProfileName.SPEED: 1024}


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a renamed field fails loudly, never vanishes

    name: ProfileName
    variant: GgufVariant
    ctx: int = 0  # llama.cpp profiles
    kv_type: KvType = KvType.F16
    ubatch: int | None = None  # set for embedding/rerank: the whole input in one step
    # Embedding and rerank read each input in one batched pass (ubatch = ctx) and decode
    # nothing, so a decode t/s says nothing about them.
    whole_input: bool = False
    # sd.cpp and whisper.cpp profiles: exactly what the recipe will run (llama.cpp's are
    # built from ctx, kv_type and ubatch above).
    engine_args: SdCppArgs | WhisperArgs | AudioCppArgs | VllmArgs | None = None
    seconds_per_image: float = 0.0  # rough, compute-bound, uncalibrated
    kit_bytes: int = 0  # every companion file an image kit downloads
    # llama.cpp across several cards: `-ts` and `-mg` from the budget's cards
    tensor_split: tuple[int, ...] | None = None
    main_gpu: int | None = None
    # A MoE that does not fit whole on several cards, its blocks and experts per card
    placement: Placement | None = None
    estimate: MemoryEstimate
    fit: Fit
    tokens_per_second: float = 0.0
    why: str

    @property
    def quant(self) -> str:
        return self.variant.quant

    @property
    def download_bytes(self) -> int:
        """The files `add` fetches: the variant's own (whatever part of them runs on the GPU)
        plus the projector, the MTP head and the kit's companions."""
        own = self.variant.size_bytes + self.variant.mtp_bytes
        return own + self.estimate.projector_bytes + self.kit_bytes

    @property
    def image(self) -> SdCppArgs | None:
        return self.engine_args if isinstance(self.engine_args, SdCppArgs) else None

    @property
    def audio(self) -> WhisperArgs | None:
        return self.engine_args if isinstance(self.engine_args, WhisperArgs) else None

    @property
    def vllm(self) -> VllmArgs | None:
        return self.engine_args if isinstance(self.engine_args, VllmArgs) else None

    @property
    def voice(self) -> AudioCppArgs | None:
        return self.engine_args if isinstance(self.engine_args, AudioCppArgs) else None

    @property
    def ram_bytes(self) -> int:
        """What an offload keeps in RAM (layers, or a MoE's experts); 0 when it all goes on
        the card, or nothing fits."""
        total, card = self.estimate.total_bytes, self.fit.gpu_bytes
        if self.fit.level is FitLevel.NO_FIT or not 0 < card < total:
            return 0
        return total - card

    @property
    def vram_words(self) -> str:
        """`8.4 GiB on the card + 15.3 GiB in RAM` for an offload, else the whole estimate:
        a "VRAM" column is read as the card's, and a MoE plan's whole estimate can be
        23.7 GiB on a 12 GiB card."""
        if not self.ram_bytes:
            return f"{self.estimate.total_bytes / GIB:.1f} GiB"
        card = self.fit.gpu_bytes
        return f"{card / GIB:.1f} GiB on the card + {self.ram_bytes / GIB:.1f} GiB in RAM"

    @property
    def settings_summary(self) -> str:
        """One cell for tables: what the profile decided besides the quant."""
        if self.vllm is not None:
            return self.vllm.plan_summary
        if self.engine_args is not None:
            return self.engine_args.summary
        return f"ctx {self.ctx:,}, KV {self.kv_type.value}"

    @property
    def speed_summary(self) -> str:
        if self.image is not None:
            unit = "clip" if self.image.frames else "image"
            return f"~{self.seconds_per_image:.0f} s/{unit}"
        if self.audio is not None:
            return "faster than real time"
        if self.voice is not None and self.voice.music:  # Work per second of audio
            return f"~{1 / self.tokens_per_second:.2g} s per second of audio"
        if self.voice is not None:
            return f"~{self.tokens_per_second:.0f}x realtime"
        if self.whole_input:
            return "one pass per input"
        return f"~{self.tokens_per_second:.0f} t/s"


def plan_profiles(
    facts: ModelFacts, header: GgufHeader, budget: Budget
) -> dict[ProfileName, Profile]:
    variants = [v for v in facts.gguf_variants if plannable(v)]
    if not variants:
        return {}
    if not header.block_count:  # said as its own reason, never as "no recognisable quant"
        raise ValueError("the GGUF header has no block count, so the KV cache cannot be sized")
    planner = Planner(header, budget, facts.projector_bytes)
    model_ctx = header.context_length or facts.context_length or 32768
    if model_ctx < MIN_CTX:  # LlamaCppArgs' floor: a recipe could not hold it
        raise ValueError(
            f"its native context is {model_ctx} tokens, below the {MIN_CTX} this tool plans "
            "llama-server with; it is too small a model to be useful here"
        )
    by_quality = sorted(variants, key=lambda v: (quant_bits(v.quant), v.size_bytes), reverse=True)
    if facts.modality in EMBEDDING_MODALITIES:
        return _embedding_profiles(planner, variants, by_quality, model_ctx)
    ladder = [c for c in CTX_LADDER if c <= model_ctx] or [model_ctx]
    check_floor(planner, facts, variants, ladder[-1])

    quality_ctxs = [c for c in ladder if MIN_USEFUL_CTX <= c <= QUALITY_MAX_CTX] or ladder
    balanced = planner.place_or_step(
        ProfileName.BALANCED,
        variants,
        preferred(variants, BALANCED_PREFERENCE),
        ladder,
        KvType.Q8_0,
        "4-bit weights with as much context as fits; KV cache at 8-bit.",
    )
    # Quality never goes below balanced's quant: on a busy card the highest quant that fit
    # was a 1-bit build (UD-IQ1_M for Qwen3-8B and for Qwen3-30B-A3B). When no
    # quant at least as good fits comfortably, quality runs balanced's quant offloaded.
    floor = quant_bits(balanced.quant)
    candidates = [v for v in by_quality if quant_bits(v.quant) >= floor]
    best = planner.best_fitting(candidates, quality_ctxs, KvType.F16)
    quality_variant = best or balanced.variant
    speed_ctxs = [c for c in ladder if c <= SPEED_CTX] or ladder[-1:]

    return {
        ProfileName.QUALITY: planner.place(
            ProfileName.QUALITY,
            quality_variant,
            quality_ctxs,
            KvType.F16,
            "Highest-quality quant that fits comfortably with an unquantised KV cache."
            if best is not None
            else "No quant fits comfortably: the balanced quant with an unquantised KV cache.",
        ),
        ProfileName.BALANCED: balanced,
        # Never "all on the GPU": a busy card offloads speed too, and the Fit column says so.
        ProfileName.SPEED: planner.place_or_step(
            ProfileName.SPEED,
            variants,
            preferred(variants, SPEED_PREFERENCE),
            speed_ctxs,
            KvType.F16,
            "Lightest 4-bit weights, 8k context, f16 cache: the least work per token.",
        ),
    }


def plannable(variant: GgufVariant) -> bool:
    """Whether the planner plans this variant at all: its bits are known and it loads."""
    return quant_bits(variant.quant) > 0 and _loads(variant)


def _loads(variant: GgufVariant) -> bool:
    """Whether mnm plans it (an Engram table of type MXFP8 is refused: llama.cpp loaded none
    until 68d20b884, and reads it as plain FP8 since)."""
    return variant.engram is None or not variant.engram.unreadable


STEPPED_WHY = "No 4-bit quant fits this machine: {quant}, the largest smaller quant that does."


def _embedding_profiles(
    planner: Planner, variants: list[GgufVariant], by_quality: list[GgufVariant], model_ctx: int
) -> dict[ProfileName, Profile]:
    best = by_quality[0]
    eight_bit = next((v for v in by_quality if quant_bits(v.quant) <= 8.5), best)
    fastest = preferred(variants, SPEED_PREFERENCE)
    picks = (
        (
            ProfileName.QUALITY,
            best,
            f"{best.quant} weights: the most faithful vectors; inputs up to {{ctx}} tokens "
            "(long documents).",
        ),
        (
            ProfileName.BALANCED,
            eight_bit,
            _embedding_why(eight_bit, "indistinguishable vectors")
            + " Inputs up to {ctx} tokens: any RAG chunk (Odysseus sends at most 900 characters).",
        ),
        (
            ProfileName.SPEED,
            fastest,
            _embedding_why(fastest, "the smallest download")
            + " Inputs up to {ctx} tokens, the least VRAM.",
        ),
    )
    profiles = {}
    for name, variant, why in picks:
        ctx = min(model_ctx, EMBEDDING_CTX[name])
        text = why.format(ctx=f"{ctx:,}")
        profiles[name] = planner.profile(name, variant, ctx, KvType.F16, text, whole_input=True)
    return profiles


def _embedding_why(variant: GgufVariant, benefit: str) -> str:
    """Prose that matches the pick: a repo with one build must not promise 8-bit or 4-bit."""
    bits = quant_bits(variant.quant)
    if bits <= 5:
        return f"4-bit weights for {benefit}; slight loss in vector quality."
    if bits <= 8.5:
        return f"8-bit weights: {benefit}, at about half the size of 16-bit."
    return f"{variant.quant} is the only build this repo offers; {benefit} comes with it."


def plan_custom(
    facts: ModelFacts,
    header: GgufHeader,
    budget: Budget,
    quant: str,
    ctx: int,
    kv: KvType,
    *,
    floor: bool = False,
) -> Profile:
    """One explicit (quant, ctx, kv) choice, estimated the same way as the named profiles;
    `floor` refuses one this machine's RAM and VRAM cannot hold (the user's pick)."""
    variant = facts.variant(quant)
    if variant is None:
        available = ", ".join(v.quant for v in facts.gguf_variants) or "none"
        raise ValueError(f"{facts.repo_id} has no '{quant}' variant (available: {available})")
    planner = Planner(header, budget, facts.projector_bytes)
    if floor:
        check_pick(planner, facts, variant, ctx)  # the ladder's floor, at this context
    return planner.profile(
        ProfileName.CUSTOM,
        variant,
        ctx,
        kv,
        "Your explicit quant and context choice.",
        whole_input=facts.modality in EMBEDDING_MODALITIES,
    )


@dataclass(frozen=True)
class Planner:
    header: GgufHeader
    budget: Budget
    projector_bytes: int
    experts: Mode = "auto"  # Where experts go across cards (`mnm set experts=`)

    def profile(
        self,
        name: ProfileName,
        variant: GgufVariant,
        ctx: int,
        kv: KvType,
        why: str,
        ubatch: int | None = None,
        whole_input: bool = False,
    ) -> Profile:
        """`whole_input` sizes the batch to the context (embedding and rerank models); `ubatch`
        is a micro-batch the recipe already runs, which the plan never exceeds."""
        ubatch = LlamaCppArgs.whole_input_micro_batch(ctx, whole_input, ubatch)
        header = self.header.for_quant(variant.quant)  # this variant's density, not the file's
        estimate = estimate_memory(
            header,
            variant.memory_bytes,  # an Engram table stays on disk
            ctx,
            kv,
            self.projector_bytes,
            ubatch=ubatch or DEFAULT_UBATCH,
            whole_input=whole_input,
            device=self.budget.device,
        )
        cards = list(self.budget.cards)
        one_card = estimate
        estimate = per_card(estimate, len(cards))  # each card's context and buffer
        fit = fit_estimate(  # never within the spill margin of a WSL card's top
            estimate, header, self.budget.plan_free_bytes, self.budget.ram_free_bytes
        )
        budget = self.budget
        tps = self.speed(header, variant, fit)
        if every_expert_in_ram(fit, header):  # Only when faster than whole blocks
            plain = fit_estimate(
                estimate, header, budget.plan_free_bytes, budget.ram_free_bytes, all_experts=False
            )
            plain_tps = self.speed(header, variant, plain)
            if plain.level is not FitLevel.NO_FIT and plain_tps >= tps:
                fit, tps = plain, plain_tps
        placed = self.placed(header, one_card, fit)
        if placed is not None:  # Its blocks and experts per card, the rest in RAM
            fit, tps = placed.fit, placed.tokens_per_second
        return Profile(
            name=name,
            variant=variant,
            ctx=ctx,
            kv_type=kv,
            ubatch=ubatch,
            whole_input=whole_input,
            estimate=estimate,
            fit=fit,
            tokens_per_second=tps,
            why=why,
            tensor_split=tensor_split(cards) if len(cards) > 1 and placed is None else None,
            main_gpu=main_gpu(cards) if len(cards) > 1 and placed is None else None,
            placement=placed.placement if placed is not None else None,
        )

    def speed(self, header: GgufHeader, variant: GgufVariant, fit: Fit) -> float:
        budget = self.budget
        return estimate_tokens_per_second(
            header,
            variant.memory_bytes,
            fit,
            budget.gpu_bandwidth,
            budget.expert_rate,
            device=budget.device,
            layer_rate=budget.layer_rate,
        )

    def placed(self, header: GgufHeader, estimate: MemoryEstimate, fit: Fit) -> Placed | None:
        """A MoE that does not fit whole on several cards, placed per card; None on one
        card, for a whole fit, a dense model, or when the cards cannot hold its dense part
        (then the split by free VRAM stands)."""
        if len(self.budget.cards) < 2 or not header.is_moe:
            return None
        if fit.level in (FitLevel.COMFORTABLE, FitLevel.TIGHT):
            return None
        budget = self.budget
        return place_experts(
            header,
            estimate,
            budget.cards,
            budget.ram_free_bytes,
            budget.expert_rate,
            self.experts,
            device=budget.device,
        )

    def place(
        self,
        name: ProfileName,
        variant: GgufVariant,
        ctxs: list[int],
        kv: KvType,
        why: str,
        whole_input: bool = False,
        ubatch: int | None = None,
    ) -> Profile:
        """Largest context at the best achievable fit level (contexts are given largest
        first); `whole_input` and `ubatch` as for `profile`."""
        candidates = [
            self.profile(name, variant, ctx, kv, why, ubatch=ubatch, whole_input=whole_input)
            for ctx in ctxs
        ]
        for level in _PREFERENCE:
            for candidate in candidates:
                if candidate.fit.level is level:
                    return candidate
        return candidates[-1]

    def floor(self, variant: GgufVariant, ctx: int) -> Floor:
        """The variant's RAM floor at `ctx` (`fit.plan_floor`)."""
        return floor_at(self, variant, ctx)

    def place_or_step(
        self,
        name: ProfileName,
        variants: list[GgufVariant],
        start: GgufVariant,
        ctxs: list[int],
        kv: KvType,
        why: str,
    ) -> Profile:
        """`place` from `start`; when it fits nowhere, or the machine's RAM and VRAM cannot
        hold it at the context placed (the floor the ladder reads), the largest smaller
        quant that does, down to the smallest the repo has; `start` again when none."""
        first = self.place(name, start, ctxs, kv, why)
        if first.fit.level is not FitLevel.NO_FIT and holds(self, start, first.ctx):
            return first
        for variant in below(variants, start):
            stepped = self.place(name, variant, ctxs, kv, STEPPED_WHY.format(quant=variant.quant))
            if stepped.fit.level is not FitLevel.NO_FIT and holds(self, variant, stepped.ctx):
                return stepped
        return first

    def best_fitting(
        self, variants: list[GgufVariant], ctxs: list[int], kv: KvType
    ) -> GgufVariant | None:
        """Best quant with a comfortable fit at any listed context; failing that, a tight one."""
        for level in (FitLevel.COMFORTABLE, FitLevel.TIGHT):
            for variant in variants:
                if any(
                    self.profile(ProfileName.CUSTOM, variant, ctx, kv, "").fit.level is level
                    for ctx in ctxs
                ):
                    return variant
        return None
