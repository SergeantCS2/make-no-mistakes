"""What vLLM puts on the GPU, and the three profiles for a safetensors chat model.

vLLM's accounting (its `gpu_memory_utilization` share of the whole card): the weights as
stored, a profile run's activation peak (`--max-num-batched-tokens` tokens through the
model, plus the sampler's logits), the CUDA graphs it captures, and its runtime (CUDA
context, cuBLAS workspaces). The KV cache takes the rest of the share and must hold one
full `--max-model-len` sequence, or the server refuses to start. The plan asks for the share
that holds that sequence, so the rest of the card stays free for other instances.

Calibrated on an RTX 4070 (vLLM 0.31.0, Qwen3-8B-AWQ at ctx 16,384, share 0.83): vLLM's
own start log read weights 5.71 GiB, non-torch 0.08 GiB, a profile-run activation peak of 1.22 GiB
at 2,048 batched tokens and a 4,096-wide model (156 bytes a token per hidden unit), CUDA graphs
0.01 GiB at the small capture sizes it picks on this card, and 2.94 GiB left for the KV cache.
One model, one point: the activation term scales with the hidden width only, so models with a
wide feed-forward layer (Gemma), MoE workspaces and vision encoders are unmeasured.
"""

from __future__ import annotations

import math

from makenomistakes.engine_args_vllm import (
    CODER_TOOL_PARSER,
    REASONING_PARSERS,
    TOOL_PARSERS,
    VllmArgs,
)
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.device_row import VLLM_DECODE_EFFICIENCY
from makenomistakes.fit.estimate import (
    Fit,
    FitLevel,
    MemoryEstimate,
    headroom_level,
)
from makenomistakes.fit.profiles import Profile, ProfileName
from makenomistakes.hub.facts import Modality, ModelFacts
from makenomistakes.hub.hf_config import HfConfig
from makenomistakes.hub.variants import GgufVariant

GIB = 1024**3
RUNTIME_BYTES = GIB // 4  # vLLM's non-torch memory (0.08 GiB read) with room for workspaces
CUDA_GRAPH_BYTES = GIB // 16  # 0.01-0.04 GiB read; none with --enforce-eager
BATCHED_TOKENS = 2048  # vLLM's chunked-prefill budget on a small card (`--max-num-batched-tokens`)
ACTIVATION_BYTES_PER_TOKEN_WIDTH = 156  # profile-run peak per batched token per hidden unit
DECODE_EFFICIENCY = VLLM_DECODE_EFFICIENCY  # the 4070's: the card's row has its own
UTIL_MARGIN = 0.02  # of the card on top of the need, so the KV cache is not cut to the token
MAX_SHARE = 0.95  # the most of the card a plan asks for: the driver and display keep the rest
# What a plan leaves free at the card's peak: a qwen-awq plan (share 0.82, Windows holding
# 1.4 GiB) peaked at 11,856 of 12,288 MiB, 0.42 GiB from the top.
SPARE_BYTES = GIB // 2
# What vLLM held beyond its share at that peak (CUDA context and the like, outside
# `gpu_memory_utilization`): 11,856 - 1,434 (Windows) - 0.82 x 12,288 MiB.
OUTSIDE_SHARE_BYTES = 346 * 2**20
# The most of a unified pool (a GB10, an APU's GTT) a plan's share may be until a box
# measures: a serving framework at 0.85 took a Spark down and 0.65 was safe (berkorbay's
# gist); the share is then of the whole pool, MemTotal.
POOL_SHARE_CAP = 0.70
CONTEXTS = (32768, 16384, 8192, 4096, 2048)
USEFUL_CTX = 8192  # a chat turn with some history; Balanced and Speed go below only when forced
_KV_BYTES = {"auto": 2, "fp8": 1}


def vllm_estimate(weights_bytes: int, config: HfConfig, args: VllmArgs) -> MemoryEstimate:
    """What the server needs from its share for these settings: on each card, with tensor
    parallel, where the weights and the KV cache are split and every card keeps its
    own activations, runtime and graphs (vLLM's TP shards each layer's heads)."""
    tp = args.tensor_parallel or 1
    text = config.text
    hidden = text.hidden_size or 0
    vocab = text.vocab_size or 0
    activations = BATCHED_TOKENS * hidden * ACTIVATION_BYTES_PER_TOKEN_WIDTH
    logits = args.max_num_seqs * vocab * 4
    graphs = 0 if args.enforce_eager else CUDA_GRAPH_BYTES
    kv = config.kv_bytes_per_token(_KV_BYTES[args.kv_cache_dtype]) * args.max_model_len // tp
    return MemoryEstimate(
        weights_bytes=weights_bytes // tp,
        compute_bytes=activations + logits + graphs,
        overhead_bytes=RUNTIME_BYTES,
        kv_bytes=kv,
        ctx=args.max_model_len,
    )


def share_for(estimate: MemoryEstimate, card_bytes: int) -> float:
    """The share of the card these settings need: the need plus a margin, in whole percent.
    Never capped: above `MAX_SHARE` the settings do not fit at all."""
    return math.ceil((estimate.total_bytes / card_bytes + UTIL_MARGIN) * 100) / 100


def utilization(estimate: MemoryEstimate, card_bytes: int) -> float:
    """The share a recipe stores (the flag's own bound); `vllm_fit` refuses past it."""
    return min(share_for(estimate, card_bytes), MAX_SHARE)


def share_cap(budget: Budget) -> float:
    """The most of the card a plan's share may be: `MAX_SHARE`, and never more than leaves
    `SPARE_BYTES` free at the peak beside what other programs hold now (Windows, a game: what
    the budget does not count as free; mnm's own running instances are counted free when
    planning) and what vLLM holds outside its share (`OUTSIDE_SHARE_BYTES`)."""
    card = card_bytes(budget)
    if budget.pool:  # A fraction of the pool, firmly capped, never past what is free
        return min(POOL_SHARE_CAP, budget.vram_free_bytes / card)
    busy = budget.vram_used_bytes or 0
    return min(MAX_SHARE, (card - busy - OUTSIDE_SHARE_BYTES - SPARE_BYTES) / card)


def card_bytes(budget: Budget) -> int:
    """What vLLM's share is a fraction of: the card, or on a pool the machine's whole RAM
    (vLLM reads the device's total, MemTotal on a GB10); the free VRAM for a `--vram` plan."""
    if budget.pool and budget.ram_total_bytes:
        return budget.ram_total_bytes
    return budget.vram_total_bytes or budget.vram_free_bytes


def vllm_fit(estimate: MemoryEstimate, budget: Budget) -> Fit:
    """vLLM refuses to start when less than its share is free, and gives the KV cache only
    what is left inside the share, so the uncapped share must fit now, under `share_cap`.
    Without the card's total (a `--vram` plan) the free VRAM stands in, and the start sizes
    the share again against the real card (`start_fixes.vllm_share_upgrade`)."""
    card = card_bytes(budget)
    share = share_for(estimate, card)
    taken = int(share * card)
    level = (
        FitLevel.NO_FIT
        if share > share_cap(budget)
        else headroom_level(taken, budget.vram_free_bytes)
    )
    return Fit(
        level=level,
        gpu_layers=0,
        headroom_bytes=budget.vram_free_bytes - taken,
        gpu_bytes=taken,
        room_bytes=budget.vram_free_bytes,
    )


def vllm_args(
    facts: ModelFacts, config: HfConfig, ctx: int, kv: str, tensor_parallel: int | None = None
) -> VllmArgs:
    """The flags for one context and KV precision; the share is filled in by the planner."""
    model_type = config.model_type
    return VllmArgs(
        max_model_len=ctx,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=kv,  # "auto" | "fp8", from _KV_BYTES
        tool_parser=(
            CODER_TOOL_PARSER
            if model_type.startswith("qwen3") and "coder" in facts.repo_id.lower()
            else TOOL_PARSERS.get(model_type)
        ),
        reasoning_parser=REASONING_PARSERS.get(model_type),
        images_per_prompt=1 if facts.modality is Modality.VISION_CHAT else None,
        tensor_parallel=tensor_parallel,
    )


def vllm_profile(  # noqa: PLR0913, PLR0917 - one profile's every choice
    name: ProfileName, facts: ModelFacts, config: HfConfig, budget: Budget, ctx: int, kv: str,
    why: str, tensor_parallel: int | None = None,
) -> Profile:  # fmt: skip
    card = card_bytes(budget)
    args = vllm_args(facts, config, ctx, kv, tensor_parallel)
    estimate = vllm_estimate(facts.vllm_bytes, config, args)
    args = args.model_copy(update={"gpu_memory_utilization": utilization(estimate, card)})
    variant = GgufVariant(
        quant=config.quant_label, files=list(facts.vllm_files), size_bytes=facts.vllm_bytes
    )
    return Profile(
        name=name,
        variant=variant,
        ctx=ctx,
        engine_args=args,
        estimate=estimate,
        fit=vllm_fit(estimate, budget),
        tokens_per_second=budget.gpu_bandwidth
        * budget.device.vllm_decode_efficiency
        / max(facts.vllm_bytes, 1),
        why=why,
    )


def _best(tried: list[Profile], levels: tuple[FitLevel, ...], floor: int = 0) -> Profile:
    """The first tried profile (longest context first) whose share fits at the first of
    `levels` that any does, else the last (NO_FIT). vLLM never grows past its share, so a
    tight fit only leaves less room for other instances: quality takes it, the others keep
    room. A context under `floor` is a choice only when nothing at or above it fits at all
    (else Balanced takes 2,048 tokens, "comfortable", over 16,384 that fit tightly)."""
    useful = [p for p in tried if p.ctx >= floor]
    for pool in (useful, tried):
        for level in levels:
            fitting = [p for p in pool if p.fit.level is level]
            if fitting:
                return fitting[0]
    return tried[-1]


ROOMY = (FitLevel.COMFORTABLE, FitLevel.TIGHT)
LONGEST = (FitLevel.TIGHT, FitLevel.COMFORTABLE)


def plan_vllm_profiles(
    facts: ModelFacts, config: HfConfig, budget: Budget, tensor_parallel: int | None = None
) -> dict[ProfileName, Profile]:
    """`budget` is one card's; `tensor_parallel` the equal cards the model is split over."""
    native = config.text.max_position_embeddings or 8192
    ctxs = [c for c in CONTEXTS if c <= native] or [native]
    floor = min(USEFUL_CTX, native)

    def tries(name: ProfileName, limit: int, kvs: tuple[str, ...], why: str) -> list[Profile]:
        contexts = [c for c in ctxs if c <= limit] or ctxs
        return [
            vllm_profile(name, facts, config, budget, c, kv, why, tensor_parallel)
            for c in contexts
            for kv in kvs
        ]

    balanced = _best(
        tries(
            ProfileName.BALANCED,
            16384,
            ("auto", "fp8"),
            "Up to 16,384 tokens, with room left on the card; an FP8 KV cache when the "
            "model's own precision would cost context.",
        ),
        ROOMY,
        floor,
    )
    quality = _best(
        tries(
            ProfileName.QUALITY,
            native,
            ("auto",),
            "The longest context that fits, KV cache at the model's own precision.",
        ),
        LONGEST,
    )
    if quality.ctx < balanced.ctx:  # never less context than Balanced: FP8 KV buys it back
        quality = _best(
            tries(
                ProfileName.QUALITY,
                native,
                ("fp8",),  # one precision: the list runs longest context first
                "The longest context that fits; an FP8 KV cache where the model's own "
                "precision would leave less context than Balanced.",
            ),
            LONGEST,
        )
    return {
        ProfileName.QUALITY: quality,
        ProfileName.BALANCED: balanced,
        ProfileName.SPEED: _best(
            tries(ProfileName.SPEED, 8192, ("fp8",),
                  "Up to 8,192 tokens with an FP8 KV cache: the smallest share of the card."),
            ROOMY, floor,
        ),
    }  # fmt: skip


def plan_vllm_custom(
    facts: ModelFacts,
    config: HfConfig,
    budget: Budget,
    ctx: int,
    kv: str,
    tensor_parallel: int | None = None,
) -> Profile:
    """A context or KV the user chose, on the cards the named profiles split over (so
    `--for-agents` and `--ctx` keep tensor parallel on equal cards)."""
    return vllm_profile(
        ProfileName.CUSTOM,
        facts,
        config,
        budget,
        ctx,
        kv,
        f"Context {ctx:,}, KV {kv}.",
        tensor_parallel,
    )
