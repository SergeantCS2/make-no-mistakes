"""Estimate what a llama.cpp instance will need and whether it fits.

Everything here is an estimate and is labelled as such in the UI. The KV cache term uses
the real layer count, KV-head count and head size from the GGUF header, which is what the
Odysseus heuristic (8 KB per token per billion parameters) gets wrong for GQA models.
Constants are calibrated against two measured runs on an RTX 4070 under WSL2
(nvidia-smi `memory.used` delta before/after `mnm start`, which is also what the budget
reads, so estimate and budget see the same accounting):

    Qwen3-8B   Q4_K_M (4.68 GiB file)  ctx 8192  KV q8_0  -> 5.13 GiB used
    Qwen3-8B   Q4_K_M                  ctx 16384 KV q8_0  -> 5.83 GiB used, 75.4 tok/s
    Qwen3-0.6B Q4_K_M (0.37 GiB file)  ctx 16384 KV q8_0  -> ~1.55 GiB used, 325 tok/s

The 8B run uses *less* than file + KV because llama.cpp keeps the input embedding table in
system RAM; the estimate does not subtract it, so it stays a few percent conservative.
Decode time is bandwidth-bound plus a fixed per-layer kernel-launch cost, which is what
keeps a 0.6B model at 325 tok/s instead of the 1,200 that bandwidth alone would allow. The
speeds were read with every block on the GPU: earlier readings of 52 and 180 tok/s ran
`-ngl <blocks>`, which leaves block 0 on the CPU for every token.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel

from makenomistakes.engine_args import DRAFT_KV, KvType, LlamaCppArgs
from makenomistakes.fit.cpu_rates import CPU_BANDWIDTH, CPU_EXPERT_RATE, EXPERT_TOUCH_SECONDS
from makenomistakes.fit.device_row import (
    GPU_EFFICIENCY,
    IQ_GRID_EFFICIENCY,
    LAYER_OVERHEAD_SECONDS,
    RTX_4070,
    RUNTIME_OVERHEAD_BYTES,
    DeviceRow,
)
from makenomistakes.fit.devices import (
    DEFAULT_GPU_BANDWIDTH,
    DEFAULT_GPU_TFLOPS,
)
from makenomistakes.fit.kv_terms import (
    DEFAULT_UBATCH,
    DEFAULT_VOCAB,
    KV_BYTES_PER_VALUE,
    KV_PAD,
    LOGIT_ROWS,
    VISION_COMPUTE_BYTES,
    compute_buffer_bytes,
    kv_cache_bytes,
    recurrent_state_bytes,
)
from makenomistakes.gguf.file_types import GRID_QUANT_IDS
from makenomistakes.gguf.reader import GgufHeader

__all__ = [  # the 4070's: `fit/devices.py` is the table
    "CPU_BANDWIDTH",
    "CPU_EXPERT_RATE",
    "DEFAULT_GPU_BANDWIDTH",
    "DEFAULT_GPU_TFLOPS",
    "DEFAULT_UBATCH",
    "DEFAULT_VOCAB",
    "GPU_EFFICIENCY",
    "IQ_GRID_EFFICIENCY",
    "KV_BYTES_PER_VALUE",
    "KV_PAD",
    "LAYER_OVERHEAD_SECONDS",
    "LOGIT_ROWS",
    "RUNTIME_OVERHEAD_BYTES",
    "VISION_COMPUTE_BYTES",
    "compute_buffer_bytes",
    "kv_cache_bytes",
    "recurrent_state_bytes",
]

MIB = 1024 * 1024
GIB = 1024 * MIB
# GPU_EFFICIENCY, IQ_GRID_EFFICIENCY, LAYER_OVERHEAD_SECONDS and RUNTIME_OVERHEAD_BYTES are the
# 4070's (`fit.device_row`, where their readings are told): `fit/devices.py` is the table.
# CPU_BANDWIDTH, CPU_EXPERT_RATE and the touch cost are the CPU side: `fit.cpu_rates`.
COMFORTABLE_HEADROOM = 0.25  # a fit is comfortable with this fraction of free VRAM to spare


class MemoryEstimate(BaseModel):
    weights_bytes: int
    compute_bytes: int
    overhead_bytes: int
    kv_bytes: int = 0  # llama.cpp only
    state_bytes: int = 0  # llama.cpp hybrids: the linear-attention layers' recurrent state
    projector_bytes: int = 0  # llama.cpp vision only
    companion_bytes: int = 0  # image kits: VAE and text encoders kept on the GPU
    # llama.cpp only: the output head (vocab x embedding length), which `-ngl` offloads first,
    # and the input embedding table of the same size, which stays in RAM.
    head_bytes: int = 0
    ctx: int = 0
    kv_type: KvType = KvType.F16

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.kv_bytes
            + self.state_bytes
            + self.compute_bytes
            + self.projector_bytes
            + self.companion_bytes
            + self.overhead_bytes
        )


class FitLevel(StrEnum):
    COMFORTABLE = "comfortable"  # at most 75 % of free VRAM
    TIGHT = "tight"  # fits, with less than 25 % headroom
    OFFLOAD = "offload"  # some layers stay in system RAM; slower
    NO_FIT = "no_fit"


class Fit(BaseModel):
    level: FitLevel
    gpu_layers: int  # the `-ngl` count before `ngl_for`: blocks for a full fit (it adds the
    # output layer), up to blocks for an offload (block 0 in RAM), 0 for NO_FIT
    headroom_bytes: int  # negative when it does not fit on the GPU
    gpu_bytes: int  # what this fit puts on the GPU (less than the estimate when offloading)
    # Mixture-of-experts offload: this many layers keep their routed experts in RAM
    # (`--n-cpu-moe`), everything else on the GPU; 0 for dense models and full fits.
    cpu_moe_layers: int = 0
    # The free room this fit was judged against, in the planner's own terms: each planner's
    # (llama.cpp keeps WSL's margin out of it, `Budget.plan_free_bytes`; the others judge the
    # free VRAM). `mnm why` says it as the planner saw it (one rule assumed for every engine
    # would say a margin only llama.cpp keeps). 0: not judged (a speed model's fit).
    room_bytes: int = 0


def headroom_level(estimate_bytes: int, vram_free_bytes: int) -> FitLevel:
    """The one rule for "does this amount fit in that much free VRAM" (never OFFLOAD)."""
    headroom = vram_free_bytes - estimate_bytes
    if headroom >= COMFORTABLE_HEADROOM * vram_free_bytes:
        return FitLevel.COMFORTABLE
    if headroom >= 0:
        return FitLevel.TIGHT
    return FitLevel.NO_FIT


def estimate_memory(  # noqa: PLR0913 - the card's row is the eighth, by keyword
    header: GgufHeader,
    weights_bytes: int,
    ctx: int,
    kv_type: KvType,
    projector_bytes: int = 0,
    ubatch: int = DEFAULT_UBATCH,
    whole_input: bool = False,
    *,
    device: DeviceRow = RTX_4070,
) -> MemoryEstimate:
    """`device` is the card's row: its backend's context and scratch."""
    return MemoryEstimate(
        weights_bytes=weights_bytes,
        head_bytes=head_bytes(header, weights_bytes),
        kv_bytes=kv_cache_bytes(header, ctx, kv_type, ubatch),
        state_bytes=recurrent_state_bytes(header),
        compute_bytes=compute_buffer_bytes(
            header, ubatch, with_projector=projector_bytes > 0, whole_input=whole_input
        ),
        projector_bytes=projector_bytes,
        overhead_bytes=device.context_bytes,
        ctx=ctx,
        kv_type=kv_type,
    )


def per_card(estimate: MemoryEstimate, cards: int) -> MemoryEstimate:
    """The estimate on `cards` cards (`fit.cards`): a CUDA context and a compute buffer
    on each, which llama.cpp allocates per device (unmeasured). One card is as it was."""
    if cards < 2:
        return estimate
    extra = (cards - 1) * (estimate.overhead_bytes + estimate.compute_bytes)
    return estimate.model_copy(update={"overhead_bytes": estimate.overhead_bytes + extra})


def _expert_offload(
    estimate: MemoryEstimate, header: GgufHeader, vram_free_bytes: int, ram_free_bytes: int
) -> Fit | None:
    """A mixture-of-experts model that does not fit whole keeps the routed experts of its
    first N layers in RAM (`--n-cpu-moe N`) before any block goes there: a token reads
    only its active experts, so the GPU still runs every block's attention. The fewest
    layers that fit comfortably, else the fewest that fit at all; None when the experts of
    every layer in RAM would still not fit, or RAM cannot hold them."""
    per_layer, layers = header.expert_bytes_per_layer, header.block_count or 0
    if not header.uniform_experts:
        return _uneven_expert_offload(estimate, header, vram_free_bytes, ram_free_bytes)
    if not per_layer or not layers:
        return None
    # The most the GPU may hold at each level, so the fewest layers whose experts must go.
    for allowed in ((1 - COMFORTABLE_HEADROOM) * vram_free_bytes, vram_free_bytes):
        cpu_layers = math.ceil((estimate.total_bytes - allowed) / per_layer)
        if cpu_layers > layers or cpu_layers * per_layer > ram_free_bytes:
            continue  # that level is out of reach (or RAM cannot hold it): try the next
        return Fit(
            level=FitLevel.OFFLOAD,
            gpu_layers=layers,
            headroom_bytes=vram_free_bytes - estimate.total_bytes,
            room_bytes=vram_free_bytes,
            gpu_bytes=estimate.total_bytes - max(cpu_layers, 1) * per_layer,
            cpu_moe_layers=max(cpu_layers, 1),
        )
    return None


def _uneven_expert_offload(
    estimate: MemoryEstimate, header: GgufHeader, vram_free_bytes: int, ram_free_bytes: int
) -> Fit | None:
    """`_expert_offload` for blocks whose experts differ (from the tensor infos):
    `--n-cpu-moe N` moves the experts of blocks 0 to N - 1, which may hold none (Bolt's
    Mamba and attention blocks), so N is the first block count whose experts free enough."""
    per_block = header.expert_layer_bytes()
    first = [0]
    for size in per_block:
        first.append(first[-1] + size)
    if not first[-1]:
        return None
    for allowed in ((1 - COMFORTABLE_HEADROOM) * vram_free_bytes, vram_free_bytes):
        need = estimate.total_bytes - allowed
        n = next((n for n in range(1, len(first)) if first[n] >= need and first[n]), None)
        if n is None or first[n] > ram_free_bytes:
            continue
        return Fit(
            level=FitLevel.OFFLOAD,
            gpu_layers=len(per_block),
            headroom_bytes=vram_free_bytes - estimate.total_bytes,
            room_bytes=vram_free_bytes,
            gpu_bytes=estimate.total_bytes - first[n],
            cpu_moe_layers=n,
        )
    return None


def _all_experts_offload(
    estimate: MemoryEstimate, header: GgufHeader, vram_free_bytes: int, ram_free_bytes: int
) -> Fit | None:
    """`--cpu-moe -ngl N`: a MoE whose dense part
    does not fit the card keeps every block's experts in RAM and as many dense blocks on the
    card as fit, the last ones (llama.cpp leaves the first `blocks + 1 - ngl` in RAM). Rendered
    as `--n-cpu-moe <blocks>` and `-ngl N + 1` (`ngl_for` adds the output layer): no new
    args. `gpu_layers` is the blocks on the card; None for a dense model, when not one block
    fits, or when RAM cannot hold the experts and the RAM-held blocks."""
    layers = header.block_count or 0
    experts = header.expert_layer_bytes() if header.is_moe else ()
    if not layers or not any(experts):
        return None
    per_block = block_bytes(estimate, layers)
    dense = [max(per_block - size, 0.0) for size in experts]
    fixed = estimate.total_bytes - estimate.weights_bytes + estimate.head_bytes
    room, held = float(vram_free_bytes - fixed), 0  # the whole room, as a layer offload's
    while held < layers and dense[layers - 1 - held] <= room:
        room -= dense[layers - 1 - held]
        held += 1
    on_card = int(fixed + sum(dense[layers - held :]))
    if not held or estimate.total_bytes - on_card > ram_free_bytes:
        return None
    return Fit(
        level=FitLevel.OFFLOAD,
        gpu_layers=held,
        headroom_bytes=vram_free_bytes - estimate.total_bytes,
        room_bytes=vram_free_bytes,
        gpu_bytes=on_card,
        cpu_moe_layers=layers,
    )


def head_bytes(header: GgufHeader, weights_bytes: int) -> int:
    """The output head (and the input table, the same shape) as the header sizes it, never
    more than a quarter of the file (a tiny vocab-heavy model keeps the split sane)."""
    return min(header.token_embedding_bytes, weights_bytes // 4)


def block_bytes(estimate: MemoryEstimate, layers: int) -> float:
    """One transformer block's weights: the file less the embedding table and the output
    head (each `head_bytes`), over the block count."""
    if not layers:
        return 0.0
    return max(estimate.weights_bytes - 2 * estimate.head_bytes, 0) / layers


def gpu_bytes(estimate: MemoryEstimate, layers: int, gpu_layers: int) -> int:
    """What sits on the GPU with `-ngl gpu_layers` of `layers` blocks: everything but the
    weights of the layers left in RAM. llama.cpp offloads the output layer first, then
    blocks from the top: `-ngl k` is the head plus k - 1 blocks, so `-ngl <blocks>`
    keeps block 0 in RAM (`ngl_for`) and only blocks + 1 is the whole file. The one offload
    split; the planner and `settings_gpu_bytes` both use it."""
    if not layers or gpu_layers > layers:
        return estimate.total_bytes
    non_weight = estimate.total_bytes - estimate.weights_bytes
    if gpu_layers <= 0:
        return int(non_weight)
    return int(non_weight + estimate.head_bytes + (gpu_layers - 1) * block_bytes(estimate, layers))


def settings_gpu_bytes(
    header: GgufHeader,
    weights_bytes: int,
    projector_bytes: int,
    args: LlamaCppArgs,
    whole_input: bool = False,
    rollback: int = 0,
) -> int:
    """What llama-server puts on the GPU with exactly these saved settings (`whole_input`:
    an embedding or rerank recipe). The planner always emits `-np 1`; a hand-set `parallel`
    multiplies the sliding-window layers' cache and a hybrid's recurrent state, which an
    MTP draft multiplies by `1 + rollback` too (`recurrent_state_bytes`). `cpu_moe_layers`
    keeps that many layers' experts in RAM, and `moe_cache_mib` of the card caches the
    hottest of them."""
    ubatch = args.ubatch or DEFAULT_UBATCH
    estimate = estimate_memory(
        header, weights_bytes, args.ctx, args.kv_type, projector_bytes, ubatch, whole_input
    )
    if args.parallel > 1 or rollback:
        kv = kv_cache_bytes(header, args.ctx, args.kv_type, ubatch, args.parallel)
        state = recurrent_state_bytes(header, args.parallel, rollback)
        estimate = estimate.model_copy(update={"kv_bytes": kv, "state_bytes": state})
    placement = args.placement  # Each card's blocks and experts, the rest in RAM
    cards = len(placement.cards) if placement is not None else len(args.tensor_split or ())
    estimate = per_card(estimate, cards)  # several cards: a context and buffer on each
    blocks = header.block_count or 0
    # The cache costs its budget and nothing more on the card (llama.cpp c7fc509,
    # src/llama-moe-cache.cpp: each group's banks take `n_slots + 1` expert slots, the
    # slot count chosen so the whole allocation stays within its share of the budget; the
    # slot maps live in host memory; no staging buffer beside it). A full card seen beside the
    # cache was another instance still running a draft its recipe had dropped
    # (`instance_vram.started_with`), not the cache's own size.
    cache = (args.moe_cache_mib or 0) * MIB if args.experts_in_ram else 0
    if placement is not None:
        sizes = header.expert_layer_bytes()
        in_ram = sum(sizes[b] for b in placement.in_ram if b < len(sizes))
        return estimate.total_bytes - in_ram + placement.copy_bytes + cache
    # Both count from layer 0: `-ngl` leaves the first `blocks + 1 - ngl` blocks in RAM whole
    # (their experts with them, already out of `gpu_bytes`), `--n-cpu-moe N` the experts of
    # the first N. Only the expert layers above the RAM-held blocks come off again.
    cpu_blocks = max(blocks + 1 - args.gpu_layers, 0)  # gpu_layers >= 1: at most `blocks`
    experts_only = max(min(args.cpu_moe_layers or 0, blocks) - cpu_blocks, 0)
    if header.uniform_experts:
        in_ram = experts_only * header.expert_bytes_per_layer
    else:  # the blocks above the RAM-held ones, each its own
        in_ram = sum(header.expert_layer_bytes()[cpu_blocks : cpu_blocks + experts_only])
    return gpu_bytes(estimate, blocks, args.gpu_layers) - in_ram + cache


def ngl_for(fit: Fit) -> int:
    """llama.cpp's `-ngl` for a fit. It offloads `n_layer + 1` layers, the output layer last,
    so `-ngl <block count>` keeps block 0 on the CPU for every token (llama-model.cpp
    `i_gpu_start = n_layer + 1 - n_gpu_layers`). Everything on the GPU is
    the block count plus one; a partial offload keeps the planner's count. An expert offload
    (`cpu_moe_layers`) keeps every block on the GPU: only the experts move."""
    whole = fit.level in (FitLevel.COMFORTABLE, FitLevel.TIGHT) or fit.cpu_moe_layers > 0
    return fit.gpu_layers + 1 if whole else fit.gpu_layers


def every_expert_in_ram(fit: Fit, header: GgufHeader) -> bool:
    """The `--cpu-moe -ngl N` plan: every block's experts in RAM, some blocks too."""
    blocks = header.block_count or 0
    return bool(blocks) and fit.cpu_moe_layers >= blocks and fit.gpu_layers < blocks


def fit_estimate(
    estimate: MemoryEstimate,
    header: GgufHeader,
    vram_free_bytes: int,
    ram_free_bytes: int,
    *,
    all_experts: bool = True,
) -> Fit:
    """The best fit level; `all_experts` False leaves out the `--cpu-moe -ngl N` plan (the
    planner prices both and keeps the faster, `fit.profiles`)."""
    headroom = vram_free_bytes - estimate.total_bytes
    layers = header.block_count or 0
    level = headroom_level(estimate.total_bytes, vram_free_bytes)
    if level is not FitLevel.NO_FIT:
        return Fit(
            level=level,
            gpu_layers=layers,
            headroom_bytes=headroom,
            gpu_bytes=estimate.total_bytes,
            room_bytes=vram_free_bytes,
        )

    experts = _expert_offload(estimate, header, vram_free_bytes, ram_free_bytes)
    if experts is None and all_experts:  # Every expert in RAM, dense blocks on card
        experts = _all_experts_offload(estimate, header, vram_free_bytes, ram_free_bytes)
    if experts is not None:
        return experts
    non_weight = estimate.total_bytes - estimate.weights_bytes
    per_block = block_bytes(estimate, layers)
    room = vram_free_bytes - non_weight - estimate.head_bytes  # the head goes first
    # At most `layers` (block 0 in RAM): the whole file did not fit, so blocks + 1 is out.
    gpu_layers = min(1 + int(room // per_block), layers) if per_block and room >= 0 else 0
    on_gpu = gpu_bytes(estimate, layers, gpu_layers)
    spill = estimate.total_bytes - on_gpu
    if gpu_layers > 0 and spill <= ram_free_bytes:
        return Fit(
            level=FitLevel.OFFLOAD,
            gpu_layers=gpu_layers,
            headroom_bytes=headroom,
            gpu_bytes=on_gpu,
            room_bytes=vram_free_bytes,
        )
    return Fit(
        level=FitLevel.NO_FIT,
        gpu_layers=0,
        headroom_bytes=headroom,
        gpu_bytes=estimate.total_bytes,
        room_bytes=vram_free_bytes,
    )


def estimate_tokens_per_second(
    header: GgufHeader,
    weights_bytes: int,
    fit: Fit,
    gpu_bandwidth: int,
    expert_rate: float = CPU_EXPERT_RATE,
    *,
    device: DeviceRow = RTX_4070,
    layer_rate: float = CPU_BANDWIDTH,
) -> float:
    """Seconds per token = bytes streamed over effective bandwidth + per-layer launch cost.
    The input embedding table is read one row per token, never streamed. A
    mixture-of-experts model reads its dense weights and only the active experts; the
    experts of `cpu_moe_layers` layers come from RAM at `expert_rate` (this box's,
    `machine_facts`, else the RTX 4070 machine's `CPU_EXPERT_RATE`), each used expert held
    there at `EXPERT_TOUCH_SECONDS` more (`fit.cpu_rates`); whole blocks left in RAM at
    `layer_rate` (the box's probe x `LLAMA_RAM_SHARE`, else the RTX 4070 machine's
    `CPU_BANDWIDTH`)."""
    fixed, from_ram = decode_parts(
        header, weights_bytes, fit, gpu_bandwidth, device=device, layer_rate=layer_rate
    )
    return 1.0 / (fixed + from_ram / expert_rate)


def decode_parts(
    header: GgufHeader,
    weights_bytes: int,
    fit: Fit,
    gpu_bandwidth: int,
    *,
    device: DeviceRow = RTX_4070,
    layer_rate: float = CPU_BANDWIDTH,
) -> tuple[float, float]:
    """A decoded token's seconds apart from the RAM-held experts' bytes, and those bytes: the
    speed model split so a reading can be turned back into the expert rate. The card's
    row gives the efficiency and the launch term; the seconds include each RAM-held
    used expert's touch (`EXPERT_TOUCH_SECONDS`), which the rate does not carry."""
    layers = header.block_count or 1
    streamed = weights_bytes - head_bytes(header, weights_bytes)  # the table stays in RAM
    gpu_fraction = 0.0 if fit.level is FitLevel.NO_FIT else min(fit.gpu_layers / layers, 1.0)
    # The blocks whose experts the CPU reads: those `--n-cpu-moe` names, and those
    # left in RAM whole (`-ngl` below the block count), whose experts it reads the same way
    in_ram = min(max(fit.cpu_moe_layers, round(layers * (1 - gpu_fraction))), layers)
    if header.uniform_experts:
        experts = min(header.expert_bytes_per_layer * layers, streamed)
        active_fraction, used = 1.0, 0
        if header.is_moe and header.expert_count and header.expert_used_count:
            active_fraction = header.expert_used_count / header.expert_count  # No floor
            used = header.expert_used_count
        per_token = (streamed - experts) + experts * active_fraction
        from_ram = experts * (in_ram / layers) * active_fraction
        touched = used * in_ram if experts else 0
    else:
        per_token, from_ram = _uneven_reads(header, streamed, in_ram)
        touched = sum(header.experts_used()[:in_ram])
    effective = 1.0 / ((gpu_fraction / gpu_bandwidth) + ((1 - gpu_fraction) / layer_rate))
    efficiency = decode_efficiency(header, device)
    fixed = (per_token - from_ram) / (effective * efficiency) + layers * device.launch_seconds
    return fixed + touched * EXPERT_TOUCH_SECONDS, from_ram


def decode_efficiency(header: GgufHeader, device: DeviceRow = RTX_4070) -> float:
    """The card's decode efficiency for this file: the codebook i-quants' own, a MoE's where
    the row has one, else the row's."""
    if header.file_type in GRID_QUANT_IDS:
        return device.iq_grid_efficiency
    if header.is_moe and device.moe_efficiency is not None:
        return device.moe_efficiency
    return device.efficiency


def _uneven_reads(header: GgufHeader, streamed: int, cpu_moe_layers: int) -> tuple[float, float]:
    """A decoded token's bytes and those of them read from RAM, block by block: each
    block's active share of its own experts; the first `cpu_moe_layers` blocks' from RAM."""
    sizes = header.expert_layer_bytes()
    shares = header.expert_shares()
    scale = min(1.0, streamed / sum(sizes)) if sum(sizes) else 1.0  # never more than the file
    active = [size * scale * share for size, share in zip(sizes, shares, strict=True)]
    experts = sum(sizes) * scale
    return (streamed - experts) + sum(active), sum(active[:cpu_moe_layers])


def draft_gpu_bytes(
    header: GgufHeader, file_bytes: int, args: LlamaCppArgs, mtp: bool = False
) -> int:
    """What a draft model adds on the GPU beside its target: every layer of it (`-ngld
    all`) less the input embedding table (`head_bytes`, read one row per token from RAM, as
    the target's is), its KV cache at the target's context and slots (llama.cpp sizes the
    draft's context from the target's) at f16 (`DRAFT_KV`, `-ctkd/-ctvd`), and its own compute
    buffer. An MTP head (`mtp`) runs in an MTP context: only its MTP blocks keep KV
    (none for Gemma 4's assistant, which shares the model's). Its speed is not modelled: what
    a draft gains is read on the box (`mnm try --speed`)."""
    ubatch = args.ubatch or DEFAULT_UBATCH
    weights = file_bytes - head_bytes(header, file_bytes)
    kv = kv_cache_bytes(header, args.ctx, DRAFT_KV, ubatch, args.parallel, mtp=mtp)
    return weights + kv + compute_buffer_bytes(header, ubatch)
