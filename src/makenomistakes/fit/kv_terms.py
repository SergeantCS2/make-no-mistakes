"""The memory terms of a llama.cpp estimate that need no fit (`fit/estimate.py` re-exports
them): the KV cache layer by layer, a hybrid's recurrent state and the CUDA compute buffer."""

from __future__ import annotations

import math

from makenomistakes.engine_args import KvType
from makenomistakes.fit.kv_compressed import compressed_kv_bytes
from makenomistakes.gguf.reader import GgufHeader

MIB = 1024 * 1024
VISION_COMPUTE_BYTES = 512 * MIB  # image encoder activations when an mmproj is loaded
DEFAULT_UBATCH = 512
DEFAULT_VOCAB = 150_000
LOGIT_ROWS = 8  # llama-server keeps logits for a handful of tokens, not a whole ubatch
KV_BYTES_PER_VALUE: dict[KvType, float] = {
    KvType.F16: 2.0,
    KvType.Q8_0: 34 / 32,  # 32 values + fp16 scale per block
    KvType.Q4_0: 18 / 32,
}


KV_PAD = 256  # llama.cpp pads a sliding-window cache to this many cells


def kv_cache_bytes(
    header: GgufHeader,
    ctx: int,
    kv_type: KvType,
    ubatch: int = DEFAULT_UBATCH,
    parallel: int = 1,
    *,
    mtp: bool = False,
) -> int:
    """Standard/GQA attention, layer by layer; latent attention (MLA: one compressed key row
    a token a layer, the value a view of it) and DeepSeek-V4's compressed attention
    (`kv_compressed`) as llama.cpp keeps them.

    A sliding-window layer (Gemma 2/3/4, gpt-oss …) keeps a window per sequence: `window x
    parallel + ubatch` cells, padded to 256, as llama.cpp's iSWA cache does
    (llama-kv-cache-iswa.cpp, `n_swa * n_seq_max + n_ubatch`), at its own head size when
    the GGUF gives one (Gemma 4); a recurrent layer (0 KV heads: LFM2, Granite-H, Jamba; the
    linear-attention layers of Qwen3.5/3.6, 3 of every 4) and a layer sharing another's
    KV keep none, nor do the MTP blocks, which only an MTP context (`mtp`) holds."""
    compressed = compressed_kv_bytes(
        header, ctx, KV_BYTES_PER_VALUE.__getitem__, kv_type, ubatch, parallel
    )
    if compressed is not None:
        return compressed
    layers = header.block_count
    per_layer = header.kv_heads_per_layer()
    if not (layers and header.head_dim and any(per_layer)):
        raise ValueError("GGUF header lacks block_count, head counts or head size")
    sliding = header.swa_layers or (False,) * layers
    window = header.sliding_window or 0
    swa_cells = min(ctx, -(-(window * parallel + ubatch) // KV_PAD) * KV_PAD) if window else ctx
    latent = header.key_length_mla is not None  # keys only: GLM-4.7-Flash 1,692 MiB at 32k
    values = 0
    for il, holds in enumerate(header.kv_layers(mtp)):
        if holds and per_layer[il]:
            cells = swa_cells if sliding[il] else ctx
            width = header.head_dims(il)[0] if latent else sum(header.head_dims(il))
            values += per_layer[il] * cells * width
    return math.ceil(values * KV_BYTES_PER_VALUE[kv_type])


def recurrent_state_bytes(header: GgufHeader, parallel: int = 1, rollback: int = 0) -> int:
    """The recurrent state of a hybrid model's linear-attention layers: one row
    a sequence of each, in f32, whatever the context (llama-memory-recurrent.cpp), and
    `1 + rollback` rows when a draft head may need to step back that many tokens (an MTP
    draft: llama.cpp's `need_n_rs_seq`, its draft's most tokens a step)."""
    trunk = header.trunk_layers
    count = sum(1 for il, r in enumerate(header.recurrent_layers or ()) if r and il < trunk)
    rows = max(parallel, 1) * (1 + rollback)
    return count * header.recurrent_row_values * 4 * rows


def compute_buffer_bytes(
    header: GgufHeader,
    ubatch: int = DEFAULT_UBATCH,
    with_projector: bool = False,
    whole_input: bool = False,
) -> int:
    """Rough size of llama.cpp's CUDA compute buffer: activations for one ubatch plus logits.

    In embedding/rerank mode (`whole_input`) every input token is an output, so a model with
    an output head computes a float per vocabulary entry for each of them: Qwen3-Embedding
    at ubatch 8192 needs 8192 x 151,669 x 4 bytes, 4.6 GiB (measured: compute buffer 5,028
    MiB on llama.cpp 79e2e74; an RTX 4070 saw 6.6 GiB in use for the whole instance)."""
    n_embd = header.embedding_length or 4096
    n_vocab = header.vocab_size or DEFAULT_VOCAB
    rows = ubatch if whole_input and header.computes_logits else LOGIT_ROWS
    base = ubatch * n_embd * 16 + n_vocab * 4 * rows
    return base + (VISION_COMPUTE_BYTES if with_projector else 0)
