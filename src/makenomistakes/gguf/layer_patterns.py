"""Which blocks of a GGUF keep only a sliding window of KV, and which are linear attention
(a recurrent state, no KV cache), decided per architecture the way llama.cpp's
`load_arch_hparams` decides them when the file carries no per-layer array of its own."""

from __future__ import annotations

# llama.cpp's `load_swa_pattern(ml, n, dense_first)` per architecture (src/models/*.cpp,
# 2026-10): every n-th layer keeps the whole context, the rest only the sliding window.
# A GGUF's own `attention.sliding_window_pattern` (a number, or a bool per layer) wins.
_SWA_PATTERNS: dict[str, tuple[int, bool]] = {
    "gemma2": (2, False),
    "gemma3": (6, False),
    "gemma3n": (5, False),
    "gemma-embedding": (6, False),
    "gpt-oss": (2, False),
    "cohere2": (4, False),
    "exaone4": (4, False),
    "olmo2": (4, False),
    "afmoe": (4, False),
    "mellum": (4, False),
    "plamo3": (8, False),
    "smallthinker": (4, True),
}

# The hybrid architectures whose linear-attention layers llama.cpp marks by an interval when
# the GGUF has no `attention.recurrent_layers` array (src/models/<arch>.cpp, 2026-10): every
# n-th block is full attention (`(il + 1) % n == 0`), the rest recurrent; the default when
# `full_attention_interval` is absent.
_FULL_ATTENTION_INTERVALS: dict[str, int] = {
    "qwen3next": 4,
    "qwen35": 4,
    "qwen35moe": 4,
    "minimax-01": 8,
}


def swa_layers(
    architecture: str,
    block_count: int | None,
    window: int | None,
    per_layer: tuple[int, ...] | None,
    pattern: int | None,
) -> tuple[bool, ...] | None:
    """Which blocks keep only a sliding window of KV (`hparams.set_swa_pattern`)."""
    if not window or not block_count:
        return None
    if per_layer is not None and len(per_layer) == block_count:
        return tuple(bool(v) for v in per_layer)
    known = _SWA_PATTERNS.get(architecture)
    n, dense_first = (pattern, False) if pattern else known or (0, False)
    if not n:
        return None  # no pattern known: every layer full, as llama.cpp does (an over-estimate)
    if dense_first:
        return tuple(il % n != 0 for il in range(block_count))
    return tuple(il % n < n - 1 for il in range(block_count))


def recurrent_layers(
    architecture: str,
    block_count: int | None,
    nextn: int,
    per_layer: tuple[int, ...] | None,
    interval: int | None,
) -> tuple[bool, ...] | None:
    """Which blocks are linear attention (a recurrent state, no KV cache), as llama.cpp's
    `load_arch_hparams` marks them for the hybrid architectures it knows: a per-layer array
    when the GGUF has one, else every block but each `full_attention_interval`-th; the MTP
    blocks after the trunk are attention. None for every other architecture."""
    default = _FULL_ATTENTION_INTERVALS.get(architecture)
    if default is None or not block_count:
        return None
    if per_layer is not None and len(per_layer) == block_count:
        return tuple(bool(v) for v in per_layer)
    n = interval or default
    trunk = block_count - nextn
    return tuple(il < trunk and (il + 1) % n != 0 for il in range(block_count))


# Nemotron-H: three kinds of block, told apart by their per-layer arrays the way
# llama.cpp's `nemotron_h` loader does: attention (KV heads), an FFN or MoE block (a feed-
# forward width) and Mamba-2 (neither), which keeps a recurrent state.
_MAMBA_HYBRIDS = frozenset({"nemotron_h", "nemotron_h_moe"})


def mamba_layers(
    architecture: str,
    block_count: int | None,
    kv_heads: tuple[int, ...] | None,
    feed_forward: tuple[int, ...] | None,
) -> tuple[bool, ...] | None:
    """Nemotron-H's Mamba blocks: no KV heads and no feed-forward width. None elsewhere."""
    if architecture not in _MAMBA_HYBRIDS or not block_count:
        return None
    if not (kv_heads and feed_forward and len(kv_heads) == len(feed_forward) == block_count):
        return None
    return tuple(kv == 0 and ff == 0 for kv, ff in zip(kv_heads, feed_forward, strict=True))
