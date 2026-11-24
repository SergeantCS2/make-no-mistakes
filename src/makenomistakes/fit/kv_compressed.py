"""The KV cache of DeepSeek-V4 and V4.1's compressed attention, as llama.cpp's
`llama_kv_cache_dsv4` allocates it (src/llama-kv-cache-dsv4.cpp, master d854d98 after the
V4.1 merge 74bca39). Four parts, every one keys only:

- the raw cache: each block keeps a sliding window of uncompressed rows, `window + ubatch`
  cells padded to 256 per sequence (a stream per sequence, whatever `-np` says);
- the compressed caches: a block of compression r keeps one row per r tokens. V4 (`deepseek4`)
  compresses by 4 (CSA) or 128 (HCA) and gives each CSA block a lightning-indexer cache too;
  V4.1 (`deepseek41`) marks its blocks 2 or 1 (two tokens a row, or one), stores the index
  key beside each row, and only the blocks named in `attention.kv_source_layers` own rows:
  the blocks after a source read its cache (three CSA sources and one HCA source);
- the compressor states, f32 rows of the window being compressed.

Checked against `llama-fit-params --fit-print on --no-repack` on the CPU build over sparse
files rebuilt from the repos' headers: V4.1 UD-IQ1_S 55, 130 and 430 MiB at 8k,
32k and 128k (exactly); V4-Flash-0731 101, 258 and 903 MiB (within 1.6 MiB).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from makenomistakes.engine_args import KvType
from makenomistakes.gguf.reader import GgufHeader

PAD = 256  # each cache's cells are padded to this (GGML_PAD in the constructor)
_V4 = "deepseek4"
_V41 = "deepseek41"
COMPRESSED_ARCHITECTURES = frozenset({_V4, _V41})
_F32 = 4


def _pad(cells: int) -> int:
    return -(-cells // PAD) * PAD


def _cells(ctx: int, ratio: int) -> int:
    return _pad(max(1, -(-ctx // ratio)))


def compressed_kv_bytes(
    header: GgufHeader,
    ctx: int,
    bytes_per_value: Callable[[KvType], float],
    kv_type: KvType,
    ubatch: int,
    parallel: int,
) -> int | None:
    """The cache in bytes, or None when the header is not a compressed-attention model."""
    ratios = header.compress_ratios
    layers = header.block_count or 0
    key = header.key_length or 0
    if header.architecture not in COMPRESSED_ARCHITECTURES or not ratios or not layers or not key:
        return None
    ratios = tuple(ratios[:layers]) + (0,) * max(layers - len(ratios), 0)
    streams = max(parallel, 1)
    per_stream = max(ctx // streams, 1)
    window = header.sliding_window or 0
    raw_cells = min(per_stream, _pad(window + ubatch)) if window else per_stream
    values = layers * raw_cells * key  # the raw cache: every block's window
    states = 0  # f32 compressor rows
    index = header.indexer_key_length or 0
    if header.architecture == _V41:
        sources = set(header.kv_source_layers or ())
        for il, code in enumerate(ratios):
            if code and il in sources:  # only a source owns rows; the blocks after it read them
                values += _cells(per_stream, code) * (key + index)
                states += 2 * code * key  # kv and score: `code` rows of the head's width
    else:
        for code in ratios:
            if code == 4:  # CSA, its indexer, and their states (2 x ratio rows, 2 x width)
                values += _cells(per_stream, 4) * (key + index)
                states += 2 * 8 * 2 * key + 2 * 8 * 2 * index
            elif code:  # HCA
                values += _cells(per_stream, code) * key
                states += 2 * code * key
    return math.ceil(streams * (values * bytes_per_value(kv_type) + states * _F32))
