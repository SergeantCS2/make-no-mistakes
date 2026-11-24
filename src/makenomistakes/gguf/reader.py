"""Read the metadata at the front of a GGUF file from any ByteSource (local or HTTP range).

Layout (GGUF v3, little-endian): magic "GGUF", version u32, tensor_count u64, kv_count u64,
then key/value pairs. A string is a u64 byte length followed by UTF-8 bytes; an array is an
element type u32, a count u64, then the elements. The architecture keys we need come before
the tokenizer arrays, so by default reading stops at the first `tokenizer.ggml.` array and
costs a few hundred kilobytes instead of several megabytes. The token list's element count
is read on the way (no elements): most GGUFs carry the vocabulary size only there.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from makenomistakes.gguf import layer_patterns
from makenomistakes.gguf.cursor import (
    CHUNK_BYTES,
    GGUF_MAGIC,
    MAX_LAYER_ARRAY,
    MAX_METADATA_BYTES,
    SUPPORTED_VERSIONS,
    ByteSource,
    FileSource,
    GgufError,
    Scalar,
    ValueType,
    _check_count,
    _Cursor,
    _read_value,
    _skip_array,
    _small_int_array,
    _value_type,
)
from makenomistakes.gguf.file_types import BITS_BY_ID, file_type_of_quant
from makenomistakes.gguf.tensors import ModelTensors

__all__ = [
    "CHUNK_BYTES",
    "GGUF_MAGIC",
    "MAX_LAYER_ARRAY",
    "MAX_METADATA_BYTES",
    "SUPPORTED_VERSIONS",
    "ByteSource",
    "FileSource",
    "GgufError",
    "GgufHeader",
    "Scalar",
    "ValueType",
    "_Cursor",
    "_check_count",
    "_read_value",
    "_skip_array",
    "_small_int_array",
    "_value_type",
    "file_type_of_quant",
]

DEFAULT_STOP_PREFIX = "tokenizer.ggml."
_TOKENS_KEY = "tokenizer.ggml.tokens"

# llama.cpp architectures that encode (no output head over the vocabulary).
_ENCODER_ARCHS = (
    "bert",
    "nomic-bert",
    "jina-bert",
    "neo-bert",
    "modern-bert",
    "eurobert",
    "t5encoder",
)


class GgufLayout(StrEnum):
    """Which converter wrote a DeepSeek-V4.1 GGUF: three layouts no runtime reads all
    of. llama.cpp's merged one (#31086, unsloth's: `ple.layers`, the Engram table as
    `per_layer_token_embd.weight`), ik_llama.cpp's (vcruz305's: `engram.layer_ids`, a
    `blk.N.engram_embd` table) and antirez's DwarfStar (`engram.encoding` and the whole HF
    config as `{arch}.config`, no `block_count`). Every other GGUF is mainline."""

    MAINLINE = "mainline"
    IK = "ik_llama.cpp"
    DWARFSTAR = "DwarfStar"


def _layout(arch: str, values: dict[str, Scalar], arrays: dict[str, tuple[int, ...]]) -> GgufLayout:
    if f"{arch}.engram.encoding" in values or f"{arch}.config" in values:
        return GgufLayout.DWARFSTAR
    if f"{arch}.engram.layer_ids" in arrays and f"{arch}.ple.layers" not in arrays:
        return GgufLayout.IK
    return GgufLayout.MAINLINE


class GgufHeader(BaseModel):
    """The architecture facts the fit engine needs, plus what was read to get them."""

    version: int
    tensor_count: int
    kv_count: int
    architecture: str
    name: str | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    key_length: int | None = None
    value_length: int | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None
    expert_feed_forward_length: int | None = None  # `{arch}.expert_feed_forward_length`
    vocab_size: int | None = None  # `{arch}.vocab_size`, else the token list's length
    # Per-layer head counts (Nemotron-NAS, OpenELM; LFM2, Granite-H, Jamba with 0 on their
    # recurrent layers), and the layers that keep only a sliding window of the KV cache.
    head_count_layers: tuple[int, ...] | None = None
    head_count_kv_layers: tuple[int, ...] | None = None
    sliding_window: int | None = None
    swa_layers: tuple[bool, ...] | None = None
    # A sliding-window layer's own head size (Gemma 4: 256 against 512 on full layers).
    key_length_swa: int | None = None
    value_length_swa: int | None = None
    # Hybrid attention: the linear-attention layers (Gated DeltaNet in Qwen3.5 and
    # 3.6, lightning attention in MiniMax-01) keep a recurrent state instead of a KV cache.
    recurrent_layers: tuple[bool, ...] | None = None
    ssm_conv_kernel: int | None = None
    ssm_inner_size: int | None = None
    ssm_state_size: int | None = None
    ssm_group_count: int | None = None
    # The multi-token-prediction blocks counted in `block_count` (`nextn_predict_layers`):
    # the trunk's caches skip them, an MTP context holds only them.
    nextn_layers: int = 0
    # Gemma 3n/4: the last this many layers reuse an earlier layer's KV and keep none.
    shared_kv_layers: int = 0
    causal: bool | None = None  # `{arch}.attention.causal` when the GGUF declares it
    # Latent attention converted for llama.cpp's MLA path (DeepSeek-2, GLM-4.7-Flash,
    # Mistral-Small-4: `key_length_mla` written) caches one compressed row a token a layer,
    # `key_length` wide (the latent rank + the rope part), keys only.
    key_length_mla: int | None = None
    # DeepSeek-V4/V4.1's compressed attention (`fit.kv_compressed`): each block's compression
    # (V4 4 or 128, V4.1 2 or 1, 0 none), the blocks whose compressed rows others read (V4.1),
    # and the sparse indexer's key size.
    compress_ratios: tuple[int, ...] | None = None
    kv_source_layers: tuple[int, ...] | None = None
    indexer_key_length: int | None = None
    # Routed experts per block: `expert_used_count` as an array (Nemotron Bolt: 0 on its
    # Mamba and attention blocks), and from the tensor infos (`with_tensors`) each block's
    # expert weights and experts; None: every block alike, as the scalar keys say.
    expert_used_layers: tuple[int, ...] | None = None
    expert_weight_layers: tuple[int, ...] | None = None
    expert_count_layers: tuple[int, ...] | None = None
    layout: GgufLayout = GgufLayout.MAINLINE
    # DeepSeek-V4.1's Engram (`ple.layers`, the merged layout): the blocks that read the
    # n-gram table, which a split GGUF keeps in a shard of its own (`plan_large` finds it).
    engram_layers: tuple[int, ...] | None = None
    file_type: int | None = None
    size_label: str | None = None
    parameter_count: int | None = None  # `general.parameter_count`, when written
    complete: bool

    def own_quant(self, label: str) -> GgufHeader:
        """This file's own header at the density `label` names, keeping a file type it read
        when the label names none (unlike `for_quant`, which stands in for other files)."""
        return self if file_type_of_quant(label) is None else self.for_quant(label)

    def for_quant(self, label: str) -> GgufHeader:
        """This header with the density of the variant `label` names (the plan reads one file's
        header for every variant of a repo, and real files carry no readable file type)."""
        file_type = file_type_of_quant(label)
        # An unknown label must not inherit the density of whichever file was read.
        return self.model_copy(update={"file_type": file_type})

    @property
    def bits_per_weight(self) -> float | None:
        """The file type's bits per weight (llama.cpp's `general.file_type` enum); None for
        a type this table does not know, which the fit model treats as "do not split"."""
        return None if self.file_type is None else BITS_BY_ID.get(self.file_type)

    @property
    def token_embedding_bytes(self) -> int:
        """The input embedding table (vocab x embedding length at the file's density). llama.cpp
        keeps it in RAM and reads one row per token, so it is never streamed from the GPU;
        the output head has the same shape (a tied model keeps a GPU copy of the table as
        its head). 0 when the header cannot size it."""
        bits = self.bits_per_weight
        if not (bits and self.vocab_size and self.embedding_length):
            return 0
        return int(self.vocab_size * self.embedding_length * bits / 8)

    @property
    def head_dim(self) -> int | None:
        heads = self.head_count or max(self.head_count_layers or (0,))
        if self.key_length:
            return self.key_length
        if self.embedding_length and heads:
            return self.embedding_length // heads
        return None

    @property
    def kv_heads(self) -> int | None:
        """The widest layer's KV heads (`kv_heads_per_layer` has every layer's)."""
        return (
            self.head_count_kv
            or max(self.head_count_kv_layers or (0,))
            or self.head_count
            or max(self.head_count_layers or (0,))
            or None
        )

    def kv_heads_per_layer(self) -> tuple[int, ...]:
        """Each block's KV heads: 0 on a recurrent layer, which keeps no KV cache."""
        layers = self.block_count or 0
        per_layer = self.head_count_kv_layers or (
            self.head_count_layers if self.head_count_kv is None else None
        )
        if per_layer and len(per_layer) == layers:
            return per_layer
        return (self.kv_heads or 0,) * layers

    @property
    def trunk_layers(self) -> int:
        """The blocks the model runs for every token: `block_count` less its MTP blocks."""
        return max((self.block_count or 0) - self.nextn_layers, 0)

    def kv_layers(self, mtp: bool = False) -> tuple[bool, ...]:
        """Which blocks keep a KV cache in a context, as llama.cpp's memory filters decide
        (`llama_model::create_memory`): the trunk's (an MTP context: its MTP blocks only, or
        every block of a head that is all MTP blocks), less the recurrent layers and the
        layers that share another's KV (`shared_kv_layers`)."""
        layers, trunk = self.block_count or 0, self.trunk_layers
        recurrent = self.recurrent_layers or (False,) * layers
        own = layers - self.shared_kv_layers  # llama.cpp's `n_layer_kv_from_start`
        split = self.nextn_layers > 0 and trunk > 0
        return tuple(
            (not split or (il >= trunk if mtp else il < trunk)) and not recurrent[il] and il < own
            for il in range(layers)
        )

    def head_dims(self, il: int) -> tuple[int, int]:
        """Block `il`'s key and value sizes: a sliding-window layer's own when the GGUF
        gives them (`key_length_swa`), else the model's."""
        key = self.head_dim or 0
        value = self.value_length or key
        if self.swa_layers is not None and self.swa_layers[il] and self.key_length_swa:
            return self.key_length_swa, self.value_length_swa or self.key_length_swa
        return key, value

    @property
    def recurrent_row_values(self) -> int:
        """One recurrent layer's state per sequence, in floats: the convolution's history
        and the state matrix (llama.cpp's `n_embd_r` + `n_embd_s`, the Mamba/DeltaNet shape),
        0 when the header does not size them."""
        conv, inner = self.ssm_conv_kernel, self.ssm_inner_size
        state, groups = self.ssm_state_size, self.ssm_group_count or 0
        if not (conv and inner and state):
            return 0
        return (conv - 1) * (inner + 2 * groups * state) + state * inner

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_count and self.expert_count > 1)

    @property
    def expert_bytes_per_layer(self) -> int:
        """One layer's routed experts as stored: gate, up and down projections of every expert
        (`ffn_{gate,up,down}_exps`, 3 x n_expert x n_ff_exp x n_embd weights) at the file
        type's bits. What `--n-cpu-moe N` keeps in RAM for N layers. 0 when the header
        cannot size them (a dense model, or a file type this reader does not know)."""
        bits = self.bits_per_weight
        if not (self.is_moe and bits and self.expert_feed_forward_length and self.embedding_length):
            return 0
        weights = 3 * (self.expert_count or 0) * self.expert_feed_forward_length
        return int(weights * self.embedding_length * bits / 8)

    def with_tensors(self, found: ModelTensors) -> GgufHeader:
        """This header with what the tensor infos say per block: which blocks carry
        routed experts, how many weights and experts each (the scalar keys say one number
        for all; Bolt's differ block by block and 48 of its 88 blocks carry none)."""
        if not any(found.expert_weights) or len(found.expert_weights) != (self.block_count or 0):
            return self
        return self.model_copy(
            update={
                "expert_weight_layers": found.expert_weights,
                "expert_count_layers": found.expert_counts,
            }
        )

    @property
    def uniform_experts(self) -> bool:
        """Every block's experts alike, as the scalar keys say (no per-block facts read)."""
        return self.expert_weight_layers is None and self.expert_used_layers is None

    def expert_layer_bytes(self) -> tuple[int, ...]:
        """Each block's routed experts as stored at the variant's bits (the quant label's,
        never a split's `general.file_type`): from the tensor infos when they were read,
        else `expert_bytes_per_layer` on every block."""
        layers = self.block_count or 0
        bits = self.bits_per_weight
        weights = self.expert_weight_layers
        if weights is None or not bits:
            return (self.expert_bytes_per_layer,) * layers
        return tuple(int(w * bits / 8) for w in weights)

    def _used_and_counts(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        layers = self.block_count or 0
        used = self.expert_used_layers or (self.expert_used_count or 0,) * layers
        counts = self.expert_count_layers or (self.expert_count or 0,) * layers
        if len(used) != layers or len(counts) != layers:
            used, counts = (
                (self.expert_used_count or 0,) * layers,
                (self.expert_count or 0,) * layers,
            )
        return tuple(used), tuple(counts)

    def expert_shares(self) -> tuple[float, ...]:
        """Each block's active share of its experts: its used count over its experts, from
        the per-block arrays and tensors when there are, else the keys. No floor: a 0.05 floor
        priced DeepSeek's 8 of 256 as 13 of 256; what it stood in for is a cost per used
        expert, `fit.cpu_rates.EXPERT_TOUCH_SECONDS`."""
        used, counts = self._used_and_counts()
        return tuple(u / c if u and c else 1.0 for u, c in zip(used, counts, strict=True))

    def experts_used(self) -> tuple[int, ...]:
        """Each block's used experts a token (0 where the block holds no routed experts)."""
        used, counts = self._used_and_counts()
        return tuple(u if u and c else 0 for u, c in zip(used, counts, strict=True))

    @property
    def latent_kv_unpriced(self) -> bool:
        """DeepSeek-style attention whose cache the fit model does not size exactly (no
        `key_length_mla`, no compressed-attention keys): its KV estimate is pessimistic."""
        priced = self.key_length_mla is not None or self.compress_ratios is not None
        return self.uses_latent_attention and not priced

    @property
    def computes_logits(self) -> bool:
        """Whether the graph ends in an output head over the vocabulary. Decoder models used
        for embeddings (Qwen3-Embedding) still compute it for every output token; encoders
        (BERT, XLM-R rerankers) do not (measured on llama.cpp 79e2e74: 5,028 vs 416 MiB of
        compute buffer at ubatch 8192)."""
        if self.causal is not None:
            return self.causal
        return not self.architecture.startswith(_ENCODER_ARCHS)

    @property
    def uses_latent_attention(self) -> bool:
        """DeepSeek MLA stores a compressed latent per token; our KV math over-estimates it."""
        return self.architecture.startswith("deepseek")


def read_header(source: ByteSource, stop_prefix: str | None = DEFAULT_STOP_PREFIX) -> GgufHeader:
    cursor = _Cursor(source)
    if cursor.read(4) != GGUF_MAGIC:
        raise GgufError("not a GGUF file (bad magic)")
    version = cursor.u32()
    if version not in SUPPORTED_VERSIONS:
        raise GgufError(f"unsupported GGUF version {version}")
    tensor_count = cursor.u64()
    kv_count = cursor.u64()

    values: dict[str, Scalar] = {}
    arrays: dict[str, tuple[int, ...]] = {}
    complete = True
    tokens: int | None = None
    for _ in range(kv_count):
        key = cursor.string()
        value_type = _value_type(cursor.u32())
        stop = stop_prefix is not None and key.startswith(stop_prefix)
        if stop and value_type is ValueType.ARRAY:  # the big ones: count the tokens, read none
            cursor.u32()  # element type
            tokens = cursor.u64() if key == _TOKENS_KEY else None
            complete = False
            break
        if value_type is ValueType.ARRAY and key.endswith(_LAYER_ARRAYS):
            small = _small_int_array(cursor)
            if small is not None:
                arrays[key] = small
            continue
        value = _read_value(cursor, value_type)
        if value is not None:
            values[key] = value

    architecture = values.get("general.architecture")
    if not isinstance(architecture, str):
        raise GgufError("GGUF metadata has no general.architecture")
    attention = f"{architecture}.attention"
    block_count = _int(values, f"{architecture}.block_count")
    window = _int(values, f"{attention}.sliding_window")
    nextn = _int(values, f"{architecture}.nextn_predict_layers") or 0
    return GgufHeader(
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        architecture=architecture,
        name=_str(values, "general.name"),
        context_length=_int(values, f"{architecture}.context_length"),
        embedding_length=_int(values, f"{architecture}.embedding_length"),
        block_count=block_count,
        head_count=_int(values, f"{architecture}.attention.head_count"),
        head_count_kv=_int(values, f"{architecture}.attention.head_count_kv"),
        key_length=_int(values, f"{architecture}.attention.key_length"),
        value_length=_int(values, f"{architecture}.attention.value_length"),
        expert_count=_int(values, f"{architecture}.expert_count"),
        expert_used_count=_int(values, f"{architecture}.expert_used_count"),
        expert_used_layers=arrays.get(f"{architecture}.expert_used_count"),
        expert_feed_forward_length=_int(values, f"{architecture}.expert_feed_forward_length"),
        vocab_size=_int(values, f"{architecture}.vocab_size") or tokens,
        causal=_bool(values, f"{architecture}.attention.causal"),
        key_length_mla=_int(values, f"{attention}.key_length_mla"),
        compress_ratios=arrays.get(f"{attention}.compress_ratios"),
        kv_source_layers=arrays.get(f"{attention}.kv_source_layers"),
        indexer_key_length=_int(values, f"{attention}.indexer.key_length"),
        layout=_layout(architecture, values, arrays),
        engram_layers=arrays.get(f"{architecture}.ple.layers")
        or arrays.get(f"{architecture}.engram.layer_ids"),
        head_count_layers=arrays.get(f"{attention}.head_count"),
        head_count_kv_layers=arrays.get(f"{attention}.head_count_kv"),
        sliding_window=window,
        swa_layers=layer_patterns.swa_layers(
            architecture,
            block_count,
            window,
            arrays.get(f"{attention}.sliding_window_pattern"),
            _int(values, f"{attention}.sliding_window_pattern"),
        ),
        key_length_swa=_int(values, f"{attention}.key_length_swa"),
        value_length_swa=_int(values, f"{attention}.value_length_swa"),
        recurrent_layers=layer_patterns.recurrent_layers(
            architecture,
            block_count,
            nextn,
            arrays.get(f"{attention}.recurrent_layers"),
            _int(values, f"{architecture}.full_attention_interval"),
        )
        or layer_patterns.mamba_layers(
            architecture,
            block_count,
            arrays.get(f"{attention}.head_count_kv"),
            arrays.get(f"{architecture}.feed_forward_length"),
        ),
        ssm_conv_kernel=_int(values, f"{architecture}.ssm.conv_kernel"),
        ssm_inner_size=_int(values, f"{architecture}.ssm.inner_size"),
        ssm_state_size=_int(values, f"{architecture}.ssm.state_size"),
        ssm_group_count=_int(values, f"{architecture}.ssm.group_count"),
        nextn_layers=nextn,
        shared_kv_layers=_int(values, f"{attention}.shared_kv_layers") or 0,
        file_type=_int(values, "general.file_type"),
        size_label=_str(values, "general.size_label"),
        parameter_count=_int(values, "general.parameter_count"),
        complete=complete,
    )


_LAYER_ARRAYS = (
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.sliding_window_pattern",
    ".attention.recurrent_layers",
    # Per-layer experts and FFNs (Nemotron-H), DeepSeek-V4's compressed attention, and
    # the keys that tell the V4.1 layouts apart
    ".expert_used_count",
    ".feed_forward_length",
    ".attention.compress_ratios",
    ".attention.kv_source_layers",
    ".ple.layers",
    ".engram.layer_ids",
)


def _int(values: dict[str, Scalar], key: str) -> int | None:
    value = values.get(key)
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _bool(values: dict[str, Scalar], key: str) -> bool | None:
    value = values.get(key)
    return value if isinstance(value, bool) else None


def _str(values: dict[str, Scalar], key: str) -> str | None:
    value = values.get(key)
    return value if isinstance(value, str) else None
