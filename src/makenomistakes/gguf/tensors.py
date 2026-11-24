"""A GGUF's tensor infos: each tensor's name, shape and type, read after the whole metadata
section. `read_header` stops before the tokenizer arrays and never sees them; the
fit model reads them for what the keys do not say: which blocks carry routed experts and
how many weights each holds (Nemotron Bolt: 40 of 88 blocks, 240 to 496 experts each), and
which tensor is an Engram table that stays on disk (`per_layer_token_embd.weight`).

unsloth's split GGUFs keep every key in shard 1 (no tensors) and each later shard's own
tensor list behind three `split.*` keys, so a split model's infos cost one small read per
shard; a single file's sit after its tokenizer (about 8 MB into Bolt's).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from makenomistakes.gguf.cursor import (
    GGUF_MAGIC,
    SUPPORTED_VERSIONS,
    ByteSource,
    GgufError,
    _Cursor,
    _read_value,
    _value_type,
)

if TYPE_CHECKING:  # reader imports this module for `ModelTensors`
    from makenomistakes.gguf.reader import GgufHeader

ENGRAM_TENSOR = "per_layer_token_embd.weight"  # mainline llama.cpp's name for the table
# unsloth's MXFP8 Engram tables (V4.1 Q3_K_XL and up, 1.03125 bytes a weight); ggml's own 43
# is F8_E4M3, a byte a weight, since llama.cpp 68d20b884 (2026-10-29): same id, other layout
MXFP8_TYPE = 43
_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")
_EXPERTS_RE = re.compile(r"^blk\.\d+\.ffn_(?:gate|up|down|gate_up)_(?:ch)?exps\.weight$")
MAX_TENSORS = 1 << 20

# ggml's types (ggml.h, 2026-10): the weights a block holds and the block's bytes.
GGML_BLOCKS: dict[int, tuple[int, int]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
}


class TensorInfo(BaseModel):
    name: str
    dims: tuple[int, ...]
    type: int
    offset: int  # from the start of the file's tensor data

    @property
    def elements(self) -> int:
        return math.prod(self.dims)

    @property
    def nbytes(self) -> int | None:
        """Its size on disk, None for a type this table does not know (type 43)."""
        block = GGML_BLOCKS.get(self.type)
        return None if block is None else self.elements // block[0] * block[1]

    @property
    def block(self) -> int | None:
        match = _BLOCK_RE.match(self.name)
        return int(match.group(1)) if match else None

    @property
    def routed_experts(self) -> bool:
        return _EXPERTS_RE.match(self.name) is not None


class GgufTensors(BaseModel):
    """One file's tensor infos and where they end (the length a recorded header needs)."""

    tensors: tuple[TensorInfo, ...]
    end: int  # the byte offset after the last tensor info


def read_tensors(source: ByteSource) -> GgufTensors:
    """Every tensor info of a GGUF file: the metadata walked (arrays skipped), then the list."""
    cursor = _Cursor(source, grow=True)
    if cursor.read(4) != GGUF_MAGIC:
        raise GgufError("not a GGUF file (bad magic)")
    if cursor.u32() not in SUPPORTED_VERSIONS:
        raise GgufError("unsupported GGUF version")
    count = cursor.u64()
    kv_count = cursor.u64()
    if count > MAX_TENSORS:
        raise GgufError(f"{count} tensors is more than any model holds")
    for _ in range(kv_count):
        cursor.string()
        _read_value(cursor, _value_type(cursor.u32()))
    tensors = []
    for _ in range(count):
        name = cursor.string()
        dims = tuple(cursor.u64() for _ in range(cursor.u32()))
        kind = cursor.u32()
        tensors.append(TensorInfo(name=name, dims=dims, type=kind, offset=cursor.u64()))
    return GgufTensors(tensors=tuple(tensors), end=cursor.offset)


class ModelTensors(BaseModel):
    """What the planner reads from a model's tensor infos (every shard's together)."""

    # Each block's routed-expert weights (`ffn_*_exps`) and experts; 0 on a block without.
    expert_weights: tuple[int, ...]
    expert_counts: tuple[int, ...]
    engram: TensorInfo | None = None


def model_tensors(tensors: list[TensorInfo], block_count: int) -> ModelTensors:
    weights = [0] * block_count
    counts = [0] * block_count
    for tensor in tensors:
        il = tensor.block
        if tensor.routed_experts and il is not None and il < block_count:
            weights[il] += tensor.elements
            counts[il] = tensor.dims[-1] if tensor.dims else 0
    engram = next((t for t in tensors if t.name == ENGRAM_TENSOR), None)
    return ModelTensors(expert_weights=tuple(weights), expert_counts=tuple(counts), engram=engram)


def needs_tensors(header: GgufHeader) -> bool:
    """Whether the keys leave a block's experts unknown (a per-layer `expert_used_count`), so
    the plan and Fits now read the tensor infos (`GgufHeader.with_tensors`)."""
    return header.is_moe and header.expert_used_layers is not None
