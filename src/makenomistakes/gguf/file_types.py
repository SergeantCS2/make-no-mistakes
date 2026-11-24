"""llama.cpp's file types (`enum llama_ftype` in `llama.h`): the id `general.file_type` holds,
the name quant labels use, and the bits per weight of the tensors a mix of that type
quantizes alike (the embedding table and output head follow it).

Real files put `general.file_type` after the tokenizer arrays, where the header reader stops
(every unsloth, Qwen and Gemma GGUF checked, 2026-10-15: at ~5.9 MB), so a plan reads a
variant's density from its quant label (`file_type_of_quant`).
"""

from __future__ import annotations

from typing import NamedTuple


class FileType(NamedTuple):
    name: str
    id: int
    bits: float


FILE_TYPES: tuple[FileType, ...] = (
    FileType("F32", 0, 32.0),
    FileType("F16", 1, 16.0),
    FileType("Q4_0", 2, 4.5),
    FileType("Q4_1", 3, 5.0),
    FileType("Q8_0", 7, 8.5),
    FileType("Q5_0", 8, 5.5),
    FileType("Q5_1", 9, 6.0),
    FileType("Q2_K", 10, 2.6),
    FileType("Q3_K_S", 11, 3.4),
    FileType("Q3_K_M", 12, 3.9),
    FileType("Q3_K_L", 13, 4.3),
    FileType("Q4_K_S", 14, 4.6),
    FileType("Q4_K_M", 15, 4.9),
    FileType("Q5_K_S", 16, 5.5),
    FileType("Q5_K_M", 17, 5.7),
    FileType("Q6_K", 18, 6.6),
    FileType("IQ2_XXS", 19, 2.1),
    FileType("IQ2_XS", 20, 2.3),
    FileType("Q2_K_S", 21, 2.5),
    FileType("IQ3_XS", 22, 3.3),
    FileType("IQ3_XXS", 23, 3.1),
    FileType("IQ1_S", 24, 1.6),
    FileType("IQ4_NL", 25, 4.5),
    FileType("IQ3_S", 26, 3.4),
    FileType("IQ3_M", 27, 3.7),
    FileType("IQ2_S", 28, 2.5),
    FileType("IQ2_M", 29, 2.7),
    FileType("IQ4_XS", 30, 4.3),
    FileType("IQ1_M", 31, 1.8),
    FileType("BF16", 32, 16.0),
    FileType("TQ1_0", 36, 1.7),
    FileType("TQ2_0", 37, 2.1),
    FileType("MXFP4_MOE", 38, 4.25),
)
BITS_BY_ID: dict[int, float] = {t.id: t.bits for t in FILE_TYPES}
# The codebook i-quants: a weight group is an index into a lattice grid (with sign bits);
# their dequantisation, not the bytes, bounds a decode step (`fit.estimate.IQ_GRID_EFFICIENCY`).
GRID_QUANT_IDS: frozenset[int] = frozenset(
    t.id for t in FILE_TYPES if t.name.startswith(("IQ1_", "IQ2_", "IQ3_"))
)
_ID_BY_NAME: dict[str, int] = {t.name: t.id for t in FILE_TYPES} | {"MXFP4": 38, "Q8_K": 7}


def file_type_of_quant(label: str) -> int | None:
    """The file type a quant label names. unsloth's `UD-` mixes keep a few layers higher and
    are priced as their base: `UD-Q4_K_XL` as `Q4_K_M`, `Q2_K_XL`/`Q6_K_XL` (no `_M` of their
    own) as `Q2_K`/`Q6_K`, `Q8_K_XL` as `Q8_0`; `Q2_K_L` as `Q2_K`. None for a label that
    names no file type (`mmproj`, a whisper size)."""
    name = label.upper().removeprefix("UD-")
    if name in _ID_BY_NAME:
        return _ID_BY_NAME[name]
    for suffix in ("_XL", "_L"):
        if not name.endswith(suffix):
            continue
        base = name.removesuffix(suffix)
        for candidate in (f"{base}_M", base):
            if candidate in _ID_BY_NAME:
                return _ID_BY_NAME[candidate]
    return None
