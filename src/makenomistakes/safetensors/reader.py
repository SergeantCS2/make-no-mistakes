"""Read the JSON header at the front of a .safetensors file from any ByteSource.

Layout: a little-endian u64 header length N, then N bytes of UTF-8 JSON mapping each tensor
name to `{"dtype", "shape", "data_offsets": [begin, end]}`, plus an optional `__metadata__`
map of strings. The header of a 6.9 GB SDXL checkpoint is about 360 KB, so a plan reads two
ranges over HTTP instead of downloading the model.
"""

from __future__ import annotations

import json
import math
import struct

from pydantic import BaseModel, ValidationError, model_validator

from makenomistakes.gguf.reader import ByteSource

MAX_HEADER_BYTES = 32 * 1024 * 1024  # real headers are well under 2 MB


class SafetensorsError(ValueError):
    pass


class TensorInfo(BaseModel):
    dtype: str  # "F16", "BF16", "F32", "F8_E4M3", …
    shape: list[int]
    data_offsets: tuple[int, int]

    @model_validator(mode="after")
    def _sizes_are_sizes(self) -> TensorInfo:
        """A negative dimension or reversed offsets would count negative bytes."""
        begin, end = self.data_offsets
        if any(dim < 0 for dim in self.shape) or not 0 <= begin <= end:
            raise ValueError("negative shape or reversed data offsets")
        return self

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


class SafetensorsHeader(BaseModel):
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str] = {}


def read_safetensors_header(source: ByteSource) -> SafetensorsHeader:
    first = source.read_range(0, 8)
    if len(first) < 8:
        raise SafetensorsError("file too short for a safetensors header")
    (length,) = struct.unpack("<Q", first)
    if not 2 <= length <= MAX_HEADER_BYTES:
        raise SafetensorsError(f"implausible safetensors header length {length}")
    raw = source.read_range(8, length)
    if len(raw) < length:
        raise SafetensorsError("unexpected end of file inside the safetensors header")
    try:
        data = json.loads(raw)
    except ValueError as exc:  # UnicodeDecodeError is a ValueError too
        raise SafetensorsError(f"the safetensors header is not JSON: {exc}") from None
    except RecursionError:  # a header is one level of objects; brackets nested past Python's
        raise SafetensorsError("the safetensors header nests too deep to be one") from None
    if not isinstance(data, dict):
        raise SafetensorsError("the safetensors header is not a JSON object")
    meta = data.pop("__metadata__", None)
    tensors: dict[str, TensorInfo] = {}
    for name, value in data.items():
        try:
            tensors[name] = TensorInfo.model_validate(value)
        except ValidationError as exc:
            first_error = exc.errors()[0]
            where = ".".join(str(part) for part in (name, *first_error["loc"]))
            raise SafetensorsError(f"bad tensor entry ({where}: {first_error['msg']})") from None
    metadata = {str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {}
    return SafetensorsHeader(tensors=tensors, metadata=metadata)
