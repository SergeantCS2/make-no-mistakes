"""The byte-level side of a GGUF file: the cursor over any ByteSource (local or an HTTP
range), the value types, and reading or skipping one value. `reader` (the header),
`tensors` (the tensor infos) and `vocab` (the tokenizer) walk the metadata with it."""

from __future__ import annotations

import struct
from enum import IntEnum
from pathlib import Path
from typing import Protocol

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = {2, 3}
CHUNK_BYTES = 256 * 1024
MAX_CHUNK_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024 * 1024


class GgufError(ValueError):
    pass


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


def _value_type(raw: int) -> ValueType:
    """A damaged or newer-format file is a GgufError like any other unreadable header, so a
    caller that tolerates bad files (the instance list) never sees a bare ValueError."""
    try:
        return ValueType(raw)
    except ValueError:
        raise GgufError(f"unknown GGUF value type {raw}") from None


_SCALARS: dict[ValueType, str] = {
    ValueType.UINT8: "<B",
    ValueType.INT8: "<b",
    ValueType.UINT16: "<H",
    ValueType.INT16: "<h",
    ValueType.UINT32: "<I",
    ValueType.INT32: "<i",
    ValueType.FLOAT32: "<f",
    ValueType.BOOL: "<?",
    ValueType.UINT64: "<Q",
    ValueType.INT64: "<q",
    ValueType.FLOAT64: "<d",
}

Scalar = int | float | bool | str


class ByteSource(Protocol):
    def read_range(self, start: int, length: int) -> bytes:
        """Return up to `length` bytes starting at `start`; fewer only at end of file."""
        ...


class FileSource:
    def __init__(self, path: Path) -> None:
        self._path = path

    def read_range(self, start: int, length: int) -> bytes:
        with self._path.open("rb") as handle:
            handle.seek(start)
            return handle.read(length)


class _Cursor:
    def __init__(self, source: ByteSource, grow: bool = False) -> None:
        self._source = source
        # The least each read asks for (one range request over HTTP); `grow` doubles it after
        # each read up to 8 MiB: a whole metadata section in a few requests, while a small
        # shard header still costs one 256 KiB read.
        self._chunk = CHUNK_BYTES
        self._grow = grow
        self._buffer = bytearray()
        self._end = 0
        self.offset = 0

    def read(self, length: int) -> bytes:
        if self.offset + length > MAX_METADATA_BYTES:
            raise GgufError("metadata section is larger than the supported limit")
        while self._end < self.offset + length:
            want = max(self._chunk, self.offset + length - self._end)
            chunk = self._source.read_range(self._end, want)
            if not chunk:
                raise GgufError("unexpected end of file inside GGUF metadata")
            self._buffer.extend(chunk)
            self._end += len(chunk)
            if self._grow:
                self._chunk = min(self._chunk * 2, MAX_CHUNK_BYTES)
        data = bytes(self._buffer[self.offset : self.offset + length])
        self.offset += length
        return data

    def unpack(self, fmt: str) -> int | float | bool:
        value: int | float | bool = struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]
        return value

    def u32(self) -> int:
        return int(self.unpack("<I"))

    def u64(self) -> int:
        return int(self.unpack("<Q"))

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")


MAX_LAYER_ARRAY = 4096


_MIN_ELEMENT_BYTES = {ValueType.STRING: 8, ValueType.ARRAY: 12}  # their length fields alone


def _check_count(cursor: _Cursor, element_type: ValueType, count: int) -> None:
    """A count no metadata section this size could hold is a damaged or hostile file: said at
    once, not after walking 2**40 elements (which took 40 s and fetched 64 MiB)."""
    smallest = _MIN_ELEMENT_BYTES.get(element_type) or struct.calcsize(_SCALARS[element_type])
    if cursor.offset + count * smallest > MAX_METADATA_BYTES:
        raise GgufError(f"an array of {count} elements runs past the metadata limit")


def _skip_array(cursor: _Cursor, element_type: ValueType, count: int) -> None:
    _check_count(cursor, element_type, count)
    if element_type in _SCALARS:  # fixed size: one read past the whole block
        cursor.read(count * struct.calcsize(_SCALARS[element_type]))
        return
    for _ in range(count):
        _read_value(cursor, element_type)


def _small_int_array(cursor: _Cursor) -> tuple[int, ...] | None:
    """A short array of integers or booleans (per-layer settings); anything else is walked
    past and dropped, like every other array."""
    element_type = _value_type(cursor.u32())
    count = cursor.u64()
    if count > MAX_LAYER_ARRAY or element_type in (ValueType.STRING, ValueType.ARRAY):
        _skip_array(cursor, element_type, count)
        return None
    values = [_read_value(cursor, element_type) for _ in range(count)]
    if not all(isinstance(v, int | bool) for v in values):
        return None
    return tuple(int(v) for v in values if isinstance(v, int | bool))


def _read_value(cursor: _Cursor, value_type: ValueType) -> Scalar | None:
    """Scalars and strings are returned; arrays are walked and dropped (we never need them)."""
    if value_type is ValueType.STRING:
        return cursor.string()
    if value_type is ValueType.ARRAY:
        element_type = _value_type(cursor.u32())
        _skip_array(cursor, element_type, cursor.u64())
        return None
    return cursor.unpack(_SCALARS[value_type])
