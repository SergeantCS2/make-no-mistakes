"""Small media files read and written without an imaging library: a WAV's length, a flat grey
PNG (an edit's test picture, a video's first frame), and a picture's type from its first bytes."""

from __future__ import annotations

import dataclasses
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IMAGE_TYPES = (
    (PNG_SIGNATURE, "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


@dataclass(frozen=True)
class WavHeader:
    """A RIFF/WAVE file's format and the size its data chunk says it holds."""

    rate: int
    channels: int
    byte_rate: int
    block: int
    data_bytes: int  # as the header declares it (a cut file holds less)
    data_start: int  # where the samples begin

    @property
    def frames(self) -> int:
        return self.data_bytes // self.block if self.block else 0

    @property
    def seconds(self) -> float:
        return self.data_bytes / self.byte_rate if self.byte_rate else 0.0


def wav_header(head: bytes) -> WavHeader | None:
    """The format and data chunks from a WAV file's first bytes (the whole file, or its
    first 64 KiB: the samples are never needed); None for anything else."""
    if len(head) < 44 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    offset, fmt, data, start = 12, None, 0, 0
    while offset + 8 <= len(head):
        kind, size = (
            head[offset : offset + 4],
            struct.unpack("<I", head[offset + 4 : offset + 8])[0],
        )
        if kind == b"fmt " and size >= 16:
            fmt = struct.unpack("<HHIIH", head[offset + 8 : offset + 22])
        elif kind == b"data":
            data, start = size, offset + 8
            break
        offset += 8 + size + (size % 2)
    if fmt is None or not data:
        return None
    _format, channels, rate, byte_rate, block = fmt
    return WavHeader(rate, channels, byte_rate, block, data, start)


def wav_seconds(wav: bytes) -> float | None:
    """The clip's length from a RIFF/WAVE header's format and data chunks, at most what the
    bytes hold; None when they are not a WAV file (the engine's JSON error, say)."""
    header = wav_header(wav)
    if header is None or not header.byte_rate:
        return None
    held = min(header.data_bytes, len(wav) - header.data_start)
    return held / header.byte_rate if held > 0 else None


def wav_file_header(path: Path) -> WavHeader | None:
    """A WAV file's header from its first 64 KiB, its data size held to what the file holds:
    a WAV written to a pipe keeps the streamed-length placeholder (0xFFFFFFFF, ffmpeg 6.1) as
    its data size, which would read a 10 s reference as 13.5 hours."""
    try:
        with path.open("rb") as handle:
            header = wav_header(handle.read(65536))
            size = os.fstat(handle.fileno()).st_size
    except OSError:
        return None
    if header is None:
        return None
    held = max(size - header.data_start, 0)
    return dataclasses.replace(header, data_bytes=min(header.data_bytes, held))


def wav_file_seconds(path: Path) -> float | None:
    """A WAV file's length from its header alone (`wav_file_header`)."""
    header = wav_file_header(path)
    return header.seconds if header is not None and header.byte_rate else None


def grey_png(side: int, height: int | None = None) -> bytes:
    """A flat mid-grey RGB PNG, `side` pixels square (or `side` x `height`): the picture an
    edit request carries, or a video's first frame when `mnm try` is given none."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = height or side
    header = struct.pack(">IIBBBBB", side, rows, 8, 2, 0, 0, 0)  # 8-bit RGB, no interlace
    row = b"\x00" + b"\x80\x80\x80" * side  # filter byte, then the pixels
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * rows, 9))
        + chunk(b"IEND", b"")
    )


def image_mime(data: bytes) -> str | None:
    """`image/png`, `image/jpeg`, `image/gif` or `image/webp` from the magic bytes; None for
    anything else (a client-supplied name or type is never trusted)."""
    for magic, mime in _IMAGE_TYPES:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def tone_wav(seconds: float = 1.5, hertz: float = 440.0, rate: int = 16000) -> bytes:
    """A mono 16-bit WAV of one sine tone: a clip any speech-to-text engine takes (`mnm try
    --all` sends it, so a whisper instance is tried with no clip of the user's)."""
    count = int(seconds * rate)
    samples = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * hertz * i / rate)))
        for i in range(count)
    )
    header = b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return header + fmt + b"data" + struct.pack("<I", len(samples)) + samples
