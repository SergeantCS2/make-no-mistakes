"""Bytes in words, one way everywhere: the CLI, the web pages and refusals (the CLI never
imports the web app, so this lives outside `web/`)."""

from __future__ import annotations


def size_words(size: int) -> str:
    """Bytes as GiB or MiB with one decimal, KiB whole, or bytes under 1 KiB: a 300-byte
    answer would read "0 KiB"."""
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    if size >= 1024:
        return f"{size / 1024:.0f} KiB"
    return "1 byte" if size == 1 else f"{size} bytes"


def count_words(n: int, thing: str) -> str:
    """`1 file`, `2 files`."""
    return f"{n} {thing}" if n == 1 else f"{n} {thing}s"
