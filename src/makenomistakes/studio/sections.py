"""Section edits as pieces of the take: what a cut, a crop, a duplicate or a move keeps of the
take, in order, as `(start, end)` seconds the house format joins with a crossfade
(`audio_format.Splice`). Pure arithmetic over the take's length and the ranges the edit names
(snapped to the structure map's downbeats by `song_edits`); the pieces a joint leaves shorter
than a tick are dropped.
"""

from __future__ import annotations

from typing import NamedTuple

TICK = 0.05  # seconds: a piece shorter than this is not kept (a joint on the take's edge)


class SectionError(ValueError):
    """A section edit the take cannot take, worded for the user."""


class Range(NamedTuple):
    start: float
    end: float


def _kept(pieces: list[tuple[float, float]]) -> list[tuple[float, float]]:
    found = [(round(a, 6), round(b, 6)) for a, b in pieces if b - a >= TICK]
    if not found:
        raise SectionError("nothing of the take would be left")
    return found


def cut(length: float, part: Range) -> list[tuple[float, float]]:
    """The take less `part`, the two sides joined."""
    if part.start <= TICK and part.end >= length - TICK:
        raise SectionError("a cut of the whole take leaves nothing")
    return _kept([(0.0, part.start), (part.end, length)])


def crop(length: float, part: Range) -> list[tuple[float, float]]:
    """Only `part` of the take."""
    if part.start <= TICK and part.end >= length - TICK:
        raise SectionError("a crop to the whole take changes nothing")
    return _kept([(part.start, min(part.end, length))])


def duplicate(length: float, part: Range, at: float) -> list[tuple[float, float]]:
    """`part` played once more at `at` (a joint: the part's own end, or another part's)."""
    if part.start < at < part.end:
        raise SectionError(
            f"the copy would land inside the part it copies ({at:.1f} s is within "
            f"{part.start:.1f}-{part.end:.1f} s)"
        )
    return _kept([(0.0, at), (part.start, part.end), (at, length)])


def move(length: float, part: Range, at: float) -> list[tuple[float, float]]:
    """`part` taken out and played at `at` instead (a joint of the take as it was)."""
    if abs(at - part.start) <= TICK or abs(at - part.end) <= TICK:
        raise SectionError(
            f"the part is at {part.start:.1f} s already: a move there changes nothing"
        )
    if part.start < at < part.end:
        raise SectionError(
            f"the part would move into itself ({at:.1f} s is within "
            f"{part.start:.1f}-{part.end:.1f} s)"
        )
    if at < part.start:
        return _kept([(0.0, at), (part.start, part.end), (at, part.start), (part.end, length)])
    return _kept([(0.0, part.start), (part.end, at), (part.start, part.end), (at, length)])


def joints(pieces: list[tuple[float, float]]) -> int:
    return max(len(pieces) - 1, 0)


def length_after(pieces: list[tuple[float, float]], fade: float) -> float:
    """The new take's length: the pieces less one crossfade per joint."""
    return sum(b - a for a, b in pieces) - fade * joints(pieces)
