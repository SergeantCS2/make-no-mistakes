"""Musical keys as the user writes them and the structure map names them: a tonic and
a mode, read from `A minor`, `F# major`, `Bb`, `c#m` or `Ebmin`, written back with sharps or
flats as each key is usually spelled. A key change moves the tonic and keeps the mode: from A
minor to C major is the same notes (a relative key), which no shift makes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLATS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
# The tonics spelled with flats (F major, B-flat major … and their relative minors)
FLAT_MAJORS = {1, 3, 5, 8, 10}  # Db Eb F Ab Bb
FLAT_MINORS = {0, 2, 3, 5, 7, 10}  # C D Eb F G Bb
_LETTERS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_KEY = re.compile(
    r"^\s*([a-gA-G])\s*(#|♯|b|♭|sharp|flat)?\s*"
    r"(major|maj|minor|min|m|moll|dur)?\s*$",
    re.IGNORECASE,
)


class KeyNameError(ValueError):
    """A key the user wrote that mnm cannot read or shift to, worded for them."""


@dataclass(frozen=True)
class Key:
    tonic: int  # 0 = C … 11 = B
    minor: bool

    def words(self) -> str:
        flats = self.tonic in (FLAT_MINORS if self.minor else FLAT_MAJORS)
        name = (FLATS if flats else SHARPS)[self.tonic]
        return f"{name} {'minor' if self.minor else 'major'}"

    def shifted(self, semitones: int) -> Key:
        return Key((self.tonic + semitones) % 12, self.minor)


def parse(text: str) -> Key:
    """`A minor`, `F#`, `Bb major`, `c#m`: a tonic and a mode (major when none is said)."""
    match = _KEY.match(text.replace("-", " "))
    if match is None:
        raise KeyNameError(f"'{text}' is not a key mnm reads: write it as A minor, F# major or Bb")
    letter, accidental, mode = match.groups()
    tonic = _LETTERS[letter.lower()]
    if accidental:
        tonic += 1 if accidental.lower() in ("#", "♯", "sharp") else -1
    minor = bool(mode) and mode.lower() in ("minor", "min", "m", "moll")
    if mode == "M":  # `CM` is C major by convention; `Cm` is C minor
        minor = False
    return Key(tonic % 12, minor)


def shift_to(now: Key, wanted: Key) -> int:
    """The semitones from `now` to `wanted`, the shorter way round (-5 to +6); refused when
    the modes differ (a shift keeps the mode)."""
    if now.minor != wanted.minor:
        mode = "minor" if now.minor else "major"
        raise KeyNameError(
            f"The take is in {now.words()}, and a key change keeps the mode: name a {mode} key "
            f"(to={Key(wanted.tonic, now.minor).words()} moves the tonic to "
            f"{wanted.words().split()[0]})"
        )
    step = (wanted.tonic - now.tonic) % 12
    return step - 12 if step > 6 else step


TAKE_KEY = "take_key"  # an edit's setting: the key its take is in, handed on
TAKE_KEY_CLARITY = "take_key_clarity"  # how clearly that key was first heard ("" unknown)


@dataclass(frozen=True)
class Carried:
    """The key an edit hands on to the take it made, and how clearly it was first heard: the
    one place its settings are written and read."""

    key: str
    clarity: float | None = None

    def settings(self) -> dict[str, str]:
        clarity = "" if self.clarity is None else f"{self.clarity:.3f}"
        return {TAKE_KEY: self.key, TAKE_KEY_CLARITY: clarity}

    @classmethod
    def from_settings(cls, settings: Mapping[str, str]) -> Carried | None:
        key = settings.get(TAKE_KEY)
        if not key:
            return None
        try:
            clarity = float(settings[TAKE_KEY_CLARITY]) if settings.get(TAKE_KEY_CLARITY) else None
        except ValueError:
            clarity = None  # a hand-edited record: the key stands, its clarity unknown
        return cls(key, clarity)
