"""The engine self-test's bar and its verdict: pure functions over figures, no Docker and no
card."""

from __future__ import annotations

from dataclasses import dataclass, replace

from makenomistakes.calibration import plausible
from makenomistakes.hub.facts import Engine
from makenomistakes.measurements import Reading, SpeedUnit
from makenomistakes.readings_filter import Here
from makenomistakes.recipes import Recipe

VRAM_OVER = 0.15  # the new build may hold at most this much more than the old one did
SPEED_UNDER = 0.25  # and run at most this much slower
GIB = 2**30


@dataclass(frozen=True)
class Speed:
    value: float
    unit: SpeedUnit

    def slower_than(self, other: Speed) -> bool:
        """More than `SPEED_UNDER` slower (seconds per request: more than 1/(1-x) longer)."""
        if self.unit is SpeedUnit.TOKENS:
            return self.value < other.value * (1 - SPEED_UNDER)
        return self.value > other.value / (1 - SPEED_UNDER)


@dataclass(frozen=True)
class Figures:
    """VRAM held and speed: a copy's on one build, or the instance's best readings."""

    vram_bytes: int | None = None
    speed: Speed | None = None
    build: str | None = None  # where they came from (for the result's line)


def baseline(readings: list[Reading], settings: str, build: str | None) -> Figures:
    """The bar without the previous build's own run: the least VRAM any other build was
    measured holding for these settings (a build rolled back for being greedy never sets
    it). No speed: clients' requests do other work than the copy's."""
    held: dict[str, int] = {}
    for r in readings:
        mine = r.settings == settings and r.build and r.build != build
        if r.vram_bytes and mine and r.build and plausible(r):
            held[r.build] = max(held.get(r.build, 0), r.vram_bytes)
    if not held:
        return Figures()
    least = min(held.items(), key=lambda item: item[1])
    return Figures(least[1], build=least[0])


def judge(measured: Figures, before: Figures) -> str | None:
    """Why the new build fails against the bar; None when it holds."""
    held, bar = measured.vram_bytes, before.vram_bytes
    if held and bar and held > bar * (1 + VRAM_OVER):
        return (
            f"took {held / GIB:.2f} GiB, {held / bar - 1:.0%} over the "
            f"{bar / GIB:.2f} GiB measured on {before.build}"
        )
    new, old = measured.speed, before.speed
    if new is not None and old is not None and new.unit is old.unit and new.slower_than(old):
        return (
            f"ran at {new.value:g} {new.unit}, against {old.value:g} on {before.build} "
            f"(more than {SPEED_UNDER:.0%} slower)"
        )
    return None


def with_tuned_speed(bar: Figures, recipe: Recipe, previous: str | None, here: Here) -> Figures:
    """No speed to compare with (no previous build ran the same work): a tuned instance's
    measured decode is the bar, with the same 25% bound, only when the tune was
    measured on this box and on the build being replaced (`previous`, an image id): a tune
    from another box (kept by `mnm machine adopt`) or another build is other work (else
    a good build is rolled back against a faster card's figure)."""
    tune = recipe.tuned
    if bar.speed is not None or tune is None or recipe.engine is not Engine.LLAMACPP:
        return bar
    replaced = previous.removeprefix("sha256:")[:12] if previous else None
    if replaced is None or tune.key.build != replaced or not here.owns(tune.key.machine):
        return bar
    tuned = Speed(tune.score.decode, SpeedUnit.TOKENS)
    return replace(bar, speed=tuned, build=bar.build or "its tune")


def compared(measured: Figures, before: Figures) -> str:
    """What the pass rests on beyond the start and the answer, for its one line."""
    if before.build is None:
        return "; no figures from an earlier build to compare with yet"
    held = []
    if measured.vram_bytes and before.vram_bytes:
        held.append(f"VRAM within {VRAM_OVER:.0%}")
    if measured.speed and before.speed and measured.speed.unit is before.speed.unit:
        held.append(f"speed within {SPEED_UNDER:.0%}")
    if not held:
        return f"; {before.build}'s figures were not comparable (one was busy or not measured)"
    return f", {' and '.join(held)} of {before.build}"
