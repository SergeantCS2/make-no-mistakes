"""Video repos' file shapes the generic GGUF listing does not know.

Wan 2.2 A14B ships two 14B diffusion models that run in turn, a high-noise and a low-noise
one, one file of each per quant (`HighNoise/…-HighNoise-Q4_K_M.gguf` beside `LowNoise/…`;
ComfyUI's `…_high_noise_14B_…` beside `…_low_noise_…`). One instance loads both, so the
pair is one variant: the low-noise file is the model (`--diffusion-model`), the high-noise
one its typed `high_noise` (`--high-noise-diffusion-model`), the size the two together.
Halves pair by their names with the noise token taken out, so a repo holding a T2V and an
I2V pair of one quant pairs each with its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from makenomistakes.hub.variants import GgufVariant

NOISE_RE = re.compile(r"(high|low)[-_]?noise", re.I)


def noise_half(path: str) -> tuple[str, str] | None:
    """(the path without its noise tokens, `high` or `low`) for one half of a Wan 2.2 A14B
    pair, else None. Folder names count too: `HighNoise/x-HighNoise-Q4.gguf` and
    `LowNoise/x-LowNoise-Q4.gguf` share one key."""
    found = NOISE_RE.search(path.rsplit("/", 1)[-1])
    if found is None:
        return None
    return NOISE_RE.sub("", path).lower(), found.group(1).lower()


class _HasFiles(Protocol):
    @property
    def files(self) -> list[str]: ...


def _single_file(item: _HasFiles) -> str | None:
    """A half is one file: a sharded variant or a folder is never one."""
    return item.files[0] if len(item.files) == 1 else None


def pair_by_noise[T: _HasFiles](
    items: list[T], join: Callable[[T, T], T], path_of: Callable[[T], str | None] = _single_file
) -> list[T]:
    """Each low-noise item joined with its high-noise sibling (`join(low, high)`); a lone
    low-noise half stays, a lone high-noise half goes: it is a first stage, and on its own it
    would run as the whole model. Items whose `path_of` is None, or that are no half, stay."""
    halves: dict[str, dict[str, T]] = {}
    out: list[T] = []
    for item in items:
        path = path_of(item)
        half = noise_half(path) if path is not None else None
        if half is None:
            out.append(item)
        else:
            halves.setdefault(half[0], {})[half[1]] = item
    for pair in halves.values():
        low, high = pair.get("low"), pair.get("high")
        if low is not None:
            out.append(low if high is None else join(low, high))
    return out


def pair_noise_halves(variants: list[GgufVariant]) -> list[GgufVariant]:
    """The GGUF variants with each pair as one (`pair_by_noise`), smallest first."""

    def join(low: GgufVariant, high: GgufVariant) -> GgufVariant:
        return GgufVariant(
            quant=low.quant,
            files=list(low.files),
            size_bytes=low.size_bytes + high.size_bytes,
            high_noise=high.files[0],
        )

    paired = pair_by_noise(variants, join)
    return sorted(paired, key=lambda v: v.size_bytes)
