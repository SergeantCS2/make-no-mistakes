"""Which quant a profile starts from, and how far down it may step: bits read from a
quant label, the 4-bit picks Balanced and Speed prefer, and the walk below 4 bits when no
4-bit quant fits this machine (Nemotron Bolt and Mistral-Small-4 read `no_fit` with nothing
smaller tried)."""

from __future__ import annotations

import re

from makenomistakes.hub.variants import GgufVariant

_BITS_RE = re.compile(r"(?i)(?:^|[-_])(?:UD-)?(?:I?Q|TQ)(\d)")
BALANCED_PREFERENCE = ("Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0", "MXFP4")
SPEED_PREFERENCE = ("Q4_K_S", "Q4_0", "IQ4_XS", "Q4_K_M", "UD-Q4_K_XL", "IQ4_NL", "MXFP4")


def quant_bits(quant: str) -> float:
    """Approximate bits per weight from a quant label; higher means closer to the original."""
    upper = quant.upper()
    if upper == "F32":
        return 32
    if upper in {"F16", "BF16", "FP16"}:
        return 16
    if "MXFP4" in upper:
        return 4
    match = _BITS_RE.search(quant)
    if not match:
        return 0
    bits = float(match.group(1))
    if "_XL" in upper or "_L" in upper or "_M" in upper:
        bits += 0.4
    elif "_XS" in upper or "XXS" in upper:
        bits -= 0.3
    return bits


def preferred(variants: list[GgufVariant], preference: tuple[str, ...]) -> GgufVariant:
    """First variant on the preference list; otherwise the one closest to 4.4 bits."""
    by_name = {v.quant.upper(): v for v in variants}
    for preferred in preference:
        if preferred in by_name:
            return by_name[preferred]
    four_bit = [v for v in variants if 3.5 <= quant_bits(v.quant) <= 4.9]
    return min(four_bit or variants, key=lambda v: abs(quant_bits(v.quant) - 4.4))


def below(variants: list[GgufVariant], start: GgufVariant) -> list[GgufVariant]:
    """The quants smaller than `start`, the most faithful first: where a profile steps when
    `start` fits nowhere, down to the smallest the repo has."""
    smaller = [v for v in variants if v.memory_bytes < start.memory_bytes]
    return sorted(smaller, key=lambda v: (quant_bits(v.quant), v.memory_bytes), reverse=True)
