"""Fit for speech-to-text models (whisper.cpp): small, so the choice is which size, not how.

VRAM = the model file + the engine's buffers + the CUDA runtime. The buffers (KV self, cross
and pad, plus the conv, encode, cross and decode compute buffers) come from what
`whisper-server` reports at load, with the self cache at the size a temperature fallback
grows it to; they follow the architecture, not the quant, so they are a table per size
family, measured on a CPU build. The runtime term is the one the llama.cpp fit calibrated
on an RTX 4070 (same ggml CUDA backend). Profiles rank every file the repo has by a parsed
size (family, English-only, quant), so any label works, fine-tunes included, and take the
first that fits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial

from makenomistakes.engine_args_whisper import WhisperArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.estimate import (
    RUNTIME_OVERHEAD_BYTES,
    Fit,
    FitLevel,
    MemoryEstimate,
    headroom_level,
)
from makenomistakes.fit.profiles import Profile, ProfileName
from makenomistakes.fit.quant_ladder import quant_bits
from makenomistakes.hub.facts import ModelFacts
from makenomistakes.hub.variants import GgufVariant

MB = 10**6  # whisper.cpp logs decimal megabytes

# whisper-server's buffers per size family, steady state (decimal MB as logged; any quant):
# KV pad + cross + 4x self (a temperature fallback reallocates the self cache for
# best_of=2 decoders, +2 spare) + the conv, encode, cross and decode compute buffers. Summed
# from each family's load log on the CPU build; tiny's sum is recomputed from
# the recorded log in tests. A request with beam_size > 1 grows the self cache further.
# large-v3-turbo keeps large-v3's encoder with a 4-layer decoder, hence far less KV.
SELF_KV_FACTOR = 4
BUFFER_BYTES: dict[str, int] = {
    "tiny": 155 * MB,
    "base": 188 * MB,
    "small": 297 * MB,
    "medium": 538 * MB,
    "large-v3-turbo": 281 * MB,
    "large": 795 * MB,
}
UNKNOWN_FAMILY_BUFFER_BYTES = max(BUFFER_BYTES.values())  # a fine-tune we cannot place

# Families each profile tries, best first (any quant and any fine-tune of the family), and
# the weight precision it aims for within a family. large-v3-turbo comes first for quality:
# near large-v3's accuracy at a third of the time. Labels naming no family come last.
_FAMILY_ORDER: dict[ProfileName, tuple[str, ...]] = {
    ProfileName.QUALITY: ("large-v3-turbo", "large", "medium", "small", "base", "tiny"),
    ProfileName.BALANCED: ("large-v3-turbo", "small", "medium", "base", "tiny", "large"),
    ProfileName.SPEED: ("base", "tiny", "small", "medium", "large-v3-turbo", "large"),
}
_TARGET_BITS = {ProfileName.QUALITY: 8.0, ProfileName.BALANCED: 5.0, ProfileName.SPEED: 8.0}
_QUANT_RE = re.compile(r"[-.](?P<quant>q\d+_[0-9a-z]+|f16|f32|fp16|fp32)$")  # `-q5_0`, `.fp32`
_SIZES = ("tiny", "base", "small", "medium")
_WHY = {
    ProfileName.QUALITY: (
        "Best transcription the card holds; large-v3-turbo is near large-v3 at a third of the time."
    ),
    ProfileName.BALANCED: "Good accuracy at a fraction of the size; fine for dictation and chat.",
    ProfileName.SPEED: "Fastest; fine for short English commands.",
}


@dataclass(frozen=True)
class WhisperSize:
    """What a size label says: `distil-small.en-q5_1` is small, English-only, 5-bit."""

    family: str | None  # a key of BUFFER_BYTES, or None when the label names no family
    english_only: bool
    bits: float  # weight precision; the plain file is f16


def parse_size(label: str) -> WhisperSize:
    lowered = label.lower()
    match = _QUANT_RE.search(lowered)
    name = lowered[: match.start()] if match else lowered
    tokens = re.split(r"[-.]", name)
    if "large" in tokens:
        family: str | None = "large-v3-turbo" if "turbo" in tokens else "large"
    else:
        family = next((t for t in tokens if t in _SIZES), None)
    return WhisperSize(
        family=family,
        english_only="en" in tokens,
        bits=quant_bits(match.group("quant").replace("fp", "f")) if match else 16.0,
    )


def audio_estimate(size: str, model_bytes: int) -> MemoryEstimate:
    family = parse_size(size).family
    return MemoryEstimate(
        weights_bytes=model_bytes,
        compute_bytes=BUFFER_BYTES[family] if family else UNKNOWN_FAMILY_BUFFER_BYTES,
        overhead_bytes=RUNTIME_OVERHEAD_BYTES,
    )


def audio_fit(estimate: MemoryEstimate, budget: Budget) -> Fit:
    """No layer offload for whisper: it fits on the GPU or it does not."""
    headroom = budget.vram_free_bytes - estimate.total_bytes
    level = headroom_level(estimate.total_bytes, budget.vram_free_bytes)
    return Fit(
        level=level,
        gpu_layers=0,
        headroom_bytes=headroom,
        gpu_bytes=estimate.total_bytes,
        room_bytes=budget.vram_free_bytes,
    )


def audio_profile(name: ProfileName, variant: GgufVariant, budget: Budget, why: str) -> Profile:
    estimate = audio_estimate(variant.quant, variant.size_bytes)
    return Profile(
        name=name,
        variant=variant,
        # an English-only model ignores `auto` with a warning on every request
        engine_args=WhisperArgs(
            language="en" if parse_size(variant.quant).english_only else "auto"
        ),
        estimate=estimate,
        fit=audio_fit(estimate, budget),
        why=why,
    )


def plan_audio_profiles(facts: ModelFacts, budget: Budget) -> dict[ProfileName, Profile]:
    """Always all three profiles: each takes the first of its ranked sizes that fits, or,
    when none does, the smallest file as NO_FIT (which `Plan.choose` refuses with a reason)."""
    profiles: dict[ProfileName, Profile] = {}
    if not facts.gguf_variants:
        return profiles
    smallest = min(facts.gguf_variants, key=lambda v: v.size_bytes)
    for name in _FAMILY_ORDER:
        ranked = sorted(facts.gguf_variants, key=partial(_rank, name))
        fitting = (audio_profile(name, v, budget, _WHY[name]) for v in ranked)
        profiles[name] = next(
            (p for p in fitting if p.fit.level is not FitLevel.NO_FIT),
            audio_profile(name, smallest, budget, _WHY[name]),
        )
    return profiles


def _rank(name: ProfileName, variant: GgufVariant) -> tuple[int, bool, float, float, int]:
    """Sort key: the profile's family order, multilingual before English-only, the quant
    closest to its target (higher on a tie); unknown families by size, largest first for
    quality and smallest first otherwise."""
    size = parse_size(variant.quant)
    order = _FAMILY_ORDER[name]
    family = order.index(size.family) if size.family in order else len(order)
    by_size = -variant.size_bytes if name is ProfileName.QUALITY else variant.size_bytes
    return (family, size.english_only, abs(size.bits - _TARGET_BITS[name]), -size.bits, by_size)


def plan_audio_custom(facts: ModelFacts, budget: Budget, size: str) -> Profile:
    variant = facts.variant(size)
    if variant is None:
        available = ", ".join(v.quant for v in facts.gguf_variants) or "none"
        raise ValueError(f"{facts.repo_id} has no '{size}' model (available: {available})")
    return audio_profile(ProfileName.CUSTOM, variant, budget, "Your explicit size choice.")
