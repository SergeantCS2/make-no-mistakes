"""Fit for text-to-speech models (audio.cpp): small, so the choice is which quant
of the one package, not how to place it.

VRAM = the GGUF (PocketTTS: plus its voice embeddings, a few MB each) + the engine's working
buffers for the family + the CUDA runtime term the llama.cpp fit calibrated on the 4070.
The buffers are a table per family: audio.cpp's `--log` reports the weight buffers at load
(PocketTTS Q8_0: flow_lm 87 MiB, mimi encoder 13, decoder 14; Kokoro Q8_0: 282 MiB, on the
CPU build) but not its compute graphs, so each family carries a working-set allowance until
nvidia-smi is read on a card. Speed is said as a rate (seconds of speech per second of work),
from the CPU build's timing scaled by the GPU; uncalibrated until read on a card.
"""

from __future__ import annotations

from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.estimate import (
    MIB,
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
from makenomistakes.hub.voices import VoiceFamily, VoiceKit, kit_for, package_language

# Working buffers per family while a sentence is synthesised (weights are counted from the
# files): allowances until nvidia-smi is read on a card.
BUFFER_BYTES: dict[VoiceFamily, int] = {
    VoiceFamily.KOKORO: 256 * MIB,
    VoiceFamily.POCKET: 192 * MIB,
    VoiceFamily.QWEN3: 1024 * MIB,  # an LLM's KV cache for the text plus the codec
}
# Seconds of speech a second of work makes on the 4070, request included. PocketTTS is a
# reading on an RTX 4070 (2.2 s of speech in 0.44 s, 5.1x; the CPU build's rates and audio.cpp's
# CUDA figures had said 15x); the others are their CPU-derived figures scaled by the same 1/3.
REALTIME_FACTOR: dict[VoiceFamily, float] = {
    VoiceFamily.KOKORO: 7.0,
    VoiceFamily.POCKET: 5.0,
    VoiceFamily.QWEN3: 1.0,
}
_TARGET_BITS = {ProfileName.QUALITY: 16.0, ProfileName.BALANCED: 8.0, ProfileName.SPEED: 8.0}
_WHY = {
    ProfileName.QUALITY: "The package's full-precision weights.",
    ProfileName.BALANCED: "8-bit weights: the same voices at half the size.",
    ProfileName.SPEED: "8-bit weights; speech models are small, so this is balanced's choice.",
}


def voice_estimate(family: VoiceFamily, model_bytes: int) -> MemoryEstimate:
    return MemoryEstimate(
        weights_bytes=model_bytes,
        compute_bytes=BUFFER_BYTES[family],
        overhead_bytes=RUNTIME_OVERHEAD_BYTES,
    )


def voice_fit(estimate: MemoryEstimate, budget: Budget) -> Fit:
    """No offload: a voice fits on the GPU or it does not."""
    headroom = budget.vram_free_bytes - estimate.total_bytes
    level = headroom_level(estimate.total_bytes, budget.vram_free_bytes)
    return Fit(
        level=level,
        gpu_layers=0,
        headroom_bytes=headroom,
        gpu_bytes=estimate.total_bytes,
        room_bytes=budget.vram_free_bytes,
    )


def voice_args(package: str, kit: VoiceKit) -> AudioCppArgs:
    """The config's model entry for this package: its family, default voice and, for
    PocketTTS, the language its folder names."""
    return AudioCppArgs(
        family=kit.family.value,
        voice=kit.default_voice,
        language=package_language(package) if kit.language_option else None,
        task="vdes" if "voicedesign" in package.lower() else "tts",
    )


def voice_profile(
    name: ProfileName, facts: ModelFacts, variant: GgufVariant, budget: Budget, why: str
) -> Profile:
    assert facts.voice_family is not None  # the planner only runs for a voice package
    kit = kit_for(facts.voice_family)
    estimate = voice_estimate(kit.family, variant.size_bytes + variant.voice_bytes)
    return Profile(
        name=name,
        variant=variant,
        engine_args=voice_args(facts.voice_package or "", kit),
        estimate=estimate,
        fit=voice_fit(estimate, budget),
        tokens_per_second=REALTIME_FACTOR[kit.family],  # read as x realtime by the summary
        why=why,
    )


def plan_voice_profiles(facts: ModelFacts, budget: Budget) -> dict[ProfileName, Profile]:
    """Always all three: each profile takes the quant nearest its target bits that fits, or,
    when none does, the smallest file as NO_FIT (which `Plan.choose` refuses with a reason)."""
    profiles: dict[ProfileName, Profile] = {}
    if not facts.gguf_variants or facts.voice_family is None:
        return profiles
    smallest = min(facts.gguf_variants, key=lambda v: v.size_bytes)
    for name, bits in _TARGET_BITS.items():
        ranked = sorted(
            facts.gguf_variants, key=lambda v: (abs(quant_bits(v.quant) - bits), v.size_bytes)
        )
        fitting = (voice_profile(name, facts, v, budget, _WHY[name]) for v in ranked)
        profiles[name] = next(
            (p for p in fitting if p.fit.level is not FitLevel.NO_FIT),
            voice_profile(name, facts, smallest, budget, _WHY[name]),
        )
    return profiles


def plan_voice_custom(facts: ModelFacts, budget: Budget, quant: str) -> Profile:
    variant = facts.variant(quant)
    if variant is None:
        available = ", ".join(v.quant for v in facts.gguf_variants) or "none"
        raise ValueError(f"{facts.repo_id} has no '{quant}' file (available: {available})")
    return voice_profile(ProfileName.CUSTOM, facts, variant, budget, "Your explicit choice.")
