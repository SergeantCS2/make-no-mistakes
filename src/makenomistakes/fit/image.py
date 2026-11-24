"""Fit and profiles for diffusion models served by stable-diffusion.cpp: the planner that
picks quant, size and placement. What each choice puts on the GPU is `fit/image_memory.py`;
this module plans with them and exports its planners only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from makenomistakes.engine_args import SdCppArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.estimate import Fit, FitLevel, MemoryEstimate
from makenomistakes.fit.image_memory import (
    CheckpointSizes,
    KitSizes,
    Placement,
    _Weights,
    image_fit,
    kit_sizes,
    placement_flags,
    restreamed_bytes,
    vae_decode_bytes_per_pixel,
)
from makenomistakes.fit.profiles import Profile, ProfileName
from makenomistakes.fit.quant_ladder import quant_bits
from makenomistakes.fit.sdcpp import sd_seconds, sd_settings_estimate
from makenomistakes.fit.video import (
    VIDEO_VAE_BYTES_PER_PIXEL_FRAME,
    VIDEO_VAE_FIXED_BYTES_PER_PIXEL,
)
from makenomistakes.hub.facts import ModelFacts
from makenomistakes.hub.image_files import ImageWeights
from makenomistakes.hub.kits import GenerationDefaults, ImageKit, defaults_for
from makenomistakes.hub.variants import GgufVariant
from makenomistakes.safetensors.sd import DitParts

__all__ = [
    "dit_precisions",
    "plan_checkpoint_profiles",
    "plan_dit_custom",
    "plan_dit_profiles",
    "plan_image_custom",
    "plan_image_profiles",
    "preference_order",
]

_SPEED_PREFERENCE = ("Q4_0", "Q4_K", "Q4_K_S", "Q4_K_M", "Q3_K", "Q5_0", "Q5_K_S", "Q8_0")
_BALANCED_PREFERENCE = ("Q4_K", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_0", "Q5_K_S", "Q8_0")


def plan_image_profiles(
    facts: ModelFacts, kit: ImageKit, sizes: KitSizes, budget: Budget
) -> dict[ProfileName, Profile]:
    """A GGUF diffusion model: its quants, with the kit's companions (`sizes`)."""
    variants = [v for v in facts.gguf_variants if quant_bits(v.quant) > 0]
    if not variants:
        return {}
    planner = _ImagePlanner(facts, kit, sizes, budget, _defaults(facts, kit), sizes.total)
    by_quality = sorted(variants, key=lambda v: (quant_bits(v.quant), v.size_bytes), reverse=True)
    usual = planner.weights(_preferred(variants, _BALANCED_PREFERENCE))
    # The usual 4-bit quant, unless not even streaming fits it (a 14B video pair in 20 GB of
    # RAM): then the heaviest lighter quant that runs at all, so balanced is a plan, not a
    # refusal. Quality never runs lighter weights than balanced: when nothing heavier than
    # balanced's quant fits comfortably, both profiles share it.
    lighter = [
        planner.weights(v)
        for v in by_quality
        if quant_bits(v.quant) < quant_bits(usual.variant.quant)
    ]
    balanced = planner.first_running(usual, lighter)
    heavier = [
        planner.weights(v)
        for v in by_quality
        if quant_bits(v.quant) >= quant_bits(balanced.variant.quant)
    ]
    speed = planner.first_running(
        planner.weights(_preferred(variants, _SPEED_PREFERENCE)), [balanced]
    )
    return planner.three(
        planner.best_fitting(heavier) or balanced,
        balanced,
        speed,
        _GGUF_WHY,
    )


def plan_checkpoint_profiles(
    facts: ModelFacts,
    kit: ImageKit,
    weights: ImageWeights,
    checkpoint: CheckpointSizes,
    kit_bytes: int,
    budget: Budget,
) -> dict[ProfileName, Profile]:
    """A safetensors checkpoint or diffusers folder: one set of weights, F16 as loaded, placed
    by `checkpoint`; downloads its files plus the kit's companions (`kit_bytes`)."""
    planner = _ImagePlanner(facts, kit, checkpoint.sizes, budget, _defaults(facts, kit), kit_bytes)
    unet = _Weights(
        weights.as_variant("F16"),
        checkpoint.diffusion_bytes,
        checkpoint.diffusion_bytes / 2 / 1e9,  # two bytes per parameter
    )
    return planner.three(unet, unet, unet, _CHECKPOINT_WHY)


def plan_dit_profiles(
    facts: ModelFacts,
    kit: ImageKit,
    weights: ImageWeights,
    parts: DitParts,
    sizes: KitSizes,
    kit_bytes: int,
    budget: Budget,
) -> dict[ProfileName, Profile]:
    """A transformer's diffusion model in one safetensors file, with the kit's companions.
    As stored (BF16, FP8) when that stays in VRAM comfortably; otherwise converted to Q8_0
    while loading (near-lossless, about half of BF16), which also lightens the speed profile.
    Quality never runs lighter weights than balanced."""
    planner = _ImagePlanner(facts, kit, sizes, budget, _defaults(facts, kit), kit_bytes)
    stored, q8 = _dit_precisions(weights, parts)
    lighter = q8 if q8.loaded_bytes < stored.loaded_bytes else stored
    best = planner.best_fitting([stored, lighter])
    return planner.three(best or lighter, best or lighter, lighter, _DIT_WHY)


def plan_dit_custom(
    facts: ModelFacts,
    kit: ImageKit,
    weights: ImageWeights,
    parts: DitParts,
    companion_sizes: Mapping[str, int],
    budget: Budget,
    quant: str,
) -> Profile:
    """A DiT file at the precision asked for (`--quant`), placed at the family's full size
    and steps like any explicit quant. ValueError names the choices."""
    choices = {w.variant.quant.upper(): w for w in _dit_precisions(weights, parts)}
    chosen = choices.get(quant.upper())
    if chosen is None:
        raise ValueError(
            f"{quant} is not a choice for this file (choices: {', '.join(choices)}: as stored, "
            "or quantized while loading)."
        )
    sizes = kit_sizes(kit.companions, companion_sizes)
    kit_bytes = sum(companion_sizes.values())
    planner = _ImagePlanner(facts, kit, sizes, budget, _defaults(facts, kit), kit_bytes)
    return planner.place(
        ProfileName.CUSTOM, chosen, kit.sizes, planner.defaults.steps, "Your explicit quant choice."
    )


def dit_precisions(weights: ImageWeights, parts: DitParts) -> list[str]:
    """What a DiT file can run at, whatever fits now: as stored, or Q8_0 while loading."""
    return [w.variant.quant for w in _dit_precisions(weights, parts)]


def _dit_precisions(weights: ImageWeights, parts: DitParts) -> tuple[_Weights, _Weights]:
    """A DiT file's diffusion model as stored, and converted to Q8_0 while loading."""
    params_b = parts.diffusion_params / 1e9

    def loaded(diffusion_type: Literal["bf16", "q8_0"] | None, label: str) -> _Weights:
        variant = weights.as_variant(label)
        loaded_bytes = parts.loaded_diffusion_bytes(diffusion_type)
        return _Weights(variant, loaded_bytes, params_b, diffusion_type)

    stored = loaded(parts.diffusion_type, "BF16" if parts.diffusion_type else parts.dtype)
    return stored, loaded("q8_0", "Q8_0")


@dataclass(frozen=True)
class _WhyTexts:
    """Each profile's why-text: `{size}` is where it runs, `{fewer}` the speed step note."""

    quality: str
    balanced: str
    speed: str


_GGUF_WHY = _WhyTexts(
    quality="Highest-quality weights that fit at {size} with the family's step count.",
    balanced="4-bit weights at {size}; the usual choice on 12 GB.",
    speed="Lightest 4-bit weights at {size}{fewer}: the quickest preview.",
)
_DIT_WHY = _WhyTexts(
    quality="The diffusion model at {size} with the family's step count.",
    balanced="The diffusion model at {size} with the family's step count.",
    speed="At {size}{fewer}: the quickest preview.",
)
_CHECKPOINT_WHY = _WhyTexts(
    quality="The F16 weights at {size} with the family's step count.",
    balanced="The F16 weights at {size} with the family's step count.",
    speed="At {size}{fewer}: the quickest preview.",
)


def plan_image_custom(
    facts: ModelFacts, kit: ImageKit, sizes: KitSizes, budget: Budget, quant: str
) -> Profile:
    variant = facts.variant(quant)
    if variant is None:
        available = ", ".join(v.quant for v in facts.gguf_variants) or "none"
        raise ValueError(f"{facts.repo_id} has no '{quant}' variant (available: {available})")
    planner = _ImagePlanner(facts, kit, sizes, budget, _defaults(facts, kit), sizes.total)
    return planner.place(
        ProfileName.CUSTOM,
        planner.weights(variant),
        kit.sizes,
        planner.defaults.steps,
        "Your explicit quant choice.",
    )


def _defaults(facts: ModelFacts, kit: ImageKit) -> GenerationDefaults:
    """The kit's defaults for this repo and the version planned: a repo of several builds
    (`circlestone-labs/Anima`) names its turbo one only in the version."""
    return defaults_for(" ".join(filter(None, (facts.repo_id, facts.version))), kit)


def _preferred(variants: list[GgufVariant], preference: tuple[str, ...]) -> GgufVariant:
    by_name = {v.quant.upper(): v for v in variants}
    for wanted in preference:
        if wanted in by_name:
            return by_name[wanted]
    four_bit = [v for v in variants if 3.5 <= quant_bits(v.quant) <= 4.9]
    return min(four_bit or variants, key=lambda v: abs(quant_bits(v.quant) - 4.4))


_ENCODERS_ON_CPU = (
    " Text encoders run on the CPU to leave VRAM for the diffusion model (a few seconds per "
    "new prompt)."
)
_TILES = " The VAE decodes in tiles (an untiled decode at this size needs {decode} on its own)."
_WHY_SUFFIX = {
    # Said for every placement: a speed profile (768x768) can show more VRAM than its
    # 1024x1024 quality one, whose larger size decodes in tiles
    Placement.GPU: " The VAE decodes whole frames at once ({decode} of the estimate): "
    "quicker than tiles.",
    Placement.GPU_TILED: _TILES,
    Placement.ENCODERS_IN_RAM: _ENCODERS_ON_CPU,
    Placement.ENCODERS_IN_RAM_TILED: _ENCODERS_ON_CPU + _TILES,
    Placement.STREAMED: " Weights stream from RAM: the engine fills the VRAM free at start "
    "and re-reads {restream} each step (slower).",
}


_CONVERTED = {
    "q8_0": " Converted to Q8_0 while loading (near-lossless; a few seconds per start).",
    "bf16": " Loads as BF16 (stored as F32).",
}
_RESIDENT = (  # weights stay in VRAM; the cheaper concession first
    Placement.GPU,
    Placement.GPU_TILED,
    Placement.ENCODERS_IN_RAM,
    Placement.ENCODERS_IN_RAM_TILED,
)
Candidate = tuple[int, int, Placement, FitLevel]


def preference_order(resolutions: tuple[tuple[int, int], ...]) -> list[Candidate]:
    """The order `place` takes the first fit from: largest size first; within it, comfortable
    before tight, then all-GPU, a tiled decode, encoders in RAM, both; streamed weights only
    after every resident fit at every size failed."""
    resident = [
        (w, h, placement, level)
        for w, h in resolutions
        for level in (FitLevel.COMFORTABLE, FitLevel.TIGHT)
        for placement in _RESIDENT
    ]
    return resident + [(w, h, Placement.STREAMED, FitLevel.OFFLOAD) for w, h in resolutions]


def _why(why: str, width: int, height: int, resolutions: tuple[tuple[int, int], ...]) -> str:
    """`why` names the size it runs at (`{size}`), and says so when the first one missed."""
    text = why.format(size=f"{width}x{height}")
    if (width, height) == resolutions[0]:
        return text
    big_w, big_h = resolutions[0]
    return f"{text} {big_w}x{big_h} does not fit."


@dataclass(frozen=True)
class _ImagePlanner:
    facts: ModelFacts
    kit: ImageKit
    sizes: KitSizes  # what is placed besides the diffusion weights
    budget: Budget
    defaults: GenerationDefaults
    kit_bytes: int  # what the kit's companions download

    def weights(self, variant: GgufVariant) -> _Weights:
        """A GGUF quant loads as stored. A two-stage pair (Wan 2.2 A14B) downloads and loads
        both halves (`size_bytes`), and each step runs one of them, so the time uses one
        half's parameters (the Hub's count, when it gives one, is a single file's too)."""
        stages = 2 if variant.high_noise is not None else 1
        params = self.facts.params_b or _params_from_bytes(variant) / stages
        return _Weights(variant, variant.size_bytes, params)

    def _args(self, width: int, height: int, placement: Placement, **more: object) -> SdCppArgs:
        video = self.kit.video
        return SdCppArgs(
            width=width,
            height=height,
            frames=video.frames if video is not None else None,
            fps=video.fps if video is not None else None,
            high_noise_steps=video.high_noise_steps if video is not None else None,
            **placement_flags(placement),
            **more,
        )

    def _seconds(self, weights: _Weights, args: SdCppArgs, restream: int) -> float:
        """The fit model's time, times what this card measured for the family."""
        params = weights.params_b * 1e9
        budget = self.budget
        speeds = budget.device.picture_speeds  # the card's row
        found = sd_seconds(
            params, args, self.kit.video, budget.gpu_compute, restream, self.kit.family, speeds
        )
        return found * self.budget.speed_ratio(self.kit.family)

    def _decode_bytes(self, width: int, height: int) -> int:
        """What an untiled decode of one output needs: the why-text's figure."""
        if self.kit.video is not None:
            per_pixel = VIDEO_VAE_FIXED_BYTES_PER_PIXEL
            per_pixel += self.kit.video.frames * VIDEO_VAE_BYTES_PER_PIXEL_FRAME
            return width * height * per_pixel
        return width * height * vae_decode_bytes_per_pixel(self.kit.family)

    def three(
        self, quality: _Weights, balanced: _Weights, speed: _Weights, why: _WhyTexts
    ) -> dict[ProfileName, Profile]:
        steps = self.defaults.steps
        speed_steps = steps if steps <= 8 else steps // 2
        fewer = ", fewer steps" if speed_steps < steps else ""
        return {
            ProfileName.QUALITY: self.place(
                ProfileName.QUALITY, quality, self.kit.sizes, steps, why.quality
            ),
            ProfileName.BALANCED: self.place(
                ProfileName.BALANCED, balanced, self.kit.sizes, steps, why.balanced
            ),
            ProfileName.SPEED: self.place(
                ProfileName.SPEED,
                speed,
                self.kit.sizes[1:] or self.kit.sizes,  # a kit with one size keeps it
                speed_steps,
                why.speed.replace("{fewer}", fewer),
            ),
        }

    def estimate_for(
        self, weights: _Weights, width: int, height: int, placement: Placement
    ) -> tuple[MemoryEstimate, Fit]:
        args = self._args(width, height, placement)
        estimate = sd_settings_estimate(weights.loaded_bytes, self.sizes, args, self.kit.family)
        streamed_ram = weights.loaded_bytes + self.sizes.total
        if placement is Placement.STREAMED and streamed_ram > self.budget.ram_free_bytes:
            # Streamed weights live in RAM; without room there nothing fits.
            headroom = self.budget.vram_free_bytes - estimate.total_bytes
            return estimate, Fit(
                level=FitLevel.NO_FIT,
                gpu_layers=0,
                headroom_bytes=headroom,
                gpu_bytes=estimate.total_bytes,
                room_bytes=self.budget.vram_free_bytes,
            )
        return estimate, image_fit(estimate, self.budget, placement)

    def place(
        self,
        name: ProfileName,
        weights: _Weights,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
        why: str,
    ) -> Profile:
        """Largest resolution first; within it, comfortable before tight, all-GPU before
        encoders in RAM. Streaming the weights is the last resort at any resolution.

        A comfortable fit with the encoders in RAM beats a tight all-GPU fit on purpose: the
        constants are uncalibrated, and running out of VRAM mid-image costs more than a second
        of prompt encoding. Resolution is never traded for headroom: a tight fit at full size
        beats a comfortable one at 768x768 (that is the speed profile's job).
        """
        for w, h, placement, level in preference_order(resolutions):
            if self.estimate_for(weights, w, h, placement)[1].level is level:
                break
        else:  # nothing fits: the smallest size, streamed, carries the NO_FIT verdict
            (w, h), placement = resolutions[-1], Placement.STREAMED
        return self.profile(name, weights, w, h, steps, _why(why, w, h, resolutions), placement)

    def runs(self, weights: _Weights) -> bool:
        """Whether these weights run at any size and placement (streamed included)."""
        for w, h, placement, level in preference_order(self.kit.sizes):
            if self.estimate_for(weights, w, h, placement)[1].level is level:
                return True
        return False

    def first_running(self, preferred: _Weights, fallbacks: list[_Weights]) -> _Weights:
        """`preferred` when it runs at all, else the first fallback that does, else
        `preferred` (whose NO_FIT verdict then says why)."""
        if self.runs(preferred):
            return preferred
        return next((w for w in fallbacks if self.runs(w)), preferred)

    def best_fitting(self, candidates: list[_Weights]) -> _Weights | None:
        """Best quant that fits comfortably at full resolution without streaming. Unlike
        llama.cpp's `Planner.best_fitting` there is no tight fallback: the caller falls back to
        balanced's quant, so a tight fit never buys heavier weights here."""
        for weights in candidates:
            for placement in _RESIDENT:
                level = self.estimate_for(weights, *self.kit.sizes[0], placement)[1].level
                if level is FitLevel.COMFORTABLE:
                    return weights
        return None

    def profile(
        self,
        name: ProfileName,
        weights: _Weights,
        width: int,
        height: int,
        steps: int,
        why: str,
        placement: Placement,
    ) -> Profile:
        estimate, fit = self.estimate_for(weights, width, height, placement)
        restream = (
            restreamed_bytes(estimate.total_bytes, self.budget.vram_free_bytes)
            if placement is Placement.STREAMED
            else 0
        )
        args = self._args(
            width,
            height,
            placement,
            steps=steps,
            cfg_scale=self.defaults.cfg_scale,
            sampler=self.defaults.sampler,
            flow_shift=self.defaults.flow_shift,
            main_flag=self.kit.main_flag,
            diffusion_type=weights.diffusion_type,
        )
        return Profile(
            name=name,
            variant=weights.variant,
            engine_args=args,
            seconds_per_image=self._seconds(weights, args, restream),
            kit_bytes=self.kit_bytes,
            estimate=estimate,
            fit=fit,
            why=why
            + _WHY_SUFFIX[placement].format(
                decode=_gib(self._decode_bytes(width, height)),
                restream="nothing extra" if not restream else f"about {_gib(restream)}",
            )
            + _CONVERTED.get(weights.diffusion_type or "", ""),
        )


def _gib(value: int) -> str:
    return f"{value / 2**30:.1f} GiB"


def _params_from_bytes(variant: GgufVariant) -> float:
    """Parameter count in billions from the file size and the quant's bits per weight."""
    bits = quant_bits(variant.quant) or 8.0
    return variant.size_bytes * 8 / bits / 1e9
