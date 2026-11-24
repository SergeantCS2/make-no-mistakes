"""Fit for the studio's models on audio.cpp: one row per family, the shape of
`fit/voice.py` (no offload: a model fits on the card or it does not, and the choice is which
quant of the one package).

VRAM = the GGUF + the family's working allowance (`MusicFamilyRow.working_bytes`: the
engine's buffers while a song or a split runs) + the CUDA runtime term. Time = `rate`
seconds of work per second of audio on the 4070. No figure here was read on a card: each row
names where it came from and says `estimated`; the box's first readings replace an instance's VRAM
through `calibration`'s instance figure, and a plan is the row's estimate priced with the
engine's correction, naming this box's readings of the family from three (`basis_words`).
Another card's time is the 4070's scaled by TFLOPS, the processor's the CPU image's reading
scaled by its cores (`music_time`), said as an estimate too, until this box's own rate
replaces it (`music_speed`).

A CPU build on 4 vCPUs read: ACE-Step 1.5 turbo Q8 (a 5.76 GiB file)
resident at 6.29 GiB after a 10 s song, 83 s cold and 49 s warm; HTDemucs 6 stems on a 30 s
mix in 137 s on one thread; BS-RoFormer on 5 s in 146 s on four. CPU figures say nothing of
a card's buffers, so they stand beside the rows, never in them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel

from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.engine_args_music import (
    OPTIONS_KIND,
    AceStepOptions,
    ConvertOptions,
    MusicOptions,
    SeparatorOptions,
)
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.cpu_plan import CPU_TFLOPS_PER_CORE
from makenomistakes.fit.device_row import RTX_4070, DeviceRow
from makenomistakes.fit.estimate import RUNTIME_OVERHEAD_BYTES, FitLevel, MemoryEstimate
from makenomistakes.fit.profiles import Profile, ProfileName
from makenomistakes.fit.quant_ladder import quant_bits
from makenomistakes.fit.voice import voice_fit
from makenomistakes.hub.facts import ModelFacts
from makenomistakes.hub.music import MusicFamily, kit_for
from makenomistakes.hub.variants import GgufVariant

GIB = 1024**3
REPORT = "published figures"
ESTIMATE_WORDS = f"estimate ({REPORT})"
READINGS_TO_TRUST = 3  # as the picture corrections: three of this box's own
CPU_SLOWER = 70.0  # the CPU image against the 4070 (ACE-Step: 8.3 s/s cold over 0.12)
CPU_FAMILY = "cpu"  # the processor's device row (`fit.cpu_plan.cpu_row`)
CPU_READ_CORES = 4  # the vCPUs CPU_SLOWER was read on
CPU_READ_TFLOPS = CPU_READ_CORES * CPU_TFLOPS_PER_CORE  # as `cpu_row` prices them
# ACE-Step's `num_inference_steps` when a request names none, the steps its row's rate is for
# (audio.cpp's own parameters for the model: default 8, turbo's cap)
ACE_STEP_STEPS = 8


class MusicFamilyRow(BaseModel):
    """What a family costs beyond its weights, and how fast it works, on the 4070."""

    family: MusicFamily
    working_bytes: int  # the engine's buffers while it runs, beyond the GGUF
    rate: float  # seconds of work per second of audio on the 4070
    source: str
    estimated: bool = True


def _row(family: MusicFamily, working_gib: float, rate: float, source: str) -> MusicFamilyRow:
    return MusicFamilyRow(
        family=family, working_bytes=int(working_gib * GIB), rate=rate, source=source
    )


F = MusicFamily
ROWS: dict[MusicFamily, MusicFamilyRow] = {
    r.family: r
    for r in (
        _row(
            F.ACE_STEP,
            2.25,
            0.12,
            f"{REPORT}: the 2B turbo with the 1.7B LM in the '8 to 16 GB' tier, its floor less "
            "the Q8 package's 5.76 GiB; '<10 s per song on a 3090', 15 to 25 s on a 4070 for "
            "a 3-minute song",
        ),
        _row(F.HEARTMULA, 4.5, 1.0, f"{REPORT}: 12 GB with lazy loading, RTF about 1.0"),
        _row(F.YUE2, 6.0, 0.75, f"{REPORT}: 11.0 to 11.2 GiB peak on a 4090, 95 s for 3:16"),
        _row(F.MINIMAX_MUSIC, 7.0, 3.1, f"{REPORT}: about 14 GB at Q8 and RTF 2.1 on a 3090"),
        _row(F.STABLE_AUDIO, 3.0, 0.2, f"{REPORT}: 5.1 to 6.5 GB for the Medium model"),
        _row(F.HTDEMUCS, 6.5, 0.1, f"{REPORT}: 3 to 7 GB by segment length (htdemucs_ft)"),
        _row(F.HTDEMUCS_6STEMS, 6.5, 0.1, f"{REPORT}: 3 to 7 GB by segment length"),
        _row(F.BS_ROFORMER, 7.3, 0.3, f"{REPORT}: '8 GB recommended' (UVR), time not published"),
        _row(F.MEL_BAND_ROFORMER, 7.3, 0.3, f"{REPORT}: '8 GB recommended' (UVR)"),
        _row(F.RVC, 0.5, 0.5, f"{REPORT}: a small model, realtime-capable"),
        _row(F.SEED_VC, 2.0, 1.0, "the model's notes: zero-shot from 1 to 30 s, no VRAM figure"),
        _row(F.VEVO2, 6.0, 1.0, "the model's notes: about 6 / 10 / 12 GB at 15 / 30 / 45 s"),
        _row(F.MUSCRIPTOR, 2.0, 0.5, "no figure read: a placeholder until a reading"),
        _row(F.SHEETSAGE2, 4.0, 0.5, f"{REPORT}: up to a 3.6 GB tensor unless tiled"),
    )
}
_TARGET_BITS = {ProfileName.QUALITY: 16.0, ProfileName.BALANCED: 8.0, ProfileName.SPEED: 8.0}
ONLY_ONE = "Its one set of weights (every profile is the same model)."
_WHY = {
    ProfileName.QUALITY: "Full-precision weights where they fit, else the 8-bit ones.",
    ProfileName.BALANCED: "8-bit weights: the same model at about half the size.",
    ProfileName.SPEED: "8-bit weights: the package's smallest, as balanced chose.",
}


# A checkpoint mnm converts (`hub/converted.py`) is not its family's packaged model:
# its own row, by its file, from mnm's reading on audio.cpp's full-cpu image (4 vCPUs)
CONVERTED_ROWS: dict[str, MusicFamilyRow] = {
    "dereverb_room_anvuew_sdr_13.7432.ckpt": MusicFamilyRow(
        family=MusicFamily.BS_ROFORMER,
        working_bytes=int(1.3 * GIB),
        rate=9.0 / CPU_SLOWER,
        source="mnm's reading on audio.cpp's full-cpu image: 1.41 GiB resident with "
        "its 0.11 GiB of weights, 8.4 to 9.7 s of work a second of audio on 4 vCPUs; the "
        "4070's rate is that over the CPU image's 70x",
    ),
}


def row_for(family: MusicFamily, file: str | None = None) -> MusicFamilyRow:
    """The family's row, or a converted checkpoint's own when `file` is one."""
    own = CONVERTED_ROWS.get(PurePosixPath(file).name) if file else None
    return own if own is not None else ROWS[family]


def music_estimate(
    family: MusicFamily, model_bytes: int, file: str | None = None
) -> MemoryEstimate:
    return MemoryEstimate(
        weights_bytes=model_bytes,
        compute_bytes=row_for(family, file).working_bytes,
        overhead_bytes=RUNTIME_OVERHEAD_BYTES,
    )


def rate_key(family: str, file: str) -> str:
    """How a learned rate is looked up (`Budget.music_rates`, `music_speed`): the family and
    the model's file name, since turbo and base are one family."""
    return f"{family}/{PurePosixPath(file).name}"


class TimeBasis(StrEnum):
    """Where a music plan's time came from."""

    MEASURED = "measured"  # this box's own readings (`music_speed`)
    ESTIMATE = "estimate"  # the row's figure for the 4070
    SCALED = "scaled"  # the row's figure scaled to this card or processor


@dataclass(frozen=True)
class MusicTime:
    rate: float  # seconds of work a second of audio
    basis: TimeBasis
    device: str  # the card or processor it is for

    @property
    def words(self) -> str:
        """How the time was come by, in a phrase: "is measured on this box" …"""
        if self.basis is TimeBasis.MEASURED:
            return "is measured on this box"
        if self.basis is TimeBasis.ESTIMATE:
            return f"is an estimate ({REPORT})"
        if self.device == CPU_FAMILY:
            return "is the CPU image's reading scaled to this processor's cores, an estimate"
        return f"is the 4070's estimate scaled by TFLOPS to the {self.device}"


def music_time(
    family: MusicFamily,
    device: DeviceRow,
    file: str | None = None,
    learned: dict[str, float] | None = None,
) -> MusicTime:
    """Seconds of work per second of audio on this card: this box's own rate for the model's
    file once it has three readings (`learned`, `music_speed`), else the 4070's figure
    scaled by its TFLOPS over this card's; on the processor the CPU image's reading
    (`CPU_SLOWER`, 4 vCPUs) scaled by its cores, so a plan and a client's wait agree."""
    if learned and file:
        found = learned.get(rate_key(family.value, file))
        if found is not None:
            return MusicTime(found, TimeBasis.MEASURED, device.name)
    rate = row_for(family, file).rate
    if device.family == CPU_FAMILY:
        scale = CPU_READ_TFLOPS / device.tflops if device.tflops > 0 else 1.0
        return MusicTime(rate * CPU_SLOWER * scale, TimeBasis.SCALED, CPU_FAMILY)
    if device.name == RTX_4070.name:
        return MusicTime(rate, TimeBasis.ESTIMATE, device.name)
    scale = RTX_4070.tflops / device.tflops if device.tflops > 0 else 1.0
    return MusicTime(rate * scale, TimeBasis.SCALED, device.name)


def rate_on(
    family: MusicFamily,
    device: DeviceRow,
    file: str | None = None,
    learned: dict[str, float] | None = None,
) -> float:
    return music_time(family, device, file, learned).rate


def basis_words(readings: int) -> str:
    """How the plan's VRAM was come by: always the row's estimate (a plan never takes an
    instance's figure; the engine's correction, the plan's Calibration line, prices it), and
    from three readings of the family the count this box holds (never "measured on this
    box" beside the unchanged estimate)."""
    if readings >= READINGS_TO_TRUST:
        return f"an {ESTIMATE_WORDS}, with {readings} readings of the family on this box"
    return ESTIMATE_WORDS


def default_options(family: MusicFamily) -> MusicOptions | None:
    kind = OPTIONS_KIND.get(family.value)
    if kind == "ace_step":
        return AceStepOptions()
    if kind == "separator":
        return SeparatorOptions()
    if kind == "convert":
        return ConvertOptions()
    return None


def music_args(family: MusicFamily) -> AudioCppArgs:
    """The config's model entry: the family, its task, its options at their defaults."""
    kit = kit_for(family)
    return AudioCppArgs(family=family.value, task=kit.task, options=default_options(family))


def music_profile(
    name: ProfileName, facts: ModelFacts, variant: GgufVariant, budget: Budget, why: str
) -> Profile:
    assert facts.music_family is not None  # the planner only runs for a music package
    family = facts.music_family
    estimate = music_estimate(family, variant.size_bytes, variant.files[0])
    rate = rate_on(family, budget.device, variant.files[0], budget.music_rates)
    return Profile(
        name=name,
        variant=variant,
        engine_args=music_args(family),
        estimate=estimate,
        fit=voice_fit(estimate, budget),
        tokens_per_second=1.0 / rate,  # seconds of audio a second of work (x realtime)
        why=why,
    )


def plan_music_profiles(facts: ModelFacts, budget: Budget) -> dict[ProfileName, Profile]:
    """All three, as the voices plan: the quant nearest each target that fits, else the
    smallest as NO_FIT (`Plan.choose` refuses it with the reason)."""
    profiles: dict[ProfileName, Profile] = {}
    if not facts.gguf_variants or facts.music_family is None:
        return profiles
    smallest = min(facts.gguf_variants, key=lambda v: v.size_bytes)
    for name, bits in _TARGET_BITS.items():
        ranked = sorted(
            facts.gguf_variants, key=lambda v: (abs(quant_bits(v.quant) - bits), v.size_bytes)
        )
        why = _WHY[name] if len(facts.gguf_variants) > 1 else ONLY_ONE
        fitting = (music_profile(name, facts, v, budget, why) for v in ranked)
        profiles[name] = next(
            (p for p in fitting if p.fit.level is not FitLevel.NO_FIT),
            music_profile(name, facts, smallest, budget, why),
        )
    return profiles


def plan_music_custom(facts: ModelFacts, budget: Budget, quant: str) -> Profile:
    variant = facts.variant(quant)
    if variant is None:
        available = ", ".join(v.quant for v in facts.gguf_variants) or "none"
        raise ValueError(f"{facts.repo_id} has no '{quant}' file (available: {available})")
    return music_profile(ProfileName.CUSTOM, facts, variant, budget, "Your explicit choice.")


def task_timeout(  # noqa: PLR0913 - where it runs (a card's row, or cores) and what it asks
    family: MusicFamily,
    seconds: float,
    cpu: bool = False,
    file: str | None = None,
    learned: float | None = None,
    *,
    cores: int | None = None,
    steps: int | None = None,
    device: DeviceRow | None = None,
) -> float:
    """How long a client waits for one task, and the router for the engine's answer: the rate
    times the audio's length, four times over (a cold load, a longer piece), never under the
    ten minutes `mnm try` waits. On the CPU image (4 vCPUs) ACE-Step took 4.9 to 8.3 s of
    work per second of audio, about seventy times the card's estimate (`CPU_SLOWER`), scaled
    by its `cores` as `music_time` scales it; on a card the 4070's estimate scaled to the
    `device` (its row's TFLOPS, `music_time`). A song asked at more denoising `steps` than
    the row's rate is for waits that much longer (the full tier's base at 50: about six
    times). This box's own rate (`learned`) only lengthens the wait: a learned warm
    rate never shortens it below the estimate's, since a first task after a load runs far
    slower (54 s/s against 5.5)."""
    if cpu:
        scale = CPU_SLOWER * CPU_READ_CORES / cores if cores else CPU_SLOWER
        estimate = row_for(family, file).rate * scale
    else:
        estimate = music_time(family, device or RTX_4070, file).rate
    estimate *= steps_scale(family, steps)
    rate = max(estimate, learned) if learned is not None else estimate
    return max(600.0, rate * seconds * 4)


def steps_scale(family: MusicFamily, steps: int | None) -> float:
    """A song's denoising steps over the ones its row's rate is for (`ACE_STEP_STEPS`); 1 for
    other families, steps not named, or fewer (a wait never shortens for them)."""
    if family is not MusicFamily.ACE_STEP or steps is None:
        return 1.0
    return max(steps / ACE_STEP_STEPS, 1.0)
