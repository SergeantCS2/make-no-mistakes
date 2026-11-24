"""How a style training is planned: the tight profile on a card under 16 GB, the box measures
the rest.

**The estimate** is arithmetic, never a reading: the DiT's decoder in bf16, the adapter with its
gradients and optimizer state, the activations of one song's latents under gradient
checkpointing (linear in its length), the encoders and VAE unless they are moved to the
processor, the CUDA context, and 12% for fragmentation. Upstream publishes no measured figure
for a 12 GB card, so the estimate refuses only a training that cannot fit by its own
arithmetic; the probe decides the rest.

**The probe** is the first epoch of every training on a card: the router reads the card's
memory in use as the training's container starts (what other programs hold), samples it once
the training stage begins (preprocessing is not the training, and a resumed training skips
it), and once that start's first checkpoint is saved judges the peak (`judged`). The
training's own share is the peak less what was in use at the start; within `margin_gib()` of
the card's top that share alone stops the training for good with what to change, and when it
is the outside use that takes it there the training pauses until the room is free
(`room_words`). The own share is kept per box and settings
(`<state>/studio/style-probes.json`), so the next plan says the measured peak and the time an
epoch takes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from makenomistakes.fit.budget import WSL_SPILL_MARGIN
from makenomistakes.hardware import platform as platforms
from makenomistakes.studio.style_trainer import CPU_PRESET, Preset, StyleJob, preset_for
from makenomistakes.studio.trainer import Stop

GIB = 2**30
DECODER_GIB = 2.8  # the DiT's decoder in bf16: about 1.52 B weights
ENCODERS_GIB = 1.5  # condition encoders, VAE and text encoder, when they stay on the card
ADAPTER_GIB_PER_RANK = 0.3 / 32  # the adapter, its gradients and AdamW's state
ACTIVATIONS_GIB_PER_SECOND = 0.65 / 240  # one song's latents, gradient checkpointing on
CONTEXT_GIB = 0.5
FRAGMENTATION = 1.12
NATIVE_MARGIN_GIB = 0.25  # native Linux: a later epoch's fragmentation, before it runs out
CPU_RAM_GIB = 13.0  # the whole model in fp32 with the encoders, on the processor
CPU_RAM_SPARE_GIB = 2.0  # what the system keeps beside it
PROBES = "studio/style-probes.json"


class ProbeReading(BaseModel):
    peak_gib: float  # the training's own: the card's highest in epoch 1 less `outside_gib`
    card_gib: float
    outside_gib: float = 0.0  # in use by other programs as the training's container started
    epoch_seconds: float | None = None
    songs: int
    machine: str | None = None

    @property
    def seconds_per_song(self) -> float | None:
        return self.epoch_seconds / self.songs if self.epoch_seconds and self.songs else None


@dataclass(frozen=True)
class StylePlan:
    preset: Preset
    estimate_gib: float
    card_gib: float | None  # None: trained on the processor
    max_seconds: float
    measured: ProbeReading | None = None

    @property
    def words(self) -> str:
        where = (
            f"on the processor (about {CPU_RAM_GIB:.0f} GiB of RAM)"
            if self.card_gib is None
            else f"about {self.estimate_gib:.1f} GiB of the card's {self.card_gib:.1f}"
        )
        measured = (
            f"; measured on this box: a peak of {self.measured.peak_gib:.1f} GiB"
            if self.measured is not None
            else "; the first epoch measures the card's real peak"
            if self.card_gib is not None
            else ""
        )
        preset = self.preset
        whose = "mnm's" if preset.name == CPU_PRESET.name else f"upstream's {preset.name}:"
        offload = ", encoders on the processor" if preset.offload_encoder else ""
        return (
            f"{preset.words} settings ({whose} rank {preset.rank}, {preset.optimizer}{offload}), "
            f"{where}{measured}"
        )

    def hours(self, epochs: int, songs: int) -> float | None:
        """The training's hours by the probe's reading; None before one."""
        per_song = self.measured.seconds_per_song if self.measured else None
        return epochs * songs * per_song / 3600 if per_song else None


def estimate_gib(preset: Preset, max_seconds: float) -> float:
    on_card = (
        DECODER_GIB
        + preset.rank * ADAPTER_GIB_PER_RANK
        + max_seconds * ACTIVATIONS_GIB_PER_SECOND
        + (0.0 if preset.offload_encoder else ENCODERS_GIB)
        + CONTEXT_GIB
    )
    return round(on_card * FRAGMENTATION, 1)


def plan(
    card_gib: float | None, max_seconds: float, measured: ProbeReading | None = None
) -> StylePlan:
    preset = preset_for(card_gib) if card_gib is not None else CPU_PRESET
    return StylePlan(preset, estimate_gib(preset, max_seconds), card_gib, max_seconds, measured)


ON_THE_PROCESSOR = (
    "the style trainer built for the processor trains it, slowly "
    "(`mnm studio trainer install --for styles --cpu`)"
)


def margin_gib() -> float:
    """How close to the card's top a training may come: under WSL the driver spills to system
    memory past it (`WSL_SPILL_MARGIN`, what every plan keeps free there), and a training that
    spills crawls for hours; on native Linux the card runs out instead, so only a later
    epoch's fragmentation is kept free."""
    return WSL_SPILL_MARGIN / GIB if platforms.current().is_wsl else NATIVE_MARGIN_GIB


def _top_words() -> str:
    if platforms.current().is_wsl:
        return "where the driver spills to system memory and a training crawls"
    return "where a later epoch runs out of memory"


def _shorter(max_seconds: float) -> str:
    """`--max-seconds N` at half the songs' share, when there is a half to take."""
    halved = max(30, int(max_seconds // 2))
    return (
        f"`--max-seconds {halved}` (shorter songs, less of the card), or "
        if (halved < max_seconds)
        else ""
    )


@dataclass(frozen=True)
class Fitted:
    job: StyleJob
    preset: Preset | None  # the preset the job now follows; None: its own settings kept


def fit_job(job: StyleJob, *, on_card: bool, card_gib: float | None, resumed: bool) -> Fitted:
    """The one rule for a style's job on the build that runs it, applied when it is queued
    and again at each start (a reinstall or a rollback may have moved it between the CPU and
    CUDA): on the processor plain AdamW in fp32, nothing offloaded; on a card bf16 and
    upstream's preset for the card's size. The rank, alpha, accumulation and optimizer follow
    the preset only while no checkpoint holds them (`resumed`: a resumed adapter keeps its
    shape and its optimizer's state); a card that cannot be read keeps the job's own."""
    preset = CPU_PRESET if not on_card else preset_for(card_gib) if card_gib is not None else None
    update: dict[str, object] = {
        "device": "cuda" if on_card else "cpu",
        # the processor trains in fp32: upstream's bf16-mixed there stops at the first step
        # ("Found dtype BFloat16 but expected Float")
        "precision": "bf16" if on_card else "fp32",
    }
    if preset is not None:
        update["offload_encoder"] = preset.offload_encoder
        if not resumed:
            update.update(
                rank=preset.rank,
                alpha=preset.alpha,
                grad_accum=preset.grad_accum,
                optimizer=preset.optimizer,
            )
    if not on_card:
        update["optimizer"] = "adamw"  # bitsandbytes is in the CUDA builds only
    return Fitted(job.model_copy(update=update), preset if not resumed else None)


def refusal(planned: StylePlan) -> str | None:
    """Why the training cannot fit by its own arithmetic; None when the probe may decide."""
    card = planned.card_gib
    if card is None or planned.estimate_gib + margin_gib() <= card:
        return None
    return (
        f"A style training needs about {planned.estimate_gib:.1f} GiB of the card by its "
        f"estimate ({planned.preset.words} settings at {planned.max_seconds:.0f} s a song), and "
        f"the card has {card:.1f} GiB: {_shorter(planned.max_seconds)}{ON_THE_PROCESSOR}."
    )


def judged(own_gib: float, outside_gib: float, card_gib: float, max_seconds: float) -> Stop | None:
    """The probe's verdict on the training's own peak beside what other programs held as it
    started: None when it may go on; a pause when only the outside use takes it to the card's
    top (it waits for the room, `room_words`); else a stop for good with what to change."""
    margin = margin_gib()
    if own_gib + outside_gib + margin <= card_gib:
        return None
    if own_gib + margin <= card_gib:
        return Stop(
            f"its first epoch peaked at {own_gib:.1f} GiB with {outside_gib:.1f} GiB of the card "
            f"in use by {platforms.current().others}, within {margin:.2f} GiB of the card's "
            f"{card_gib:.1f}, {_top_words()}",
            pause=True,
            until=f"once {own_gib + margin:.1f} GiB of the card is free",
        )
    return Stop(
        f"The training's first epoch peaked at {own_gib:.1f} GiB of the card's {card_gib:.1f} on "
        f"its own, within {margin:.2f} GiB of the top, {_top_words()}: train again with "
        f"{_shorter(max_seconds)}{ON_THE_PROCESSOR}."
    )


def room_words(own_gib: float, free_gib: float) -> str | None:
    """Why a training whose own peak is known waits before it starts again; None when the
    card has the room (the free VRAM counts what mnm's running models would give back)."""
    need = own_gib + margin_gib()
    if free_gib >= need:
        return None
    return (
        f"{need:.1f} GiB of the card must be free first (its first epoch peaked at "
        f"{own_gib:.1f} GiB)"
    )


def probe_key(machine: str | None, variant: str, preset: str, max_seconds: float) -> str:
    """A probe's reading is reused for the same box, variant, preset and song length."""
    return f"{machine or 'unknown'}:{variant}:{preset}:{max_seconds:.0f}"


def read_probes(state_dir: Path) -> dict[str, ProbeReading]:
    try:
        raw = json.loads((state_dir / PROBES).read_text(encoding="utf-8"))
        return {key: ProbeReading.model_validate(value) for key, value in raw.items()}
    except (OSError, ValueError, ValidationError, AttributeError):
        return {}


def merged_probes(here: bytes, given: bytes) -> bytes:
    """A restore's probes: the backup's, with this box's own for every key both hold (a key
    names the box, and what it holds here was measured here, since the backup was made); the
    backup's whole when this file does not read (replacing the file would lose the probes
    measured after the backup)."""
    try:
        kept, backed = json.loads(here), json.loads(given)
    except ValueError:
        return given
    if not isinstance(kept, dict) or not isinstance(backed, dict):
        return given
    return json.dumps({**backed, **kept}, indent=2).encode()


def keep_probe(state_dir: Path, key: str, reading: ProbeReading) -> None:
    probes = read_probes(state_dir)
    probes[key] = reading
    path = state_dir / PROBES
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({k: v.model_dump() for k, v in probes.items()}, indent=2), encoding="utf-8"
    )
    tmp.replace(path)
