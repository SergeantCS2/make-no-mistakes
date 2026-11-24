"""The studio's models and the cards that hold them (every packaged model and the three
without a package yet). audio.cpp offloads nothing, so a music model runs on the card whole or
not at all, one stage at a time: what it needs is the card's free VRAM, or on a box that runs
every model in RAM the RAM's room (`ram_room`).

A packaged rung's figure is `fit.music`'s plan from the recorded Hub tree (an estimate until
this box's readings). YuE2, MiniMax Music 3 and
ACE-Step 1.5 XL have no audio.cpp package in that tree: their rows carry published figures
(upstream's or third parties'), said as such, so the ladder says which card holds them
before they can be added. `mnm studio models` and System → Studio show the rows with this box's
verdict and, once measured, its own time for each (`music_speed`).
"""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.fit.estimate import FitLevel, headroom_level
from makenomistakes.hub.music import MusicFamily

GIB = 2**30
AUDIO_CPP = "audio-cpp/audio.cpp-gguf"
PLANNED = "fit.music's plan from the recorded Hub tree (an estimate)"
REPORT = "a published figure"


@dataclass(frozen=True)
class MusicRung:
    """One studio model at one quant: what it needs of the card, what it downloads."""

    model: str
    stage: str  # the studio stage it plays: song, stems, voice, midi
    family: MusicFamily
    quant: str
    card_gib: float  # the weights, the family's working allowance and the CUDA runtime
    download_gb: float
    folder: str | None = None  # its package in audio.cpp-gguf; None: none yet
    file: str | None = None  # the package's GGUF at this quant
    source: str = PLANNED

    @property
    def reference(self) -> str | None:
        """What `mnm fit`/`mnm add` take; None while no package exists."""
        return f"{AUDIO_CPP}/{self.folder}" if self.folder else None

    @property
    def card_bytes(self) -> int:
        return int(self.card_gib * GIB)

    @property
    def label(self) -> str:
        return f"{self.model} {self.quant}"


F = MusicFamily
# The stages registry's seeded models first (`studio/stages.json`), then every other packaged
# music model mnm plans, then the three without a package; smallest need first in each group.
MUSIC_LADDER: tuple[MusicRung, ...] = (
    MusicRung("HTDemucs 6 stems", "stems", F.HTDEMUCS_6STEMS, "Q8_0", 6.79, 0.04,
              "HTDemucs-6stems-GGUF", "htdemucs-6stems-q8_0.gguf"),
    MusicRung("BS-RoFormer ep368", "stems", F.BS_ROFORMER, "Q8_0", 7.71, 0.17,
              "BS-RoFormer-ep368-GGUF", "bs-roformer-ep368-q8_0.gguf"),
    MusicRung("ACE-Step 1.5 turbo", "song", F.ACE_STEP, "Q8_0", 8.26, 6.19,
              "ACE-Step1.5-GGUF/turbo", "ace-step-1.5-turbo-q8_0.gguf"),
    MusicRung("ACE-Step 1.5 base", "song", F.ACE_STEP, "Q8_0", 8.26, 6.19,
              "ACE-Step1.5-GGUF/base", "ace-step-1.5-base-q8_0.gguf"),
    MusicRung("RVC", "voice", F.RVC, "F16", 1.92, 1.26, "RVC-GGUF", "rvc-f16.gguf"),
    MusicRung("MuScriptor Small", "midi", F.MUSCRIPTOR, "F32", 2.63, 0.41,
              "MuScriptor-Small-GGUF", "muscriptor-small-f32.gguf"),
    MusicRung("Stable Audio 3 Small Music", "song", F.STABLE_AUDIO, "Q8_0", 4.82, 1.68,
              "Stable-Audio-3-Small-Music-GGUF", "stable-audio-3-small-music-q8_0.gguf"),
    MusicRung("Seed-VC", "voice", F.SEED_VC, "Q8_0", 5.16, 3.12,
              "SeedVC-MLX-GGUF", "seed-vc-mlx-q8_0.gguf"),
    MusicRung("Stable Audio 3 Medium", "song", F.STABLE_AUDIO, "Q8_0", 6.59, 3.58,
              "Stable-Audio-3-Medium-GGUF", "stable-audio-3-medium-q8_0.gguf"),
    MusicRung("HTDemucs 4 stems", "stems", F.HTDEMUCS, "Q8_0", 6.81, 0.06,
              "HTDemucs-GGUF", "htdemucs-q8_0.gguf"),
    MusicRung("Mel-Band RoFormer", "stems", F.MEL_BAND_ROFORMER, "Q8_0", 7.78, 0.25,
              "Mel-Band-RoFormer-GGUF", "mel-band-roformer-q8_0.gguf"),
    MusicRung("Vevo2", "voice", F.VEVO2, "Q8_0", 9.27, 3.24, "Vevo2-GGUF", "vevo2-q8_0.gguf"),
    MusicRung("HeartMuLa 3B", "song", F.HEARTMULA, "Q8_0", 11.88, 7.66,
              "HeartMuLa-GGUF", "heartmula-q8_0.gguf"),
    MusicRung("MiniMax Music 3", "song", F.MINIMAX_MUSIC, "Q8", 13.04, 8.39,
              source=f"{REPORT}: 'peaks ~14 GB on one 3090' at Q8 (third party)"),
    MusicRung("YuE2-3B", "song", F.YUE2, "int8", 14.9, 6.0,
              source=f"{REPORT}: its int8 checkpoint ran on a 16 GB 4080 in ComfyUI; "
              "'24 GB' upstream"),
    MusicRung("ACE-Step 1.5 XL", "song", F.ACE_STEP, "BF16", 18.63, 9.0,
              source=f"{REPORT}: 'XL ≥12 GB with offload and quantisation, ≥20 GB without'"),
)  # fmt: skip


def card_holds(rung: MusicRung, card_free_bytes: int) -> bool:
    """One stage at a time: the model alone on the card's free VRAM, by the planner's
    own rule (`headroom_level`, the one `fit.voice.voice_fit` plans with)."""
    return headroom_level(rung.card_bytes, card_free_bytes) is not FitLevel.NO_FIT


# What a card's driver, runtime and display hold of it before any model: a nominal 12 GiB card
# never offers 12 GiB free (else HeartMuLa's 11.9 GiB would read "a 12 GiB card" as the card it
# needs and "0.4 GiB short" on that very card) [est]
CARD_KEPT = GIB // 2
# The card sizes a model's need is said in: the common ones, GiB.
CARD_SIZES = (8, 12, 16, 24, 32, 48, 80)


def smallest_card(rung: MusicRung) -> int | None:
    """The smallest common card (GiB) that holds it alone, less what the card keeps."""
    return next((gib for gib in CARD_SIZES if card_holds(rung, gib * GIB - CARD_KEPT)), None)


def need_words(rung: MusicRung, where: str = "the card") -> str:
    said = "an estimate" if rung.source == PLANNED else "a published figure"
    return f"needs about {rung.card_gib:.1f} GiB of {where} ({said})"


def verdict_words(rung: MusicRung, free_bytes: int) -> str:
    """This box's word on one rung, the planner's level: it holds it (near the limit when the
    plan would read tight), or by how much it falls short."""
    level = headroom_level(rung.card_bytes, free_bytes)
    if level is FitLevel.COMFORTABLE:
        return "holds it"
    if level is FitLevel.TIGHT:
        return "holds it, near the limit"
    return f"{(rung.card_bytes - free_bytes) / GIB:.1f} GiB short"


def card_words(rung: MusicRung) -> str:
    """The smallest common card that holds it, in words: "a 12 GiB card", "an 8 GiB card"."""
    gib = smallest_card(rung)
    if gib is None:
        return f"no single card of up to {CARD_SIZES[-1]} GiB"
    article = "an" if str(gib).startswith("8") or gib in (11, 18) else "a"  # an 8, an 18
    return f"{article} {gib} GiB card"
