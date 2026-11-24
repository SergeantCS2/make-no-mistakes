"""The studio's packages audio.cpp runs: which family a Hub folder is, what kind of
work it does, and what it gives back.

They live beside the speech packages in one repo, `audio-cpp/audio.cpp-gguf`, one folder per
model (`ACE-Step1.5-GGUF/turbo/`, `BS-RoFormer-ep368-GGUF/`, `HTDemucs-6stems-GGUF/`), so the
reference the user pastes names the folder, exactly as `hub/voices.py` reads a voice
package: the family from the folder's name, the quant from each GGUF's name. Every package
read on 2026-10-31 is one self-contained GGUF per quant (ACE-Step's carries its DiT, the 1.7B
planner LM, the text encoder and the VAE; a separator's config is embedded), so a variant's
companions are empty today; `with_companions` keeps a `config.json` or `.yaml` beside the
GGUF should a later package ship one. Licences and the commercial column are audio.cpp's
own table (its Hub README and `docs/model_licenses.md`, read 2026-10-31).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from makenomistakes.hub.modality import Modality
from makenomistakes.hub.variants import GgufVariant

Commercial = Literal["yes", "no", "conditional", "unclear"]


class MusicFamily(StrEnum):
    """audio.cpp's `family` names for the studio's models, as its server config spells them
    (`engine_args_music.MusicFamilyName` lists the same names)."""

    ACE_STEP = "ace_step"
    HEARTMULA = "heartmula"
    YUE2 = "yue2"
    MINIMAX_MUSIC = "minimax_music3"
    STABLE_AUDIO = "stable_audio"
    HTDEMUCS = "htdemucs"
    HTDEMUCS_6STEMS = "htdemucs_6stems"
    BS_ROFORMER = "bs_roformer"
    MEL_BAND_ROFORMER = "mel_band_roformer"
    RVC = "rvc"
    SEED_VC = "seed_vc"
    VEVO2 = "vevo2"
    MUSCRIPTOR = "muscriptor"
    SHEETSAGE2 = "sheetsage2"


class MusicKit(BaseModel):
    family: MusicFamily
    label: str
    modality: Modality
    task: Literal["gen", "sep", "vc", "svc", "midi"]
    served_id: str  # the config's model id; never listed to chat clients (`/v1/models`)
    licence: str  # the words audio.cpp's table gives
    commercial: Commercial
    stems: tuple[str, ...] = ()  # what a separator gives back, by `named_audio_outputs` id
    sample_rate: int | None = None  # what it reads (a separator) or writes (a song model)
    channels: int = 2  # what it reads: a mono model refuses two (anvuew's room dereverb)
    note: str = ""


def _kit(family: MusicFamily, label: str, modality: Modality, task: str, **kw: object) -> MusicKit:
    served = family.value.replace("_", "-")
    return MusicKit.model_validate(
        {"family": family, "label": label, "modality": modality, "task": task, "served_id": served}
        | kw
    )


_SONG, _STEMS, _CONVERT, _SCORE = (
    Modality.SONG,
    Modality.STEMS,
    Modality.VOICE_CONVERT,
    Modality.MUSIC_SCORE,
)
KITS: dict[MusicFamily, MusicKit] = {
    k.family: k
    for k in (
        _kit(
            MusicFamily.ACE_STEP,
            "ACE-Step 1.5 (2B DiT + 1.7B planner)",
            _SONG,
            "gen",
            licence="MIT",
            commercial="yes",
            sample_rate=48_000,
            note="Tags and lyrics to a song of 10 s to 10 min; the base build also edits "
            "(extract, lego, complete).",
        ),
        _kit(
            MusicFamily.HEARTMULA,
            "HeartMuLa (3B)",
            _SONG,
            "gen",
            licence="Apache-2.0",
            commercial="yes",
            sample_rate=48_000,
            note="Lyrics and comma-separated tags are both required.",
        ),
        _kit(MusicFamily.YUE2, "YuE2", _SONG, "gen", licence="CC-BY-NC-4.0", commercial="no"),
        _kit(
            MusicFamily.MINIMAX_MUSIC,
            "MiniMax Music 3",
            _SONG,
            "gen",
            licence="MiniMax licence (non-commercial)",
            commercial="conditional",
        ),
        _kit(
            MusicFamily.STABLE_AUDIO,
            "Stable Audio 3",
            _SONG,
            "gen",
            licence="Stability AI Community License",
            commercial="conditional",
            note="Instrumentals and sound effects only: it sings no lyrics.",
        ),
        _kit(
            MusicFamily.HTDEMUCS,
            "HTDemucs (4 stems)",
            _STEMS,
            "sep",
            licence="MIT",
            commercial="yes",
            stems=("drums", "bass", "other", "vocals"),
            sample_rate=44_100,
        ),
        _kit(
            MusicFamily.HTDEMUCS_6STEMS,
            "HTDemucs (6 stems)",
            _STEMS,
            "sep",
            licence="MIT",
            commercial="yes",
            stems=("drums", "bass", "other", "vocals", "guitar", "piano"),
            sample_rate=44_100,
        ),
        _kit(
            MusicFamily.BS_ROFORMER,
            "BS-RoFormer (vocals)",
            _STEMS,
            "sep",
            licence="Apache-2.0 (checkpoint source undocumented)",
            commercial="unclear",
            stems=("vocals", "instrumental"),
            sample_rate=44_100,
        ),
        _kit(
            MusicFamily.MEL_BAND_ROFORMER,
            "Mel-Band RoFormer (vocals)",
            _STEMS,
            "sep",
            licence="MIT",
            commercial="yes",
            stems=("vocals", "instrumental"),
            sample_rate=44_100,
        ),
        _kit(
            MusicFamily.RVC,
            "RVC",
            _CONVERT,
            "vc",
            licence="MIT (the packaged voices' sources unknown)",
            commercial="unclear",
        ),
        # singing conversion (`v1_svc`): the studio converts sung stems
        _kit(MusicFamily.SEED_VC, "Seed-VC", _CONVERT, "svc", licence="GPL-3.0", commercial="yes"),
        _kit(
            MusicFamily.VEVO2, "Vevo2", _CONVERT, "svc", licence="CC-BY-NC-ND-4.0", commercial="no"
        ),
        _kit(
            MusicFamily.MUSCRIPTOR,
            "MuScriptor",
            _SCORE,
            "midi",
            licence="CC-BY-NC-4.0",
            commercial="no",
        ),
        _kit(
            MusicFamily.SHEETSAGE2,
            "SheetSage2",
            _SCORE,
            "midi",
            licence="CC-BY-NC-4.0",
            commercial="no",
        ),
    )
}

# The folder names audio.cpp's repo uses; the more specific pattern first (6 stems before 4).
_FAMILY_RE: tuple[tuple[re.Pattern[str], MusicFamily], ...] = (
    (re.compile(r"ace[-_ ]?step", re.I), MusicFamily.ACE_STEP),
    (re.compile(r"heart[-_ ]?mula", re.I), MusicFamily.HEARTMULA),
    (re.compile(r"\byue[-_ ]?2", re.I), MusicFamily.YUE2),
    (re.compile(r"minimax[-_ ]?music", re.I), MusicFamily.MINIMAX_MUSIC),
    (re.compile(r"stable[-_ ]?audio", re.I), MusicFamily.STABLE_AUDIO),
    (re.compile(r"htdemucs[-_ ]?6", re.I), MusicFamily.HTDEMUCS_6STEMS),
    (re.compile(r"htdemucs", re.I), MusicFamily.HTDEMUCS),
    (re.compile(r"bs[-_ ]?roformer", re.I), MusicFamily.BS_ROFORMER),
    (re.compile(r"mel[-_ ]?band[-_ ]?roformer", re.I), MusicFamily.MEL_BAND_ROFORMER),
    (re.compile(r"(?:^|/)rvc[-_]", re.I), MusicFamily.RVC),
    (re.compile(r"seed[-_ ]?vc", re.I), MusicFamily.SEED_VC),
    (re.compile(r"vevo[-_ ]?2", re.I), MusicFamily.VEVO2),
    (re.compile(r"muscriptor", re.I), MusicFamily.MUSCRIPTOR),
    (re.compile(r"sheet[-_ ]?sage", re.I), MusicFamily.SHEETSAGE2),
)


def music_family(package: str) -> MusicFamily | None:
    """The family a package folder's name says (`ACE-Step1.5-GGUF/turbo`), or None."""
    for pattern, family in _FAMILY_RE:
        if pattern.search(package):
            return family
    return None


def kit_for(family: MusicFamily) -> MusicKit:
    return KITS[family]


def one_package(variants: list[GgufVariant], package: str) -> bool:
    """Every variant sits in the pasted folder itself: a parent folder holding several builds
    (`ACE-Step1.5-GGUF` holds `base/` and `turbo/`) names no one package."""
    return bool(variants) and all(v.files[0].rsplit("/", 1)[0] == package for v in variants)
