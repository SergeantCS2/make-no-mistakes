"""Text-to-speech packages audio.cpp runs: which family a Hub folder is, what it
ships beside its GGUF, and the voices it speaks with.

The packages live in one repo, `audio-cpp/audio.cpp-gguf`, one folder per model
(`Kokoro-82M-GGUF/`, `PocketTTS-GGUF/english/`, `Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/`), so the
reference the user pastes names the folder; the family comes from that folder's name, the
quant from the file name, and PocketTTS's built-in voices are `embeddings/*.safetensors`
beside its GGUF, which travel with the model as a kit's companions do.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel

from makenomistakes.hub.variants import GgufVariant

AUDIOCPP_REPO = "audio-cpp/audio.cpp-gguf"


class VoiceFamily(StrEnum):
    """audio.cpp's `family` names, as its server config spells them."""

    KOKORO = "kokoro_tts"
    POCKET = "pocket_tts"
    QWEN3 = "qwen3_tts"


class VoiceKit(BaseModel):
    family: VoiceFamily
    label: str
    served_id: str  # carries `tts`: Odysseus's voice picker lists only such ids
    default_voice: str  # audio.cpp's `voice_id` for the family's default voice
    voices: tuple[str, ...]  # built-in voices a request may name (`voice`)
    task: str = "tts"  # `vdes` for Qwen3-TTS VoiceDesign (an instruction, not a reference)
    clones: bool = False  # a reference WAV in the voice library gives a new voice
    language_option: bool = False  # PocketTTS loads one language's weights (`load_options`)
    note: str = ""
    # Why the engine's official image cannot speak with this family; empty when it can.
    missing: str = ""


# Kokoro's and PocketTTS's voices are the GGUF's and the embeddings folder's; Qwen3-TTS
# CustomVoice ships nine speakers (its card). The server resolves a request's `voice` through
# the presets the config declares, so each kit's list becomes those presets.
KITS: dict[VoiceFamily, VoiceKit] = {
    VoiceFamily.KOKORO: VoiceKit(
        family=VoiceFamily.KOKORO,
        label="Kokoro (82M)",
        served_id="kokoro-tts",
        default_voice="af_heart",
        voices=(
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ),
        note="English first (American and British voices), quick on the CPU too; no cloning.",
        # audio.cpp turns its English (and Spanish, French, Hindi, Italian, Portuguese) text
        # into phonemes with eSpeak-ng, loaded from the system or linked in with
        # AUDIOCPP_STATIC_ESPEAK; the official CUDA image has neither, and every request
        # answered 500 "Could not load eSpeak-ng".
        missing="audio.cpp's official image has no eSpeak-ng, which Kokoro needs to turn "
        'text into phonemes: every request answers HTTP 500 "Could not load eSpeak-ng"',
    ),
    VoiceFamily.POCKET: VoiceKit(
        family=VoiceFamily.POCKET,
        label="PocketTTS (100M)",
        served_id="pocket-tts",
        default_voice="alba",
        voices=(
            "alba",
            "anna",
            "azelma",
            "charles",
            "cosette",
            "eponine",
            "estelle",
            "eve",
            "fantine",
            "george",
            "jane",
            "javert",
            "jean",
            "marius",
        ),
        clones=True,
        language_option=True,
        note="One language per package folder (english, french, german, italian, portuguese, "
        "spanish); clones a reference WAV without a transcript.",
    ),
    VoiceFamily.QWEN3: VoiceKit(
        family=VoiceFamily.QWEN3,
        label="Qwen3-TTS (0.6B / 1.7B)",
        served_id="qwen3-tts",
        default_voice="Vivian",
        voices=(
            "Vivian",
            "Ryan",
            "Serena",
            "Aiden",
            "Dylan",
            "Eric",
            "Ono_Anna",
            "Sohee",
            "Uncle_Fu",
        ),
        clones=True,
        note="CustomVoice builds speak with built-in speakers; Base builds clone a reference "
        "WAV plus its transcript; VoiceDesign builds take a description of the voice.",
    ),
}

_FAMILY_RE: tuple[tuple[re.Pattern[str], VoiceFamily], ...] = (
    (re.compile(r"kokoro", re.I), VoiceFamily.KOKORO),
    (re.compile(r"pocket[-_ ]?tts", re.I), VoiceFamily.POCKET),
    (re.compile(r"qwen3[-_ ]?tts", re.I), VoiceFamily.QWEN3),
)
_LANGUAGE_RE = re.compile(r"/(english|french|german|italian|portuguese|spanish)(?:/|$)", re.I)
VOICE_FILE_SUFFIXES = (".safetensors",)  # PocketTTS's embeddings beside the GGUF


def voice_family(*names: str) -> VoiceFamily | None:
    """The family named by a repo id, a pasted folder or a file name (the first that says)."""
    for text in names:
        for pattern, family in _FAMILY_RE:
            if pattern.search(text):
                return family
    return None


def package_language(path: str) -> str | None:
    """PocketTTS's language from its folder (`PocketTTS-GGUF/english/…`)."""
    found = _LANGUAGE_RE.search("/" + path)
    return found.group(1).lower() if found else None


def kit_for(family: VoiceFamily) -> VoiceKit:
    return KITS[family]


def with_voice_files(
    variants: list[GgufVariant], files: dict[str, int], package: str
) -> list[GgufVariant]:
    """PocketTTS's built-in voices (`embeddings/*.safetensors` beside the GGUF) travel with
    every quant of the package: the engine reads them from the GGUF's folder, and the plan
    counts their bytes (English ships 26, more than its Q8_0 GGUF)."""
    prefix = f"{package}/embeddings/" if package else "embeddings/"
    voices = sorted(f for f in files if f.startswith(prefix) and f.endswith(VOICE_FILE_SUFFIXES))
    for variant in variants:
        if voices:
            variant.voice_files = voices
            variant.voice_bytes = sum(files[f] for f in voices)
    return variants


def package_voices(variants: list[GgufVariant], family: VoiceFamily) -> list[str]:
    """The voices a package offers before it is downloaded: its embeddings files' names when it
    ships them (PocketTTS English: 26), else the kit's list, which may name
    fewer (14 for PocketTTS)."""
    files = next((v.voice_files for v in variants if v.voice_files), [])
    return sorted(PurePosixPath(f).stem for f in files) or list(kit_for(family).voices)
