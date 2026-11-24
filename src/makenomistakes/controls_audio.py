"""whisper.cpp's (speech-to-text) and audio.cpp's (voice) settings."""

from __future__ import annotations

from makenomistakes.controls_catalog import (
    AUDIO_MODALITIES,
    VOICE_MODALITIES,
    Control,
    Kind,
    check_controls,
)
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.engine_args_whisper import WhisperArgs

AUDIO_CONTROLS: tuple[Control, ...] = (
    Control(
        key="language",
        field="language",
        label="Spoken language",
        help="ISO code (en, de, fr, ja …) or `auto` to detect per request. English-only "
        "models (`.en`) ignore it.",
        kind=Kind.TEXT,
        recommended="auto, or your language for faster, steadier results",
        modalities=AUDIO_MODALITIES,
    ),
    Control(
        key="threads",
        field="threads",
        label="CPU threads",
        help="For the parts whisper runs on the CPU. Empty uses the engine default.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=AUDIO_MODALITIES,
    ),
    Control(
        key="convert",
        field="convert",
        label="Convert uploads with ffmpeg",
        help="Accept webm/mp3/ogg uploads (browsers and Odysseus send webm); off accepts WAV only.",
        kind=Kind.BOOL,
        recommended="on",
        modalities=AUDIO_MODALITIES,
    ),
)


VOICE_CONTROLS: tuple[Control, ...] = (
    Control(
        key="voice",
        field="voice",
        label="Default voice",
        help="The voice used when a request names none (Odysseus sends its own `voice` "
        "field: name one of this model's voices there, or set it here). `mnm show` lists "
        "the voices the model has.",
        kind=Kind.TEXT,
        recommended="the family's default (af_heart for Kokoro, alba for PocketTTS, Vivian for "
        "Qwen3-TTS)",
        modalities=VOICE_MODALITIES,
    ),
    Control(
        key="speed",
        field="speed",
        label="Speaking rate",
        help="1.0 is the model's natural pace; a request's `speed` overrides it.",
        kind=Kind.FLOAT,
        recommended="1.0",
        modalities=VOICE_MODALITIES,
    ),
    Control(
        key="seed",
        field="seed",
        label="Seed",
        help="The same seed and text give the same reading; empty means the engine's choice.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=VOICE_MODALITIES,
    ),
    Control(
        key="threads",
        field="threads",
        label="CPU threads",
        help="For the parts the engine runs on the CPU. Empty uses the engine default.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=VOICE_MODALITIES,
    ),
)


check_controls(
    [*((c, WhisperArgs) for c in AUDIO_CONTROLS), *((c, AudioCppArgs) for c in VOICE_CONTROLS)]
)
