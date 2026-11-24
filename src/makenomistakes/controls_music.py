"""The studio's settings on audio.cpp: the per-family options a music instance carries
(`AudioCppArgs.options`, `on_options` controls) and the request defaults beside them. Each
applies at the next start: the config is rendered from the recipe then."""

from __future__ import annotations

from typing import get_args

from makenomistakes.controls_catalog import Control, Kind, check_controls
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.engine_args_music import (
    AceRoute,
    AceStepOptions,
    ConvertOptions,
    SeparatorOptions,
)
from makenomistakes.hub.facts import MUSIC_MODALITIES, Modality

SONG = frozenset({Modality.SONG})
STEMS = frozenset({Modality.STEMS})
CONVERT = frozenset({Modality.VOICE_CONVERT})

SONG_CONTROLS: tuple[Control, ...] = (
    Control(
        key="duration",
        field="duration",
        label="Song length (seconds)",
        help="What a request gets when it names no length (`duration_sec`); `mnm try` asks "
        "for 10 s whatever this says. ACE-Step makes 10 s to 10 min.",
        kind=Kind.FLOAT,
        recommended="30 to explore, the whole song for a keeper",
        modalities=SONG,
        on_options=True,
    ),
    Control(
        key="route",
        field="route",
        label="Operation",
        help="ACE-Step's default operation: text2music makes a song; the others edit a "
        "source clip a request sends (repaint, cover, extract, lego, complete).",
        kind=Kind.CHOICE,
        choices=get_args(AceRoute),
        recommended="text2music",
        modalities=SONG,
        on_options=True,
    ),
    Control(
        key="steps",
        field="steps",
        label="Diffusion steps",
        help="Empty is the engine's 8; turbo builds cap it at 8, base builds take more.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=SONG,
        on_options=True,
    ),
    Control(
        key="guidance",
        field="guidance",
        label="Guidance",
        help="Diffusion guidance; empty is the engine's 1.0. Turbo builds ignore it.",
        kind=Kind.FLOAT,
        nullable=True,
        recommended="empty",
        modalities=SONG,
        on_options=True,
    ),
    Control(
        key="mem-saver",
        field="mem_saver",
        label="Memory saver",
        help="Releases the engine's staged buffers between a song's phases: less held on the "
        "card, a little slower (`ace_step.mem_saver`).",
        kind=Kind.BOOL,
        recommended="off, unless the card is tight",
        modalities=SONG,
        on_options=True,
    ),
    Control(
        key="style",
        field="style",
        label="Style",
        help="A style from the library (`mnm studio styles ls`), trained against this "
        "instance's build (base or turbo) and merged into the model when it loads; a song's "
        "style line carries its trigger word. Empty is none. Applies at the next start.",
        kind=Kind.TEXT,
        nullable=True,
        recommended="empty, or a style you trained",
        modalities=SONG,
        on_options=True,
    ),
    Control(
        key="style-scale",
        field="style_scale",
        label="Style strength",
        help="How strongly the style is merged (`ace_step.lora_scale`, times its alpha over its "
        "rank): 1 as trained, 0.7 to 0.9 when it overwhelms the prompt, 0 none.",
        kind=Kind.FLOAT,
        recommended="1.0",
        modalities=SONG,
        on_options=True,
    ),
)
STEMS_CONTROLS: tuple[Control, ...] = (
    Control(
        key="overlap",
        field="num_overlap",
        label="Overlapping passes",
        help="Passes over the mix the separator blends; empty is the package's own (4 for "
        "BS-RoFormer). Fewer is faster and can blur the joins.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=STEMS,
        on_options=True,
    ),
)
CONVERT_CONTROLS: tuple[Control, ...] = (
    Control(
        key="semitones",
        field="semitone_shift",
        label="Pitch shift (semitones)",
        help="Shifts the converted voice; 12 is an octave up.",
        kind=Kind.INT,
        recommended="0, or ±12 across voices of another range",
        modalities=CONVERT,
        on_options=True,
    ),
    Control(
        key="retrieval",
        field="retrieval_blend",
        label="Retrieval blend",
        help="How much of the voice's own index is mixed in (0 to 1); 0 turns it off.",
        kind=Kind.FLOAT,
        recommended="0, or 0.5 with a voice that ships an index",
        modalities=CONVERT,
        on_options=True,
    ),
)
MUSIC_ARGS_CONTROLS: tuple[Control, ...] = (
    Control(
        key="seed",
        field="seed",
        label="Seed",
        help="The same seed and words give the same song; empty is a new one each time.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty",
        modalities=SONG,
    ),
    Control(
        key="threads",
        field="threads",
        label="CPU threads",
        help=(
            "The cores the engine works on: on the CPU image (`full-cpu`) the whole task, where "
            "empty is one core and the plan sets the physical cores; on a card only the parts "
            "it runs on the CPU, where empty is the engine's default."
        ),
        kind=Kind.INT,
        nullable=True,
        recommended="the plan's: the physical cores on the CPU image, empty on a card",
        modalities=MUSIC_MODALITIES,
    ),
)
MUSIC_CONTROLS = (*SONG_CONTROLS, *STEMS_CONTROLS, *CONVERT_CONTROLS, *MUSIC_ARGS_CONTROLS)

check_controls(
    [
        *((c, AceStepOptions) for c in SONG_CONTROLS),
        *((c, SeparatorOptions) for c in STEMS_CONTROLS),
        *((c, ConvertOptions) for c in CONVERT_CONTROLS),
        *((c, AudioCppArgs) for c in MUSIC_ARGS_CONTROLS),
    ]
)
