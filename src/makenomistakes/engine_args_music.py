"""The studio's typed settings on audio.cpp (recipe schema 24): the music families as
the server's config spells them, the task each runs, and the per-family options a music
instance carries (`AudioCppArgs.options`, a union discriminated on `kind`).

Read from audio.cpp's own pages on 2026-10-31 (`docs/music_generation.md`,
`docs/models/ace_step.md`, `docs/audio_tools.md`, `app/server/api.md`) and checked against
the `full-cpu` image the same day: music generation is task `gen` with ACE-Step's operation
as the request's `task_route`, separation is task `sep` (stems come back as
`named_audio_outputs`), conversion `vc`, audio to a score `midi`. Like `engine_args`, it
imports nothing from `fit`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# audio.cpp's `family` names for the studio's models. `minimax_music3` and `htdemucs_6stems`
# are the engine's spellings (the plan wrote `minimax_music`, and `htdemucs` for both).
MusicFamilyName = Literal[
    "ace_step",
    "heartmula",
    "yue2",
    "minimax_music3",
    "stable_audio",
    "htdemucs",
    "htdemucs_6stems",
    "bs_roformer",
    "mel_band_roformer",
    "rvc",
    "seed_vc",
    "vevo2",
    "muscriptor",
    "sheetsage2",
]
# The server's `models[].task` per kind of work (`docs/usage.md`'s `--task` list)
MusicTask = Literal["gen", "sep", "vc", "svc", "midi"]
AceRoute = Literal["text2music", "complete", "lego", "extract", "cover", "cover-nofsq", "repaint"]
SPEECH_FAMILIES = frozenset({"kokoro_tts", "pocket_tts", "qwen3_tts"})  # text-to-speech
# The options each music family takes (none for the others: HeartMuLa, YuE2 … run on defaults)
OPTIONS_KIND: dict[str, str] = {
    "ace_step": "ace_step",
    **dict.fromkeys(
        ("htdemucs", "htdemucs_6stems", "bs_roformer", "mel_band_roformer"), "separator"
    ),
    "rvc": "convert",
}
MAX_SONG_SECONDS = 600.0  # ACE-Step's own range is 10 s to 10 min


STYLES_FOLDER = "styles"  # The style library's files, under the models folder
LIBRARY_NAME = r"^[a-z0-9][a-z0-9._-]{0,63}$"  # a voice's or a style's name in its library
STYLE_ADAPTER = "adapter_model.safetensors"  # a style's PEFT weights in its folder


class AceStepOptions(BaseModel):
    """ACE-Step 1.5 (`docs/models/ace_step.md`): what a request asks for unless it says
    otherwise, and the one session option that moves what it holds (`mem_saver`). The GGUF
    package carries the 1.7B planner LM inside it; a 0.6B planner is not packaged."""

    kind: Literal["ace_step"] = "ace_step"
    route: AceRoute = "text2music"
    duration: float = Field(default=30.0, ge=1.0, le=MAX_SONG_SECONDS)  # `duration_sec`
    steps: int | None = Field(default=None, ge=1, le=200)  # turbo caps at 8 itself
    guidance: float | None = Field(default=None, ge=0.0, le=30.0)  # ignored by turbo
    mem_saver: bool = False  # `ace_step.mem_saver`: releases staged graphs between phases
    # A style from the library (`<models>/styles/<name>/`), merged at load as
    # `ace_step.lora` at `ace_step.lora_scale` (times the adapter's alpha over its rank)
    style: str | None = Field(default=None, pattern=LIBRARY_NAME)
    style_scale: float = Field(default=1.0, ge=0.0, le=2.0)

    @property
    def summary(self) -> str:
        steps = f", {self.steps} steps" if self.steps is not None else ""
        saver = ", memory saver" if self.mem_saver else ""
        style = f", style {self.style} at {self.style_scale:g}" if self.style else ""
        return f"{self.route}, {self.duration:g} s{steps}{saver}{style}"


class SeparatorOptions(BaseModel):
    """A separator (the RoFormers, HTDemucs): `<family>.num_overlap` passes over the mix
    (the package's own count when unset: 4 for BS-RoFormer ep368); fewer is faster."""

    kind: Literal["separator"] = "separator"
    num_overlap: int | None = Field(default=None, ge=1, le=8)

    @property
    def summary(self) -> str:
        return f"{self.num_overlap} overlaps" if self.num_overlap else "the package's overlaps"


class ConvertOptions(BaseModel):
    """A voice converter (RVC's request options, `docs/audio_tools.md`)."""

    kind: Literal["convert"] = "convert"
    semitone_shift: int = Field(default=0, ge=-24, le=24)
    retrieval_blend: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def summary(self) -> str:
        return f"{self.semitone_shift:+d} semitones, retrieval {self.retrieval_blend:g}"


MusicOptions = AceStepOptions | SeparatorOptions | ConvertOptions
