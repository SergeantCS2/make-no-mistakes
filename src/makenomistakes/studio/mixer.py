"""The stem mixer: per stem a level, mute, solo and fades; Render sums the audible stems through
ffmpeg (`volume`, `afade`, `amix`) in the house format into a new mix, kept as a child run of
the song (role `mix`), which the export then uses. The song's own mix (the stems summed as they
are) stays; picking it again is undo.

Solo works as on a desk: when any stem is soloed only soloed stems sound (a soloed stem's
mute is ignored); otherwise every stem not muted sounds.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from makenomistakes.audio_format import HOUSE_RATE, Codec, HouseFormatError

MIN_DB, MAX_DB = -60.0, 12.0
MAX_FADE = 60.0
SETTING = "levels"  # a mix run's setting: its levels as JSON
_STEM = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class MixError(ValueError):
    """A mix that cannot be rendered, worded for the user."""


class StemLevel(BaseModel):
    stem: str
    gain_db: float = Field(default=0.0, ge=MIN_DB, le=MAX_DB)
    mute: bool = False
    solo: bool = False
    fade_in: float = Field(default=0.0, ge=0, le=MAX_FADE)
    fade_out: float = Field(default=0.0, ge=0, le=MAX_FADE)

    @field_validator("stem")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _STEM.match(value):
            raise ValueError(f"'{value}' is not a stem's name")
        return value

    @property
    def file(self) -> str:
        return f"stem-{self.stem}.wav"

    @property
    def gain(self) -> float:
        """Linear, as REAPER's track volume takes it."""
        return 10 ** (self.gain_db / 20)

    def words(self) -> str:
        said = [f"{self.gain_db:+g} dB"] if self.gain_db else []
        said += ["muted"] if self.mute else []
        said += ["solo"] if self.solo else []
        said += [f"fade in {self.fade_in:g} s"] if self.fade_in else []
        said += [f"fade out {self.fade_out:g} s"] if self.fade_out else []
        return f"{self.stem} " + (", ".join(said) or "as made")


class MixSettings(BaseModel):
    levels: list[StemLevel] = Field(min_length=1)

    @field_validator("levels")
    @classmethod
    def _once(cls, value: list[StemLevel]) -> list[StemLevel]:
        names = [level.stem for level in value]
        if len(names) != len(set(names)):
            raise ValueError("a stem is named twice")
        return value

    def audible(self) -> list[StemLevel]:
        soloed = [level for level in self.levels if level.solo]
        return soloed or [level for level in self.levels if not level.mute]

    def sounds(self, stem: str) -> bool:
        return any(level.stem == stem for level in self.audible())

    def level(self, stem: str) -> StemLevel:
        return next((lv for lv in self.levels if lv.stem == stem), StemLevel(stem=stem))

    def words(self) -> str:
        changed = [lv.words() for lv in self.levels if lv.words() != f"{lv.stem} as made"]
        return "; ".join(changed) or "every stem as made"

    def setting(self) -> str:
        return json.dumps(self.model_dump(mode="json")["levels"], separators=(",", ":"))

    @classmethod
    def from_setting(cls, text: str) -> MixSettings:
        try:
            return cls.model_validate({"levels": json.loads(text)})
        except (json.JSONDecodeError, ValidationError) as exc:
            raise MixError(f"the mix's levels cannot be read ({exc})") from None

    @classmethod
    def flat(cls, stems: list[str]) -> MixSettings:
        """Every stem as made: the song's own mix."""
        return cls(levels=[StemLevel(stem=s) for s in stems])


def _chain(level: StemLevel, length: float) -> str:
    """A stem's filters: its level, then its fades (a fade out ends at the stem's end)."""
    filters = [f"volume={level.gain_db:g}dB"]
    if level.fade_in:
        filters.append(f"afade=t=in:st=0:d={level.fade_in:g}")
    if level.fade_out and length > 0:
        start = max(0.0, length - level.fade_out)
        filters.append(f"afade=t=out:st={start:g}:d={level.fade_out:g}")
    return ",".join(filters)


def render_args(
    settings: MixSettings,
    stems: dict[str, float | None],
    target: str,
    codec: Codec = Codec.WAV24,
) -> list[str]:
    """ffmpeg's arguments for the mix: each audible stem through its chain, summed with no
    normalising (a flat mix equals the song's own), at the house rate in `codec` (float for a
    sum checked for clipping after). `stems`: the song's stem files and their lengths (None
    where the header cannot say; only a fade out needs it)."""
    audible = settings.audible()
    if not audible:
        raise MixError("Every stem is muted: nothing would sound. Unmute one, or solo one.")
    lengths: dict[str, float] = {}
    for level in audible:
        if level.file not in stems:
            raise MixError(f"This song has no {level.stem} stem.")
        length = stems[level.file]
        if level.fade_out and length is None:
            raise MixError(f"The {level.stem} stem's length cannot be read: it cannot fade out.")
        lengths[level.file] = length or 0.0
    args: list[str] = []
    graph: list[str] = []
    for i, level in enumerate(audible):
        args += ["-i", level.file]
        graph.append(f"[{i}:a]{_chain(level, lengths[level.file])}[s{i}]")
    inputs = "".join(f"[s{i}]" for i in range(len(audible)))
    graph.append(
        f"{inputs}amix=inputs={len(audible)}:normalize=0:duration=longest:dropout_transition=0"
    )
    return [
        *args, "-filter_complex", ";".join(graph),
        "-ar", str(HOUSE_RATE), "-ac", "2", *codec.argv(), target,
    ]  # fmt: skip


__all__ = ["HouseFormatError", "MixError", "MixSettings", "StemLevel", "render_args"]
