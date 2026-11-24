"""Tempo and key: the whole take faster or slower with its key kept, or in another key with its
tempo kept, by Rubber Band through the house format's ffmpeg (Ubuntu's ffmpeg carries its
`rubberband` filter, and so the engines' images do; `python-stretch`, the Signalsmith binding,
has no arm64 build). A key change shifts the take's own stems but the drums when they are made,
and the mix when they are not; the vocals keep their formants.

A stretch past 1.5x either way, or a shift of more than four semitones, smears what it
moves: the edit is made, and its note offers a Cover, which sings the take anew at the tempo
or key its tags name.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from makenomistakes.studio import keys
from makenomistakes.studio.edit_base import BaseAsk, Edit, EditError, EditHost, Span, form_hint
from makenomistakes.studio.tools_answers import StructureMap

FAR_RATIO = 1.5  # a stretch past this either way is offered as a Cover
FAR_SEMITONES = 4  # likewise a shift past this many semitones
DRUMS = "stem-drums.wav"  # the stem a key change leaves where it is


class RetimeAsk(BaseAsk):
    """An edit of the whole take that ffmpeg's Rubber Band makes."""

    host: ClassVar[EditHost] = EditHost.HOUSE

    def far(self, structure: StructureMap | None) -> bool:
        """Whether it moves the take far enough to offer a Cover instead."""
        raise NotImplementedError

    def note(self, structure: StructureMap | None, run_id: str) -> str:
        """What the edit's line adds: a Cover offered when it went far."""
        if not self.far(structure):
            return ""
        return (
            f"; this far, Rubber Band smears the attacks: a Cover (`mnm song edit {run_id} "
            f'cover tags="…, {self.aim(structure)}"`) sings the take anew there'
        )

    def aim(self, structure: StructureMap | None) -> str:
        """Where the edit takes the take, as a Cover's tags would name it."""
        raise NotImplementedError


class TempoAsk(RetimeAsk):
    label: ClassVar[str] = "Change the tempo"
    says: ClassVar[str] = (
        "Faster or slower, the key kept (Rubber Band); the structure map is made again."
    )
    edit: Literal[Edit.TEMPO] = Edit.TEMPO
    bpm: float | None = Field(
        default=None, ge=30, le=300, title="To BPM", json_schema_extra=form_hint("number", 0.1)
    )
    percent: float | None = Field(
        default=None, ge=50, le=200, title="Or percent of the tempo now",
        json_schema_extra=form_hint("number", 1),
    )  # fmt: skip

    @model_validator(mode="after")
    def _one(self) -> TempoAsk:
        if (self.bpm is None) == (self.percent is None):
            raise ValueError("say how fast: bpm= (the tempo wanted) or percent= (of the tempo now)")
        if self.percent == 100:
            raise ValueError("percent=100 is the tempo it has")
        return self

    def ratio(self, structure: StructureMap | None) -> float:
        """How much faster (above 1) or slower the take plays."""
        if self.percent is not None:
            return self.percent / 100
        assert self.bpm is not None  # `_one`
        if structure is None:
            raise EditError(
                "bpm= is read against the take's tempo, which its structure map gives, and it "
                "has none: give percent= instead (`mnm studio install`, then `mnm song redo "
                "<run> structure`, maps it)"
            )
        ratio = self.bpm / structure.tempo
        if not 0.5 <= ratio <= 2.0:
            raise EditError(
                f"The take is {structure.tempo:.0f} BPM: {self.bpm:g} is past half or double it"
            )
        if abs(ratio - 1) < 0.002:
            raise EditError(f"The take is {structure.tempo:.1f} BPM already")
        return ratio

    def far(self, structure: StructureMap | None) -> bool:
        ratio = self.ratio(structure)
        return max(ratio, 1 / ratio) > FAR_RATIO

    def aim(self, structure: StructureMap | None) -> str:
        if self.bpm is not None:
            return f"{self.bpm:g} bpm"
        if structure is not None:
            return f"{structure.tempo * self.ratio(structure):.0f} bpm"
        return f"{self.percent:g}% of its tempo"

    def words(self, span: Span | None = None) -> str:
        return f"tempo to {self.bpm:g} BPM" if self.bpm is not None else f"tempo {self.percent:g}%"

    def filters(self, ratio: float) -> list[str]:
        """ffmpeg's chain: Rubber Band's tempo with crisp attacks, the channels together (the
        stereo image kept)."""
        return [f"rubberband=tempo={ratio:.6f}:transients=crisp:channels=together"]


class KeyAsk(RetimeAsk):
    label: ClassVar[str] = "Change the key"
    says: ClassVar[str] = (
        "Up or down by semitones, or to a key; the tempo kept. The stems but the drums are "
        "shifted when the take's are made, the vocals keeping their formants."
    )
    edit: Literal[Edit.KEY] = Edit.KEY
    semitones: int | None = Field(
        default=None, ge=-12, le=12, title="Semitones (+ up, - down)",
        json_schema_extra=form_hint("number", 1),
    )  # fmt: skip
    to: str | None = Field(
        default=None, max_length=24, title="Or to a key (A minor)",
        json_schema_extra=form_hint("text"),
    )  # fmt: skip

    @field_validator("to")
    @classmethod
    def _readable(cls, value: str | None) -> str | None:
        if value is not None and value.strip():
            keys.parse(value)  # a KeyNameError is a ValueError: refused as the field's
            return value.strip()
        return None

    @model_validator(mode="after")
    def _one(self) -> KeyAsk:
        if (self.semitones is None) == (self.to is None):
            raise ValueError("say where to: semitones= (up or down) or to= (a key, A minor)")
        if self.semitones == 0:
            raise ValueError("semitones=0 is the key it has")
        return self

    def shift(self, structure: StructureMap | None) -> int:
        """The semitones the take moves (`to=` read against the map's key)."""
        if self.semitones is not None:
            return self.semitones
        assert self.to is not None  # `_one`
        if structure is None or structure.key is None:
            why = "has none" if structure is None else "was made before mnm read keys"
            raise EditError(
                f"to= is read against the take's key, which its structure map gives, and it "
                f"{why}: give semitones= instead (`mnm song redo <run> structure` maps it)"
            )
        try:
            step = keys.shift_to(keys.parse(structure.key), keys.parse(self.to))
        except keys.KeyNameError as exc:
            raise EditError(str(exc)) from exc
        if step == 0:
            raise EditError(f"The take is in {structure.key} already")
        return step

    def far(self, structure: StructureMap | None) -> bool:
        return abs(self.shift(structure)) > FAR_SEMITONES

    def note(self, structure: StructureMap | None, run_id: str) -> str:
        """A `to=` says the key it moved from and how the map knew it (a key heard unsurely
        can move the take from the wrong one)."""
        said = super().note(structure, run_id)
        if self.to is None or structure is None or structure.key is None:
            return said
        parts = ["kept from the take before"] if structure.key_carried else []
        if structure.unsure:
            heard = "first heard" if structure.key_carried else "heard"
            parts.append(f"{heard} unsurely: check it by ear, or give semitones=")
        how = "".join(f", {' and '.join(parts)}" if parts else "")
        return f"; moved from {structure.key}, the structure map's key{how}{said}"

    def aim(self, structure: StructureMap | None) -> str:
        if self.to is not None:
            return keys.parse(self.to).words()
        if structure is not None and structure.key is not None:
            return keys.parse(structure.key).shifted(self.shift(structure)).words()
        return f"{self.semitones:+d} semitones"

    def words(self, span: Span | None = None) -> str:
        if self.to is not None:
            return f"key to {keys.parse(self.to).words()}"
        return f"key {self.semitones:+d} semitone{'s' if abs(self.semitones or 0) != 1 else ''}"

    @staticmethod
    def filters(semitones: int, *, vocals: bool) -> list[str]:
        """ffmpeg's chain for one file: Rubber Band's pitch, the best quality, the formants
        kept for a voice and crisp attacks for the rest."""
        ratio = 2 ** (semitones / 12)
        voice = "formant=preserved:transients=smooth" if vocals else "transients=crisp"
        return [f"rubberband=pitch={ratio:.6f}:pitchq=quality:channels=together:{voice}"]
