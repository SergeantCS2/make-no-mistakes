"""The section tools: cut, crop, duplicate and move a part of the take, each a new take whose
pieces ffmpeg joins with a short equal-power crossfade (`audio_format.Splice`). A part is one of
the structure map's, or a range in seconds snapped to the nearest bar line (never out to whole
bars, as a repaint is: a cut keeps the music on the beat); where a part goes is "after part N"
(0 is the start) or a time in seconds, snapped the same way.

Nothing is sung again, so no model plays these; the lyrics stay the take's (a cut verse's
words are still in its lyrics: the user edits them). Inserting new bars is a cut's mirror
that needs ACE-Step to fill the gap: a repaint over a duplicated part does that today.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from makenomistakes.audio_format import DEFAULT_FADE
from makenomistakes.run_parts import ActsOn
from makenomistakes.studio import sections
from makenomistakes.studio.edit_base import Edit, EditHost, RangedAsk, Span, form_hint


class SectionAsk(RangedAsk):
    """A section edit: a part or a range, the crossfade at each joint, ffmpeg the host."""

    host: ClassVar[EditHost] = EditHost.HOUSE
    snap_out: ClassVar[bool] = False
    snap_words: ClassVar[str] = "on the bar lines"

    snap: bool = Field(
        default=True, title="Snap to the nearest bar lines", json_schema_extra=form_hint("check")
    )
    fade: float = Field(
        default=DEFAULT_FADE, ge=0, le=2, title="Crossfade (s)",
        json_schema_extra=form_hint("number", 0.01),
    )  # fmt: skip

    @model_validator(mode="after")
    def _a_part(self) -> SectionAsk:
        if self.part is None and (self.start is None or self.end is None):
            raise ValueError("say which part: part=, or start= and end=")
        return self

    def pieces(self, span: Span, length: float, at: float | None) -> list[tuple[float, float]]:
        """The take's pieces in their new order (`at`: where a moved or copied part goes)."""
        raise NotImplementedError

    def acts_on(self, span: Span | None, length: float) -> ActsOn:
        start, end = (span.start, span.end) if span is not None else (0.0, length)
        part = str(self.part) if self.part is not None else None
        return ActsOn(start_s=start, end_s=end, section=part)


class CutAsk(SectionAsk):
    label: ClassVar[str] = "Cut a part"
    says: ClassVar[str] = "Take a part out; the music either side of it is joined on the beat."
    edit: Literal[Edit.CUT] = Edit.CUT

    def words(self, span: Span | None = None) -> str:
        return f"cut{self._where(span)}"

    def pieces(self, span: Span, length: float, at: float | None) -> list[tuple[float, float]]:
        return sections.cut(length, sections.Range(span.start, span.end))


class CropAsk(SectionAsk):
    label: ClassVar[str] = "Crop to a part"
    says: ClassVar[str] = "Keep only a part or a range (a shorter take: an intro or outro gone)."
    edit: Literal[Edit.CROP] = Edit.CROP

    def words(self, span: Span | None = None) -> str:
        return f"crop to{self._where(span)}"

    def pieces(self, span: Span, length: float, at: float | None) -> list[tuple[float, float]]:
        return sections.crop(length, sections.Range(span.start, span.end))


class PlacedAsk(SectionAsk):
    """A section edit that puts a part somewhere: after part N (0: the start), or at a time."""

    after: int | None = Field(
        default=None, ge=0, le=64, title="After part (0: the start)",
        json_schema_extra=form_hint("after", blank="(at the time below)"),
    )  # fmt: skip
    at: float | None = Field(
        default=None, ge=0, le=600, title="Or at (s)", json_schema_extra=form_hint("number", 0.1)
    )

    @model_validator(mode="after")
    def _one_place(self) -> PlacedAsk:
        if self.after is not None and self.at is not None:
            raise ValueError("after= (a part) or at= (seconds), not both")
        return self

    def _to(self) -> str:
        if self.after == 0:
            return " to the start"
        if self.after is not None:
            return f" after part {self.after}"
        return f" at {self.at:g} s" if self.at is not None else ""


class DuplicateAsk(PlacedAsk):
    label: ClassVar[str] = "Duplicate a part"
    says: ClassVar[str] = (
        "Play a part once more: right after itself, or after another part (a chorus twice)."
    )
    edit: Literal[Edit.DUPLICATE] = Edit.DUPLICATE
    after: int | None = Field(
        default=None, ge=0, le=64, title="After part (0: the start)",
        json_schema_extra=form_hint("after", blank="(right after itself)"),
    )  # fmt: skip

    def words(self, span: Span | None = None) -> str:
        return f"duplicate{self._where(span)}{self._to()}"

    def pieces(self, span: Span, length: float, at: float | None) -> list[tuple[float, float]]:
        where = span.end if at is None else at
        return sections.duplicate(length, sections.Range(span.start, span.end), where)


class MoveAsk(PlacedAsk):
    label: ClassVar[str] = "Move a part"
    says: ClassVar[str] = "Take a part out and play it after another part instead."
    edit: Literal[Edit.MOVE] = Edit.MOVE

    @model_validator(mode="after")
    def _a_place(self) -> MoveAsk:
        if self.after is None and self.at is None:
            raise ValueError("say where it goes: after= (a part, 0 for the start) or at=")
        return self

    def words(self, span: Span | None = None) -> str:
        return f"move{self._where(span)}{self._to()}"

    def pieces(self, span: Span, length: float, at: float | None) -> list[tuple[float, float]]:
        assert at is not None  # `_a_place`: a move always names where
        return sections.move(length, sections.Range(span.start, span.end), at)
