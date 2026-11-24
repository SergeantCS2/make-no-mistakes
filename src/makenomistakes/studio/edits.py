"""The song's edits, one typed ask per edit. Each edit says what it is, how the user asks for it
(its form, and `mnm song edit <run> <edit> key=value`), and how ACE-Step on audio.cpp runs it
(its route and options). A field an edit does not take is refused, never ignored. An edit is a
child run of the song holding a new take; the song then uses that take
(`pipeline/song_edits.py`), and Undo is the take before it.

The model-side edits are the ones ACE-Step plays (here): repaint a range, re-sing a section with
new lyrics, extend, cover or remaster, add an instrument and swap one. The tool-side ones are
the section tools (`studio/edits_sections.py`: cut, crop, duplicate, move, joined by ffmpeg).
The base every edit shares is `studio/edit_base.py`.

ACE-Step's routes on audio.cpp (`models-ace-step-params.json`, read on the CPU image on
2026-10-31): `repaint` sings a range again and keeps the rest; a range past the end makes
the song longer (Extend). `cover` follows the whole take at a strength. `lego` plays one
track class over the take and answers **the track alone** (its answer correlated 0.01 with
the take it was sent), so an Add or a Swap mixes the track over the take or over the other
stems (`pipeline/song_edits.py`). `lego` is not held to its source's length, so it is sent
the take's.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, get_args

from pydantic import (
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from makenomistakes.music_api import OptionValue
from makenomistakes.run_parts import ActsOn
from makenomistakes.studio.edit_base import (
    BaseAsk,
    Edit,
    EditError,
    RangedAsk,
    Span,
    form_hint,
)
from makenomistakes.studio.edits_retime import KeyAsk, TempoAsk
from makenomistakes.studio.edits_sections import CropAsk, CutAsk, DuplicateAsk, MoveAsk

# ACE-Step 1.5's track classes for `lego` (and `extract`, `complete`)
Track = Literal[
    "vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard", "percussion",
    "strings", "synth", "fx", "brass", "woodwinds",
]  # fmt: skip
TRACKS: tuple[str, ...] = get_args(Track)
# the separator's stems a swap can take out, and the track class that plays each again
Swappable = Literal["vocals", "drums", "bass", "guitar", "piano"]
SWAPPABLE: dict[str, Track] = {
    "vocals": "vocals", "drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "keyboard",
}  # fmt: skip
MAX_EXTEND = 120.0  # seconds one Extend adds (ACE-Step's song is 10 minutes at most)
DEFAULT_COVER = 0.5  # a cover's strength when none is said: half held to the take


class Styled(BaseAsk):
    """An edit ACE-Step sings: the style line (the take's own when not given), and the route
    and options it is sent."""

    tags: str | None = Field(
        default=None, min_length=1, max_length=600, title="Style",
        json_schema_extra=form_hint("text"),
    )  # fmt: skip

    def options(self, span: Span | None, length: float) -> dict[str, OptionValue]:
        """ACE-Step's route and options for this edit of a take `length` seconds long."""
        raise NotImplementedError


class SungRange(RangedAsk, Styled):
    """A range ACE-Step sings again (a repaint, a re-sing)."""

    def _repaint(self, span: Span | None, length: float) -> dict[str, OptionValue]:
        start, end = (span.start, span.end) if span is not None else (0.0, length)
        return {"route": "repaint", "repainting_start": round(start, 3),
                "repainting_end": round(end, 3)}  # fmt: skip


class RepaintAsk(SungRange):
    label: ClassVar[str] = "Repaint a range"
    says: ClassVar[str] = (
        "Sing the range again (the rest stays), snapped to whole bars by the structure map; "
        "the style line may change for the range."
    )
    edit: Literal[Edit.REPAINT] = Edit.REPAINT
    strength: float | None = Field(
        default=None, ge=0, le=1, title="Strength", json_schema_extra=form_hint("number", 0.05)
    )

    def words(self, span: Span | None = None) -> str:
        return f"repaint{self._where(span)}"

    def options(self, span: Span | None, length: float) -> dict[str, OptionValue]:
        options = self._repaint(span, length)
        if self.strength is not None:
            options["repaint_strength"] = self.strength
        return options


class ResingAsk(SungRange):
    label: ClassVar[str] = "Re-sing a section's lyrics"
    says: ClassVar[str] = (
        "New lines for one section of the take's lyrics, sung over the range or part you "
        "give (where the section falls: mnm does not place lyrics on the music)."
    )
    edit: Literal[Edit.RESING] = Edit.RESING
    section: int = Field(ge=1, le=16, title="Section", json_schema_extra=form_hint("section"))
    lines: list[str] = Field(
        min_length=1, max_length=16, title="New lines", json_schema_extra=form_hint("lines")
    )

    @field_validator("lines", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        if isinstance(value, str):  # one line each, or ` / ` between them on a command line
            parts = [p.strip() for p in value.replace(" / ", "\n").splitlines()]
            return [p for p in parts if p]
        return value

    @model_validator(mode="after")
    def _a_place(self) -> ResingAsk:
        if self.part is None and self.start is None and self.end is None:
            raise ValueError("say where the section is sung: start= and end=, or part=")
        return self

    def words(self, span: Span | None = None) -> str:
        return f"re-sing section {self.section}{self._where(span)}"

    def options(self, span: Span | None, length: float) -> dict[str, OptionValue]:
        return self._repaint(span, length)

    def acts_on(self, span: Span | None, length: float) -> ActsOn:
        return super().acts_on(span, length).model_copy(update={"section": str(self.section)})


class ExtendAsk(Styled):
    label: ClassVar[str] = "Extend"
    says: ClassVar[str] = "Carry the song on past its end by this many seconds."
    edit: Literal[Edit.EXTEND] = Edit.EXTEND
    seconds: float = Field(
        ge=1, le=MAX_EXTEND, title="Seconds to add", json_schema_extra=form_hint("number", 1, 20)
    )

    def words(self, span: Span | None = None) -> str:
        return f"extend +{self.seconds:g} s"

    def options(self, span: Span | None, length: float) -> dict[str, OptionValue]:
        # a repaint from the end to the new end: audio.cpp's `repaint` takes a range past
        # the source's end and answers the longer song (8-14 s over a 10 s take gave 14.0 s)
        return {"route": "repaint", "repainting_start": round(length, 3),
                "repainting_end": round(length + self.seconds, 3)}  # fmt: skip

    def acts_on(self, span: Span | None, length: float) -> ActsOn:
        return ActsOn(start_s=length, end_s=length + self.seconds)

    def seconds_after(self, length: float) -> float:
        return length + self.seconds


class CoverAsk(Styled):
    label: ClassVar[str] = "Cover or remaster"
    says: ClassVar[str] = (
        "The whole take again, held to it by the strength: near 1 a remaster of the same song, "
        "lower a new performance of it (in the style line given)."
    )
    edit: Literal[Edit.COVER] = Edit.COVER
    strength: float = Field(
        default=DEFAULT_COVER,
        ge=0,
        le=1,
        title="Strength",
        json_schema_extra=form_hint("number", 0.05),
    )

    def words(self, span: Span | None = None) -> str:
        return f"cover at strength {self.strength:g}"

    def options(self, span: Span | None, length: float) -> dict[str, OptionValue]:
        return {"route": "cover", "audio_cover_strength": self.strength}


class LegoAsk(Styled):
    """An edit `lego` plays: one track over the take (Add) or over the other stems (Swap)."""

    track: str

    def words(self, span: Span | None = None) -> str:
        return f"{self.edit} {self.track}"

    def lego_track(self) -> str:
        """The track class `lego` plays."""
        return self.track

    def options(self, span: Span | None, length: float) -> dict[str, OptionValue]:
        # `lego` is not held to its source's length: without one it sang the instance's 30 s
        # over a 10 s take (the CPU image, 2026-10-31)
        return {"route": "lego", "track_name": self.lego_track(), "duration_sec": round(length, 3)}

    def acts_on(self, span: Span | None, length: float) -> ActsOn:
        return ActsOn(stem=self.track)


class AddAsk(LegoAsk):
    label: ClassVar[str] = "Add an instrument"
    says: ClassVar[str] = "A new track played over the take (the style line says how)."
    edit: Literal[Edit.ADD] = Edit.ADD
    track: Track = Field(title="Track", json_schema_extra=form_hint("choice"))


class SwapAsk(LegoAsk):
    label: ClassVar[str] = "Swap an instrument"
    says: ClassVar[str] = (
        "Take one stem out and have that part played again (the style line says how), over "
        "the other stems."
    )
    edit: Literal[Edit.SWAP] = Edit.SWAP
    track: Swappable = Field(title="Stem", json_schema_extra=form_hint("choice"))

    def lego_track(self) -> str:
        return SWAPPABLE[self.track]


EditAsk = Annotated[
    RepaintAsk | ResingAsk | ExtendAsk | CoverAsk | AddAsk | SwapAsk
    | CutAsk | CropAsk | DuplicateAsk | MoveAsk | TempoAsk | KeyAsk,
    Field(discriminator="edit"),
]  # fmt: skip
ASKS: dict[Edit, type[BaseAsk]] = {
    Edit.REPAINT: RepaintAsk, Edit.RESING: ResingAsk, Edit.EXTEND: ExtendAsk,
    Edit.COVER: CoverAsk, Edit.ADD: AddAsk, Edit.SWAP: SwapAsk,
    Edit.CUT: CutAsk, Edit.CROP: CropAsk, Edit.DUPLICATE: DuplicateAsk, Edit.MOVE: MoveAsk,
    Edit.TEMPO: TempoAsk, Edit.KEY: KeyAsk,
}  # fmt: skip
_ASK: TypeAdapter[EditAsk] = TypeAdapter(EditAsk)


def taken(edit: Edit) -> list[str]:
    """The keys an edit takes, as the user types them."""
    return [f.name for f in ASKS[edit].form()]


def ask_of(edit: str, values: dict[str, str]) -> EditAsk:
    """An edit from the CLI's `key=value` words or the web form's fields (empty ones left
    out), as a refusal that names what the edit takes when one does not fit."""
    try:
        chosen = Edit(edit)
    except ValueError:
        raise EditError(f"'{edit}' is not an edit: {', '.join(e.value for e in Edit)}.") from None
    given = {k: v for k, v in values.items() if v.strip() != ""}  # pydantic reads on/off
    try:
        return _ASK.validate_python({**given, "edit": chosen})
    except ValidationError as exc:
        error = exc.errors()[0]
        loc = [str(p) for p in error["loc"]]
        where = ".".join(loc[1:] if loc[:1] == [chosen.value] else loc)
        said = {
            "extra_forbidden": "not one this edit takes",
            "missing": "needed",
        }.get(error["type"], str(error["msg"]).removeprefix("Value error, "))
        raise EditError(
            f"{chosen}: {where + ': ' if where else ''}{said} (it takes "
            f"{', '.join(taken(chosen))})."
        ) from exc
