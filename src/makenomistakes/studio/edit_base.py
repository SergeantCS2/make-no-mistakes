"""What every edit of a song shares: the edits by name, the range one is sung or cut
over, its form's fields, and the base asks (`studio/edits.py` holds ACE-Step's edits,
`studio/edits_sections.py` the section tools). Each edit says which host plays it: ACE-Step on
audio.cpp, or ffmpeg in the house format."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Literal, NamedTuple, cast, get_args

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.config import JsonDict

from makenomistakes.run_parts import ActsOn

FormKind = Literal["number", "part", "after", "section", "choice", "lines", "text", "check"]


class EditError(ValueError):
    """An edit asked with what it cannot take, worded for the user."""


class Edit(StrEnum):
    REPAINT = "repaint"
    RESING = "resing"
    EXTEND = "extend"
    COVER = "cover"
    ADD = "add"
    SWAP = "swap"
    CUT = "cut"  # The section tools
    CROP = "crop"
    DUPLICATE = "duplicate"
    MOVE = "move"
    TEMPO = "tempo"  # Rubber Band through the house format's ffmpeg
    KEY = "key"


class EditHost(StrEnum):
    """What makes an edit's new take."""

    ACE_STEP = "ace_step"  # ACE-Step on audio.cpp (an `edit` stage instance)
    HOUSE = "house"  # ffmpeg in the house format's image (`audio_format`)


class Span(NamedTuple):
    """The range an edit is sung over, in seconds; `barred` when the structure map's
    downbeats moved it out to whole bars."""

    start: float
    end: float
    barred: bool = False


@dataclass(frozen=True)
class FormField:
    """One field of an edit's form, as the Song page draws it."""

    name: str
    label: str
    kind: FormKind
    required: bool
    default: str = ""
    choices: tuple[str, ...] = ()
    low: float | None = None
    high: float | None = None
    step: float | None = None
    blank: str = ""  # a select's empty choice, in words


def form_hint(
    kind: FormKind, step: float | None = None, suggest: float | None = None, blank: str = ""
) -> JsonDict:
    """A field's place on the form: its kind, a number's step, a value offered to start from,
    and what a select's empty choice means."""
    return {"form": kind, "step": step, "suggest": suggest, "blank": blank}


class BaseAsk(BaseModel):
    """What every edit shares: its name, its host, its label and words; a field it does not
    take is refused (`extra="forbid"`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: ClassVar[str]
    says: ClassVar[str]  # what the edit does, for its form and the guide
    host: ClassVar[EditHost] = EditHost.ACE_STEP

    edit: Edit

    def words(self, span: Span | None = None) -> str:
        """The edit in a few words (`repaint 12.0-19.5 s`, `cut part 2`); with `span` the
        range it acts on."""
        raise NotImplementedError

    def acts_on(self, span: Span | None, length: float) -> ActsOn:
        """What the edit changes, for the edit's record and the provenance."""
        return ActsOn()

    def seconds_after(self, length: float) -> float:
        """The take's length once edited."""
        return length

    @classmethod
    def form(cls) -> list[FormField]:
        """The edit's form, from its fields: a label, a kind, the bounds and the default."""
        found = []
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra
            if name == "edit" or not isinstance(extra, dict):
                continue
            kind = cast(FormKind, extra["form"])
            low = next((m.ge for m in field.metadata if isinstance(m, Ge)), None)
            high = next((m.le for m in field.metadata if isinstance(m, Le)), None)
            default = None if field.is_required() else field.default
            if default is None:
                default = extra.get("suggest")
            step = extra.get("step")
            blank = extra.get("blank")
            found.append(
                FormField(
                    name=name,
                    label=field.title or name,
                    kind=kind,
                    required=field.is_required(),
                    default=_said(default),
                    choices=tuple(get_args(field.annotation)) if kind == "choice" else (),
                    low=float(low) if isinstance(low, int | float) else None,
                    high=float(high) if isinstance(high, int | float) else None,
                    step=float(step) if isinstance(step, int | float) else None,
                    blank=blank if isinstance(blank, str) else "",
                )
            )
        return found


def _said(value: object) -> str:
    """A default as the form shows it: `on` for a ticked box, `20` for 20.0, `` for none."""
    if value is None:
        return ""
    if value is True:
        return "on"
    return f"{value:g}" if isinstance(value, float) else str(value)


class RangedAsk(BaseAsk):
    """An edit of a range of the take: start and end in seconds, or a part of the structure
    map, snapped to the map's downbeats unless `snap=off`: out to whole bars for what is sung
    again (`snap_out`), to the nearest bar line for a section tool."""

    snap_out: ClassVar[bool] = True
    snap_words: ClassVar[str] = "out to whole bars"

    start: float | None = Field(
        default=None, ge=0, le=600, title="Start (s)", json_schema_extra=form_hint("number", 0.1)
    )
    end: float | None = Field(
        default=None, ge=0, le=600, title="End (s)", json_schema_extra=form_hint("number", 0.1)
    )
    part: int | None = Field(
        default=None, ge=1, le=64, title="Or a part", json_schema_extra=form_hint("part")
    )
    snap: bool = Field(
        default=True, title="Snap the range out to whole bars", json_schema_extra=form_hint("check")
    )

    @model_validator(mode="after")
    def _a_range(self) -> RangedAsk:
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError(f"the range ends ({self.end:g} s) before it starts ({self.start:g} s)")
        if self.part is not None and (self.start is not None or self.end is not None):
            raise ValueError("a part or a range (start, end), not both")
        return self

    def _where(self, span: Span | None) -> str:
        if span is not None:
            part = f" part {self.part}," if self.part is not None else ""
            bars = f" ({self.snap_words})" if span.barred else ""
            return f"{part} {span.start:.1f}-{span.end:.1f} s{bars}"
        if self.part is not None:
            return f" part {self.part}"
        if self.start is not None or self.end is not None:
            start = self.start or 0.0
            return f" {start:g}-{self.end:g} s" if self.end else f" from {start:g} s"
        return ""

    def acts_on(self, span: Span | None, length: float) -> ActsOn:
        start, end = (span.start, span.end) if span is not None else (0.0, length)
        return ActsOn(start_s=start, end_s=end)
