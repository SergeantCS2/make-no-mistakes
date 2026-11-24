"""What the studio's tools answer, typed at the edge: `mnm-tools` inside the
`mnm/studio-tools` image prints one JSON line per job (`studio/docker/mnm_tools.py`), and
nothing past this module sees it as text.

The score's one number is decided here, never in the image: Audiobox Aesthetics measures four
axes (content enjoyment CE, content usefulness CU, production complexity PC, production
quality PQ; about 1 to 10), and a candidate's score is the mean of CE and PQ, the two that ask
"is this good" (PC says how busy a mix is, which is no merit by itself). Every axis is kept in
the reading beside it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from makenomistakes.studio.built_image import ImageError, ImageMissingError, json_line, tool_error

SCORE_AXES = ("CE", "PQ")  # the mean of these is a candidate's score
AXIS_WORDS = {
    "CE": "enjoyment",
    "CU": "usefulness",
    "PC": "complexity",
    "PQ": "production quality",
}


class ToolsError(ImageError):
    """A studio tool that could not answer, worded for the user."""


class ToolsMissingError(ToolsError, ImageMissingError):
    """The tools image is not on this box (`mnm studio install` builds it)."""


class ScoreAnswer(BaseModel):
    tool: Literal["score"]
    scorer: str = Field(min_length=1, max_length=200)
    axes: dict[str, float]
    seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _axes(self) -> ScoreAnswer:
        missing = [axis for axis in SCORE_AXES if axis not in self.axes]
        if missing:
            raise ValueError(f"the scorer gave no {', '.join(missing)}")
        return self

    @property
    def score(self) -> float:
        return round(sum(self.axes[a] for a in SCORE_AXES) / len(SCORE_AXES), 2)

    def words(self) -> str:
        """`6.84 (enjoyment 6.2, production quality 7.5)`."""
        parts = ", ".join(f"{AXIS_WORDS.get(a, a)} {self.axes[a]:.1f}" for a in SCORE_AXES)
        return f"{self.score:.2f} ({parts})"


class Part(BaseModel):
    """One part of the song between two boundaries; a repeated part shares its letter."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    label: str = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _order(self) -> Part:
        if self.end <= self.start:
            raise ValueError(f"part {self.label} ends before it starts")
        return self


UNSURE_KEY = 0.1  # a key clarity under this is said "unsure" (Temperley's margins: 0.02-0.18)


class StructureMap(BaseModel):
    """`structure.json`: the take's beats, downbeats, tempo, bar and parts, kept with the
    run and read by the export (the REAPER project's tempo and markers) and the page."""

    tool: Literal["structure"] = "structure"
    analyser: str = Field(min_length=1, max_length=200)
    duration: float = Field(ge=0)
    tempo: float = Field(ge=20, le=400)
    beats_per_bar: int = Field(ge=1, le=16)
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    sections: list[Part] = Field(default_factory=list)
    # The take's key ("A minor") and how far it stands out from the next best one
    # (0 to 1); a map made before reads None, and `key to=` asks for it made again
    key: str | None = Field(default=None, max_length=24)
    key_clarity: float | None = Field(default=None, ge=0, le=1)
    # the key handed on by the edits that made the take (a tempo, section or key edit knows
    # what it did to the key), never heard again: a key heard again drifts
    key_carried: bool = False
    seconds: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _within(self) -> StructureMap:
        for part in self.sections:
            if part.end > self.duration + 0.5:
                raise ValueError(f"part {part.label} runs past the song's end")
        return self

    def words(self) -> str:
        """`120 BPM, 4/4, parts A B A`."""
        parts = " ".join(p.label for p in self.sections) or "none found"
        key = f", {self.key}" if self.key else ""
        if self.key and self.key_carried:
            key += (
                " (kept from the take before, first heard unsurely)"
                if self.unsure
                else (" (kept from the take before)")
            )
        elif self.key and self.unsure:
            key += " (unsure)"
        return f"{self.tempo:.0f} BPM, {self.beats_per_bar}/4{key}, parts {parts}"

    @property
    def unsure(self) -> bool:
        """Whether the key barely stood out from the next best (most real songs: a template
        key detector is right about half the time on real music)."""
        return self.key_clarity is not None and self.key_clarity < UNSURE_KEY


class MidiFile(BaseModel):
    midi: str = Field(min_length=1, max_length=200)  # the `.mid` the job wrote in its out folder
    notes: int = Field(ge=0)


class MidiAnswer(BaseModel):
    """Each file's notes as a MIDI file, by the file it was read from."""

    tool: Literal["midi"]
    model: str = Field(min_length=1, max_length=200)
    files: dict[str, MidiFile]
    seconds: float = Field(ge=0)


class MasterAnswer(BaseModel):
    """A mix mastered to a reference: the file written and Matchering's own lines."""

    tool: Literal["master"]
    matchering: str = Field(min_length=1, max_length=40)
    file: str = Field(min_length=1, max_length=200)
    notes: list[str] = Field(default_factory=list, max_length=20)
    seconds: float = Field(ge=0)


class TuneAnswer(BaseModel):
    """The vocals moved toward the key's notes: the file written, the share of its
    frames CREPE heard sung, and the median distance it moved them."""

    tool: Literal["tune"]
    model: str = Field(min_length=1, max_length=80)
    file: str = Field(min_length=1, max_length=200)
    sung: float = Field(ge=0, le=1)
    moved_cents: float = Field(ge=0, le=1200)
    seconds: float = Field(ge=0)


class SimilarityAnswer(BaseModel):
    """Each file's voice against the reference's: a speaker embedding's cosine, -1 to 1
    (the same voice reads high; another voice low, never a judgement of the singing)."""

    tool: Literal["similarity"]
    model: str = Field(min_length=1, max_length=200)
    similarity: dict[str, float]
    seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _bounds(self) -> SimilarityAnswer:
        wrong = [n for n, v in self.similarity.items() if not -1.0001 <= v <= 1.0001]
        if wrong:
            raise ValueError(f"a cosine outside -1 to 1 for {', '.join(wrong)}")
        return self


def _last_line(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ToolsError("the studio tools printed nothing")
    return lines[-1]


def answered_error(stdout: str) -> bool:
    """Whether the tool's last line is its own worded error (else it was cut off)."""
    return tool_error(stdout) is not None


def parse[T: BaseModel](stdout: str, model: type[T]) -> T:
    """The job's one JSON line (the last on stdout: a library's warnings go to stderr) as
    `model`; the tool's own error line, or anything else, as a worded ToolsError."""
    line = _last_line(stdout)
    data = json_line(line)
    if data is None:
        raise ToolsError(f"the studio tools answered something mnm cannot read: {line[:200]}")
    if (said := tool_error(line)) is not None:
        raise ToolsError(said[:400])
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "the answer"
        raise ToolsError(
            f"the studio tools' answer is not usable ({where}: {first['msg']})"
        ) from None
