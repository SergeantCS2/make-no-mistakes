"""The Song template: a theme → lyrics (the desk's `chat`) → N candidates (the song stage's
model) → each candidate scored and the picked one's structure mapped (the studio's tools) → the
picked one split into stems → the stems summed into a mix → an export folder.

The stages run in that order on purpose: every candidate is one request to the
same loaded song model, then the separators, so the router loads each model once and parks
the one before by its own rules. The lyrics and the candidates are review points: the user
edits a section or replaces the whole, and picks one candidate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from makenomistakes.comfy.form import SEED_NAMES

if TYPE_CHECKING:
    from makenomistakes.runs import RunRecord

NAME = "song"  # the runs' target
LYRICS, SONG, STEMS, MIX, EXPORT = "lyrics", "song", "stems", "mix", "export"
SCORE, STRUCTURE = "score", "structure"  # The studio's tools
VOCALS = "vocals"  # The separator's vocals cleaned (skipped when nothing is asked)
VOICE = "voice"  # The vocals sung again in a library voice (skipped without one)
MIDI = "midi"  # Each stem but the drums as MIDI (skipped without the studio tools)
STAGES = (LYRICS, SONG, SCORE, STRUCTURE, STEMS, VOCALS, VOICE, MIDI, MIX, EXPORT)
# Where a run waits for the user, unless it runs straight through: the lyrics, and the pick,
# which follows the scores (a run made before mnm scored candidates has none and waits after
# the candidates).
PICKS = (SONG, SCORE)
REVIEWS = (LYRICS, *PICKS)
STAGE_LABELS = {
    LYRICS: "Lyrics",
    SONG: "Candidate",
    SCORE: "Score",
    STRUCTURE: "Structure",
    STEMS: "Stems",
    VOCALS: "Vocals",
    VOICE: "Voice",
    MIDI: "MIDI",
    MIX: "Mix",
    EXPORT: "Export",
}
ITEMS = (SONG, SCORE, STEMS)  # the stages with an item per candidate or per stem package
# what each stage's items are made from: a stage made again makes these again too (the one
# graph `_stale`, a new pick and a new take read)
MADE_FROM: dict[str, frozenset[str]] = {
    LYRICS: frozenset(STAGES),
    SONG: frozenset({STRUCTURE, STEMS, VOCALS, VOICE, MIDI, MIX, EXPORT}),
    SCORE: frozenset({EXPORT}),  # its provenance lists the scores
    STRUCTURE: frozenset({EXPORT}),
    STEMS: frozenset({VOCALS, VOICE, MIDI, MIX, EXPORT}),
    VOCALS: frozenset({VOICE, MIDI, MIX, EXPORT}),  # a voice sings the cleaned vocals
    VOICE: frozenset({MIDI, MIX, EXPORT}),  # the vocals' notes are the voice's
    MIDI: frozenset({EXPORT}),
    MIX: frozenset({EXPORT}),
    EXPORT: frozenset(),
}
LYRICS_FILE = "lyrics.json"
MIX_RUN = "mix_run"  # the run's setting: the mixer's render in use (a child run), if any
MASTER_RUN = "master_run"  # the run's setting: the master in use (a child run), if any
TAKE_RUN = "take_run"  # the run's setting: the edit whose take the song uses
STEMS_TAKE = "stems_take"  # the run's setting: the take its stems were split from
STRUCTURE_FILE = "structure.json"  # the picked take's beats, tempo and parts
MIX_FILE = "mix.wav"
MASTER_FILE = "master.wav"  # a master's take
# A song's child runs that are no song, by role: what one is and where its song lists them
CHILD_ROLES: dict[str, tuple[str, str]] = {
    "mix": ("a mix", "`mnm song mix {parent}` shows the song's mixes"),
    "edit": ("an edit", "`mnm song take {parent}` lists its takes"),
    "master": ("a master", "`mnm song master {parent}` lists its masters"),
}
EXPORT_FOLDER = "export"
MAX_CANDIDATES = 8
DEFAULT_CANDIDATES = 4
MIN_SECONDS, MAX_SECONDS = 10.0, 600.0  # ACE-Step 1.5's range
DEFAULT_SECONDS = 30.0
MAX_SECTIONS = 16
MAX_LINES = 16
DESK_CHAT = "chat"  # the desk's role, through `/upstream/chat/`
# How a recording the user gives is used, and the ACE-Step route each maps to on
# audio.cpp (docs/models/ace_step.md, read 2026-10-31: text2music, complete, lego, extract,
# cover, cover-nofsq, repaint). `vocal2bgm` is ACE-Step's own Python name for "complete
# around a vocal"; audio.cpp spells it `complete` with the accompaniment's track classes.
# `reference` has no route of its own: `cover` follows a source at a strength (1.0 keeps
# its structure, 0.1 is loose); a loose cover is the nearest, and its length is the source's.
RecordingUse = Literal["complete", "vocal2bgm", "reference"]
ROUTES: dict[str, str] = {"complete": "complete", "vocal2bgm": "complete", "reference": "cover"}
ACCOMPANIMENT = ["drums", "bass", "guitar", "keyboard", "strings"]
REFERENCE_STRENGTH = 0.3
Tier = Literal["fast", "full"]


class SongError(ValueError):
    """The lyrics or a setting the user gave cannot be used (the message says why)."""


def seconds_of(text: str) -> float:
    """`30s`, `3m`, `2m30s`, `90` (seconds) as seconds, within ACE-Step's range."""
    found = re.fullmatch(r"\s*(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s?)?\s*", text.lower())
    if not text.strip() or found is None or not any(found.groups()):
        raise SongError(f"'{text}' is not a length: write 30s, 3m or 2m30s.")
    minutes, seconds = found.groups()
    total = int(minutes or 0) * 60 + float(seconds or 0)
    if not MIN_SECONDS <= total <= MAX_SECONDS:
        raise SongError(f"A song is 10 s to 10 minutes long ({total:g} s asked).")
    return total


class SongInputs(BaseModel):
    """What the user gives the template; kept in the run's settings as strings."""

    theme: str = Field(min_length=3, max_length=4000)
    seconds: float = Field(default=DEFAULT_SECONDS, ge=MIN_SECONDS, le=MAX_SECONDS)
    candidates: int = Field(default=DEFAULT_CANDIDATES, ge=1, le=MAX_CANDIDATES)
    tier: Tier = "fast"
    chat: str = DESK_CHAT  # the desk's role, or an instance by name
    tags: str | None = Field(default=None, max_length=600)  # the user's own style line
    own_lyrics: bool = False  # the user gave the lyrics: the stage is skipped
    recording: RecordingUse | None = None  # how the kept recording is used
    voice: str | None = Field(default=None, max_length=64)  # a library voice
    voice_shift: int = Field(default=0, ge=-12, le=12)  # semitones the voice moves the vocals
    # What is done to the separator's vocals, as `song_vocals.parse` reads it
    vocals_clean: str | None = Field(default=None, max_length=200)
    auto: bool = False  # run straight through the review points (the first candidate)

    def settings(self) -> dict[str, str]:
        values = self.model_dump(exclude_none=True)
        return {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in values.items()
        }

    @classmethod
    def from_settings(cls, settings: dict[str, str]) -> SongInputs:
        known = set(cls.model_fields)
        values: dict[str, object] = {k: v for k, v in settings.items() if k in known}
        for flag in ("auto", "own_lyrics"):
            values[flag] = settings.get(flag) == "true"
        return cls.model_validate(values)


class Section(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    lines: list[str] = Field(min_length=1, max_length=MAX_LINES)

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        return " ".join(value.replace("[", "").replace("]", "").split())

    @field_validator("lines")
    @classmethod
    def _lines(cls, value: list[str]) -> list[str]:
        kept = [" ".join(line.split()) for line in value]
        kept = [line[:200] for line in kept if line]
        if not kept:
            raise ValueError("a section needs at least one line")
        return kept


class Lyrics(BaseModel):
    title: str = Field(default="", max_length=200)
    tags: str = Field(min_length=1, max_length=600)  # the song model's style line
    sections: list[Section] = Field(min_length=1, max_length=MAX_SECTIONS)

    @field_validator("tags", "title")
    @classmethod
    def _one_line(cls, value: str) -> str:
        return " ".join(value.split())

    def sung(self) -> str:
        """The lyrics as ACE-Step reads them: `[Verse]` headers, one line each."""
        return "\n\n".join(
            "\n".join([f"[{section.label}]", *section.lines]) for section in self.sections
        )


LYRICS_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "tags": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "lines": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "lines"],
            },
        },
    },
    "required": ["title", "tags", "sections"],
}


def read_lyrics(folder: Path) -> Lyrics:
    """A folder's `lyrics.json` (a song's, or an edit's: the lyrics its take was sung with);
    a SongError saying what is wrong."""
    path = folder / LYRICS_FILE
    try:
        return Lyrics.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SongError(f"{path} cannot be read ({exc.strerror or exc})") from exc
    except ValueError as exc:
        raise SongError(f"{path} is not lyrics mnm wrote") from exc


def section_budget(seconds: float) -> int:
    """About one section per 20 seconds, two at least (a verse and a chorus)."""
    return max(2, min(MAX_SECTIONS, round(seconds / 20)))


def lines_budget(seconds: float) -> int:
    """About one sung line per four seconds (with no budget, a chat model wrote twelve lines
    for a 10 s song)."""
    return max(2, min(MAX_SECTIONS * 4, round(seconds / 4)))


def lyrics_prompt(inputs: SongInputs) -> str:
    sections = section_budget(inputs.seconds)
    style = f"The style is fixed: {inputs.tags}\n" if inputs.tags else ""
    return (
        f"Write the lyrics of a song about {inputs.seconds:g} seconds long, in about "
        f"{sections} sections.\n"
        f"The theme: {inputs.theme}\n{style}\n"
        "Answer with JSON only, in this shape: "
        '{"title": "...", "tags": "...", "sections": [{"label": "Verse", "lines": ["..."]}]}.\n'
        "- tags: one line for a music model, chosen for this theme: its genre, its mood, two "
        "or three instruments, a tempo in bpm and the singer's voice, comma-separated.\n"
        f"- sections: in order; label is Intro, Verse, Pre-Chorus, Chorus, Bridge or Outro; "
        f"short singable lines, about {lines_budget(inputs.seconds)} lines in all, no stage "
        "directions."
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_lyrics(text: str) -> Lyrics:
    """The lyrics from a chat answer: plain JSON, JSON in a code fence, or the first `{…}`."""
    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise SongError("the answer holds no JSON object")
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SongError(f"the JSON does not parse ({exc.msg} at character {exc.pos})") from None
    return _validated(data)


def _validated(data: object) -> Lyrics:
    try:
        return Lyrics.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "the lyrics"
        raise SongError(f"{where}: {first['msg']}") from None


_HEADER = re.compile(r"^\s*\[([^\]]{1,40})\]\s*$")


def lyrics_from_text(text: str, tags: str, title: str = "") -> Lyrics:
    """The user's own lyrics (a file, or Replace all): `[Verse]`-style headers start the
    sections; text before the first header, or with none, is one section."""
    sections: list[dict[str, object]] = []
    for raw in text.splitlines():
        header = _HEADER.match(raw)
        if header:
            sections.append({"label": header.group(1), "lines": []})
        elif raw.strip():
            if not sections:
                sections.append({"label": "Verse", "lines": []})
            lines = sections[-1]["lines"]
            assert isinstance(lines, list)
            lines.append(raw)
    sections = [s for s in sections if s["lines"]]
    if not sections:
        raise SongError("the lyrics are empty: write at least one line")
    return _validated({"title": title, "tags": tags, "sections": sections})


def candidate_label(index: int) -> str:
    return f"candidate {index + 1}"


def stem_file(stem: str) -> str:
    """`stem-vocals.wav`: a stem in the house format, in the run's folder."""
    return f"stem-{stem}.wav"


PICKED = "picked"  # the run's setting: the picked candidate's index, from 0


def picked_index(settings: dict[str, str]) -> int | None:
    value = settings.get(PICKED, "")
    return int(value) if value.isdigit() else None


def candidate_seed(child: RunRecord) -> str | None:
    """The seed a candidate was sung with: an audio.cpp candidate's `seed`, a ComfyUI
    workflow's form seed, kept under its field's key (`<node>.seed`)."""
    if "seed" in child.settings:
        return child.settings["seed"]
    if child.kind.value != "workflow":
        return None
    seeds = (v for k, v in child.settings.items() if k.rpartition(".")[2] in SEED_NAMES)
    return next(seeds, None)


def again_refusal(child: RunRecord, parent: RunRecord | None) -> str | None:
    """Why Run again makes no copy of a song's candidate (None for any other run): outside
    its song it would lose the user's recording and the song's length; the song makes it
    again in place."""
    if child.role != "candidate" or parent is None or not is_song(parent):
        return None
    found = next((s for s in parent.steps if s.stage == SONG and s.text == child.id), None)
    which = str(found.index + 1) if found is not None else "<n>"
    return (
        f"{child.id} is a candidate of the song {parent.id}: `mnm song redo {parent.id} song "
        f"{which}` makes it again in the song, with its recording and its length (or Make "
        "again on the song's page)."
    )


def is_song(run: RunRecord) -> bool:
    """A song's own run: a pipeline run of this template made under no other run (the
    mixer's renders are pipeline runs of the song too, under it: never a song to run, list
    or open)."""
    return run.kind.value == "pipeline" and run.target == NAME and run.parent is None


def stem_files(names: list[str]) -> list[str]:
    """The stems among a run's file names (`stem-vocals.wav` …), in the order they came."""
    return [n for n in names if n.startswith("stem-") and n.endswith(".wav")]


def past_pick(reviewed: list[str]) -> bool:
    """Whether the user passed the pick (at the scores, or at the candidates for a run made
    before the scores)."""
    return any(stage in reviewed for stage in PICKS)
