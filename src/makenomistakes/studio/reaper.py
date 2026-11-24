"""The REAPER project an export carries: `song.rpp`, plain text rendered from a
typed `ReaperProject` through one template (`templates/song.rpp.j2`), and read back by a
parser for the subset mnm writes, so a project round-trips.

REAPER's format has no published schema. What is written here follows `.rpp` files as
REAPER saves them (public projects on GitHub, read 2026-10-31): nested blocks opened by a
`<NAME args` line and closed by a line holding `>`, one setting a line, strings in double
quotes, a project's notes as `|`-prefixed lines in `<NOTES 0 2`. REAPER fills every setting
a file leaves out with its default. A track holds one item whose source is the stem's file,
named relative to the project (REAPER resolves it beside the `.rpp`, so the folder opens from
`\\\\wsl$` on Windows). Opening it in REAPER 7.x is the real test.
"""

from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

TEMPLATE = "song.rpp.j2"
VERSION = "7.0"  # the REAPER version the file says it is for
DEFAULT_TEMPO = 120.0  # REAPER's own defaults, used when the song's tempo is not known
DEFAULT_BEATS = 4


class ReaperError(ValueError):
    """A project text that is not the subset mnm writes."""


def _quoted(text: str) -> str:
    """A string REAPER reads between double quotes: no double quote, no line break."""
    return " ".join(text.replace('"', "'").split())


class Marker(BaseModel):
    position: float = Field(ge=0)
    name: str


class Track(BaseModel):
    name: str
    file: str  # relative to the project file: `stems/vocals.wav`
    length: float = Field(ge=0)
    volume: float = Field(default=1.0, ge=0, le=4)  # linear gain, 1 = 0 dB
    mute: bool = False
    guid: str

    @property
    def file_name(self) -> str:
        return Path(self.file).name

    @property
    def source_kind(self) -> str:
        """REAPER's source type by the file's suffix (a `.mid` is a MIDI file it reads in
        place)."""
        suffix = Path(self.file).suffix.lower()
        return {".flac": "FLAC", ".mid": "MIDI"}.get(suffix, "WAVE")


class ReaperProject(BaseModel):
    version: str = VERSION
    created: int = 0  # seconds since the epoch, as REAPER stamps its files
    tempo: float = Field(default=DEFAULT_TEMPO, gt=0, le=960)
    beats: int = Field(default=DEFAULT_BEATS, ge=1, le=32)
    beat_unit: int = Field(default=4, ge=1, le=64)
    rate: int = 48_000
    notes: list[str] = Field(default_factory=list)
    markers: list[Marker] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)


def track_guid(run_id: str, name: str) -> str:
    """A track's GUID, the same for the same run and stem (an export rendered again keeps
    REAPER's track identities)."""
    return "{" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"mnm:{run_id}:{name}")).upper() + "}"


def _num(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    env.filters["num"] = _num
    return env


def render(project: ReaperProject) -> str:
    """The project's text; every string REAPER reads quoted is made safe first."""
    safe = project.model_copy(
        update={
            "notes": [" ".join(line.split()) for line in project.notes],
            "markers": [m.model_copy(update={"name": _quoted(m.name)}) for m in project.markers],
            "tracks": [
                t.model_copy(update={"name": _quoted(t.name), "file": _quoted(t.file)})
                for t in project.tracks
            ],
        }
    )
    return _environment().get_template(TEMPLATE).render(project=safe)


# -- reading back ---------------------------------------------------------------------------


@dataclass
class Node:
    """One `<NAME args` block: its own lines (each split into words) and its blocks."""

    name: str
    args: list[str]
    lines: list[list[str]] = field(default_factory=list)
    children: list[Node] = field(default_factory=list)
    text: list[str] = field(default_factory=list)  # `|` lines (a NOTES block's)

    def child(self, name: str) -> Node | None:
        return next((c for c in self.children if c.name == name), None)

    def line(self, key: str) -> list[str] | None:
        return next((words[1:] for words in self.lines if words and words[0] == key), None)


def _words(line: str) -> list[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError as exc:
        raise ReaperError(f"an unreadable line ({exc}): {line[:80]}") from None


def parse(text: str) -> Node:  # noqa: PLR0912 - one branch per kind of line
    """The block tree of a project text."""
    stack: list[Node] = []
    root: Node | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<"):
            words = _words(line[1:])
            if not words:
                raise ReaperError("a block without a name")
            node = Node(words[0], words[1:])
            if stack:
                stack[-1].children.append(node)
            elif root is None:
                root = node
            else:
                raise ReaperError("text after the project's end")
            stack.append(node)
        elif line == ">":
            if not stack:
                raise ReaperError("a '>' that closes nothing")
            stack.pop()
        elif not stack:
            raise ReaperError(f"a line outside the project: {line[:80]}")
        elif line.startswith("|"):
            stack[-1].text.append(line[1:])
        else:
            stack[-1].lines.append(_words(line))
    if root is None or stack:
        raise ReaperError("the project is not closed" if root else "no REAPER_PROJECT block")
    if root.name != "REAPER_PROJECT":
        raise ReaperError(f"the file is a {root.name} block, not a REAPER project")
    return root


def read(text: str) -> ReaperProject:
    """The subset mnm writes, from a project text (`mnm song show`)."""
    root = parse(text)
    tempo = root.line("TEMPO") or [str(DEFAULT_TEMPO), str(DEFAULT_BEATS), "4"]
    rate = root.line("SAMPLERATE") or ["48000"]
    notes = root.child("NOTES")
    tracks = []
    for node in (c for c in root.children if c.name == "TRACK"):
        item = node.child("ITEM")
        source = item.child("SOURCE") if item else None
        if item is None or source is None:
            raise ReaperError(f"track {node.args[:1]} has no item with a source")
        tracks.append(
            Track(
                name=(node.line("NAME") or [""])[0],
                file=(source.line("FILE") or [""])[0],
                length=float((item.line("LENGTH") or ["0"])[0]),
                volume=float((node.line("VOLPAN") or ["1"])[0]),
                mute=(node.line("MUTESOLO") or ["0"])[0] == "1",
                guid=node.args[0] if node.args else "",
            )
        )
    markers = [
        Marker(position=float(words[2]), name=words[3])
        for words in root.lines
        if words[0] == "MARKER" and len(words) >= 4
    ]
    return ReaperProject(
        version=root.args[1] if len(root.args) > 1 else VERSION,
        created=int(root.args[2]) if len(root.args) > 2 else 0,
        tempo=float(tempo[0]),
        beats=int(tempo[1]),
        beat_unit=int(tempo[2]),
        rate=int(rate[0]),
        notes=list(notes.text) if notes else [],
        markers=markers,
        tracks=tracks,
    )
