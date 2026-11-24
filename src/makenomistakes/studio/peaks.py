"""A file's waveform for the studio's pages: its peak and RMS level in `BARS` equal slices,
measured by ffmpeg's `astats` in the house format's container (one `docker run` for every file
of a folder), kept as `.peaks/<file>.json` beside the files and drawn as an inline SVG by the
page; no script draws or loops.

A kept waveform is used while its file has the size and time it was measured at; a file made
again is measured again. The pipeline measures each file as it makes it (best effort: a run
never fails for a picture), and the page's Draw waveforms does the ones an older run lacks.
"""

from __future__ import annotations

import contextlib
import math
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.audio_format import HouseFormat, HouseFormatError
from makenomistakes.media_bytes import WavHeader, wav_file_header

BARS = 1000  # measured slices a file
DRAWN = 500  # bars drawn on a page (each the louder of two measured)
FOLDER = ".peaks"
_FRAME = re.compile(r"^frame:(\d+)")
_LEVEL = re.compile(r"^lavfi\.astats\.Overall\.(Peak_level|RMS_level)=(.+)$")


class Peaks(BaseModel):
    schema_version: int = 1
    file: str
    size: int  # the file's bytes and time when measured: a changed file is measured again
    mtime_ns: int
    seconds: float = Field(ge=0)
    peaks: list[float]  # 0..1, linear, one per slice
    rms: list[float]


def wav_shape(path: Path) -> WavHeader | None:
    """A WAV file's rate, channels and frames from its header (the first 64 KiB), without
    reading its samples; None for anything else."""
    return wav_file_header(path)


def _cache(folder: Path, name: str) -> Path:
    return folder / FOLDER / f"{name}.json"


def kept(folder: Path, name: str) -> Peaks | None:
    """The file's waveform as measured, while the file is the one measured."""
    path = folder / name
    try:
        found = Peaks.model_validate_json(_cache(folder, name).read_text(encoding="utf-8"))
        info = path.stat()
    except (OSError, ValidationError):
        return None
    if (found.size, found.mtime_ns) != (info.st_size, info.st_mtime_ns):
        return None
    return found


def _level(text: str) -> float:
    """astats' dBFS as linear 0..1 (`-inf` for silence)."""
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return 0.0 if not math.isfinite(value) else min(1.0, 10 ** (value / 20))


def parse_levels(text: str) -> tuple[list[float], list[float]]:
    """ametadata's print: a `frame:` line, then that slice's `Overall.Peak_level` and
    `RMS_level` (every other key ignored, as ffmpeg 4.4 prints them all)."""
    peaks: dict[int, float] = {}
    rms: dict[int, float] = {}
    frame = -1
    for line in text.splitlines():
        if found := _FRAME.match(line):
            frame = int(found.group(1))
        elif (found := _LEVEL.match(line.strip())) and frame >= 0:
            (peaks if found.group(1) == "Peak_level" else rms)[frame] = _level(found.group(2))
    order = sorted(peaks)
    return [peaks[i] for i in order], [rms.get(i, 0.0) for i in order]


def _chain(slice_frames: int, out: str) -> str:
    """One input's filters: mono, fixed slices, the slice's levels printed to `out`."""
    return (
        f"aformat=channel_layouts=mono,asetnsamples=n={slice_frames}:p=0,"
        f"astats=metadata=1:reset=1,ametadata=mode=print:file={out}"
    )


def measure(house: HouseFormat, folder: Path, names: list[str]) -> dict[str, Peaks]:
    """Every named WAV in `folder` that has no kept waveform, in one ffmpeg run; what was
    kept and what was measured, by name. A file that is not a WAV is left out."""
    found = {name: p for name in names if (p := kept(folder, name)) is not None}
    todo = [(n, s) for n in names if n not in found and (s := wav_shape(folder / n)) is not None]
    if not todo:
        return found
    try:
        (folder / FOLDER).mkdir(exist_ok=True)
    except OSError:
        return found  # a picture is never worth a failure
    args: list[str] = []
    graph: list[str] = []
    for i, (name, shape) in enumerate(todo):
        out = f"{FOLDER}/{name}.txt"
        args += ["-i", name]
        graph.append(f"[{i}:a]{_chain(max(1, math.ceil(shape.frames / BARS)), out)}[o{i}]")
    args += ["-filter_complex", ";".join(graph)]
    for i in range(len(todo)):
        args += ["-map", f"[o{i}]", "-f", "null", "-"]
    try:
        house.run(folder, args)
    except HouseFormatError:
        return found  # a picture is never worth a failure; the page says none is drawn
    for name, shape in todo:
        text_path = folder / FOLDER / f"{name}.txt"
        try:
            text = text_path.read_text(encoding="utf-8")
            info = (folder / name).stat()
        except OSError:
            continue
        finally:
            text_path.unlink(missing_ok=True)
        peaks, rms = parse_levels(text)
        if not peaks:
            continue
        made = Peaks(
            file=name, size=info.st_size, mtime_ns=info.st_mtime_ns,
            seconds=round(shape.seconds, 3), peaks=peaks, rms=rms,
        )  # fmt: skip
        with contextlib.suppress(OSError):  # drawn now, measured again next time
            _cache(folder, name).write_text(made.model_dump_json() + "\n", encoding="utf-8")
        found[name] = made
    return found


def svg_path(levels: list[float], height: float = 64.0, bars: int = DRAWN) -> tuple[str, int]:
    """An SVG path of at most `bars` vertical strokes centred on the middle line, and how
    many it drew (the viewBox's width, so a click's x is the file's time exactly)."""
    if not levels:
        return "", 0
    step = max(1, math.ceil(len(levels) / bars))
    pooled = [max(levels[i : i + step]) for i in range(0, len(levels), step)]
    middle = height / 2
    parts = []
    for x, level in enumerate(pooled):
        half = max(0.5, level * middle)
        parts.append(f"M{x}.5 {middle - half:.1f}V{middle + half:.1f}")
    return "".join(parts), len(pooled)
