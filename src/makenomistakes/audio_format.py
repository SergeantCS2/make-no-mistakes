"""The studio's house format: every file a stage hands on is a 48 kHz WAV, 24-bit
for what is kept and exported, 16-bit at the rate a model reads for what is sent to an
engine (the separators take 44.1 kHz only); FLAC to keep, MP3 on request, and the export's
loudness line by ffmpeg's `loudnorm` in two passes (-14 LUFS, -1 dBTP by default).

ffmpeg runs as the film's stitch runs it (`pipeline/steps.DockerStitcher`): inside an engine
image already on the box (audio.cpp's, which the song instance runs, this box's tag first;
then whisper.cpp's; each carries ffmpeg: 6.1 in audio.cpp's `full-cpu`, 4.4 in whisper.cpp's
`main`, read 2026-10-31), `--network none`, as the user's uid, with the run's folder as its
only mount, and nothing is ever pulled. Every command line is rendered from
the typed jobs below (`Conversion`, `Mixdown`, `Splice`, `Loudness`), never edited as a string, and
every answer is read into a pydantic model (`Measured`).

`conform_bytes` is the one edge that falls back: a single clip `mnm try`/`mnm run` sends a
separator inline is resampled by `wav_rate` (plain Python, 16-bit, linear) when Docker or an
ffmpeg image is not on the box, so a try never needs a pull. The pipeline never falls back.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from makenomistakes import wav_rate
from makenomistakes.hardware.probe import CommandRunner

HOUSE_RATE = 48_000  # the rate every stage works at
FFMPEG_SECONDS = 1800.0  # a 10-minute song's two loudness passes on a slow CPU
TIMED_OUT = 124  # the runner's exit code for a command past its time
CONTAINER_PREFIX = "mnm-ffmpeg-"
DEFAULT_LUFS = -14.0  # the loudness line: streaming's integrated target
DEFAULT_TRUE_PEAK = -1.0
DEFAULT_LRA = 11.0  # loudnorm's own default range
DEFAULT_FADE = 0.03  # seconds an equal-power crossfade takes at a section edit's joint


class HouseFormatError(RuntimeError):
    """ffmpeg could not do a step, worded for the user."""


class Codec(StrEnum):
    WAV24 = "wav24"  # the house working format, and the export's WAV
    WAV16 = "wav16"  # what an engine is sent inline
    FLOAT = "float"  # a sum before it is checked for clipping (32-bit float keeps past 0 dBFS)
    FLAC = "flac"  # 24-bit, to keep
    MP3 = "mp3"  # on request

    @property
    def suffix(self) -> str:
        return {Codec.FLAC: ".flac", Codec.MP3: ".mp3"}.get(self, ".wav")

    def argv(self) -> list[str]:
        match self:
            case Codec.WAV24:
                return ["-c:a", "pcm_s24le"]
            case Codec.WAV16:
                return ["-c:a", "pcm_s16le"]
            case Codec.FLOAT:
                return ["-c:a", "pcm_f32le"]
            case Codec.FLAC:
                return ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
            case Codec.MP3:
                return ["-c:a", "libmp3lame", "-b:a", "320k"]


_NAME = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._/-]{0,200}$")


def _plain(name: str) -> str:
    """A file name inside the mounted folder: no leading dash (ffmpeg would read an option),
    no `..`, no absolute path."""
    if not _NAME.match(name) or ".." in name.split("/"):
        raise HouseFormatError(f"'{name}' is not a file name ffmpeg is given here.")
    return name


class Conversion(BaseModel):
    """One file to another at a rate, channels and codec, with an optional filter chain."""

    source: str
    target: str
    codec: Codec = Codec.WAV24
    rate: int = Field(default=HOUSE_RATE, ge=8_000, le=192_000)
    channels: int = Field(default=2, ge=1, le=8)
    filters: list[str] = Field(default_factory=list)

    def args(self) -> list[str]:
        chain = ["-af", ",".join(self.filters)] if self.filters else []
        return [
            "-i", _plain(self.source), *chain,
            "-ar", str(self.rate), "-ac", str(self.channels),
            *self.codec.argv(), _plain(self.target),
        ]  # fmt: skip


class Mixdown(BaseModel):
    """Files summed as they are (`amix` with no normalising: the stems of a split add back up
    to the song), the longest setting the length."""

    sources: list[str] = Field(min_length=1)
    target: str
    codec: Codec = Codec.WAV24
    rate: int = HOUSE_RATE

    def args(self) -> list[str]:
        inputs = [arg for name in self.sources for arg in ("-i", _plain(name))]
        mix = f"amix=inputs={len(self.sources)}:normalize=0:duration=longest:dropout_transition=0"
        return [
            *inputs, "-filter_complex", mix,
            "-ar", str(self.rate), "-ac", "2", *self.codec.argv(), _plain(self.target),
        ]  # fmt: skip


class Splice(BaseModel):
    """Pieces of one file joined in order (a section edit): each piece a `(start, end)`
    in seconds, every joint crossfaded with an equal-power curve over `fade` seconds (a cut on
    a downbeat never clicks); `fade` 0 butts them."""

    source: str
    pieces: list[tuple[float, float]] = Field(min_length=1, max_length=64)
    target: str
    fade: float = Field(default=DEFAULT_FADE, ge=0, le=2)
    codec: Codec = Codec.WAV24

    @field_validator("pieces")
    @classmethod
    def _long_enough(cls, pieces: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for start, end in pieces:
            if start < 0 or end - start < 0.05:
                raise ValueError(f"the piece {start:g}-{end:g} s is too short to keep")
        return pieces

    def args(self) -> list[str]:
        # one input per piece: `asplit` of one input stalls when a piece comes before an
        # earlier part of the file (a move, a duplicate), and answers an empty file
        inputs = [arg for _piece in self.pieces for arg in ("-i", _plain(self.source))]
        graph = [
            f"[{i}:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[p{i}]"
            for i, (start, end) in enumerate(self.pieces)
        ]
        last = "p0"
        for i in range(1, len(self.pieces)):
            fade = min(self.fade, (self.pieces[i - 1][1] - self.pieces[i - 1][0]) / 2,
                       (self.pieces[i][1] - self.pieces[i][0]) / 2)  # fmt: skip
            join = f"acrossfade=d={fade:.4f}:c1=qsin:c2=qsin" if fade > 0 else "concat=n=2:v=0:a=1"
            graph.append(f"[{last}][p{i}]{join}[j{i}]")
            last = f"j{i}"
        return [
            *inputs, "-filter_complex", ";".join(graph), "-map", f"[{last}]",
            "-ar", str(HOUSE_RATE), "-ac", "2", *self.codec.argv(), _plain(self.target),
        ]  # fmt: skip


def _number(value: object) -> float:
    """loudnorm prints its figures as strings; silence measures `-inf`."""
    return float(str(value))


class Measured(BaseModel):
    """loudnorm's first pass, as it prints it."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float = 0.0

    @field_validator(
        "input_i", "input_tp", "input_lra", "input_thresh", "target_offset", mode="before"
    )
    @classmethod
    def _from_text(cls, value: object) -> float:
        return _number(value)

    @property
    def silent(self) -> bool:
        return not math.isfinite(self.input_i) or not math.isfinite(self.input_thresh)

    def words(self) -> str:
        if self.silent:
            return "silent"
        return f"{self.input_i:.1f} LUFS, {self.input_tp:.1f} dBTP"


class Loudness(BaseModel):
    """The export's loudness line (a setting: `mnm song export --lufs … --true-peak …`)."""

    lufs: float = Field(default=DEFAULT_LUFS, ge=-70, le=-5)
    true_peak: float = Field(default=DEFAULT_TRUE_PEAK, ge=-9, le=0)
    lra: float = Field(default=DEFAULT_LRA, ge=1, le=50)

    def measuring(self) -> str:
        return f"loudnorm=I={self.lufs}:TP={self.true_peak}:LRA={self.lra}:print_format=json"

    def applying(self, measured: Measured) -> str:
        return (
            f"loudnorm=I={self.lufs}:TP={self.true_peak}:LRA={self.lra}"
            f":measured_I={measured.input_i}:measured_TP={measured.input_tp}"
            f":measured_LRA={measured.input_lra}:measured_thresh={measured.input_thresh}"
            f":offset={measured.target_offset}:linear=true:print_format=summary"
        )

    def words(self) -> str:
        return f"{self.lufs:g} LUFS integrated, {self.true_peak:g} dBTP true peak"


def parse_measured(stderr: str) -> Measured:
    """The JSON block loudnorm prints last on stderr."""
    end = stderr.rfind("}")
    start = stderr.rfind("{", 0, end + 1)
    if start < 0 or end < start:
        raise HouseFormatError("ffmpeg's loudness pass printed no measurement.")
    try:
        return Measured.model_validate(json.loads(stderr[start : end + 1]))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HouseFormatError(f"ffmpeg's loudness measurement cannot be read: {exc}") from None


def house_images() -> list[str]:
    """The images that carry ffmpeg, in the order they are tried: audio.cpp's (the studio's
    own engine: on a box that makes songs it is pulled already), this box's tag first, then
    whisper.cpp's, as the film's stitch uses."""
    from makenomistakes.engine_images import IMAGES  # noqa: PLC0415 - heavy, and cycle-prone
    from makenomistakes.engines import box_images  # noqa: PLC0415
    from makenomistakes.hub.facts import Engine  # noqa: PLC0415

    here = box_images()
    found: list[str] = []
    for engine in (Engine.AUDIOCPP, Engine.WHISPERCPP):
        found += [here[engine]] if engine in here else []
        found.append(IMAGES[engine].default)
        for rows in IMAGES[engine].rows.values():
            found += [row.tag for row in rows]
    return list(dict.fromkeys(found))


class HouseFormat:
    """ffmpeg in Docker over one folder (a run's): convert, mix, measure and normalise."""

    def __init__(
        self,
        runner: CommandRunner,
        images: Callable[[], list[str]] = house_images,
    ) -> None:
        self.runner = runner
        self.images = images

    def image(self) -> str:
        """The first image on the box that carries ffmpeg (asked at each use, never kept)."""
        wanted = self.images()
        for image in wanted:
            if self.runner.run(["docker", "image", "inspect", image], timeout=30)[0] == 0:
                return image
        raise HouseFormatError(
            f"The studio's audio steps use the ffmpeg inside an engine image, and none is on "
            f"this box: `docker pull {wanted[0]}` (a song instance's first start pulls it "
            "too), then resume the run."
        )

    def available(self) -> bool:
        try:
            self.image()
        except HouseFormatError:
            return False
        return True

    def run(self, folder: Path, args: list[str], quiet: bool = True) -> str:
        """ffmpeg with `args` in `folder`; its stderr (the loudness pass reads it). Named, so a
        job past its time is removed, never left writing into the run's folder; sandboxed as
        the studio tools' jobs are (it reads the user's recordings and references)."""
        image = self.image()
        name = f"{CONTAINER_PREFIX}{secrets.token_hex(6)}"
        argv = [
            "docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "512",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{folder}:/w", "-w", "/w", "--entrypoint", "ffmpeg", image,
            "-hide_banner", "-nostats", "-loglevel", "error" if quiet else "info", "-y", *args,
        ]  # fmt: skip
        code, _out, err = self.runner.run(argv, timeout=FFMPEG_SECONDS)
        if code == TIMED_OUT:  # the client was killed, the container runs on: removed
            self.runner.run(["docker", "rm", "-f", name], timeout=30)
            raise HouseFormatError(
                f"ffmpeg took over {FFMPEG_SECONDS / 60:.0f} minutes and was stopped."
            )
        if code != 0:
            raise HouseFormatError(f"ffmpeg failed: {err.strip()[-400:] or f'exit {code}'}")
        return err

    def convert(self, folder: Path, job: Conversion) -> Path:
        self.run(folder, job.args())
        return self._made(folder, job.target)

    def mix(self, folder: Path, job: Mixdown) -> Path:
        self.run(folder, job.args())
        return self._made(folder, job.target)

    def splice(self, folder: Path, job: Splice) -> Path:
        self.run(folder, job.args())
        return self._made(folder, job.target)

    def measure(self, folder: Path, source: str, loudness: Loudness) -> Measured:
        """loudnorm's first pass over `source`."""
        args = ["-i", _plain(source), "-af", loudness.measuring(), "-f", "null", "-"]
        return parse_measured(self.run(folder, args, quiet=False))

    def normalised(
        self,
        folder: Path,
        source: str,
        target: str,
        codec: Codec,
        loudness: Loudness,
        measured: Measured,
    ) -> Path:
        """loudnorm's second pass (linear), resampled back to the house rate (loudnorm works
        at 192 kHz); a silent file is converted as it is."""
        filters = [] if measured.silent else [loudness.applying(measured)]
        job = Conversion(source=source, target=target, codec=codec, filters=filters)
        return self.convert(folder, job)

    @staticmethod
    def _made(folder: Path, name: str) -> Path:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise HouseFormatError(f"ffmpeg wrote no {name}.")
        return path

    def conform_bytes(self, data: bytes, rate: int, channels: int = 2) -> bytes:
        """A clip for an engine at `rate` (16-bit), through ffmpeg; through `wav_rate` when
        no ffmpeg image is on the box (a try never pulls one). Raises `wav_rate.WavError`
        for a file neither can read."""
        if not self.available():
            return wav_rate.conformed(data, rate, channels)
        work = Path(tempfile.mkdtemp(prefix="mnm-conform-"))
        try:
            (work / "in.bin").write_bytes(data)
            job = Conversion(
                source="in.bin", target="out.wav", codec=Codec.WAV16, rate=rate, channels=channels
            )
            try:
                return self.convert(work, job).read_bytes()
            except HouseFormatError as exc:
                raise wav_rate.WavError(f"not an audio file ffmpeg can read ({exc})") from None
        finally:
            shutil.rmtree(work, ignore_errors=True)


def default_house() -> HouseFormat:
    """This box's house format (tests put one without Docker in its place)."""
    from makenomistakes.hardware.runner import SubprocessRunner  # noqa: PLC0415

    return HouseFormat(SubprocessRunner())
