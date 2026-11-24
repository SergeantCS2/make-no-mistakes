"""A voice's training recordings: read from the user's folder, checked against what an
RVC voice is trained on, and copied into the training's own folder with each one's sha256
(the voice row keeps them as its dataset).

The checks are the community's figures (10 to 60 minutes of the voice alone, dry, mono,
lossless, no gating) and each is a warning, never a refusal: a short or lossy dataset trains,
and sounds less like its singer. Only a folder with nothing mnm can train on is refused. Lengths
come from the files' headers (WAV's chunks, FLAC's STREAMINFO); an MP3 or Ogg file's length is
not read (mnm has no decoder): it trains, and counts as "length not read".
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from makenomistakes.media_bytes import wav_file_header
from makenomistakes.run_parts import InputRecord

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
LOSSY = frozenset({".mp3", ".ogg"})
MIN_MINUTES = 10.0
MAX_MINUTES = 60.0
MAX_FILES = 2000
MAX_SEEN = 20_000  # entries looked at before a folder is called too large (a `~` or a `/`)
MAX_FILE_BYTES = 2 * 1024**3


class TrainDataError(ValueError):
    """A folder mnm cannot train a voice from, worded for the user."""


@dataclass(frozen=True)
class Recording:
    path: Path
    size: int
    seconds: float | None  # None: not read (MP3, Ogg, an unreadable header)
    channels: int | None
    rate: int | None

    @property
    def lossy(self) -> bool:
        return self.path.suffix.lower() in LOSSY


@dataclass(frozen=True)
class Survey:
    recordings: list[Recording]
    skipped: list[str] = field(default_factory=list)  # links, too large, …: each with why
    warnings: list[str] = field(default_factory=list)
    noun: str = "recording"  # what the files are, in words (a style's are songs)

    @property
    def seconds(self) -> float:
        return sum(r.seconds or 0.0 for r in self.recordings)

    @property
    def unread(self) -> int:
        return sum(1 for r in self.recordings if r.seconds is None)

    @property
    def words(self) -> str:
        count = len(self.recordings)
        minutes = f"{self.seconds / 60:.1f} min"
        unread = f" (and {self.unread} whose length is not read)" if self.unread else ""
        return f"{count} {self.noun}{'s' * (count != 1)}, {minutes}{unread}"


def flac_header(head: bytes) -> tuple[int, int, int] | None:
    """(rate, channels, frames) from a FLAC file's STREAMINFO block; None for anything else."""
    if len(head) < 42 or head[:4] != b"fLaC" or head[4] & 0x7F != 0:  # STREAMINFO comes first
        return None
    info = head[8:42]
    packed = int.from_bytes(info[10:18], "big")
    rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    frames = packed & 0xFFFFFFFFF
    return (rate, channels, frames) if rate else None


def read_recording(path: Path) -> Recording:
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(65536)
    suffix = path.suffix.lower()
    if suffix == ".wav" and (wav := wav_file_header(path)) is not None and wav.byte_rate:
        return Recording(path, size, wav.seconds, wav.channels, wav.rate)
    if suffix == ".flac" and (flac := flac_header(head)) is not None:
        rate, channels, frames = flac
        return Recording(path, size, frames / rate if frames else None, channels, rate)
    return Recording(path, size, None, None, None)


def survey(folder: Path, rate: int) -> Survey:
    """Every recording under `folder` (subfolders too), each read; the warnings in words."""
    found, skipped = gather(folder, AUDIO_SUFFIXES, "recordings", "one voice")
    if not found:
        raise TrainDataError(
            f"{folder} holds no recordings mnm trains from (WAV or FLAC; MP3 and Ogg train "
            "but lose detail)."
        )
    return Survey(found, skipped, _warnings(found, rate))


def gather(
    folder: Path, suffixes: tuple[str, ...], what: str, one: str
) -> tuple[list[Recording], list[str]]:
    """Every file under `folder` with one of `suffixes`, each read, and the ones skipped with
    why (links, over 2 GB); `what` and `one` word a refusal ("recordings", "one voice")."""
    if not folder.is_dir():
        raise TrainDataError(f"{folder} is not a folder of {what}.")
    found: list[Recording] = []
    skipped: list[str] = []
    for path in walk(folder, what, one):
        if path.suffix.lower() not in suffixes:
            continue
        if path.is_symlink():
            skipped.append(f"{path.name}: a link (mnm copies only files that are there)")
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            skipped.append(f"{path.name}: over 2 GB")
            continue
        found.append(read_recording(path))
        if len(found) > MAX_FILES:
            raise TrainDataError(
                f"{folder} holds over {MAX_FILES} {what}: train from a folder of {one}."
            )
    return found, skipped


def walk(folder: Path, what: str = "recordings", one: str = "one voice") -> list[Path]:
    """Every entry under `folder`, in name order, given up on past `MAX_SEEN` (lazily: a
    folder like `~` or `/` is refused in a moment, never walked whole)."""
    seen: list[Path] = []
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs.sort()
        for name in sorted(files) + [d for d in dirs if (Path(root) / d).is_symlink()]:
            seen.append(Path(root) / name)
            if len(seen) > MAX_SEEN:
                raise TrainDataError(
                    f"{folder} holds over {MAX_SEEN} files: train from a folder of {one}'s {what}."
                )
    return sorted(seen)


def _warnings(found: list[Recording], rate: int) -> list[str]:
    said: list[str] = []
    minutes = sum(r.seconds or 0.0 for r in found) / 60
    if minutes < MIN_MINUTES:
        said.append(
            f"{minutes:.1f} minutes of recordings: RVC voices are usually trained on "
            f"{MIN_MINUTES:.0f} to {MAX_MINUTES:.0f} minutes of the voice alone. It trains, "
            "and sounds less like them."
        )
    elif minutes > MAX_MINUTES:
        said.append(
            f"{minutes:.0f} minutes of recordings: past {MAX_MINUTES:.0f} minutes a voice "
            "rarely gets better, and every epoch takes longer."
        )
    lossy = [r.path.name for r in found if r.lossy]
    if lossy:
        said.append(
            f"{len(lossy)} lossy file{'s' * (len(lossy) != 1)} ({lossy[0]}…): WAV or FLAC keep "
            "what the voice sounds like; MP3 and Ogg train with their artefacts."
        )
    stereo = [r.path.name for r in found if (r.channels or 1) > 1]
    if stereo:
        said.append(
            f"{len(stereo)} stereo file{'s' * (len(stereo) != 1)}: they are mixed to mono, so "
            "a recording of the voice alone and dry trains best."
        )
    low = [r.path.name for r in found if r.rate is not None and r.rate < rate]
    if low:
        said.append(
            f"{len(low)} file{'s' * (len(low) != 1)} below {rate // 1000} kHz ({low[0]}): "
            "the voice trains at the rate asked, without the detail those files lack."
        )
    return said


def copied_names(found: Survey, prefix: str = "rec") -> list[str]:
    """The plain names `copy_in` gives each file, in order."""
    return [
        f"{prefix}-{number:04d}{recording.path.suffix.lower()}"
        for number, recording in enumerate(found.recordings, start=1)
    ]


def copy_in(
    found: Survey, into: Path, prefix: str = "rec", role: str = "recording"
) -> list[InputRecord]:
    """Each file copied into `into` under a plain name (`rec-0001.wav`, `song-0001.flac`),
    hashed as it is copied; the records keep the user's own names."""
    into.mkdir(parents=True, exist_ok=True)
    records: list[InputRecord] = []
    for name, recording in zip(copied_names(found, prefix), found.recordings, strict=True):
        target = into / name
        digest = hashlib.sha256()
        with recording.path.open("rb") as source, target.open("wb") as sink:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
                sink.write(block)
        shutil.copystat(recording.path, target)
        records.append(
            InputRecord(
                role=role,
                name=recording.path.name,
                sha256=digest.hexdigest(),
                stage="train",
            )
        )
    return records
