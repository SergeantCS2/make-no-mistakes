"""What kind of file a run keeps, by its name: the gallery's player
and the media type the web serves it with."""

from __future__ import annotations

import contextlib
import re
import secrets
import shutil
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath


class FileKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    OTHER = "other"


_KINDS = {
    ".png": FileKind.IMAGE,
    ".jpg": FileKind.IMAGE,
    ".jpeg": FileKind.IMAGE,
    ".webp": FileKind.IMAGE,
    ".gif": FileKind.IMAGE,
    ".webm": FileKind.VIDEO,
    ".mp4": FileKind.VIDEO,
    ".avi": FileKind.VIDEO,
    ".mkv": FileKind.VIDEO,
    ".wav": FileKind.AUDIO,
    ".mp3": FileKind.AUDIO,
    ".flac": FileKind.AUDIO,
    ".ogg": FileKind.AUDIO,
    ".md": FileKind.TEXT,
    ".txt": FileKind.TEXT,
    ".json": FileKind.TEXT,
}
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".md": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
}


def file_kind(name: str) -> FileKind:
    return _KINDS.get(PurePosixPath(name).suffix.lower(), FileKind.OTHER)


_SLUG = re.compile(r"[^a-z0-9-]+")
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,120}$")


def new_id(target: str, now: datetime) -> str:
    slug = _SLUG.sub("-", target.lower()).strip("-")[:40] or "run"
    return f"{now:%Y%m%d-%H%M%S}-{slug}-{secrets.token_hex(2)}"


def folder_of(run_id: str) -> str:
    """A run's folder under the outputs, `<day>/<id>`: the day is the id's own first eight
    digits (`new_id` names both from one time), so a record no mnm reads still names it."""
    return f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]}/{run_id}"


def safe_file_name(name: str, fallback: str) -> str:
    """An uploaded file's name as stored in a run: its last part, plain characters only."""
    base = PurePosixPath(name.replace("\\", "/")).name
    return base if _FILE_NAME.match(base) and not base.startswith(".") else fallback


def remove_folder(folder: Path) -> None:
    """A run's folder (a link to one only unlinked), then its day folder once empty."""
    if folder.is_symlink():
        folder.unlink()
    elif folder.is_dir():
        shutil.rmtree(folder)
    with contextlib.suppress(OSError):
        folder.parent.rmdir()  # the date folder, once its last run went
