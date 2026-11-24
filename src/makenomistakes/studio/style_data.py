"""A style's songs: the folder surveyed (`train_data.gather`), each song's lyrics from a
sidecar the user may put beside it, and the `dataset.json` ACE-Step's trainer reads.

The guidance read on 2026-11-02 (ACE-Step 1.5's `docs/sidestep/Dataset Preparation.md` and its
LoRA tutorial): at least five songs, 30 seconds to 4 minutes each (longer ones are cut at
`--max-seconds`), WAV or FLAC kept best; a caption describing the style and a trigger word the
training puts in front of it, which a song's style line carries afterwards to call the style
up. A song's lyrics come from `<song>.txt` or `<song>.lyrics.txt` beside it; without one the
song trains as instrumental, which is what upstream assumes too.
"""

from __future__ import annotations

import json
from pathlib import Path

from makenomistakes.studio.train_data import Recording, Survey, TrainDataError, gather

SONG_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")
LOSSY = frozenset({".mp3", ".ogg", ".opus", ".m4a"})
MIN_SONGS = 5
SHORT_SECONDS = 30.0
DEFAULT_MAX_SECONDS = 240.0  # upstream's `--max-duration`
INSTRUMENTAL = "[Instrumental]"
MAX_LYRICS_BYTES = 16_384
SONGS = "songs"  # the songs' folder inside the job folder
DATASET_JSON = "dataset.json"


def survey_songs(folder: Path, max_seconds: float = DEFAULT_MAX_SECONDS) -> Survey:
    """Every song under `folder`, each read where its header says its length; the warnings."""
    found, skipped = gather(folder, SONG_SUFFIXES, "songs", "one style")
    if not found:
        raise TrainDataError(
            f"{folder} holds no songs mnm trains a style from ({', '.join(SONG_SUFFIXES)})."
        )
    return Survey(found, skipped, _warnings(found, max_seconds), noun="song")


def _warnings(found: list[Recording], max_seconds: float) -> list[str]:
    said: list[str] = []
    if len(found) < MIN_SONGS:
        said.append(
            f"{len(found)} song{'s' * (len(found) != 1)}: a style is usually taught from "
            f"{MIN_SONGS} or more (10 to 20 is the community's figure). It trains, and the style "
            "comes across less."
        )
    long = [r.path.name for r in found if r.seconds is not None and r.seconds > max_seconds]
    if long:
        said.append(
            f"{len(long)} song{'s' * (len(long) != 1)} over {max_seconds:.0f} s ({long[0]}): "
            f"each is cut at {max_seconds:.0f} s (`--max-seconds` moves it; longer costs the "
            "card more)."
        )
    short = [r.path.name for r in found if r.seconds is not None and r.seconds < SHORT_SECONDS]
    if short:
        said.append(
            f"{len(short)} song{'s' * (len(short) != 1)} under {SHORT_SECONDS:.0f} s "
            f"({short[0]}): a song teaches its arrangement from 30 s up."
        )
    lossy = [r.path.name for r in found if r.path.suffix.lower() in LOSSY]
    if lossy:
        said.append(
            f"{len(lossy)} lossy file{'s' * (len(lossy) != 1)} ({lossy[0]}…): the style "
            "learns their artefacts too; WAV or FLAC keep the sound."
        )
    return said


def lyrics_for(song: Path) -> str:
    """The song's lyrics from `<song>.lyrics.txt` or `<song>.txt` beside it; instrumental
    without one."""
    for sidecar in (song.with_suffix(".lyrics.txt"), song.with_suffix(".txt")):
        if sidecar.is_file() and sidecar.stat().st_size <= MAX_LYRICS_BYTES:
            text = sidecar.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
    return INSTRUMENTAL


def dataset(found: Survey, copied: list[str], caption: str, trigger: str) -> dict[str, object]:
    """`dataset.json`: each copied song (`songs/song-0001.wav`) with the style's caption and
    its own lyrics; the trigger word put in front of every caption (`tag_position`)."""
    samples = []
    for recording, name in zip(found.recordings, copied, strict=True):
        lyrics = lyrics_for(recording.path)
        samples.append(
            {
                "audio_path": f"{SONGS}/{name}",
                "filename": name,
                "caption": caption,
                "lyrics": lyrics,
                "custom_tag": trigger,
                "is_instrumental": lyrics == INSTRUMENTAL,
            }
        )
    return {
        "metadata": {"custom_tag": trigger, "tag_position": "prepend", "genre_ratio": 0},
        "samples": samples,
    }


def write_dataset(job: Path, body: dict[str, object]) -> None:
    (job / DATASET_JSON).write_text(json.dumps(body, indent=2), encoding="utf-8")
