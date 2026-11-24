"""The export's extras: the stems' MIDI files in `midi/` with a muted MIDI track each in the
REAPER project, and the master in use, which the export's `mix.*` are made from as it was made
(the reference set its loudness)."""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel

from makenomistakes.pipeline import song, song_master, song_midi
from makenomistakes.runs import RunRecord, RunStore
from makenomistakes.studio import reaper

MIDI = "midi"  # the export's folder of MIDI files


class MasterLine(BaseModel):
    """The master the export's mix is: its run, the reference by name and hash."""

    run: str
    reference: str
    sha256: str
    made: str  # its run's line: `the mix mastered to ref.wav (Matchering 2.0.6)`


def midi_tracks(
    folder: Path, run: RunRecord, length: float
) -> tuple[list[reaper.Track], list[str]]:
    """Each MIDI file the song has, copied to `export/midi/<stem>.mid`, and its REAPER track
    (muted: it doubles its stem's audio); the copied names for the provenance."""
    tracks, copied = [], []
    for name in song_midi.midi_files([f.name for f in run.outputs]):
        if not (folder / name).is_file():
            continue
        stem = name.removeprefix(song_midi.PREFIX).removesuffix(".mid")
        target = f"{MIDI}/{stem}.mid"
        (folder / song.EXPORT_FOLDER / MIDI).mkdir(exist_ok=True)
        shutil.copyfile(folder / name, folder / song.EXPORT_FOLDER / target)
        tracks.append(
            reaper.Track(
                name=f"{stem} (MIDI)", file=target, length=round(length, 6), mute=True,
                guid=reaper.track_guid(run.id, f"midi:{stem}"),
            )
        )  # fmt: skip
        copied.append(target)
    return tracks, copied


def master_line(store: RunStore, run: RunRecord) -> tuple[Path, MasterLine] | None:
    """The master in use, as the export's mix source and its provenance line."""
    found = song_master.source(store, run)
    if found is None:
        return None
    path, child = found
    reference = next((i for i in child.inputs_recorded if i.role == song_master.REFERENCE), None)
    line = MasterLine(
        run=child.id,
        reference=song_master.reference_of(child),
        sha256=reference.sha256 if reference else "",
        made=child.answer,
    )
    return path, line
