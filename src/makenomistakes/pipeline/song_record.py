"""What a song run's record and folder hold, read for its steps and pages: the lyrics, the
kept recording and the export's options as the last export set them. The take in use is
`song_takes`'."""

from __future__ import annotations

from makenomistakes.pipeline import song
from makenomistakes.pipeline.song import Lyrics, SongError
from makenomistakes.pipeline.song_steps import Recording
from makenomistakes.pipeline.steps import StepError
from makenomistakes.runs import RunRecord, RunStore
from makenomistakes.studio import export_mix
from makenomistakes.studio.export import ExportOptions

RECORDING = "recording"  # the kept recording's role (`input-recording-<name>`)


def lyrics(store: RunStore, run: RunRecord) -> Lyrics:
    try:
        return song.read_lyrics(store.folder(run))
    except SongError as exc:
        raise StepError("The lyrics are missing or unreadable: redo the lyrics.") from exc


def save_lyrics(store: RunStore, run: RunRecord, written: Lyrics) -> None:
    path = store.folder(run) / song.LYRICS_FILE
    path.write_text(written.model_dump_json(indent=2) + "\n", encoding="utf-8")
    store.keep(run, path)


def recording(store: RunStore, run: RunRecord) -> Recording | None:
    kept = next((i for i in run.inputs_recorded if i.role == RECORDING), None)
    use = run.settings.get("recording")
    if kept is None or use is None:
        return None
    path = store.folder(run) / kept.name
    if not path.is_file():
        raise StepError(f"The recording {kept.name} is gone from the run's folder.")
    return Recording(kept.name, path, use, kept.sha256)


def export_options(run: RunRecord) -> ExportOptions:
    """What the next export the pipeline makes is asked for: the last export's MP3, release
    mark and loudness line as the user set them (`--lufs`/`--true-peak` and off kept, never
    back at -14 / -1 after an edit or a pick)."""
    return ExportOptions(
        mp3=run.settings.get("export_mp3") == "true",
        release=run.settings.get("export_release") == "true",
        loudness=export_mix.line_of(run.settings),
    )
