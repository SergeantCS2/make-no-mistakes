"""A song stage played by a ComfyUI workflow (a stage moves between hosts as a data row): the
user picks `mnm studio stage song fast=workflow:<name>`, and the Song pipeline sings each
candidate through that imported workflow (ComfyUI's own ACE-Step 1.5 template, YuE2's, a pack's)
instead of an audio.cpp instance.

The workflow's form says which fields hold a song's settings (`comfy.form.music_keys`: its
style or tags, its lyrics, its length); the candidate's style line, lyrics and length are set
on every field that holds each, its seed is the form's own (random each run). The workflow's
run is the candidate's child run: its audio is turned into the house format's `song.wav`
beside what ComfyUI saved, so stems, the score and the export read it as any candidate.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

from makenomistakes.audio_format import Conversion, HouseFormat, HouseFormatError
from makenomistakes.comfy.form import music_keys
from makenomistakes.comfy.report import analyze
from makenomistakes.comfy.run import ComfyRunError, ComfyRunner
from makenomistakes.comfy.store import WorkflowStoreError
from makenomistakes.pipeline.song import Lyrics, SongInputs
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.run_parts import ModelCredit
from makenomistakes.runs import RunError, RunRecord, RunStore

AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".opus", ".ogg", ".m4a")
NEEDED = ("style", "lyrics")  # what a song workflow must take to sing a candidate


def song_keys(comfy: ComfyRunner | None, name: str) -> dict[str, list[str]]:
    """The workflow's song fields by role; refused, with what it lacks, when it cannot sing a
    candidate (no style or no lyrics field)."""
    if comfy is None:
        raise PipelineError("ComfyUI workflows are not available here.")
    try:
        _prompt, fields = comfy.prompt(name)
    except ComfyRunError as exc:
        raise PipelineError(str(exc)) from exc
    keys = music_keys(fields)
    missing = [role for role in NEEDED if role not in keys]
    if missing:
        raise PipelineError(
            f"The workflow {name} has no {' and no '.join(missing)} field, so it cannot sing a "
            "song's candidates (`mnm comfyui workflow fields "
            f"{name}` lists what it takes)."
        )
    return keys


def make_candidate(  # noqa: PLR0913 - the candidate's parts beside the seams
    comfy: ComfyRunner | None,
    store: RunStore,
    house: HouseFormat,
    parent: RunRecord,
    name: str,
    *,
    lyrics: Lyrics,
    inputs: SongInputs,
    made: Callable[[RunRecord], None] = lambda _child: None,
) -> RunRecord:
    """One candidate from the workflow: its run as the song's child, `song.wav` beside the
    audio it saved. A failure fails the child and raises a StepError."""
    keys = song_keys(comfy, name)
    assert comfy is not None  # `song_keys` refused otherwise
    values = {key: lyrics.tags for key in keys["style"]}
    values |= {key: lyrics.sung() for key in keys["lyrics"]}
    values |= {key: f"{inputs.seconds:g}" for key in keys.get("seconds", [])}
    try:
        child = comfy.new(name, values)
    except ComfyRunError as exc:
        raise StepError(str(exc)) from exc
    child.parent, child.role, child.prompt = parent.id, "candidate", lyrics.tags
    child.models = credits(comfy, name)
    store.save(child)
    made(child)
    try:
        done = comfy.run(child, values, {})
        return _house_song(store, house, done, name)
    except (ComfyRunError, HouseFormatError, StepError, OSError) as exc:
        with contextlib.suppress(RunError):
            store.fail(store.get(child.id), str(exc))
        raise StepError(str(exc)) from exc


def credits(comfy: ComfyRunner, name: str) -> list[ModelCredit]:
    """The workflow that made a candidate: one credit per model file it loads, each with
    the packs it uses, so the disclosure names the files whose licences apply."""
    try:
        report = analyze(comfy.workflows.workflow(name), comfy.info())
    except (WorkflowStoreError, ComfyRunError, OSError, ValueError):
        return [ModelCredit(workflow=name)]
    packs = [p.pack + (f" {p.version}" if p.version else "") for p in report.packs]
    files: list[str | None] = [m.name for m in report.models]
    return [ModelCredit(workflow=name, file=file, packs=packs) for file in files or [None]]


def _house_song(store: RunStore, house: HouseFormat, done: RunRecord, name: str) -> RunRecord:
    """The first audio file the workflow saved, as the house format's `song.wav`."""
    saved = [f for f in done.files if Path(f.name).suffix.lower() in AUDIO_SUFFIXES]
    if not saved:
        raise StepError(f"The workflow {name} saved no audio (it made {done.answer or 'nothing'}).")
    folder = store.folder(done)
    house.convert(folder, Conversion(source=saved[0].name, target=CANDIDATE_FILE))
    store.keep(done, folder / CANDIDATE_FILE)
    store.save(done)
    return done
