"""MIDI out: each of the song's stems but the drums as a MIDI file, read by Basic
Pitch (its ONNX model) in the studio tools image after the voice stage, so the vocals' notes
are the ones sung last. On whenever the tools are installed; without them the step is
skipped and the run says once how to get it.

The files are `midi-<stem>.mid` in the song's folder; the export copies them into `midi/`
and its REAPER project lists each as a muted MIDI track. Basic Pitch hears pitched notes: a
drum stem would come back as noise, so it is left out.

The user may pick a MuScriptor or SheetSage2 instance on audio.cpp to play the stage instead
(`mnm studio stage midi fast=<instance>`): multi-instrument and stronger on dense parts, but
CC-BY-NC-4.0, so never the default; its credit carries the licence into the disclosure.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

import httpx

from makenomistakes.engine_answers import MalformedAnswerError
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.make import credit
from makenomistakes.media_bytes import wav_file_seconds
from makenomistakes.music_api import TaskInput, TaskRequest, parse_answer
from makenomistakes.pipeline import song, song_credits, take_audio
from makenomistakes.pipeline.song_steps import post_task
from makenomistakes.pipeline.steps import StepError
from makenomistakes.recipes.model import Recipe
from makenomistakes.runs import RunRecord, RunStore, StepRecord, StepState
from makenomistakes.studio import stages
from makenomistakes.studio.edits_retime import DRUMS
from makenomistakes.studio.stages import TierName
from makenomistakes.studio.tools import StudioTools
from makenomistakes.studio.tools_answers import ToolsError, ToolsMissingError

PREFIX = "midi-"


def midi_name(stem_file: str) -> str:
    """`midi-vocals.mid` for `stem-vocals.wav` (or the vocals as sung)."""
    stem = "vocals" if take_audio.is_vocals(stem_file) else stem_file.removeprefix("stem-")
    return f"{PREFIX}{stem.removesuffix('.wav')}.mid"


def midi_files(names: list[str]) -> list[str]:
    """The run's MIDI files, by name, sorted."""
    return sorted(n for n in names if n.startswith(PREFIX) and n.endswith(".mid"))


def no_midi_note(run_id: str) -> str:
    return (
        "No MIDI: the studio tools are not installed. `mnm studio install` builds Basic Pitch; "
        f"then `mnm song redo {run_id} midi` reads each stem's notes."
    )


def picked_name(state_dir: Path, tier: TierName) -> str | None:
    """The instance the user picked for the stage's tier, by name (None: the tools)."""
    try:
        return stages.resolve(song.MIDI, tier, state_dir).instance
    except stages.StudioError:
        return None


def transcriber(service: InstanceService, state_dir: Path, tier: TierName) -> Recipe | None:
    """The user's MuScriptor or SheetSage2 instance for the stage's tier, None for the tools;
    a pick that is gone or of another kind refused, saying how to go back."""
    try:
        name = stages.resolve(song.MIDI, tier, state_dir).instance
    except stages.StudioError as exc:  # the picks file unreadable: the stage says so
        raise StepError(str(exc)) from exc
    if name is None:
        return None
    back = f"`mnm studio stage midi {tier}=` goes back to the studio tools"
    try:
        recipe = service.get(name)
    except InstanceError:
        raise StepError(f"The MIDI stage's {tier} tier is {name}, which is gone: {back}.") from None
    family = recipe.args.family if isinstance(recipe.args, AudioCppArgs) else None
    refused = stages.pick_refusal(song.MIDI, name, family, recipe.modality)
    if refused is not None:
        raise StepError(f"{refused}: {back}.")
    return recipe


def make(
    store: RunStore,
    tools: StudioTools,
    run: RunRecord,
    step: StepRecord,
    *,
    pick: Callable[[], Recipe | None],
    http: httpx.Client,
) -> None:
    """The stems' notes as MIDI files kept with the song, by the user's picked instance or
    else the studio tools; the step skipped with the reason (and, without the tools image, the
    run's note) when it cannot be made: the mix and the export never wait on it."""
    forget(store, run, step)
    song_credits.drop(run, [step])  # a MuScriptor that read them before is no longer credited
    pitched = [s for s in song.stem_files([f.name for f in run.outputs]) if s != DRUMS]
    if not pitched:
        step.state, step.text = StepState.SKIPPED, "no pitched stems to read"
        return
    targets = {stem: midi_name(stem) for stem in pitched}
    try:
        picked = pick()  # the user's MuScriptor (a pick gone or of another kind: skipped)
        if picked is not None:
            by = _transcribed(store.folder(run), targets, picked, http)
            song_credits.give(run, step, credit(picked))  # it goes when the MIDI does
        else:
            by = tools.midi(store.folder(run), targets).model
    except ToolsMissingError:
        step.state, step.text = StepState.SKIPPED, "no MIDI (no studio tools)"
        if no_midi_note(run.id) not in run.notes:
            run.notes.append(no_midi_note(run.id))
        return
    except (ToolsError, StepError) as exc:
        for target in targets.values():  # a stem read before the failure: never half a set
            (store.folder(run) / target).unlink(missing_ok=True)
        step.state, step.text, step.error = StepState.SKIPPED, "no MIDI", str(exc)
        return
    run.notes = [n for n in run.notes if n != no_midi_note(run.id)]
    notes = 0
    for target in targets.values():
        path = store.folder(run) / target
        store.keep(run, path)
        notes += notes_in(path.read_bytes())
    step.file = targets[pitched[0]]
    step.text = f"{notes} notes from {len(targets)} stems ({by})"


def _transcribed(folder: Path, targets: dict[str, str], recipe: Recipe, http: httpx.Client) -> str:
    """Each pitched stem sent to the picked instance, its MIDI artifact written under the name
    mnm gives it; who read them, for the step's line."""
    for stem, target in targets.items():
        path = folder / stem
        request = TaskRequest(
            model=recipe.served_name,
            request=TaskInput(
                audio_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
                options={"output_format": "midi"},
            ),
        )
        raw = post_task(recipe, http, request.body(), wav_file_seconds(path) or 0.0)
        try:
            data = next((d for a in parse_answer(raw).artifacts if (d := a.midi())), None)
        except MalformedAnswerError as exc:
            raise StepError(f"{recipe.name}: {exc}.") from exc
        if data is None:
            raise StepError(f"{recipe.name} answered {stem} with no MIDI.")
        (folder / target).write_bytes(data)
    return recipe.name


def notes_in(data: bytes) -> int:
    """The notes a standard MIDI file holds (its note-ons with a velocity), read chunk by chunk
    with running status: the line says what the file holds, never what an engine's events
    list counted (MuScriptor's events have said 2,733 for files of 2,513 notes)."""
    count, at = 0, 0 if data[:4] == b"MThd" else len(data)
    while at + 8 <= len(data):  # every chunk by its own length, the header (MThd) too
        size = int.from_bytes(data[at + 4 : at + 8], "big")
        if data[at : at + 4] == b"MTrk":
            count += _track_notes(data[at + 8 : at + 8 + size])
        at += 8 + size
    return count


def _track_notes(track: bytes) -> int:
    count, at, status = 0, 0, 0
    while at < len(track):
        while at < len(track) and track[at] & 0x80:  # the delta time, a variable length
            at += 1
        at += 1
        if at >= len(track):
            break
        if track[at] & 0x80:
            status, at = track[at], at + 1
        if status in (0xFF, 0xF0, 0xF7):  # a meta or sysex event: its length, then skip
            at += 1 if status == 0xFF else 0
            length = 0
            while at < len(track):
                length, at = (length << 7) | (track[at] & 0x7F), at + 1
                if not track[at - 1] & 0x80:
                    break
            at += length
            status = 0  # a meta or sysex event cancels running status (the SMF spec)
            continue
        kind = status & 0xF0
        width = 1 if kind in (0xC0, 0xD0) else 2
        if kind == 0x90 and at + 1 < len(track) and track[at + 1] > 0:
            count += 1
        at += width
    return count


def forget(store: RunStore, run: RunRecord, step: StepRecord | None = None) -> None:
    """A MIDI step made again starts without the old stems' files (the instance that read
    them loses its credit with the item's, `song_credits.drop`, so an unpicked MuScriptor's
    licence never stays in the disclosure)."""
    folder = store.folder(run)
    for name in midi_files([f.name for f in run.files]):
        (folder / name).unlink(missing_ok=True)
    run.files = [
        f for f in run.files if not (f.name.startswith(PREFIX) and f.name.endswith(".mid"))
    ]
    if step is not None:
        step.file, step.text = None, ""
