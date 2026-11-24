"""The Song's steps on the studio's tools: score each candidate and map the picked
take's structure, through `mnm/studio-tools` (`studio/tools.py`).

The tools image is optional, as ComfyUI is: without it a score step is skipped and the run
says once how to get the scores (`mnm studio install`, then `mnm song redo <run> score`),
and the structure step is skipped with the REAPER project keeping 120 BPM. A scorer that
fails on one candidate is that step's error, never the run's: the user still picks by ear.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.media_bytes import wav_file_seconds
from makenomistakes.pipeline import song
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.runs import (
    RunError,
    RunKind,
    RunRecord,
    RunState,
    RunStore,
    StepRecord,
    StepState,
)
from makenomistakes.studio import scores
from makenomistakes.studio.keys import Carried
from makenomistakes.studio.tools import StudioTools
from makenomistakes.studio.tools_answers import StructureMap, ToolsError, ToolsMissingError


def no_scorer_note(run_id: str) -> str:
    return (
        "The candidates are not scored: the studio tools are not installed. `mnm studio "
        f"install` builds the scorer; then `mnm song redo {run_id} score` scores them."
    )


def no_structure_note(run_id: str) -> str:
    return (
        "No structure map: the studio tools are not installed, so the REAPER project keeps "
        f"REAPER's 120 BPM and places no markers. After `mnm studio install`, `mnm song redo "
        f"{run_id} structure` maps it."
    )


def _noted(run: RunRecord, note: str) -> None:
    if note not in run.notes:
        run.notes.append(note)


def score(
    store: RunStore,
    service: InstanceService,
    tools: StudioTools,
    run: RunRecord,
    step: StepRecord,
    child: RunRecord | None,
) -> None:
    """Score one candidate: the child run keeps the score, the song instance a reading."""
    if child is None or child.state is not RunState.DONE or child.files_removed:
        step.state, step.text = StepState.SKIPPED, "no candidate to score"
        return
    folder = store.folder(child)
    try:
        answer = tools.score(folder, CANDIDATE_FILE)
    except ToolsMissingError:
        step.state, step.text = StepState.SKIPPED, "not scored (no studio tools)"
        _noted(run, no_scorer_note(run.id))
        return
    except ToolsError as exc:
        step.state, step.text, step.error = StepState.SKIPPED, "not scored", str(exc)
        return
    child.score, child.scorer, child.score_axes = answer.score, answer.scorer, answer.axes
    store.save(child)
    run.notes = [n for n in run.notes if n != no_scorer_note(run.id)]
    step.text = answer.words()
    if child.kind is not RunKind.INSTANCE:  # a ComfyUI workflow's: no instance's reading
        return
    try:
        recipe = service.get(child.target)
    except InstanceError:
        return  # the song instance is gone: the child keeps its score, no reading is kept
    seconds = wav_file_seconds(folder / CANDIDATE_FILE)
    scores.record(service, recipe, answer, seconds, child.id)


def best(store: RunStore, run: RunRecord) -> tuple[int, float | None]:
    """The index of the highest-scored made candidate (its score), else the first made one."""
    made: list[tuple[int, float | None]] = []
    for step in sorted((s for s in run.steps if s.stage == song.SONG), key=lambda s: s.index):
        if not step.text:
            continue
        try:
            child = store.get(step.text)
        except RunError:  # a candidate gone is no candidate
            continue
        if child.state is RunState.DONE:
            made.append((step.index, child.score))
    if not made:
        return 0, None
    scored = [m for m in made if m[1] is not None]
    return max(scored, key=lambda m: m[1] or 0.0) if scored else made[0]


def structure(
    tools: StudioTools,
    run: RunRecord,
    step: StepRecord,
    take: Path,
    folder: Path,
    carried: Carried | None = None,
) -> None:
    """The take's structure map, written to the run's folder (with the key the edits that made
    the take handed on, when they did: `song_retime.carried`); the step skipped (and the run's
    note) without the tools image, or with the tool's error when it failed: the stems and the
    export never wait on it."""
    (folder / song.STRUCTURE_FILE).unlink(missing_ok=True)  # never another take's
    step.file = None
    try:
        found = tools.structure(take.parent, take.name)
    except ToolsMissingError:
        step.state, step.text = StepState.SKIPPED, "no structure map (no studio tools)"
        _noted(run, no_structure_note(run.id))
        return
    except ToolsError as exc:
        step.state, step.text, step.error = StepState.SKIPPED, "no structure map", str(exc)
        return
    run.notes = [n for n in run.notes if n != no_structure_note(run.id)]
    if carried is not None:
        update = {"key": carried.key, "key_clarity": carried.clarity, "key_carried": True}
        found = found.model_copy(update=update)
    path = folder / song.STRUCTURE_FILE
    path.write_text(found.model_dump_json(indent=1) + "\n", "utf-8")
    step.file, step.text = path.name, found.words()


def forget_structure(store: RunStore, run: RunRecord, step: StepRecord) -> None:
    """A structure step made again starts without the old take's map (its file, its entry
    in the run's files, the step's file and words)."""
    (store.folder(run) / song.STRUCTURE_FILE).unlink(missing_ok=True)
    run.files = [f for f in run.files if f.name != song.STRUCTURE_FILE]
    step.file, step.text = None, ""


def read_structure(folder: Path) -> StructureMap | None:
    """The run's structure map, if it has one that reads."""
    try:
        return StructureMap.model_validate_json(
            (folder / song.STRUCTURE_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
