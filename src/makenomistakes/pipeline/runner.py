"""Run a Short film: one item at a time, the run record written after each, so a
failure or a restart loses at most the item in flight.

`advance` runs every item not done yet, stage by stage, and stops at a review point the user
has not passed (unless the run goes straight through), at the end, or at a failure. The user
then continues, edits the shots (at the shot list only), or redoes one item, which resets the
items made from it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.comfy.image import ComfyError
from makenomistakes.comfy.room import park_for_comfy
from makenomistakes.comfy.run import ComfyRunner
from makenomistakes.comfy.store import WorkflowStore
from makenomistakes.engine_args import SdCppArgs
from makenomistakes.hardware.probe import SubprocessRunner
from makenomistakes.harness_media import image_size
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.pipeline import short_film as film
from makenomistakes.pipeline.clips import WorkflowClips
from makenomistakes.pipeline.short_film import FilmInputs, ShotList
from makenomistakes.pipeline.stage_runner import Report, StageRunner
from makenomistakes.pipeline.steps import (
    DockerStitcher,
    Part,
    PipelineError,
    StepError,
    Stitcher,
    ask_shotlist,
    make_media,
)
from makenomistakes.progress_words import Whole
from makenomistakes.recipes import Recipe
from makenomistakes.runs import (
    RunError,
    RunKind,
    RunRecord,
    RunState,
    RunStore,
    StepRecord,
    StepState,
)
from makenomistakes.try_run import http_client

__all__ = ["FilmRunner", "PipelineError", "comfy_runner_for", "film_runner", "instances_for"]

PER_SHOT = (film.STILLS, film.LINES, film.CLIPS)
# What redoing an item of a stage makes stale: the same shot's later items, and the film.
MADE_FROM = {film.STILLS: (film.CLIPS,), film.LINES: (), film.CLIPS: ()}


SONG = "song"  # the Song pipeline's runs (`pipeline/song.py`), which this runner never takes


class FilmRunner(StageRunner):
    name = film.NAME
    stages = film.STAGES
    labels = film.STAGE_LABELS

    def __init__(
        self,
        service: InstanceService,
        store: RunStore,
        http: Callable[[], httpx.Client],
        stitcher: Stitcher,
        comfy: ComfyRunner | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(store, http, clock)
        self.service = service
        self.stitcher = stitcher
        self.comfy = comfy  # a film's clips from a ComfyUI workflow
        self.clips = WorkflowClips(comfy, store)

    # -- starting --------------------------------------------------------------------------

    def start(self, inputs: FilmInputs) -> RunRecord:
        """A new run, checked and saved; nothing is sent until `advance`."""
        for role, name in self._members(inputs).items():
            if role == "video" and film.workflow_of(name) is not None:
                self.clips.fields(name)  # refused now when it cannot make the clips
                continue
            recipe = self._recipe(name)
            if role == "video" and (inputs.clip_size or inputs.clip_frames):
                raise PipelineError(
                    f"{name}'s clips are its own shape ({self._shape_words(inputs)}): "
                    f"`mnm set {name} width=… height=… frames=…` changes it. --clip-size and "
                    "--clip-frames are for a comfy:<workflow>'s clips."
                )
            if recipe.modality not in film.ROLES[role]:
                raise PipelineError(
                    f"{name} is {recipe.modality.a_label} model; the {role} role needs "
                    + " or ".join(sorted(m.label for m in film.ROLES[role]))
                    + "."
                )
        if film.workflow_of(inputs.video) is not None:  # kept, so a resume makes the same
            width, height, frames = inputs.clip_shape()
            inputs = inputs.model_copy(
                update={"clip_size": f"{width}x{height}", "clip_frames": frames}
            )
        try:
            run = self.store.new(RunKind.PIPELINE, film.NAME, inputs.idea, inputs.settings())
        except RunError as exc:
            raise PipelineError(str(exc)) from exc
        run.steps = [StepRecord(stage=film.SHOTLIST), StepRecord(stage=film.FILM)]
        self.store.save(run)
        return run

    @staticmethod
    def _members(inputs: FilmInputs) -> dict[str, str]:
        roles = {"chat": inputs.chat, "image": inputs.image, "video": inputs.video}
        return roles | ({"voice": inputs.voice} if inputs.voice else {})

    def _recipe(self, name: str) -> Recipe:
        try:
            return self.service.get(name)
        except InstanceError as exc:
            raise PipelineError(f"{exc} (the film's instances must exist: `mnm ls`)") from exc

    # -- running (the loop is `StageRunner.advance`) --------------------------------------

    def _ours(self, run: RunRecord) -> bool:
        return run.target != SONG  # every other pipeline run is a film's

    def _fresh(self) -> None:
        self.clips.forget()  # the workflow may have changed since the last pass

    def _context(self, run: RunRecord, report: Report) -> FilmInputs:
        stills = self.items(run, film.STILLS)
        return self.clips.kept_shape(run, stills, FilmInputs.from_settings(run.settings), report)

    def _skip(self, run: RunRecord, stage: str, context: FilmInputs, report: Report) -> bool:
        if stage == film.LINES and not context.voice:
            self._skip_lines(run, report)
            return True
        return False

    def _waits(self, run: RunRecord, stage: str, context: FilmInputs) -> bool:
        return stage in film.REVIEWS and not context.auto and stage not in run.reviewed

    def _which(self, run: RunRecord, step: StepRecord, context: FilmInputs) -> tuple[str, str]:
        which, _ = super()._which(run, step, context)
        shape = f" ({self._shape_words(context)})" if step.stage == film.CLIPS else ""
        return which, shape

    def _says_done(self, step: StepRecord) -> bool:
        return step.stage != film.SHOTLIST

    def _finish(self, run: RunRecord) -> RunRecord:
        shots = len(self.items(run, film.CLIPS))
        run.pid = None
        last = run.step(film.FILM, 0)
        name = last.file if last is not None and last.file else film.FILM_FILE
        made = [self.store.folder(run) / name]
        return self.store.finish(run, made, f"a {shots}-shot film: {name}", None)

    def _items(self, run: RunRecord, stage: str) -> list[StepRecord]:
        return self.items(run, stage)

    def _make(
        self, run: RunRecord, step: StepRecord, context: FilmInputs, http: httpx.Client
    ) -> None:
        inputs = context
        folder = self.store.folder(run)
        if step.stage == film.SHOTLIST:
            shots = ask_shotlist(self._recipe(inputs.chat), http, inputs)
            self._save_shots(run, shots)
            step.text = shots.title
            return
        shot = self.shots(run).shots[step.index] if step.stage in PER_SHOT else None
        match step.stage:
            case film.STILLS:
                assert shot is not None
                prompt = film.still_prompt(inputs, shot, *self._video_size(inputs))
                name = film.item_file(film.STILLS, step.index, ".png")
                path = make_media(self._recipe(inputs.image), http, folder, name, prompt)
                step.text = f"{shot.still_prompt}, {inputs.style}"
            case film.LINES:
                assert shot is not None
                if not inputs.voice or not shot.line:
                    step.state, step.file = StepState.SKIPPED, None
                    return
                name = film.item_file(film.LINES, step.index, ".wav")
                path = make_media(self._recipe(inputs.voice), http, folder, name, shot.line)
                step.text = shot.line
            case film.CLIPS:
                assert shot is not None
                still = self._file(run, film.STILLS, step.index)
                if film.workflow_of(inputs.video) is not None:
                    path = self.clips.make(run, step.index, inputs.video, shot, still)
                else:
                    name = film.item_file(film.CLIPS, step.index, ".webm")
                    video = self._recipe(inputs.video)
                    motion = shot.motion_prompt
                    path = make_media(video, http, folder, name, motion, still.read_bytes())
                step.text = shot.motion_prompt
            case _:  # the film
                parts = self._parts(run)
                path = self.stitcher.stitch(folder, parts, film.film_file(parts[0].clip))
        step.file = path.name
        self.store.keep(run, path)

    def _video_size(self, inputs: FilmInputs) -> tuple[int, int]:
        """The clips' size, which the stills are drawn at."""
        if film.workflow_of(inputs.video) is None:
            return image_size(self._recipe(inputs.video))
        return self.clips.size(inputs)

    def _shape_words(self, inputs: FilmInputs) -> str:
        """`832x480, 33 frames`: the clips' shape, as they are made."""
        if film.workflow_of(inputs.video) is None:
            recipe = self._recipe(inputs.video)
            width, height = image_size(recipe)
            frames = recipe.args.frames if isinstance(recipe.args, SdCppArgs) else None
            return f"{width}x{height}" + (f", {frames} frames" if frames else "")
        return self.clips.shape_words(inputs)

    def _skip_lines(self, run: RunRecord, report: Report) -> None:
        """No voice: every line is skipped, said once."""
        pending = [
            s
            for s in self._items(run, film.LINES)
            if s.state not in (StepState.DONE, StepState.SKIPPED)
        ]
        if not pending:
            return
        for step in pending:
            step.state, step.file = StepState.SKIPPED, None
        self.store.save(run)
        report(Whole("Lines: no voice given, skipped."))

    def _file(self, run: RunRecord, stage: str, index: int) -> Path:
        step = run.step(stage, index)
        if step is None or step.file is None:
            raise StepError(f"{film.STAGE_LABELS[stage]} {index + 1} is not made yet.")
        path = self.store.folder(run) / step.file
        if not path.is_file():
            raise StepError(f"{step.file} is gone from the run's folder: redo it.")
        return path

    def _parts(self, run: RunRecord) -> list[Part]:
        parts = []
        for clip in self._items(run, film.CLIPS):
            line = run.step(film.LINES, clip.index)
            if clip.file is None:
                raise StepError(f"Clip {clip.index + 1} is not made yet.")
            parts.append(Part(clip.file, line.file if line is not None else None))
        return parts

    # -- the shot list ---------------------------------------------------------------------

    def shots(self, run: RunRecord) -> ShotList:
        path = self.store.folder(run) / film.SHOTLIST_FILE
        try:
            return ShotList.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StepError("The shot list is missing or unreadable: redo the shot list.") from exc

    def _save_shots(self, run: RunRecord, shots: ShotList) -> None:
        path = self.store.folder(run) / film.SHOTLIST_FILE
        path.write_text(shots.model_dump_json(indent=2) + "\n", encoding="utf-8")
        self.store.keep(run, path)
        keep = [s for s in run.steps if s.stage in (film.SHOTLIST, film.FILM)]
        per_shot = [
            StepRecord(stage=stage, index=i) for stage in PER_SHOT for i in range(len(shots.shots))
        ]
        for step in keep:
            if step.stage == film.FILM:
                step.state, step.file = StepState.PENDING, None
        run.steps = keep[:1] + per_shot + keep[1:]

    def edit_shots(self, run_id: str, shots: ShotList) -> RunRecord:
        """New shots from the user, while the run waits at the shot list: read, checked and
        saved under the store's lock (so an edit beside a Go never undoes the other)."""
        with self.store.locked():
            run = self._get(run_id)
            if run.state is not RunState.WAITING or run.waiting_at != film.SHOTLIST:
                raise PipelineError(
                    "The shots can be edited only while the run waits at the shot list."
                )
            self._save_shots(run, shots)
            self.store.save(run)
        return run

    # -- the user's moves (`StageRunner.proceed`, `redo`) ---------------------------------

    def _stale(self, run: RunRecord, step: StepRecord) -> list[StepRecord]:
        """A redone shot list makes everything stale; any other item, the same shot's later
        items and the film."""
        if step.stage == film.SHOTLIST:
            run.reviewed = []
            return list(run.steps)
        stale = [step]
        stale += [
            s
            for s in run.steps
            if s.stage in MADE_FROM.get(step.stage, ()) and s.index == step.index
        ]
        return stale + [s for s in run.steps if s.stage == film.FILM]


def instances_for(service: InstanceService, workflows: Sequence[str] = ()) -> dict[str, list[str]]:
    """The instances each role can take, by name (the web's pickers); the imported ComfyUI
    workflows are offered for the clips as `comfy:<name>`."""
    recipes = service.all()
    found = {
        role: [r.name for r in recipes if r.modality in types] for role, types in film.ROLES.items()
    }
    found["video"] += [f"{film.WORKFLOW}{name}" for name in workflows]
    return found


def comfy_runner_for(service: InstanceService) -> ComfyRunner:
    """ComfyUI's runner for this box, recording into the box's run store."""

    settings = service.settings
    store = RunStore.for_settings(settings)
    workflows = WorkflowStore(settings.expanded_state_dir())
    return ComfyRunner(ComfyEngine(settings), workflows, store, park_for_comfy(service))


def film_runner(service: InstanceService, comfy: ComfyRunner | None = None) -> FilmRunner:
    """The runner for this box: the router through `mnm try`'s client, ffmpeg in Docker, and
    ComfyUI for clips made by a workflow."""

    settings = service.settings
    comfy = comfy or comfy_runner_for(service)
    store = comfy.runs

    def comfy_image() -> list[str]:  # mnm's ComfyUI image carries ffmpeg too
        try:
            ledger = comfy.engine.ledger()
        except ComfyError:
            return []
        return [ledger.image] if ledger is not None else []

    stitcher = DockerStitcher(SubprocessRunner(), also=comfy_image)
    return FilmRunner(service, store, lambda: http_client(settings), stitcher, comfy)
