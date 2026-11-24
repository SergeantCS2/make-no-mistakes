"""A Short film's clips from a ComfyUI workflow: the workflow's prompt and
picture fields, the clip shape set on its own width, height and frame fields, a clip made
and moved into the film's folder, and the shape a film started before mnm kept it resumes
at."""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Callable
from pathlib import Path

from makenomistakes.comfy.form import Field as FormField
from makenomistakes.comfy.form import Kind, Role
from makenomistakes.comfy.run import ComfyRunError, ComfyRunner
from makenomistakes.pipeline import short_film as film
from makenomistakes.pipeline.short_film import FilmInputs, Shot
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.progress_words import Whole
from makenomistakes.runs import RunError, RunRecord, RunStore, StepRecord, StepState
from makenomistakes.smoke_replies import TryError, png_size

Report = Callable[[str], None]
Fields = tuple[FormField, FormField | None, list[FormField]]


class WorkflowClips:
    def __init__(self, comfy: ComfyRunner | None, store: RunStore) -> None:
        self.comfy = comfy
        self.store = store
        # a clip workflow's fields, read once a pass (each still asks for the size)
        self._fields: dict[str, Fields] = {}

    def forget(self) -> None:
        """The workflow may have changed since the last pass: its fields are read again."""
        self._fields.clear()

    def fields(self, video: str) -> Fields:
        """The workflow's prompt field, its picture upload (an image-to-video one), and every
        field; refused when it has no prompt to take a shot's motion."""
        if video in self._fields:
            return self._fields[video]
        name = film.workflow_of(video) or video
        if self.comfy is None:
            raise PipelineError("ComfyUI workflows are not available here.")
        try:
            _prompt, fields = self.comfy.prompt(name)
        except ComfyRunError as exc:
            raise PipelineError(str(exc)) from exc
        prompt = next((f for f in fields if f.role is Role.PROMPT), None)
        if prompt is None:
            raise PipelineError(f"The workflow {name} has no prompt for a shot's motion.")
        picture = next((f for f in fields if f.kind is Kind.IMAGE), None)
        self._fields[video] = (prompt, picture, fields)
        return self._fields[video]

    def size(self, inputs: FilmInputs) -> tuple[int, int]:
        """A workflow's clip size, which the stills are drawn at."""
        _prompt, _picture, fields = self.fields(inputs.video)
        size = {f.means: f.default for f in fields if f.role is Role.SIZE}
        if "width" in size and "height" in size:
            return inputs.clip_shape()[:2]  # set on the workflow's own fields (`shape`)
        return film.WORKFLOW_SIZE  # a workflow that takes no size: Wan's own default

    def shape(self, inputs: FilmInputs) -> dict[str, str]:
        """The clip shape set on a workflow's fields that exist: width, height, frames."""
        _prompt, _picture, fields = self.fields(inputs.video)
        width, height, frames = inputs.clip_shape()
        wanted = {"width": width, "height": height} | dict.fromkeys(film.FRAME_FIELDS, frames)
        return {
            f.key: str(wanted[f.means]) for f in fields if f.role is Role.SIZE and f.means in wanted
        }

    def shape_words(self, inputs: FilmInputs) -> str:
        """`832x480, 33 frames`: a workflow's clips' shape, as they are made."""
        _prompt, _picture, fields = self.fields(inputs.video)
        inputs_of = {f.means for f in fields if f.role is Role.SIZE}
        width, height, frames = inputs.clip_shape()
        size = (
            f"{width}x{height}" if {"width", "height"} <= inputs_of else "the workflow's own size"
        )
        length = (
            f"{frames} frames"
            if inputs_of & set(film.FRAME_FIELDS)
            else "the workflow's own length"
        )
        return f"{size}, {length}"

    def kept_shape(
        self, run: RunRecord, stills: list[StepRecord], inputs: FilmInputs, report: Report
    ) -> FilmInputs:
        """A workflow film started before mnm kept its clip shape resumes at the size its
        first still was drawn at (else the default) and the default frames, said once and
        kept: its clips then match its stills, whatever the defaults are now."""
        if film.workflow_of(inputs.video) is None or {"clip_size", "clip_frames"} <= set(
            run.settings
        ):
            return inputs
        drawn = self._stills_size(run, stills)
        width, height, frames = inputs.clip_shape()
        if drawn is not None and "clip_size" not in run.settings:
            (width, height), why = drawn, "the size its stills were drawn at"
        else:
            why = "the size asked" if "clip_size" in run.settings else "the default size"
        kept = {"clip_size": f"{width}x{height}", "clip_frames": str(frames)}
        run.settings = {**run.settings, **kept}
        self.store.save(run)
        report(
            Whole(
                f"Clips at {width}x{height} ({why}), {frames} frames: this film was started "
                "before mnm kept its clip shape."
            )
        )
        return FilmInputs.from_settings(run.settings)

    def _stills_size(self, run: RunRecord, stills: list[StepRecord]) -> tuple[int, int] | None:
        """The first made still's width and height (its PNG header), if one is there."""
        for step in stills:
            if step.state is not StepState.DONE or not step.file:
                continue
            try:
                with (self.store.folder(run) / step.file).open("rb") as still:
                    return png_size(still.read(24))
            except (OSError, TryError):
                continue
        return None

    def make(self, run: RunRecord, index: int, video: str, shot: Shot, still: Path) -> Path:
        """One clip from the workflow (the still as its first frame when it takes a
        picture), moved into the film's folder; the workflow's own run record goes."""
        assert self.comfy is not None  # `fields` refused otherwise
        name = film.workflow_of(video) or video
        prompt, picture, _fields = self.fields(video)
        text = shot.motion_prompt if picture else f"{shot.still_prompt}. {shot.motion_prompt}"
        values = {prompt.key: text, **self.shape(FilmInputs.from_settings(run.settings))}
        uploads = {picture.key: (still.name, still.read_bytes())} if picture else {}
        child: RunRecord | None = None
        try:
            child = self.comfy.new(name, values)
            done = self.comfy.run(child, values, uploads)
            clips = [f for f in done.files if Path(f.name).suffix.lower() in film.CLIP_SUFFIXES]
            if not clips:
                raise StepError(f"The workflow {name} saved no video (it made {done.answer}).")
            source = self.store.folder(done) / clips[0].name
            target = self.store.folder(run) / film.item_file(
                film.CLIPS, index, Path(clips[0].name).suffix.lower()
            )
            target.unlink(missing_ok=True)
            shutil.move(str(source), target)
            return target
        except ComfyRunError as exc:
            raise StepError(str(exc)) from exc
        finally:  # the film's step says how it went; the workflow's own record goes either way
            if child is not None:
                with contextlib.suppress(RunError):
                    self.store.delete(child.id)
