"""Who plays a Song's stages: the desk's `chat` (or the instance named) for the lyrics, the
instance or ComfyUI workflow that sings the candidates, and the instances of a stage's tier
(`studio.stage_instances`)."""

from __future__ import annotations

from pathlib import Path

from makenomistakes.comfy.run import ComfyRunner
from makenomistakes.desk import read_aliases
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.pipeline import song, song_comfy
from makenomistakes.pipeline.song import SongInputs
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.recipes import Recipe
from makenomistakes.studio import stages, style_use
from makenomistakes.studio.stage_instances import Player, players
from makenomistakes.studio.stages import StudioError
from makenomistakes.studio.style_library import StyleError, StyleRow
from makenomistakes.workflows import WorkflowStore


def chat(service: InstanceService, state_dir: Path, inputs: SongInputs) -> Recipe:
    """The instance that writes the lyrics: the desk's `chat`, or the one named."""
    name = inputs.chat
    if name == song.DESK_CHAT:
        names = [r.name for r in service.all()]
        aliases = read_aliases(WorkflowStore(state_dir), names)
        member = aliases.instance(song.DESK_CHAT)
        if member is None and song.DESK_CHAT not in names:
            why = f" ({aliases.problem})" if aliases.problem else ""
            raise PipelineError(
                f"No desk `chat` writes the lyrics{why}: `mnm sets default <set>` with a "
                "chat role, or name a chat instance (--chat, the form's Lyrics by), or give "
                "your own lyrics (--lyrics FILE, the form's Lyrics)."
            )
        name = member or song.DESK_CHAT
    try:
        return service.get(name)
    except InstanceError as exc:
        raise PipelineError(f"{exc} (the lyrics' chat instance must exist: `mnm ls`)") from exc


def sung_by(
    service: InstanceService, state_dir: Path, comfy: ComfyRunner | None, inputs: SongInputs
) -> Recipe | str:
    """What sings the candidates: the instance that plays the song stage's tier, or the
    ComfyUI workflow the user picked for it (`workflow:<name>`), checked."""
    try:
        workflow = stages.resolve(song.SONG, inputs.tier, state_dir).workflow
    except StudioError as exc:  # the picks file unreadable: worded, never a 500
        raise PipelineError(str(exc)) from exc
    if workflow is not None:
        song_comfy.song_keys(comfy, workflow)
        return workflow
    return playing(service, state_dir, song.SONG, inputs)[0].recipe


def playing(
    service: InstanceService, state_dir: Path, stage: str, inputs: SongInputs
) -> list[Player]:
    try:
        return players(stage, inputs.tier, state_dir, service.all())
    except StudioError as exc:
        raise PipelineError(str(exc)) from exc


def name_of(sung: Recipe | str) -> str:
    """The run's `song_instance`: the instance, or `workflow:<name>`."""
    return sung.name if isinstance(sung, Recipe) else f"{stages.WORKFLOW_PICK}{sung}"


def takes_no_recording(workflow: str, tier: str) -> str:
    """A song with the user's recording sung by a workflow: refused at the start, and at a
    redo or resume after the pick moved to one (never sung without it, without a word)."""
    return (
        f"The song stage is the ComfyUI workflow {workflow}, which takes no recording: a "
        f"recording goes to an audio.cpp song model (`mnm studio stage song {tier}=` goes back "
        "to the registry's)."
    )


def style_for(service: InstanceService, recipe: Recipe) -> StyleRow | None:
    """The style `recipe` sings the candidates with, its files checked against its row (the
    candidate's credit names them)."""
    try:
        return style_use.playing(service, recipe)
    except StyleError as exc:
        raise PipelineError(str(exc)) from exc
