"""`mnm studio`: the stages registry and the user's picks for each stage's tiers;
`studio_tools_cmds` adds the tools image's commands."""

from __future__ import annotations

from typing import Annotated, cast, get_args

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.comfy.store import WorkflowStore
from makenomistakes.config import Settings
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.pipeline import song
from makenomistakes.studio import stages
from makenomistakes.studio.music_models import models
from makenomistakes.studio.stage_instances import playing
from makenomistakes.studio.stages import StudioError, TierName

studio_app = typer.Typer(
    help="The studio: its stages, which instance plays each, and the tools image."
)
app.add_typer(studio_app, name="studio")


@studio_app.command("stage")
def stage_command(
    stage: Annotated[str | None, typer.Argument(help="A stage: song, stems …")] = None,
    picks: Annotated[
        list[str] | None,
        typer.Argument(
            help="fast=<instance> or full=<instance> (the song stage: or =workflow:<name>, an "
            "imported ComfyUI workflow); `fast=` goes back to the row's."
        ),
    ] = None,
) -> None:
    """The stages and their tiers; with `fast=<instance>` or `full=<instance>`, pick the
    instance that plays that tier of the stage (the registry's model stands until you do)."""
    settings = Settings()
    state = settings.expanded_state_dir()
    try:
        if stage is not None and picks:
            for item in picks:
                said_tier, sep, name = item.partition("=")
                if not sep or said_tier not in get_args(TierName):
                    raise StudioError(f"'{item}' is not fast=<instance> or full=<instance>.")
                tier = cast(TierName, said_tier)  # one of the two, checked above
                chosen = name.strip() or None
                if chosen is not None:
                    _check_instance(settings, stage, chosen)
                stages.pick(state, stage, tier, chosen)
                said = f"plays {escape(stage)}'s {tier} tier" if chosen else "cleared"
                console.print(f"{escape(chosen or tier)} {said}.")
        rows = [stages.row(stage)] if stage is not None else list(stages.STAGES.values())
        chosen_now = stages.read_picks(state).picks
        recipes = InstanceService(settings).all()
    except StudioError as exc:
        fail_with(exc)
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Stage", "Tier", "Plays it", "Registry's model", "Licence"):
        table.add_column(column)
    for found in rows:
        for tier in found.tiers:
            mine = chosen_now.get(found.stage, {}).get(tier)
            names = playing(found.stage, tier, state, recipes)
            plays = ", ".join(names) or "not added: `mnm add` the registry's"
            if found.host == "tool" and not mine:
                plays = "the studio tools image (`mnm studio install`)"
            table.add_row(
                found.stage,
                tier,
                escape(mine if mine and not names else plays),
                escape(stages.tier_words(found, tier)),
                escape(f"{found.licence} (commercial: {found.commercial})"),
            )
    console.print(table)


@studio_app.command("models")
def models_command() -> None:
    """Every studio model against this box now: what it needs of the card (or of the RAM on
    a box without one), whether this box holds it, the smallest card that does, and its time
    for a second of audio (this box's own once measured)."""
    try:
        found = models(InstanceService(Settings()))
    except InstanceError as exc:
        fail_with(exc)
    console.print(escape(found.room))
    table = Table(box=box.SIMPLE_HEAD)
    columns = ("Model", "Stage", "Needs", "This box", "Smallest card", "Work a second of audio")
    for column in columns:
        table.add_column(column)
    for row in found.rows:
        table.add_row(
            escape(row.rung.label) + ("" if row.rung.reference else " [dim](no package yet)[/dim]"),
            row.rung.stage,
            escape(row.needs),
            escape(row.here),
            escape(row.card),
            escape(row.time),
        )
    console.print(table)
    where = "in the RAM" if found.in_ram else "on the card"
    console.print(
        f"[dim]One stage runs at a time (audio.cpp offloads nothing): each model alone {where}. "
        "`mnm fit <link>` plans one; rows without a package say so in `mnm fit`.[/dim]"
    )


def _check_instance(settings: Settings, stage: str, name: str) -> None:
    """The instance exists and is of the kind the stage runs (`TAKES`: a song stage takes a
    song instance, the voice stage a converter, the edits an ACE-Step)."""
    row = stages.row(stage)
    stages.refuse_tool_pick(stage)
    workflow = stages.workflow_of(name)
    if workflow is not None:  # A ComfyUI workflow plays the stage
        if row.stage != song.SONG:
            raise StudioError(f"Only the song stage takes a ComfyUI workflow yet, not {stage}.")
        names = [m.name for m in WorkflowStore(settings.expanded_state_dir()).all()]
        if workflow not in names:
            have = f" (imported: {', '.join(names)})" if names else ""
            raise StudioError(
                f"No imported ComfyUI workflow is named {workflow}{have}: "
                "`mnm comfyui workflow add <link or file>` imports one."
            )
        return
    try:
        recipe = InstanceService(settings).get(name)
    except InstanceError as exc:
        raise StudioError(str(exc)) from None
    family = recipe.args.family if isinstance(recipe.args, AudioCppArgs) else None
    refused = stages.pick_refusal(stage, name, family, recipe.modality)
    if refused is not None:
        raise StudioError(f"{refused}.")
