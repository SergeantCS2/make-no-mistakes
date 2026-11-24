"""`mnm lora`: the LoRAs an image instance offers, and which are always on."""

from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import app, console, fail_with, print_progress
from makenomistakes.cli.complete import instance_names
from makenomistakes.config import Settings
from makenomistakes.consumers import browser_origin
from makenomistakes.hub.facts import Engine
from makenomistakes.hub.kits import FAMILY_LABELS
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.loras import LoraService

lora_app = typer.Typer(help="LoRAs on image instances: add from the Hub, set weights, remove.")
app.add_typer(lora_app, name="lora")


def _service() -> LoraService:
    return LoraService(InstanceService(Settings()))


@lora_app.command("ls")
def list_cmd(
    name: Annotated[str, typer.Argument(help="An image instance.", autocompletion=instance_names)],
) -> None:
    """The instance's LoRAs: base family, weight, whether every request uses it."""
    try:
        recipe = _service().instances.get(name)
    except InstanceError as exc:
        fail_with(exc)
    if not recipe.loras:
        if recipe.engine is not Engine.SDCPP:
            console.print(
                f"{escape(name)} is {recipe.modality.a_label} instance: LoRAs are for image "
                "and video models."
            )
            return
        console.print(f"{escape(name)} has no LoRAs yet: `mnm lora add {escape(name)} <link>`.")
        return
    table = Table(title=f"{name} LoRAs", box=box.SIMPLE_HEAD)
    for column in ("Name", "Base", "Weight", "Always on", "Size", "From"):
        table.add_column(column, no_wrap=column != "From")
    for lora in recipe.loras:
        table.add_row(
            escape(lora.name),
            FAMILY_LABELS[lora.family],
            f"{lora.weight:g}",
            "yes" if lora.always_on else "",
            f"{lora.size_bytes / 2**20:.0f} MB",
            escape(f"{lora.repo_id}/{lora.path}"),
        )
    console.print(table)


@lora_app.command("add")
def add_cmd(
    name: Annotated[str, typer.Argument(help="An image instance.", autocompletion=instance_names)],
    reference: Annotated[
        str,
        typer.Argument(
            help="Hugging Face or Civitai (or Civitai Archive) link, or repo id, of the LoRA."
        ),
    ],
    lora_name: Annotated[
        str | None, typer.Option("--name", help="What the engine page lists it as.")
    ] = None,
    weight: Annotated[float, typer.Option(help="Strength, 1.0 = as trained.")] = 1.0,
    always: Annotated[
        bool, typer.Option("--always", help="Apply to every request (Odysseus's too).")
    ] = False,
) -> None:
    """Check the LoRA fits the instance's base model, download it, offer it."""
    service = _service()
    try:
        planned = service.plan(name, reference, lora_name, weight, always)
        lora = planned.lora
        console.print(
            f"{escape(lora.repo_id)}/{escape(lora.path)}: a LoRA for {escape(planned.info.label)}, "
            f"{lora.size_bytes / 2**20:.0f} MB, as [bold]{escape(lora.name)}[/bold]."
        )
        if planned.note:
            console.print(f"[yellow]{escape(planned.note)}[/yellow]")
        recipe = service.add(name, planned, print_progress)
    except InstanceError as exc:
        fail_with(exc)
    console.print(
        f"Added. The engine page ({browser_origin(recipe)}/) lists it now if {escape(name)} "
        "is running; pick it and set its strength per image."
    )
    if always:
        console.print(f"Always on from the next start: `mnm stop {name}` then `mnm start {name}`.")


@lora_app.command("set")
def set_cmd(
    name: Annotated[str, typer.Argument(help="An image instance.", autocompletion=instance_names)],
    lora: Annotated[str, typer.Argument(help="The LoRA's name (`mnm lora ls`).")],
    weight: Annotated[float | None, typer.Option(help="Strength, 1.0 = as trained.")] = None,
    always: Annotated[
        bool | None,
        typer.Option("--always/--not-always", help="Apply to every request, or only when picked."),
    ] = None,
) -> None:
    """Change a LoRA's default weight or whether every request uses it."""
    try:
        recipe = _service().update(name, lora, weight, always)
    except InstanceError as exc:
        fail_with(exc)
    item = next(i for i in recipe.loras if i.name == lora)
    on = "always on" if item.always_on else "used when picked"
    console.print(f"{escape(lora)}: weight {item.weight:g}, {on}. Applies on the next start.")


@lora_app.command("rm")
def remove_cmd(
    name: Annotated[str, typer.Argument(help="An image instance.", autocompletion=instance_names)],
    lora: Annotated[str, typer.Argument(help="The LoRA's name (`mnm lora ls`).")],
) -> None:
    """Stop offering a LoRA (the downloaded file stays in the cache)."""
    try:
        _service().remove(name, lora)
    except InstanceError as exc:
        fail_with(exc)
    console.print(f"Removed {escape(lora)} from {escape(name)}.")
