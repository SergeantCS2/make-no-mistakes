"""`mnm comfyui`: lay out mnm's image models for ComfyUI and print (or write) its paths block;
`install`, `start`, `stop`, `status`, `logs`, `rollback`, `uninstall` run
ComfyUI itself as an optional engine."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes import comfyui
from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.comfy.models import Models
from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService

comfy_app = typer.Typer(
    help="ComfyUI: share mnm's picture models with it (no subcommand), or run it as an "
    "optional engine for workflows (install, start, stop, status, logs).",
    invoke_without_command=True,
)
app.add_typer(comfy_app, name="comfyui")


@comfy_app.callback()
def comfyui_cmd(
    ctx: typer.Context,
    write: Annotated[
        Path | None,
        typer.Option(
            "--write",
            help="Your ComfyUI folder: add the block to its extra_model_paths.yaml (the file "
            "as it was before the first write is kept as .bak).",
        ),
    ] = None,
) -> None:
    """Link mnm's image models where ComfyUI looks, and print the extra_model_paths.yaml
    block. Run again after `mnm add` or `mnm rm`."""
    if ctx.invoked_subcommand is not None:
        return
    service = InstanceService(Settings())
    models = service.models_dir
    try:
        plan = Models(models, service.settings.expanded_state_dir()).layout(service.all())
        comfyui.write_links(models, plan)
    except (InstanceError, OSError, ValueError) as exc:
        fail_with(exc)
    console.print(f"Linked {plan.count} files under {escape(str(models / comfyui.SUBDIR))}.")
    for line in plan.skipped:
        console.print(f"[dim]Skipped {escape(line)}.[/dim]")
    block = comfyui.yaml_block(models)
    if write is None:
        console.print("Add this to ComfyUI's extra_model_paths.yaml, then restart ComfyUI:\n")
        console.print(block, markup=False, highlight=False)
        return
    target = write.expanduser() / "extra_model_paths.yaml"
    try:
        existing = target.read_text(encoding="utf-8") if target.exists() else None
        text = comfyui.merged(existing, block)
        backup = target.with_name(target.name + ".bak")
        if existing is not None and not backup.exists():  # the user's file before mnm's first
            backup.write_text(existing, encoding="utf-8")
        target.write_text(text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        fail_with(OSError(f"Could not write {target}: {exc}"))
    console.print(f"Wrote the block to {escape(str(target))}. Restart ComfyUI to load it.")
