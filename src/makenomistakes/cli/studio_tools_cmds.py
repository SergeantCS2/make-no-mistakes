"""`mnm studio install|status|rollback|uninstall`: the studio tools image, built on
this box as ComfyUI's is, and what it carries."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.cli._app import build_progress, console, fail_with
from makenomistakes.cli.image_cmds import remove_all, roll_back, show_status
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.studio_cmds import studio_app
from makenomistakes.config import Settings
from makenomistakes.studio.tools import CARRIES, IMAGE_GIB, image_tag, studio_tools
from makenomistakes.studio.tools_answers import ToolsError

GB = 1_000_000_000


@studio_app.command("install")
@notified("studio install")
def install(
    ca: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="A proxy's CA certificate, for a build behind one."
        ),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
) -> None:
    """Build the studio tools' image on this box: the candidates' scorer, the structure map,
    the voice's similarity, MIDI, mastering and pitch tuning, all on the CPU. The image before
    it stays for `rollback`."""
    tools = studio_tools(Settings())
    console.print(
        f"This builds {escape(image_tag())}: about {IMAGE_GIB:.0f} GB on disk, 10-20 minutes, "
        "downloads from PyPI, PyTorch's index, Hugging Face and CPJKU's cloud (the beat "
        "tracker's weights), each file checked by sha256. It carries:"
    )
    for line in CARRIES:
        console.print(f"  {escape(line)}")
    if not yes and not typer.confirm("Build it now?"):
        raise typer.Exit(code=1)
    try:
        ledger = tools.install(build_progress, ca)
    except ToolsError as exc:
        fail_with(exc)
    size = f", {ledger.size_bytes / GB:.1f} GB" if ledger.size_bytes else ""
    console.print(
        f"[green]Built[/green] {escape(ledger.image)}{size}. A song's candidates are scored "
        "and its picked take mapped from now on; `mnm song redo <run> score` scores an "
        "earlier run's."
    )


@studio_app.command("status")
def status() -> None:
    """The tools image: which one is in use, what it carries, and whether this mnm would
    build another."""
    show_status(studio_tools(Settings()), image_tag())


@studio_app.command("rollback")
def rollback() -> None:
    """Go back to the image before the last install."""
    roll_back(studio_tools(Settings()))


@studio_app.command("uninstall")
def uninstall(
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
) -> None:
    """Remove every studio tools image; the runs, their scores and structure maps stay."""
    remove_all(studio_tools(Settings()), yes, "the runs keep their scores and maps")
