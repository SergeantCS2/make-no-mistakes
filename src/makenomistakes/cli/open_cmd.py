"""`mnm open [name]`: the web UI's front page, or an instance's page, in this machine's
browser (`browser_open`); the address said when nothing here opens a page, and how to run the
web UI when nothing answers on its port."""

from __future__ import annotations

import os
from typing import Annotated
from urllib.parse import quote

import typer
from rich.markup import escape

from makenomistakes.browser_open import Detached, open_page
from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.cli.complete import instance_names
from makenomistakes.config import Settings
from makenomistakes.consumers import web_origin
from makenomistakes.hardware import platform as platforms
from makenomistakes.health_now import web_listens
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService


@app.command("open")
def open_web(
    name: Annotated[
        str | None,
        typer.Argument(
            help="An instance whose page to open; the front page without one.",
            autocompletion=instance_names,
        ),
    ] = None,
) -> None:
    """Open the web UI in this machine's browser (under WSL, Windows' default browser)."""
    settings = Settings()
    url = web_origin(settings.port)
    if name is not None:
        try:
            InstanceService(settings).get(name)
        except InstanceError as exc:
            fail_with(exc)
        url += f"/instances/{quote(name)}"
    if not web_listens(settings):
        console.print(
            f"[yellow]Nothing answers on port {settings.port}: `mnm service install` runs the "
            "web UI as a service, or `mnm serve` in a terminal.[/yellow]"
        )
        raise typer.Exit(code=1)
    opener = open_page(url, Detached(), platforms.current().is_wsl, os.environ)
    if opener is not None:
        console.print(f"Opened {escape(url)} ({opener}).")
        return
    console.print(f"Open {escape(url)} in a browser on this machine: nothing here opens a page.")
