"""`mnm pin` and `mnm unpin`: `keep=` and `idle=` in the words people use for them."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.cli.complete import instance_names
from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.ollama_api import KeepAliveError, keep_alive_seconds
from makenomistakes.recipes.model import IDLE_SECONDS
from makenomistakes.router_client import settings_saved


def duration_words(seconds: int) -> str:
    """`7200` → "2 h", `600` → "10 min", `90` → "90 s"."""
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60} min"
    return f"{seconds} s"


@app.command()
def pin(
    name: Annotated[str, typer.Argument(autocompletion=instance_names)],
    for_: Annotated[
        str | None,
        typer.Option(
            "--for",
            help="Keep it loaded this long after its last request (2h, 30m) instead of for good.",
        ),
    ] = None,
) -> None:
    """Keep a model loaded: for good (`keep=always`), or `--for 2h` after each request."""
    service = InstanceService(Settings())
    if for_ is None:
        updates = {"keep": "always"}
    else:
        try:
            seconds = keep_alive_seconds(for_)
        except KeepAliveError:
            fail_with(ValueError(f"--for {for_!r} is not a duration: 30m, 2h, 1h30m or seconds"))
        if seconds is None or seconds <= 0:
            fail_with(ValueError(f"--for needs a time, such as 30m or 2h, not {for_!r}"))
        updates = {"keep": "auto", "idle": str(seconds)}
    try:
        updated = service.update_settings(service.get(name), updates)
    except InstanceError as exc:
        fail_with(exc)
    settings_saved(service.settings, name)  # a client's keep_alive gives way
    if updated.pinned:
        console.print(
            f"{escape(name)} is pinned (keep=always): the router never unloads it, for idleness "
            f"or for room. `mnm unpin {escape(name)}` lets it."
        )
    else:
        console.print(
            f"{escape(name)} stays loaded {duration_words(updated.idle)} after its last request "
            f"(idle={updated.idle}); the router may still unload it to make room for another "
            "model. Applies now, over any client's keep_alive."
        )


@app.command()
def unpin(name: Annotated[str, typer.Argument(autocompletion=instance_names)]) -> None:
    """Let the router unload a model again: `keep=auto`, and its idle limit back to ten minutes."""
    service = InstanceService(Settings())
    try:
        updated = service.update_settings(
            service.get(name), {"keep": "auto", "idle": str(IDLE_SECONDS)}
        )
    except InstanceError as exc:
        fail_with(exc)
    settings_saved(service.settings, name)
    console.print(
        f"{escape(name)}: keep=auto, idle={updated.idle}: the router unloads it "
        f"{duration_words(updated.idle)} after its last request, or to make room. Applies now, "
        "over any client's keep_alive."
    )
