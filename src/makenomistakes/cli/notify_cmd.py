"""`mnm notify`: test the notifications, and turn their channels on or off; `notified`,
the decorator that sends one when a long command ends."""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable
from typing import Annotated, ParamSpec, TypeVar

import typer
from pydantic import ValidationError
from rich.markup import escape

from makenomistakes import config
from makenomistakes.cli._app import Failed, app, console, fail_with
from makenomistakes.config import Settings
from makenomistakes.notify import LONG_SECONDS, Notice, notifier
from makenomistakes.setup import write_setting

P = ParamSpec("P")
R = TypeVar("R")


def notified(
    what: str, key: str | None = None, success: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A command that sends a notice when it ends after LONG_SECONDS or more: "mnm run main:
    ended after 41 s", or its failure's own words. `key` names the argument that says which
    (an instance, a run); `success=False` tells only of a failure (`mnm start`)."""

    def wrap(command: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(command)
        def run(*args: P.args, **kwargs: P.kwargs) -> R:
            began = time.monotonic()
            title = f"mnm {what}" + (f" {kwargs[key]}" if key and kwargs.get(key) else "")
            try:
                result = command(*args, **kwargs)
            except typer.Exit as exc:
                if exc.exit_code and time.monotonic() - began >= LONG_SECONDS:
                    why = exc.reason if isinstance(exc, Failed) else f"exit code {exc.exit_code}"
                    took = time.monotonic() - began
                    tell(Notice(title, f"failed after {took:.0f} s: {why}", failed=True))
                raise
            took = time.monotonic() - began
            if success and took >= LONG_SECONDS:
                tell(Notice(title, f"ended after {took:.0f} s; its terminal says what came of it"))
            return result

        return run

    return wrap


def tell(notice: Notice) -> None:
    """Sent before the command's process ends (a thread would die with it): at most a few
    seconds after its own output, and nothing said if a channel fails."""
    with contextlib.suppress(Exception):  # best effort: the command's outcome stands
        notifier().send(notice)


@app.command()
def notify(
    text: Annotated[str | None, typer.Argument(help="A test message.")] = None,
    ntfy: Annotated[
        str | None,
        typer.Option(
            "--ntfy",
            help="An ntfy topic's URL (https://ntfy.sh/<topic>, or your own ntfy server's); "
            "`off` removes it.",
        ),
    ] = None,
    toast: Annotated[
        str | None,
        typer.Option("--toast", help="on or off: a Windows toast (WSL only; on there by default)."),
    ] = None,
) -> None:
    """Send a test notification; `--ntfy`/`--toast` save the channels in the secrets file."""
    secrets = config.SECRETS_FILE  # read now: tests point it elsewhere
    if ntfy is not None:
        url = "" if ntfy.strip().lower() == "off" else ntfy.strip()
        try:
            Settings(ntfy_url=url)  # refused here, not by every later command
        except ValidationError as exc:
            fail_with(ValueError(exc.errors()[0]["msg"].removeprefix("Value error, ")))
        write_setting(secrets, "MNM_NTFY_URL", url)
        console.print(f"Saved MNM_NTFY_URL={escape(url) or '(none)'} in {secrets}.")
    if toast is not None:
        if toast.strip().lower() not in ("on", "off"):
            fail_with(ValueError(f"--toast is on or off, not {toast!r}"))
        write_setting(secrets, "MNM_TOAST", toast.strip().lower())
        console.print(f"Saved MNM_TOAST={toast.strip().lower()} in {secrets}.")
    settings = Settings(_env_file=secrets) if secrets.exists() else Settings()
    sender = notifier(settings)
    if not sender.channels():
        console.print(
            "No channel is on: `mnm notify --ntfy https://ntfy.sh/<topic>` (or your own ntfy "
            "server's topic URL) sends to a phone; under WSL a Windows toast is on unless "
            "MNM_TOAST=off."
        )
        return
    if text is None and (ntfy is not None or toast is not None):
        return
    said = sender.send(Notice("mnm notify", text or "A test from mnm: notifications reach you."))
    for line in said:
        console.print(escape(line))
    if any(not line.split(": ", 1)[1].startswith(("sent", "shown")) for line in said):
        raise typer.Exit(code=1)
