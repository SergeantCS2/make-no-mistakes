"""What `mnm studio status|rollback|uninstall` and `mnm studio trainer status|rollback|uninstall`
share (the tools image and the trainers'; `studio/built_image.BuiltImage`): one way to say an
image's state, go back to the one before it and remove them all."""

from __future__ import annotations

import typer
from rich.markup import escape

from makenomistakes.cli._app import console, fail_with
from makenomistakes.local_time import local
from makenomistakes.studio.built_image import BuiltImage, ImageError, ImageMissingError

GB = 1_000_000_000


def show_status(image: BuiltImage, current: str, built_for: str = "") -> None:
    """The image in use, what it carries, the one before it, and whether this mnm builds
    another (`current`, its tag); `built_for` names the build ("for the CPU")."""
    spec = image.spec
    try:
        ledger = image.installed()
    except ImageMissingError as exc:
        console.print(escape(str(exc)))
        return
    except ImageError as exc:
        fail_with(exc)
    for_words = f" {built_for}" if built_for else ""
    console.print(
        f"Image: {escape(ledger.image)}{for_words} (built {local(ledger.built_at):%Y-%m-%d %H:%M})"
    )
    if ledger.size_bytes:
        console.print(f"Size: {ledger.size_bytes / GB:.1f} GB")
    if ledger.versions:
        said = ", ".join(f"{k} {v}" for k, v in sorted(ledger.versions.items()))
        console.print(f"Carries: {escape(said)}")
    rollback = spec.install.replace(" install", " rollback")
    if ledger.previous:
        console.print(f"Before it: {escape(ledger.previous)} (`{rollback}`)")
    if ledger.image != current:
        console.print(
            f"[yellow]This mnm builds {escape(current)}: `{spec.install}` builds it (the one "
            "in use stays until then).[/yellow]"
        )


def roll_back(image: BuiltImage) -> None:
    try:
        ledger = image.rollback()
    except ImageError as exc:
        fail_with(exc)
    console.print(f"Back to {escape(ledger.image)}.")


def remove_all(image: BuiltImage, yes: bool, kept: str) -> None:
    """Every image of the repository and its record; `kept` says what stays."""
    label = image.spec.label
    if not yes and not typer.confirm(f"Remove the {label} images?"):
        raise typer.Exit(code=1)
    try:
        removed = image.uninstall()
    except ImageError as exc:
        fail_with(exc)
    for item in removed:
        console.print(f"Removed {escape(item)}")
    console.print(f"The {label} {'are' if label.endswith('s') else 'is'} uninstalled; {kept}.")
