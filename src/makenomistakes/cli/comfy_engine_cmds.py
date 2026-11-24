"""`mnm comfyui install|start|stop|status|logs|rollback|uninstall`: ComfyUI as
an optional engine, built on this box, run with no internet and the models read-only."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.cli._app import build_progress, console, fail_with, print_raw
from makenomistakes.cli.comfyui_cmds import comfy_app
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.comfy.engine import (
    IMAGE_GIB,
    ComfyDockerError,
    ComfyEngine,
    ComfyError,
    State,
    Variant,
    image_tag,
)
from makenomistakes.comfy.image import VARIANT_WORDS
from makenomistakes.comfy.packs import approved_ids
from makenomistakes.comfy.preflight import preflight, refusals, warnings
from makenomistakes.comfy.status import comfy_status, ran_since
from makenomistakes.comfy.variants import install_variant
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import probe_memory


def _engine() -> ComfyEngine:
    """The engine for this box; tests replace it."""
    return ComfyEngine(Settings())


@comfy_app.command("install")
@notified("comfyui install")
def install(
    cpu: Annotated[
        bool, typer.Option("--cpu", help="PyTorch for the CPU (no NVIDIA card; slow).")
    ] = False,
    ca: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="A proxy's CA certificate, for a build behind one."
        ),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
) -> None:
    """Build ComfyUI's image on this box (an Ubuntu base, ComfyUI at mnm's pinned release,
    PyTorch). The image before it stays for `mnm comfyui rollback`."""
    engine = _engine()
    gpus = probe_memory().gpus
    variant = install_variant(cpu, isinstance(gpus, list) and bool(gpus))
    checks = preflight(variant, gpus, engine.runner)
    for warning in warnings(checks):  # a "no" is said once, in the refusal below
        console.print(f"[yellow]{escape(warning)}[/yellow]")
    if refused := refusals(checks):
        fail_with(ComfyError(" ".join(refused)))
    size = IMAGE_GIB[variant.value]
    console.print(
        f"This builds {escape(image_tag(variant))}: about {size:.0f} GB on disk, 10-30 minutes, "
        "downloads from GitHub, PyPI and PyTorch's index. Nothing else changes."
    )
    if variant is Variant.CPU and not cpu:
        console.print("[yellow]No card ComfyUI runs on is visible: building for the CPU.[/yellow]")
    elif variant is not Variant.CUDA:
        console.print(f"Built for {escape(VARIANT_WORDS[variant])}, this box's backend.")
    if not yes and not typer.confirm("Build it now?"):
        raise typer.Exit(code=1)
    try:
        ledger = engine.install(variant, build_progress, ca)
    except ComfyError as exc:
        fail_with(exc)
    if ledger.address:
        console.print(
            f"[green]Built[/green] {escape(ledger.image)}. ComfyUI keeps running the image "
            "before it until `mnm comfyui stop` and `mnm comfyui start`."
        )
    else:
        console.print(f"[green]Built[/green] {escape(ledger.image)}. Next: `mnm comfyui start`.")
    _packs_note(ledger.packs_image is None)


@comfy_app.command("start")
def start() -> None:
    """Start ComfyUI (on its own network, no internet) and wait until it answers."""
    try:
        address = _engine().start(build_progress)
    except ComfyError as exc:
        fail_with(exc)
    console.print(f"[green]ComfyUI is ready[/green] at {escape(address)} (this machine only).")


@comfy_app.command("stop")
def stop() -> None:
    """Stop ComfyUI (its image, workflows and outputs stay)."""
    try:
        _engine().stop()
    except ComfyError as exc:  # Docker could not remove it: never "stopped" on a guess
        fail_with(exc)
    console.print("ComfyUI stopped.")


@comfy_app.command("status")
def status() -> None:
    """Installed or not, running or not, and what it holds on the card."""
    engine = _engine()
    try:
        ledger = engine.ledger()
    except ComfyError as exc:
        fail_with(exc)
    try:
        state, why = engine.observe(), ""
    except ComfyDockerError as exc:  # never "stopped" on a guess
        state, why = State.UNKNOWN, f" (Docker could not be asked: {exc})"
    console.print(f"ComfyUI: {state.value}{escape(why)}")
    if ledger is not None:
        console.print(f"  image: {escape(ledger.image)} ({ledger.variant.value})")
        if ledger.previous:
            console.print(f"  previous: {escape(ledger.previous)} (`mnm comfyui rollback`)")
    if state is State.RUNNING:
        stats = engine.stats()
        if stats is not None:
            ran = ran_since(engine)
            console.print(
                f"  {escape(stats.system.comfyui_version)}, PyTorch "
                f"{escape(stats.system.pytorch_version)}, holding about "
                f"{stats.held_estimate(ran) / 2**30:.1f} GiB of VRAM "
                f"({escape(stats.held_method(ran))})"
            )
        console.print(f"  address: {escape(engine.address() or '-')}")
        console.print(f"  now: {escape(comfy_status(engine).doing())}")  # freed at …


@comfy_app.command("logs")
def logs(tail: Annotated[int, typer.Option(min=1)] = 200) -> None:
    """ComfyUI's log."""
    print_raw(_engine().logs(tail))


@comfy_app.command("rollback")
def rollback() -> None:
    """Use the image before the last install again (restart ComfyUI to apply it)."""
    try:
        ledger = _engine().rollback()
    except ComfyError as exc:
        fail_with(exc)
    console.print(f"Back to {escape(ledger.image)}. `mnm comfyui stop` and `start` to use it.")
    _packs_note(ledger.packs_image is None)


@comfy_app.command("uninstall")
def uninstall(yes: Annotated[bool, typer.Option("--yes")] = False) -> None:
    """Remove ComfyUI's container, images and network (workflows and outputs stay)."""
    engine = _engine()
    if not yes and not typer.confirm("Remove ComfyUI's container, images and network?"):
        raise typer.Exit(code=1)
    try:
        removed = engine.uninstall()
    except ComfyError as exc:
        fail_with(exc)
    console.print(f"Removed {escape(', '.join(removed) or 'nothing')}.")


def _packs_note(dropped: bool) -> None:
    """A new base (or a rollback) leaves the approved node packs out of the image: say how to
    build them in again, before a workflow that needs them refuses."""
    approved = approved_ids(Settings().expanded_state_dir())
    if dropped and approved:
        console.print(
            f"Your approved node packs ({escape(', '.join(sorted(approved)))}) are built over "
            "one image: `mnm comfyui packs install` builds them over this one."
        )
