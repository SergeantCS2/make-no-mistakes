"""`mnm comfyui packs ls|add|rm|install|rollback|find`: custom node packs.
A vetted pack is added with one command; any other pack shows what the Registry says about
it and asks. A pack off the Registry is added at a GitHub release tag or commit
(`github:owner/repo@v1.2.0`). `install` builds the approved set as a layer over
ComfyUI's image."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import build_progress, console, fail_with
from makenomistakes.cli.comfyui_cmds import comfy_app
from makenomistakes.comfy.engine import ComfyEngine, ComfyError
from makenomistakes.comfy.fetch import fetch_all
from makenomistakes.comfy.github_packs import GitHub
from makenomistakes.comfy.models import ModelError, Models
from makenomistakes.comfy.pack_build import PackBuilder, missing_weights
from makenomistakes.comfy.packs import (
    SANDBOX_NOTE,
    VETTED,
    Assessment,
    PackStore,
    approve_assessed,
    assess,
)
from makenomistakes.comfy.registry import PackError, Registry
from makenomistakes.comfy.report import analyze
from makenomistakes.comfy.store import WorkflowStore, WorkflowStoreError
from makenomistakes.config import Settings
from makenomistakes.hardware.runner import SubprocessRunner
from makenomistakes.instances import InstanceService

packs_app = typer.Typer(help="Custom node packs: the vetted list, approvals, the install.")
comfy_app.add_typer(packs_app, name="packs")


def _registry() -> Registry:
    """The Comfy Registry; tests replace it."""
    return Registry()


def _github() -> GitHub:
    """git and GitHub's API for packs off the Registry; tests replace it."""
    return GitHub(SubprocessRunner())


def _engine(settings: Settings) -> ComfyEngine:
    """The engine for this box; tests replace it."""
    return ComfyEngine(settings)


@packs_app.command("ls")
def ls() -> None:
    """The vetted packs, the approved ones and what the running layer holds."""
    settings = Settings()
    try:
        approved = {a.pack.lower(): a for a in PackStore(settings.expanded_state_dir()).approvals()}
    except PackError as exc:
        fail_with(exc)
    try:
        ledger = _engine(settings).ledger()
    except ComfyError as exc:
        fail_with(exc)
    installed = {p.lower() for p in (ledger.packs if ledger else [])}
    table = Table(box=box.SIMPLE)
    for column in ("pack", "version", "licence", "state", "what"):
        table.add_column(column)
    rows = {row.pack.lower(): (row.pack, row.version, row.licence, row.what) for row in VETTED}
    for key, a in approved.items():
        if key not in rows or rows[key][1] != a.version:
            rows[key] = (a.pack, a.version, a.licence, "approved by you")
    for key, (pack, version, licence, what) in sorted(rows.items()):
        state = "vetted"
        if key in approved and approved[key].version == version:
            state = "approved"
        if f"{key}@{version}".lower() in installed:
            state = "installed"
        table.add_row(*(escape(v) for v in (pack, version, licence, state, what)))
    console.print(table)
    console.print("`mnm comfyui packs add <pack>` approves one; `install` builds them in.")


def describe(found: Assessment) -> None:
    node = found.node
    console.print(f"[bold]{escape(found.pack)}[/bold] {escape(found.version)}")
    if found.row is not None:
        console.print(f"  vetted by mnm: {escape(found.row.what)} ({found.row.licence})")
    if found.github is not None:
        console.print(f"  source: {escape(found.github.url)} (GitHub, not the Registry)")
    if found.fetched is not None:
        console.print(f"  commit: {found.fetched.commit}")
        console.print(f"  packed: sha256 {found.fetched.sha256} (mnm's zip of it, what is pinned)")
        console.print(f"  licence: {escape(found.licence)}")
        pip = ", ".join(found.fetched.requirements) or "none"
        console.print(f"  pip packages: {escape(pip)}")
    if found.facts is not None:
        console.print(
            f"  stars: {found.facts.stargazers_count:,}, open issues: "
            f"{found.facts.open_issues_count:,}"
        )
    if node is not None:
        publisher = node.publisher.id if node.publisher else "unknown"
        console.print(f"  source: {escape(node.repository or 'not stated')}")
        console.print(f"  licence: {escape(found.licence)}")
        console.print(f"  publisher: {escape(publisher)}")
        console.print(f"  downloads: {node.downloads:,}, stars: {node.github_stars:,}")
    if found.release is not None:
        deps = ", ".join(found.release.dependencies or []) or "none listed"
        console.print(f"  pip packages: {escape(deps)}")
        console.print(f"  download: {escape(found.release.downloadUrl)}")
    for note in found.notes:
        console.print(f"  [yellow]{escape(note)}[/yellow]")
    for refusal in found.refusals:
        console.print(f"  [red]{escape(refusal)}[/red]")


def add_pack(pack: str, version: str | None, yes: bool) -> None:
    settings = Settings()
    store = PackStore(settings.expanded_state_dir())
    registry = _registry()
    try:
        found = assess(registry, pack, version, _github())
        describe(found)
        if found.refusals:  # `describe` said why, once
            raise typer.Exit(code=2)
        confirmed = yes or found.row is not None
        if found.ok and not confirmed:
            console.print(escape(SANDBOX_NOTE))
            if not typer.confirm(f"Approve {found.pack} {found.version}?"):
                raise typer.Exit(code=1)
            confirmed = True
        approval = approve_assessed(registry, store, found, confirmed)
    except PackError as exc:
        fail_with(exc)
    console.print(
        f"[green]Approved[/green] {escape(approval.pack)} {escape(approval.version)} "
        f"(sha256 {approval.sha256[:12]}…). `mnm comfyui packs install` builds it in."
    )


@packs_app.command("add")
def add(
    packs: Annotated[
        list[str] | None,
        typer.Argument(help="Registry ids, or github:owner/repo@<tag or commit>."),
    ] = None,
    version: Annotated[
        str | None, typer.Option(help="A version other than the vetted one.")
    ] = None,
    workflow: Annotated[
        str | None, typer.Option("--for", help="Every pack an imported workflow uses.")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask first.")] = False,
) -> None:
    """Approve packs: a vetted one at once, any other after showing what the Registry (or, for
    `github:owner/repo@tag`, the code and GitHub) says."""
    wanted = list(packs or [])
    if workflow is not None:
        store = WorkflowStore(Settings().expanded_state_dir())
        try:
            report = analyze(store.workflow(workflow), _engine(Settings()).object_info())
        except (WorkflowStoreError, ValueError) as exc:
            fail_with(exc)
        wanted += [p.reference for p in report.packs if not p.installed]
    if not wanted:
        fail_with(PackError("Name a pack, or `--for <workflow>`."))
    for pack in wanted:
        add_pack(pack, version, yes)


@packs_app.command("rm")
def rm(pack: str) -> None:
    """Withdraw an approval (`install` then builds the layer without it)."""
    try:
        gone = PackStore(Settings().expanded_state_dir()).remove(pack)
    except PackError as exc:
        fail_with(exc)
    console.print(f"Removed {escape(gone.pack)}. `mnm comfyui packs install` rebuilds the layer.")


@packs_app.command("install")
def install(
    ca: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="A proxy's CA certificate."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask before downloading.")] = False,
) -> None:
    """Fetch the packs' own weights, build the approved packs into ComfyUI's image, and start
    ComfyUI once to check each one loads (the layer before stays when one does not)."""
    settings = Settings()
    store = PackStore(settings.expanded_state_dir())
    engine = _engine(settings)
    models = Models(settings.expanded_models_dir(), settings.expanded_state_dir())
    try:
        weights = missing_weights(models, store.approvals())
        if weights:
            names = ", ".join(s.link for s in weights)
            console.print(f"The packs read these files, which mnm fetches: {escape(names)}")
            if not yes and not typer.confirm("Download them?"):
                raise typer.Exit(code=1)
            fetch_all(models, weights, settings, InstanceService(settings).all(), build_progress)
        ledger = PackBuilder(engine, store, _registry()).install(build_progress, ca)
    except (PackError, ComfyError, ModelError, OSError) as exc:
        fail_with(exc)
    if ledger.packs:
        console.print(f"[green]Installed[/green] {escape(', '.join(ledger.packs))}.")
    else:
        console.print("No packs are approved: ComfyUI runs its own nodes only.")


@packs_app.command("rollback")
def rollback() -> None:
    """Use the node packs layer before the last install again."""
    settings = Settings()
    try:
        ledger = PackBuilder(_engine(settings), PackStore(settings.expanded_state_dir())).rollback(
            build_progress
        )
    except (PackError, ComfyError) as exc:
        fail_with(exc)
    console.print(f"Back to {escape(', '.join(ledger.packs) or 'no packs')}.")


@packs_app.command("find")
def find(node_class: str) -> None:
    """Which Registry pack provides a node class."""
    try:
        pack = _registry().pack_of(node_class)
    except PackError as exc:
        fail_with(exc)
    if pack is None:
        fail_with(PackError(f"The Registry knows no pack with {node_class}."))
    console.print(escape(pack))
