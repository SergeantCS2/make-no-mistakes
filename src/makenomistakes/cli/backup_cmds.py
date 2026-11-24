"""`mnm backup` and `mnm restore`: this box's recipes, sets, workflows, approvals and
readings in one file, and back onto a fresh install (`backup.py`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes import config
from makenomistakes.apps.backup_files import AppsRestored, restore_app_files, to_back_up
from makenomistakes.backup import (
    NO_MODELS,
    Archive,
    BackupError,
    backup_name,
    many,
    read_backup,
    tally,
    write_backup,
)
from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.config import Settings
from makenomistakes.instances import InstanceService
from makenomistakes.local_time import local
from makenomistakes.measurements import gpu_name
from makenomistakes.report import secrets
from makenomistakes.restore_state import Restored, restore
from makenomistakes.supervisor import ContainerState, SupervisorError

backup_app = typer.Typer(
    help="Recipes, sets, ComfyUI workflows, approvals, readings and the secrets' names in one "
    ".tar.gz; never a secret's value or a model file. `--auto` into the backups folder, "
    "tested and rotated; `ls`; `schedule`.",
)
app.add_typer(backup_app, name="backup")


@backup_app.callback(invoke_without_command=True)
def backup(
    ctx: typer.Context,
    to: Annotated[
        Path | None,
        typer.Option("--to", help="The file to write (default: mnm-backup-<time>.tar.gz here)."),
    ] = None,
    auto: Annotated[
        bool,
        typer.Option(
            "--auto",
            help="Into MNM_BACKUPS_DIR as mnm-<date>-<time>.tar.gz, restore-tested, then "
            "rotated (7 daily, 4 weekly); what the nightly timer runs.",
        ),
    ] = False,
) -> None:
    """Recipes, sets, ComfyUI workflows, approvals, readings and the secrets' names in one
    .tar.gz; never a secret's value or a model file."""
    if ctx.invoked_subcommand is not None:
        if to is not None or auto:
            fail_with(BackupError(f"`--to` and `--auto` go without `{ctx.invoked_subcommand}`."))
        return
    if auto:
        if to is not None:
            fail_with(BackupError("`--auto` writes into MNM_BACKUPS_DIR: leave out `--to`."))
        from makenomistakes.cli.backup_auto_cmds import run_auto  # noqa: PLC0415 - no cycle

        run_auto()
        return
    settings = Settings()
    out = to or Path.cwd() / backup_name(datetime.now())
    if out.exists():
        fail_with(BackupError(f"{out} exists already: pass another `--to`."))
    try:
        known = secrets(settings, config.SECRETS_FILE).known  # scrubbed from every member
        card = gpu_name(InstanceService(settings).memory)
        apps = to_back_up(settings)
        manifest = write_backup(settings.expanded_state_dir(), out, known, gpu=card, app_files=apps)
    except BackupError as exc:
        fail_with(exc)
    console.print(f"Wrote {escape(str(out))}: {many(len(manifest.files), 'file', 'files')}.")
    for line in tally(manifest.files):
        console.print(f"  {escape(line)}")
    if manifest.app_files:
        console.print(f"  the apps' config files: {escape(', '.join(manifest.app_files))}")
    if manifest.secret_names:
        console.print(
            f"  the secrets file's names only ({escape(', '.join(manifest.secret_names))}): "
            "no value is in the file"
        )
    console.print(f"[dim]{escape(NO_MODELS)}[/dim]")


@app.command(name="restore")
def restore_command(
    file: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="The backup (with --test: the newest in MNM_BACKUPS_DIR unless named).",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace running instances' recipes too.")
    ] = False,
    test: Annotated[
        bool,
        typer.Option(
            "--test",
            help="Restore into a scratch state folder, read it back, remove it; this "
            "machine's state is not touched. Exit 1 when it does not restore.",
        ),
    ] = False,
) -> None:
    """Put a backup's files back into this machine's state folder; nothing is planned again
    and nothing downloads until an instance's first `mnm start`."""
    if test:
        from makenomistakes.cli.backup_auto_cmds import run_test  # noqa: PLC0415 - no cycle

        run_test(file)
        return
    if file is None:
        fail_with(BackupError("Name the backup to restore (`mnm backup ls` lists them)."))
    settings = Settings()
    service = InstanceService(settings)

    def running(name: str) -> bool:
        try:
            return service.supervisor.status(name).state is ContainerState.RUNNING
        except SupervisorError as exc:
            raise BackupError(
                f"Could not ask Docker whether {name} runs ({exc}): nothing was restored. "
                "`--force` restores regardless."
            ) from exc

    try:
        archive = read_backup(file)
        card = gpu_name(service.memory)
        done = restore(archive, settings.expanded_state_dir(), running, force, card)
        apps = restore_app_files(archive.app_files, settings)
    except BackupError as exc:
        fail_with(exc)
    except OSError as exc:  # an app's file: the state folder's are written by now
        fail_with(BackupError(f"Restoring an app's config file stopped: {exc}"))
    _say_restored(file, archive, done, apps, card)


def _say_restored(
    file: Path, archive: Archive, done: Restored, apps: AppsRestored, card: str | None
) -> None:
    made = archive.manifest
    console.print(
        f"Restored {many(len(done.written), 'file', 'files')} from {escape(file.name)} "
        f"(written {local(made.written_at):%Y-%m-%d %H:%M} by mnm {escape(made.mnm_version)})."
    )
    for line in tally(done.written):
        console.print(f"  {escape(line)}")
    if done.replaced:
        console.print(f"  replaced (they differed here): {escape(', '.join(done.replaced))}")
    if done.merged:
        console.print(
            f"  readings added to the ones taken here since: {escape(', '.join(done.merged))}"
        )
    if len(done.written) < len(archive.files) - len(done.kept):
        console.print("  the others were already here, the same")
    for name in done.rehashed:
        console.print(
            f"  workflow {escape(name)}'s file was scrubbed on backup; its hash is refreshed"
        )
    if done.other_machines:
        boxes = "; ".join(sorted(done.machines))
        console.print(
            f"  kept {many(done.other_machines, 'reading', 'readings')} of other machines "
            f"({escape(boxes)}) for when they return: this box plans from its own only"
        )
    if done.kept:
        console.print("  kept this machine's own engine ledger (its builds are the ones here)")
    for path in apps.written:
        console.print(f"  an app's config file: {escape(path)}")
    if apps.kept:
        console.print(f"  kept this machine's own app files: {escape(', '.join(apps.kept))}")
    if made.secret_names:
        console.print(
            "Secrets are never in a backup. Set these again in "
            f"~/.config/makenomistakes/secrets.env (or `mnm setup`): "
            f"{escape(', '.join(made.secret_names))}."
        )
    console.print(f"{escape(NO_MODELS)} `mnm ls` lists them.")
