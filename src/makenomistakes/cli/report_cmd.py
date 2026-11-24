"""`mnm report`: a report folder of this box (doctor, ls, engines, calibration, outputs,
ComfyUI, the services' journals, every instance's log), its secrets removed."""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape

from makenomistakes.cli._app import app, console
from makenomistakes.config import Settings
from makenomistakes.instances import InstanceService
from makenomistakes.machine_adopt import tables
from makenomistakes.report import FolderRunner, report_folder, secrets, write_report
from makenomistakes.store import StoreError


@app.command()
def report() -> None:
    """Write report-<time>/ in the current folder: this box's state and logs, secrets removed."""
    try:
        names = [r.name for r in InstanceService(Settings()).all()]
    except StoreError as exc:  # its own line in ls.txt says which file
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        names = []
    folder = report_folder(Path.cwd())
    console.print(f"Writing {escape(str(folder))}/ …")
    found = secrets(Settings())
    written = write_report(
        folder,
        names,
        FolderRunner(folder),
        found.known,
        said=lambda name: console.print(f"  {escape(name)}"),
        kept=tables(Settings().expanded_state_dir()),  # The adopt tables, if one ran
    )
    for name in found.too_short:
        console.print(
            f"[yellow]{escape(name)} is too short to scrub safely (under 4 characters): it was "
            "not replaced. Look for it before you send the folder.[/yellow]"
        )
    failed = [name for name, code in written.files if code != 0]
    console.print(
        f"[green]Wrote {len(written.files)} files[/green] to {escape(str(written.folder))}/ "
        "(secret values and Hugging Face tokens replaced by their names)."
    )
    if failed:
        console.print(
            f"[dim]{len(failed)} answered with an error, kept as written: "
            f"{escape(', '.join(failed))}.[/dim]"
        )
