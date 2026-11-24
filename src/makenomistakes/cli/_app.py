"""Shared Typer app, console, and the few helpers every command module uses."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.status import Status
from rich.table import Table

from makenomistakes import __version__
from makenomistakes.backup_auto import Tone
from makenomistakes.comfy.build_stream import BuildLine
from makenomistakes.config import SECRETS_FILE, Settings
from makenomistakes.diagnose import Diagnosis
from makenomistakes.download_progress import FileLine
from makenomistakes.fit import Budget, FitLevel
from makenomistakes.guide import Guidance
from makenomistakes.hardware.probe import SubprocessRunner
from makenomistakes.hardware.wsl_paths import windows_path
from makenomistakes.instance_types import CHANGED, InstanceError
from makenomistakes.plan import PlanError, explicit_budget
from makenomistakes.progress_words import with_ellipsis
from makenomistakes.service import ServiceError, checkout_revision, repo_root
from makenomistakes.store import NoSuchInstanceError

GIB = 1024**3
FIT_STYLE = {
    FitLevel.COMFORTABLE: "green",
    FitLevel.TIGHT: "yellow",
    FitLevel.OFFLOAD: "yellow",
    FitLevel.NO_FIT: "red",
}


def _above_zero(value: float | None) -> float | None:
    """`--vram`/`--ram` describe a machine: a size above 0 (else `mnm fit --vram -3`
    plans for -3 GiB of VRAM)."""
    if value is not None and not value > 0:
        raise typer.BadParameter(f"a size in GiB above 0, not {value:g}")
    return value


VramOption = Annotated[
    float | None,
    typer.Option(
        "--vram",
        help="Override free VRAM in GiB (plan for another machine).",
        callback=_above_zero,
    ),
]
RamOption = Annotated[
    float,
    typer.Option("--ram", help="Free system RAM in GiB, with --vram.", callback=_above_zero),
]
GpuOption = Annotated[str, typer.Option(help="GPU name for the bandwidth table, with --vram.")]
VersionOption = Annotated[
    str | None,
    typer.Option(
        "--version",
        help="Which version of an image repo (a checkpoint file, a diffusers folder or a "
        "GGUF model; `mnm fit` lists them).",
    ),
]

app = typer.Typer(
    help="A Hugging Face link in, a local model server on its own port out.",
    no_args_is_help=True,
)
console = Console()


def print_said(said: Iterable[tuple[Tone, str]]) -> None:
    """Lines with their tones (`backup_auto.Said`), each escaped: a tone is a rich style."""
    for tone, line in said:
        console.print(
            f"[{tone.value}]{escape(line)}[/{tone.value}]" if tone.value else escape(line)
        )


def _print_version(value: bool) -> None:
    if value:
        try:
            revision = checkout_revision(SubprocessRunner(), repo_root())
        except ServiceError:  # installed from a wheel, not run from the checkout
            revision = None
        console.print(f"makenomistakes {__version__}" + (f" ({revision})" if revision else ""))
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,  # runs before Click asks for a command ("Missing command.")
            help="Print the version and the checkout's commit, and exit.",
        ),
    ] = False,
) -> None:
    del version  # handled by its callback
    try:
        Settings()
    except ValidationError as exc:
        for error in exc.errors():
            key = "MNM_" + "_".join(str(part) for part in error["loc"]).upper()
            console.print(f"[red]{escape(key)}: {escape(error['msg'])}[/red]")
        console.print(f"Fix it in the shell or in {escape(str(SECRETS_FILE))}.")
        raise typer.Exit(code=2) from None


def print_block(text: str) -> None:
    """Text meant to be copied (a command, a config block, a URL): printed verbatim, never
    styled, emoji-coded or wrapped by Rich, so what is pasted is exactly what was generated."""
    console.print(text.rstrip("\n"), markup=False, highlight=False, emoji=False, soft_wrap=True)


def print_raw(text: str) -> None:
    """A log as the engine wrote it: never wrapped (a pipe or a file is 80 columns to Rich,
    which would break long lines), never read as markup."""
    console.print(text, end="", markup=False, highlight=False, emoji=False, soft_wrap=True)


def build_line(line: str) -> None:
    """One line of a `docker build` as it comes (dim, never read as markup or wrapped)."""
    console.print(line, style="dim", markup=False, highlight=False, emoji=False, soft_wrap=True)


def build_progress(message: str) -> None:
    """The ComfyUI commands' progress: a build's own line (`BuildLine`) as it came, any other
    message as a step."""
    if isinstance(message, BuildLine):
        build_line(message)
    else:
        print_progress(message)


def print_progress(message: str) -> None:
    """One line per step of a download or start ("…"); a whole sentence as it is."""
    console.print(escape(with_ellipsis(message)))


def print_changes(fixes: list[str]) -> None:
    """What auto-fix changed and saved, one line each, on success and on failure alike."""
    for what in fixes:
        console.print(f"[yellow]{CHANGED}:[/yellow] {escape(what)}")


class Failed(typer.Exit):
    """`fail_with`'s exit 2, carrying the reason for a notice (`cli/notify_cmd.notified`)."""

    def __init__(self, reason: str, code: int = 2) -> None:
        super().__init__(code=code)
        self.reason = reason


def fail_with(exc: Exception) -> NoReturn:
    """The one-line reason first, then the diagnosis or guidance that goes with it; exit 2."""
    console.print(f"[red]{escape(str(exc))}[/red]")
    if isinstance(exc.__cause__, NoSuchInstanceError):
        console.print("`mnm ls` lists the instances.")
    if isinstance(exc, InstanceError):
        print_changes(exc.fixes)
        print_diagnoses(exc.diagnoses)
    if isinstance(exc, PlanError):
        print_guidance(exc.guidance)
    raise Failed(str(exc))


def print_diagnoses(found: list[Diagnosis]) -> None:
    for item in found:
        console.print(f"[yellow]Likely cause:[/yellow] {escape(item.cause)}")
        console.print(f"[yellow]What to do:[/yellow]   {escape(item.remedy)}")


def print_guidance(guidance: Guidance | None) -> None:
    if guidance is None:
        return
    console.print(f"[yellow]What this is:[/yellow] {escape(guidance.what_it_is)}")
    console.print(f"[yellow]What to do:[/yellow]   {escape(guidance.what_to_do)}")
    if guidance.alternatives:
        table = Table(title="Try one of these instead")
        table.add_column("Repo")
        table.add_column("Downloads", justify="right")
        table.add_column("Note")
        for alt in guidance.alternatives:
            table.add_row(
                escape(alt.repo_id), f"{alt.downloads:,}" if alt.downloads else "", escape(alt.note)
            )
        console.print(table)
        console.print(f"[dim]e.g. mnm add {escape(guidance.alternatives[0].repo_id)}[/dim]")


def budget_override(vram_gb: float | None, ram_gb: float, gpu: str) -> Budget | None:
    """An explicit budget when --vram is given; None means probe the local hardware."""
    return explicit_budget(vram_gb, ram_gb, gpu) if vram_gb is not None else None


def print_run_folder(run_id: str, folder: Path) -> None:
    """`Run <id>: <folder>`, and on WSL the same folder as Windows' Explorer opens it."""
    console.print(f"Run {escape(run_id)}: {escape(str(folder))}")
    if (windows := windows_path(folder)) is not None:
        console.print(f"  From Windows: {escape(windows)}", soft_wrap=True)


def spinner_progress(status: Status) -> Callable[[str], None]:
    """A start's progress on its spinner, except a download: the spinner stops, the line is
    printed, and the download's own bar (tqdm, on stderr) draws alone until the next line
    starts the spinner again (else both rewrite one terminal line)."""
    drawing = [True]

    def said(line: str) -> None:
        if isinstance(line, FileLine):
            if drawing[0]:
                status.stop()
                drawing[0] = False
            console.print(escape(line))
            return
        if not drawing[0]:
            status.start()
            drawing[0] = True
        status.update(escape(line))

    return said
