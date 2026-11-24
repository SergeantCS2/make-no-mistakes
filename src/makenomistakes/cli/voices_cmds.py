"""`mnm voices`: the voice library: add, list, show, remove a voice, and record
the hash and date of a consent document for someone else's voice."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.config import Settings
from makenomistakes.studio.voice_library import (
    ENGINE_WORDS,
    NEEDS_CONSENT,
    SOURCE_WORDS,
    VoiceEngine,
    VoiceError,
    VoiceLibrary,
    VoiceRow,
    VoiceSource,
    refuse_packaged,
)

voices_app = typer.Typer(
    help="The voice library: the voices a song may be sung in (your own, or someone's you "
    "have the rights to), each with its source and, for someone else's, a consent record.",
    no_args_is_help=True,
)
app.add_typer(voices_app, name="voices")


def _library() -> VoiceLibrary:
    settings = Settings()
    return VoiceLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())


@voices_app.command("add")
def add(
    name: Annotated[str, typer.Argument(help="Its name in the library (`alto`, `me-2026`).")],
    engine: Annotated[
        VoiceEngine,
        typer.Option(help="seed_vc, vevo2 or soulx from a reference; rvc from a checkpoint."),
    ] = VoiceEngine.SEED_VC,
    source: Annotated[
        VoiceSource | None,
        typer.Option(help="own, generated, hired, licensed or likeness: whose voice it is."),
    ] = None,
    reference: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="1 to 30 s of the voice alone, a WAV file."),
    ] = None,
    model: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="An RVC checkpoint (.pth).")
    ] = None,
    index: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Its retrieval index (.index), optional."),
    ] = None,
    packaged: Annotated[
        str | None, typer.Option(help="One of audio.cpp's packaged RVC voices: refused, with why.")
    ] = None,
    licence: Annotated[str, typer.Option(help="The terms you use it under, in words.")] = "",
    note: Annotated[str, typer.Option(help="Anything to remember about it.")] = "",
) -> None:
    """Add a voice: its files are copied into the library and kept by their sha256."""
    try:
        if packaged is not None:
            refuse_packaged(packaged)
        if source is None:
            raise VoiceError(
                "Say whose voice it is: --source own, generated, hired, licensed or likeness "
                "(the export's disclosure names it, and someone else's asks for a consent "
                "record)."
            )
        row = _library().add(
            name, engine, source, reference=reference, model=model, index=index,
            licence=licence, note=note,
        )  # fmt: skip
    except VoiceError as exc:
        fail_with(exc)
    console.print(f"[green]Added[/green] {escape(row.words())}.")
    if row.needs_consent:
        console.print(
            f"Someone else's voice: `mnm voices consent {escape(name)} <document>` records the "
            "document's hash and date (the document stays yours). A song released without one "
            "says so in its disclosure; nothing is refused."
        )


@voices_app.command("ls")
def list_cmd() -> None:
    """The voices in the library."""
    library = _library()
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Voice", "Engine", "Source", "Consent", "Files"):
        table.add_column(column)
    for row in library.all():
        missing = library.missing_files(row)
        table.add_row(
            escape(row.name),
            ENGINE_WORDS[row.engine],
            SOURCE_WORDS[row.source],
            _consent_words(row),
            "missing: " + escape(", ".join(missing)) if missing else str(len(row.files)),
        )
    console.print(table)
    for problem in library.problems():
        console.print(f"[yellow]Not readable:[/yellow] {escape(problem)}")
    if not library.all():
        console.print("No voices yet: `mnm voices add <name> --reference voice.wav`.")


@voices_app.command("show")
def show(name: Annotated[str, typer.Argument()]) -> None:
    """One voice: its files by hash, its source and consent, how it was made."""
    library = _library()
    try:
        row = library.get(name)
    except VoiceError as exc:
        fail_with(exc)
    console.print(escape(row.words()))
    console.print(f"  folder: {escape(str(library.folder(row.name)))}")
    for item in row.files:
        console.print(f"  {item.role}: {escape(item.name)} (sha256 {item.sha256[:16]}…)")
    console.print(f"  consent: {escape(_consent_words(row))}")
    if row.licence:
        console.print(f"  licence: {escape(row.licence)}")
    if row.training is not None:
        console.print(
            f"  trained: {row.training.epochs or '?'} epochs on {len(row.training.dataset)} "
            "recordings"
        )
    if row.note:
        console.print(f"  note: {escape(row.note)}")
    missing = library.missing_files(row)
    if missing:
        gone = escape(", ".join(missing))
        console.print(
            f"[yellow]Its files are not in the library folder:[/yellow] {gone} (a restore brings "
            "rows, not recordings): copy them back, or add the voice again."
        )


@voices_app.command("rm")
def remove(
    name: Annotated[str, typer.Argument()],
    keep_files: Annotated[
        bool, typer.Option("--keep-files", help="Keep its recordings and checkpoint.")
    ] = False,
) -> None:
    """Remove a voice from the library (and its files, unless --keep-files)."""
    library = _library()
    try:
        row = library.remove(name, files=not keep_files)
    except VoiceError as exc:
        fail_with(exc)
    kept = f"; its files stay in {library.folder(row.name)}" if keep_files else ""
    console.print(f"Removed {escape(row.name)}{escape(kept)}.")


@voices_app.command("consent")
def consent(
    name: Annotated[str, typer.Argument()],
    document: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    note: Annotated[str, typer.Option(help="Who signed it, what it covers.")] = "",
) -> None:
    """Record a consent document's sha256 and today's date (the document is never copied)."""
    try:
        row = _library().consent(name, document, note)
    except VoiceError as exc:
        fail_with(exc)
    assert row.consent is not None
    console.print(
        f"Recorded {escape(document.name)} for {escape(row.name)}: sha256 "
        f"{row.consent.sha256[:16]}…, {row.consent.date}. Keep the document: mnm holds only "
        "its fingerprint."
    )


def _consent_words(row: VoiceRow) -> str:
    if row.consent is not None:
        return f"recorded {row.consent.date}"
    return "none recorded" if row.source in NEEDS_CONSENT else "not needed"
