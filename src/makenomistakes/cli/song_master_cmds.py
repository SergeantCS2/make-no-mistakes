"""`mnm song master`: the mix in use mastered to a reference track by Matchering in the
studio tools image; the export is made from the master in use, as it was made.

    mnm song master <run> --reference ref.wav    master it (any audio ffmpeg reads)
    mnm song master <run> --use <master run>     use an earlier master
    mnm song master <run> --off                  no master: the export's loudness line again
    mnm song master <run>                        the masters, and which is in use
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.cli import song_cmds
from makenomistakes.cli._app import console, fail_with
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.song_cmds import song_app
from makenomistakes.pipeline import song_master
from makenomistakes.pipeline.song_runner import SongRunner
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.runs import RunError
from makenomistakes.studio.tools_answers import ToolsError


@song_app.command("master")
@notified("song master", key="run_id")
def master(
    run_id: Annotated[str, typer.Argument()],
    reference: Annotated[
        Path | None, typer.Option("--reference", help="The track to sound like (WAV, FLAC, MP3).")
    ] = None,
    use: Annotated[str | None, typer.Option("--use", help="Use this master of the song.")] = None,
    off: Annotated[bool, typer.Option("--off", help="No master: the loudness line.")] = False,
) -> None:
    """Master the song's mix to a reference track, or choose the master its export uses."""
    runner = song_cmds._runner()  # through the module: tests replace it there
    if sum((reference is not None, use is not None, off)) > 1:
        fail_with(PipelineError("--reference, --use and --off each say a different thing."))
    try:
        if reference is not None:
            data = _read(reference)
            run = runner.master(run_id, reference.name, data, song_cmds._progress)
        elif use is not None or off:
            run = runner.use_master(run_id, None if off else use, song_cmds._progress)
        else:
            _show(runner, run_id)
            return
    except (PipelineError, RunError, ToolsError) as exc:
        fail_with(exc)
    console.print(f"{escape(run.id)}: {escape(run.state.value)}; `mnm song show {run.id}`.")
    song_cmds.exit_if_failed(run)


def _read(path: Path) -> bytes:
    if not path.is_file():
        raise PipelineError(f"There is no file {path}.")
    try:
        if path.stat().st_size > song_master.MAX_REFERENCE:
            raise PipelineError(
                f"{path.name} is past the {song_master.MAX_REFERENCE >> 20} MiB a reference may be."
            )
        return path.read_bytes()
    except OSError as exc:  # unreadable to this user: worded, never a traceback
        raise PipelineError(f"Cannot read the reference {path}: {exc.strerror or exc}.") from exc


def _show(runner: SongRunner, run_id: str) -> None:
    run = runner.song(run_id)  # a master's or an edit's id says what it is
    current = song_master.in_use(runner.store, run)
    old = song_master.stale(runner.store, run)
    if current is not None:
        console.print(f"In use: {escape(current.id)}: {escape(current.answer)}")
    elif old is not None:
        console.print(
            f"Not in use: {escape(old)} was made from another mix than the song uses now; "
            f"`mnm song master {escape(run_id)} --reference FILE` masters this one."
        )
    else:
        console.print("No master: the export's mix is at its loudness line.")
    for child in song_master.masters(runner.store, run):
        why = song_master.refusal(runner.store, run, child.id)
        mark = " (in use)" if current and child.id == current.id else ""
        mark = mark or (f" (cannot be used: {why.words})" if why is not None else "")
        line = child.answer or child.error or child.state.value
        console.print(f"  {escape(child.id)}: {escape(line)}{escape(mark)}")
    console.print(f"`mnm song master {escape(run_id)} --reference FILE` masters it to a track.")
