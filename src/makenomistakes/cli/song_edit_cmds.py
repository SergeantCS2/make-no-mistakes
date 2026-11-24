"""`mnm song edit` and `mnm song take`: a song's edits from a terminal. An edit makes a
new take from the one in use (ACE-Step on audio.cpp), which the song then uses: its stems,
voice, mix and export are made again. `take` chooses which take the song uses (Undo).

    mnm song edit <run> repaint start=12 end=19.5 tags="brighter, more drums"
    mnm song edit <run> resing section=2 lines="New first line / New second line" part=3
    mnm song edit <run> extend seconds=20
    mnm song edit <run> cover strength=0.8
    mnm song edit <run> add track=strings tags="warm strings, legato"   (mixed over the take)
    mnm song edit <run> swap track=drums tags="brushed jazz drums"
    mnm song take <run> <edit run>     (or `candidate`: the picked candidate's take)
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli import song_cmds
from makenomistakes.cli._app import console, fail_with
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.song_cmds import song_app
from makenomistakes.pipeline import song, song_takes
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.runs import RunRecord, RunState, RunStore
from makenomistakes.studio.edit_base import Edit, EditError
from makenomistakes.studio.edits import ask_of

CANDIDATE = "candidate"


def _values(words: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for word in words:
        key, sep, value = word.partition("=")
        if not sep or not key:
            raise EditError(f"'{word}' is not key=value (e.g. start=12 end=19.5).")
        values[key.strip()] = value
    return values


@song_app.command("edit")
@notified("song edit", key="run_id")
def edit_cmd(
    run_id: Annotated[str, typer.Argument()],
    edit: Annotated[str, typer.Argument(help=", ".join(e.value for e in Edit))],
    values: Annotated[
        list[str] | None, typer.Argument(help="key=value: what the edit takes.")
    ] = None,
) -> None:
    """Edit a song's take (a range repainted, a section re-sung, extended, covered, an
    instrument added or swapped): the song uses the new take, and Undo is `mnm song take`."""
    runner = song_cmds._runner()
    try:
        ask = ask_of(edit, _values(values or []))
        run = runner.edit(run_id, ask, song_cmds._progress)
    except (EditError, PipelineError, StepError) as exc:
        fail_with(exc)
    song_cmds._said(run, runner)


@song_app.command("take")
@notified("song take", key="run_id")
def take_cmd(
    run_id: Annotated[str, typer.Argument()],
    take: Annotated[
        str | None,
        typer.Argument(help="An edit's run id, or `candidate` for the picked candidate's take."),
    ] = None,
) -> None:
    """The song's takes (the picked candidate's and every edit's), or the one to use: its
    stems, voice, mix and export are made again from it."""
    runner = song_cmds._runner()
    try:
        run = runner.song(run_id)
        if take is None:
            _takes(runner.store, run)
            return
        run = runner.use_take(run_id, None if take == CANDIDATE else take, song_cmds._progress)
    except (PipelineError, StepError) as exc:
        fail_with(exc)
    song_cmds._said(run, runner)


def _takes(store: RunStore, run: RunRecord) -> None:
    edits = song_takes.edits(store, run)
    try:
        in_use: str | None = song_takes.take(store, run).run.id
        problem = ""
    except song_takes.TakeError as exc:
        in_use, problem = None, str(exc)
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Take", "Edit", "Made from", "State"):
        table.add_column(column)
    picked = song.picked_index(run.settings)
    first = song.candidate_label(picked) if picked is not None else "no candidate picked"
    candidate = song_takes.candidate(store, run, picked) if picked is not None else None
    mark = " (in use)" if candidate is not None and candidate.id == in_use else ""
    table.add_row(CANDIDATE + mark, first, "", "")
    for edit in edits:
        mark = " (in use)" if edit.id == in_use else ""
        state = edit.state.value + (f": {escape(edit.error)}" if edit.error else "")
        table.add_row(
            escape(edit.id) + mark,
            escape(edit.answer or edit.settings.get("edit", "")),
            escape(edit.settings.get(song_takes.SOURCE, "")),
            state if edit.state is not RunState.DONE else "done",
        )
    console.print(table)
    if problem:
        console.print(f"[yellow]{escape(problem)}[/yellow]")
    if not edits:
        console.print(
            f"No edits yet: `mnm song edit {escape(run.id)} repaint start=<s> end=<s>` (or "
            "resing, extend, cover, add, swap)."
        )
