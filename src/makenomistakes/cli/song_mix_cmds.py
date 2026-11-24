"""`mnm song mix`: the stem mixer from a terminal. Without settings it shows the mix in
use and the song's renders; with them it renders a new mix the export then uses.

    mnm song mix <run> vocals=+2,in=1 drums=mute bass=-3,out=4 piano=solo
    mnm song mix <run> --use <mix run>      (or --own: the stems summed as made)

A stem not named keeps its settings in the mix in use; a stem named keeps its level and fades
unless said, and is muted or soloed only when it says so."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.cli import song_cmds
from makenomistakes.cli._app import console, fail_with
from makenomistakes.cli.song_cmds import song_app
from makenomistakes.pipeline import song_mix
from makenomistakes.pipeline.song_runner import SongRunner
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.runs import RunError
from makenomistakes.studio.mixer import MixError, MixSettings, StemLevel

GRAMMAR = "stem=<dB>[,mute][,solo][,in=<s>][,out=<s>]"
OWN = "the song's own mix (the stems summed as made)"


def parse_change(text: str, levels: MixSettings) -> StemLevel:
    """One `stem=…` argument over the stem's level in use (a part not said stays)."""
    stem, sep, parts = text.partition("=")
    if not sep or not parts:
        raise MixError(f"'{text}' is not {GRAMMAR}.")
    known = [lv.stem for lv in levels.levels]
    if stem not in known:
        raise MixError(f"This song has no {stem} stem (it has {', '.join(known)}).")
    changed = levels.level(stem).model_dump()
    changed["mute"] = changed["solo"] = False
    for part in parts.split(","):
        key, _, value = part.strip().partition("=")
        try:
            match key:
                case "mute" | "solo" if not value:
                    changed[key] = True
                case "in" | "out" if value:
                    changed[f"fade_{key}"] = float(value.removesuffix("s"))
                case _ if not value:
                    changed["gain_db"] = float(key.lower().removesuffix("db"))
                case _:
                    raise ValueError(key)
        except ValueError:
            raise MixError(f"'{part}' in '{text}' is not {GRAMMAR}.") from None
    try:
        return StemLevel.model_validate(changed)
    except ValueError as exc:
        raise MixError(f"'{text}': {exc}") from None


@song_app.command("mix")
def mix(
    run_id: Annotated[str, typer.Argument()],
    changes: Annotated[
        list[str] | None, typer.Argument(help=f"{GRAMMAR} per stem (a stem not named stays).")
    ] = None,
    use: Annotated[str | None, typer.Option("--use", help="Use this render of the song.")] = None,
    own: Annotated[bool, typer.Option("--own", help="Use the song's own mix again.")] = False,
) -> None:
    """The stem mixer: show the mix in use, render a new one, or choose one to use."""
    runner = song_cmds._runner()  # through the module: tests replace it there
    try:
        run = runner.song(run_id)  # a mix's, an edit's or a master's id says whose it is
        if use is not None or own:
            runner.use_mix(run_id, None if own else use)
            chosen = escape(use) if use and not own else OWN
            console.print(f"The export now uses {chosen}.")
            return
        levels = song_mix.settings_of(runner.store, run)
        if levels is None:
            raise MixError(f"{run_id} has no stems yet: the mixer follows the stems.")
        if not changes:
            _show(runner, run_id, levels)
            return
        by_stem = {lv.stem: lv for lv in levels.levels}
        for text in changes:
            level = parse_change(text, levels)
            by_stem[level.stem] = level
        settings = MixSettings(levels=list(by_stem.values()))
        console.print(f"Rendering: {escape(settings.words())}…")
        made = runner.render_mix(run_id, settings)
    except (MixError, PipelineError, RunError) as exc:
        fail_with(exc)
    console.print(
        f"[green]Mixed[/green] {escape(made.id)}: the export uses it (`mnm song export "
        f"{escape(run_id)}` writes it); `--own` goes back to the stems summed as made."
    )


def _show(runner: SongRunner, run_id: str, levels: MixSettings) -> None:
    run = runner.store.get(run_id)
    current = song_mix.in_use(runner.store, run)
    console.print(f"In use: {escape(current.id) if current else OWN}")
    for level in levels.levels:
        console.print(f"  {escape(level.words())}")
    for render in song_mix.mixes(runner.store, run):
        mark = " (in use)" if current and render.id == current.id else ""
        console.print(f"  {escape(render.id)}: {escape(render.prompt or '')}{mark}")
    console.print(f"`mnm song mix {escape(run_id)} {GRAMMAR} …` renders a new one.")
