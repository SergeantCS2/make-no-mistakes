"""`mnm what`: find models and grade each against this box, or (`--for`) the starter
menu's picks for a use graded, three of them with why. Nothing is downloaded."""

from __future__ import annotations

import re
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import (
    FIT_STYLE,
    GpuOption,
    RamOption,
    VramOption,
    app,
    budget_override,
    console,
    fail_with,
)
from makenomistakes.cli.hub_cmds import print_budget
from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.model_uses import parse_use
from makenomistakes.plan import open_hub
from makenomistakes.plan_budget import budget_now
from makenomistakes.plan_types import PlanError
from makenomistakes.quality.readings import QuantScore, measured_by_repo
from makenomistakes.what import Answer, answer
from makenomistakes.what_grade import Graded

# a sentence ends at a full stop before a capital or a command (`mnm replan`); "7.1 GiB" is one
_SENTENCE = re.compile(r"(?<=\.)\s+(?=[A-Z`])")


@app.command()
def what(
    words: Annotated[
        list[str] | None, typer.Argument(help="What to search for: `qwen3 coder`, `flux`.")
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(
            "--type",
            help="chat, coder, vision, pictures, video, memory (and voice, ear: the curated ones).",
        ),
    ] = None,
    use: Annotated[
        str | None,
        typer.Option("--for", help="No words: three of the curated picks for this use."),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", min=1, max=20, help="The most rows a source gives.")
    ] = 6,
    vram_gb: VramOption = None,
    ram_gb: RamOption = 14.0,
    gpu: GpuOption = "RTX 4070",
) -> None:
    """Find models and grade each against this box with the planner: whether it fits, at
    which quant, how fast, what it needs, and the line that adds it. Never downloads."""
    text = " ".join(words or [])
    try:
        picked = parse_use(use or kind)
        if use is not None and kind is not None and parse_use(kind) is not picked:
            raise ValueError("--for and --type name one use: give one of them.")
        if use is not None and text:
            raise ValueError(
                "`--for` gives the curated picks for a use, with no words: to search, "
                f"`mnm what {text} --type {use}`."
            )
        if not text and picked is None:
            raise ValueError(
                "Name what to look for (`mnm what qwen3 coder`) or a use "
                "(`mnm what --for coder`, `--for chat`, `--for pictures`)."
            )
        settings = Settings()
        service = InstanceService(settings)
        budget = budget_override(vram_gb, ram_gb, gpu) or budget_now(service)
        with console.status("Searching and grading against this box…"), open_hub(settings) as hub:
            found = answer(hub, settings, budget, text, picked, limit, picks=use is not None)
    except (ValueError, PlanError, InstanceError) as exc:
        fail_with(exc)
    print_budget(budget)
    _print(found, picks=use is not None, measured=measured_by_repo(service.measurements))


def _measured(row: Graded, measured: dict[str, dict[str, QuantScore]]) -> str | None:
    """The pick's own quant as `mnm tune --quality` measured it on this box."""
    if row.pick is None or row.repo_id is None:
        return None
    score = measured.get(row.repo_id, {}).get(row.pick.quant)
    return f"measured here, {row.pick.quant}: {score.words}" if score is not None else None


def _print(found: Answer, picks: bool, measured: dict[str, dict[str, QuantScore]]) -> None:
    for note in found.notes:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    if not found.rows:
        console.print(
            "Nothing found for that: other words, or `--for <use>` for the curated picks."
        )
        return
    if picks:
        for n, row in enumerate(found.rows, 1):
            console.print(f"{n}. [bold]{escape(row.title)}[/bold]: {escape(_fit_words(row))}")
            if row.hit.why:
                console.print(f"   {escape(row.hit.why)}")
            if (scored := _measured(row, measured)) is not None:
                console.print(f"   {escape(scored)}")
            console.print(f"   {escape(row.add_line)}")
        return
    table = Table(title="Graded against this box (estimates)")
    for column in ("#", "Model", "From", "Downloads", "Fits"):
        table.add_column(column, overflow="fold")
    table.add_column("Speed", no_wrap=True)
    table.add_column("Needs", overflow="fold")
    for n, row in enumerate(found.rows, 1):
        table.add_row(
            str(n),
            escape(f"{row.hit.shown} ({row.size})" if row.size else row.hit.shown),
            row.hit.source.value,
            _count(row.hit.downloads),
            _fit_cell(row),
            row.pick.speed_summary if row.pick is not None else "",
            escape(row.pick.vram_words) if row.pick is not None else "",
        )
    console.print(table)
    told: set[str] = set()  # advice already said (Civitai's login, `mnm replan`'s offer)
    for n, row in enumerate(found.rows, 1):
        said = escape(row.add_line) if row.fits else f"[dim]{escape(_once(row, told))}[/dim]"
        console.print(f"{n}. {said}")
        if (scored := _measured(row, measured)) is not None:
            console.print(f"   {escape(scored)}")
    if found.left_out:
        console.print(f"[dim]{found.left_out} of another type left out (`--type` names one).[/dim]")


def _once(row: Graded, told: set[str]) -> str:
    """A refusal's first sentence always; each later one only the first time it is said."""
    first, *rest = _SENTENCE.split(row.refusal or "")
    new = [s for s in rest if s not in told]
    told.update(rest)
    return " ".join([first, *new, *(["(the rest as above)"] if len(new) < len(rest) else [])])


def _fit_words(row: Graded) -> str:
    if row.pick is None:
        return row.refusal or row.verdict.value
    level = row.pick.fit.level.value.replace("_", " ")
    smaller = f" ({row.verdict.value})" if row.smaller else ""
    return f"{level} at {row.pick.quant}{smaller}, {row.pick.speed_summary}, {row.pick.vram_words}"


def _fit_cell(row: Graded) -> str:
    if row.pick is None:  # the reasons follow the table, one line a row
        return f"[red]{row.verdict.value}[/red]"
    tone = FIT_STYLE[row.pick.fit.level]
    smaller = f" ({row.verdict.value})" if row.smaller else ""
    level = row.pick.fit.level.value
    return f"[{tone}]{level}[/{tone}] {escape(row.pick.quant)}{smaller}"


def _count(value: int | None) -> str:
    if value is None:
        return ""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value / 1000:.0f}k" if value >= 1000 else str(value)
