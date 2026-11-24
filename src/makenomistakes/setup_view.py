"""What the guided setup looks like: a banner, one ASCII-ruled section per stage, the
prerequisites table, and the closing summary. Plain ASCII boxes, so any terminal (and a
pasted log) shows them the same."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from makenomistakes.setup_types import Outcome, StepResult

BANNER = r"""
  _ __ ___  _ __  _ __ ___
 | '_ ` _ \| '_ \| '_ ` _ \     makenomistakes
 | | | | | | | | | | | | | |    a Hugging Face link in, a URL out
 |_| |_| |_|_| |_|_| |_| |_|
"""
RULE = "=" * 72

_STYLE = {
    Outcome.OK: "green",
    Outcome.FIXED: "green",
    Outcome.PLANNED: "yellow",
    Outcome.MANUAL: "yellow",
    Outcome.SKIPPED: "red",
    Outcome.FAILED: "red",
}


class Mark(StrEnum):
    DONE = "[ok]"
    NOTE = "[!!]"
    SKIP = "[--]"
    FAIL = "[XX]"


_MARK_STYLE = {Mark.DONE: "green", Mark.NOTE: "yellow", Mark.SKIP: "dim", Mark.FAIL: "red"}


@dataclass
class Summary:
    """What each section left: a line per section, and what the user still has to do."""

    lines: list[tuple[Mark, str, str]] = field(default_factory=list)
    todo: list[str] = field(default_factory=list)

    def add(self, mark: Mark, section: str, what: str) -> None:
        self.lines.append((mark, section, what))

    @property
    def failed(self) -> bool:
        """A FAIL row in any section (the router key on the LAN, ufw refused): setup's exit
        code says it as the prerequisites' failures do."""
        return any(mark is Mark.FAIL for mark, _section, _what in self.lines)


class View:
    def __init__(self, console: Console, sections: int) -> None:
        self.console = console
        self.sections = sections

    def banner(self, subtitle: str) -> None:
        self.console.print(BANNER, style="bold magenta", highlight=False)
        self.console.print(f"  {escape(subtitle)}\n")

    def section(self, number: int, title: str, intro: str | None = None) -> None:
        self.console.print(f"\n[bold]{RULE}[/bold]")
        self.console.print(f"[bold] [{number}/{self.sections}]  {escape(title)}[/bold]")
        self.console.print(f"[bold]{RULE}[/bold]")
        if intro:
            self.console.print(escape(intro), style="dim")

    def say(self, text: str, style: str | None = None) -> None:
        self.console.print(escape(text), style=style)

    def facts(self, rows: list[tuple[str, str]]) -> None:
        table = Table(box=box.ASCII, show_header=False, pad_edge=False)
        table.add_column(style="bold")
        table.add_column()
        for label, value in rows:
            table.add_row(escape(label), escape(value))
        self.console.print(table)

    def command(self, line: str) -> None:
        self.console.print(f"[dim]$ {escape(line)}[/dim]")

    def steps(
        self, results: list[StepResult], title: str = "mnm setup", commands: bool = True
    ) -> None:
        self.console.print(steps_table(results, title, commands))

    def summary(self, summary: Summary) -> None:
        self.console.print(f"\n[bold]{RULE}[/bold]")
        self.console.print("[bold] Summary[/bold]")
        self.console.print(f"[bold]{RULE}[/bold]")
        for mark, section, what in summary.lines:
            style = _MARK_STYLE[mark]
            self.console.print(
                f" [{style}]{escape(mark.value)}[/{style}] {escape(section):<22} {escape(what)}"
            )
        if summary.todo:
            body = "\n".join(f"{n}. {escape(item)}" for n, item in enumerate(summary.todo, 1))
            self.console.print(
                Panel(body, title="What is left for you", box=box.ASCII, border_style="yellow")
            )


def steps_table(
    results: list[StepResult], title: str = "mnm setup", commands: bool = True
) -> Table:
    """The prerequisites: each step, its outcome and detail, the commands it would run or
    ran, and its note (shared by the guided and the plain setup)."""
    table = Table(title=title, box=box.ASCII)
    for column in ("Step", "Result", "Detail"):
        table.add_column(column)
    for r in results:
        detail = escape(r.detail)
        shown = (Outcome.PLANNED, Outcome.FAILED, Outcome.MANUAL) if commands else (Outcome.FAILED,)
        if r.commands and r.outcome in shown:
            detail += "\n[dim]" + "\n".join(escape(c) for c in r.commands) + "[/dim]"
        if r.note:
            detail += f"\n[yellow]-> {escape(r.note)}[/yellow]"
        style = _STYLE[r.outcome]
        table.add_row(escape(r.name), f"[{style}]{r.outcome.value.upper()}[/{style}]", detail)
    return table
