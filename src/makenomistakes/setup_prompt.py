"""How the guided setup asks: at the terminal, or not at all (`--yes`), where every
question takes its default and a step that must be asked first (the GPU driver) runs only
when its flag says so."""

from __future__ import annotations

import sys
from typing import Protocol

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt


class Prompter(Protocol):
    interactive: bool

    def yes(self, question: str, default: bool) -> bool: ...

    def text(self, question: str, default: str, secret: bool = False) -> str: ...

    def choose(self, question: str, options: dict[str, str], default: str) -> str: ...


class TerminalPrompter:
    interactive = True

    def __init__(self, console: Console) -> None:
        self.console = console

    def yes(self, question: str, default: bool) -> bool:
        return Confirm.ask(escape(question), default=default, console=self.console)

    def text(self, question: str, default: str, secret: bool = False) -> str:
        if secret and not sys.stdin.isatty():  # piped: nothing to hide from
            # getpass warns "Can not control echo on the terminal" three times a password
            # when stdin is a pipe (a piped `mnm apps user add librechat`).
            self.console.print(f"{escape(question)}: ", end="")
            return sys.stdin.readline().rstrip("\r\n").strip()
        if secret:  # a token: never echoed, and Enter skips it
            answer = Prompt.ask(
                escape(question),
                default="",
                password=True,
                show_default=False,
                console=self.console,
            )
            return answer.strip()
        return Prompt.ask(escape(question), default=default, console=self.console).strip()

    def choose(self, question: str, options: dict[str, str], default: str) -> str:
        for key, label in options.items():
            self.console.print(f"   [bold]{escape(key)}[/bold]  {escape(label)}")
        return Prompt.ask(
            escape(question), choices=list(options), default=default, console=self.console
        )


class Defaults:
    """Every question answered with its default; tokens skipped."""

    interactive = False

    def yes(self, question: str, default: bool) -> bool:
        return default

    def text(self, question: str, default: str, secret: bool = False) -> str:
        return "" if secret else default

    def choose(self, question: str, options: dict[str, str], default: str) -> str:
        return default
