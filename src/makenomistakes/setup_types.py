"""The shapes `mnm setup` works in: a step (check, fix, notes) and what running it gave."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel

Command = list[str]


class Outcome(StrEnum):
    OK = "ok"  # already in place
    FIXED = "fixed"  # installed/configured by this run
    PLANNED = "planned"  # dry run: would install
    MANUAL = "manual"  # cannot be done from Linux; instructions printed
    SKIPPED = "skipped"  # a step it depends on failed
    FAILED = "failed"


class StepResult(BaseModel):
    name: str
    outcome: Outcome
    detail: str
    commands: list[str] = []
    note: str | None = None
    reboot: bool = False  # it worked, and takes effect at the next boot (a GPU driver)


@dataclass
class Step:
    name: str
    check: Callable[[], tuple[bool, str]]
    fix: list[Command] = field(default_factory=list)
    manual: str | None = None  # printed instead of running anything
    after: str | None = None  # note shown after a successful fix
    requires: str | None = None  # name of a step that must not have failed
    # Asked when the re-check after the fix still fails: True when that only waits for a
    # re-login (the docker group), so the step is FIXED with the note. Asked then, not when
    # the steps are built: on a fresh box Docker itself is installed by an earlier step.
    settles_at_relogin: Callable[[], bool] = lambda: False
    # Asked before the fix runs (a GPU driver): declined, the step is MANUAL with its commands.
    confirm: str | None = None
    reboot: bool = False  # the fix takes effect at the next boot: its re-check cannot pass yet
