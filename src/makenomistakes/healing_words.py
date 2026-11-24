"""The healer's journal lines, written once here and read back by the
nightly (`/__mnm/events`): a death with its cause, the restart's outcome, a crash loop's
park. Each line is a fixed sentence around an instance name (lowercase letters, digits and
dashes), so reading one back never guesses."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from makenomistakes.dead_engines import LOOPED

DIED = "{name} died: {cause}."
RESTARTED = "Restarted {name} after it died: {cause}."
NOT_RESTARTED = "Could not restart {name} after it died: {why}"
_NAME = r"(?P<name>[a-z0-9][a-z0-9-]*)"


class HealKind(StrEnum):
    DIED = "died"
    RESTARTED = "restarted"
    NOT_RESTARTED = "not restarted"
    LOOPED = "parked after two deaths"


class HealEvent(BaseModel):
    """One of the healer's lines, read back: the instance, when, and the cause (or, for a
    restart that failed, why it failed)."""

    kind: HealKind
    name: str
    at: datetime
    said: str


_LINES = (
    (
        HealKind.LOOPED,
        re.compile(rf"^{_NAME} died twice in ten minutes \((?P<said>.*)\): parked; "),
    ),
    (HealKind.DIED, re.compile(rf"^{_NAME} died: (?P<said>.*)$")),
    (HealKind.RESTARTED, re.compile(rf"^Restarted {_NAME} after it died: (?P<said>.*)$")),
    (
        HealKind.NOT_RESTARTED,
        re.compile(rf"^Could not restart {_NAME} after it died: (?P<said>.*)$"),
    ),
)


def looped_line(name: str, cause: str) -> str:
    return LOOPED.format(name=name, cause=cause)


def parse(line: str, at: datetime) -> HealEvent | None:
    """The healer's line as an event; None for any other journal line."""
    for kind, pattern in _LINES:
        if (found := pattern.match(line)) is not None:
            said = found.group("said").rstrip().removesuffix(".")
            return HealEvent(kind=kind, name=found.group("name"), at=at, said=said)
    return None
