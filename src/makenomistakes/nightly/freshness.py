"""The freshness step: a newer engine build (`mnm engines check`), a newer family
member of a saved model (`mnm upgrade`), an upgrade's `<name>-old` still kept. Each finding
is an offer on the nightly's page: never run by the nightly (an update is the user's to
run), never a regression and never a notice by itself. `fresh.py` is the box's implementation; a
run built without one says "not built"."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel


class FindingKind(StrEnum):
    ENGINE = "engine"  # a newer engine build, or one that runs an instance's architecture
    MODEL = "model"  # a newer family member of a saved model
    KEPT = "kept"  # an upgrade's `<name>-old`, still beside the new one
    UNCHECKED = "unchecked"  # something could not be asked tonight: why


class Finding(BaseModel):
    """One thing freshness found: about what, in words, and the command it offers (never
    run by the nightly: an engine or model update is the user's)."""

    kind: FindingKind
    subject: str  # `llama.cpp`, `qwen3-8b`
    figures: str  # `has a newer build (built 2026-10-27; this box's 2026-10-20)`
    offered: str | None = None  # `mnm engines update llamacpp`

    @property
    def line(self) -> str:
        return f"{self.subject} {self.figures}" + (f": `{self.offered}`" if self.offered else "")


class Freshness(Protocol):
    def check(self) -> list[Finding]: ...
