"""A doctor check: its name, a verdict and what to do about it."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, field_validator

from makenomistakes.private_ids import without_card_ids


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class Check(BaseModel):
    """A line the user reads and may paste into a report: no card's uuid or id once built or
    validated (`private_ids`), a nightly report saved before this rule included when it is read
    back (its file keeps the id until the nightly prunes it; `mnm report` redacts it). A
    `model_copy(update=…)` skips the check: build a new `Check` for text that is not mnm's."""

    name: str
    status: Status
    detail: str
    remedy: str | None = None

    @field_validator("detail", "remedy")
    @classmethod
    def _without_card_ids(cls, text: str | None) -> str | None:
        return None if text is None else without_card_ids(text)


# The base doctor's rows that a backend's view rewrites (`doctor_backend`): one name each,
# read by both, so renaming a row cannot silently break that view.
GPU_ROW = "NVIDIA GPU"
VRAM_ROW = "VRAM in use"
RUNTIME_ROW = "Docker NVIDIA runtime"
SMOKE_ROW = "Docker GPU smoke test"
