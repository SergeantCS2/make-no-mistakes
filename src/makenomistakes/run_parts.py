"""The parts of a run record: its kind, state, steps and files, and (run record schema 2)
what places it in a take's tree and says what made it.
Every schema-2 field has a default, so a schema-1 record reads as it was written; a leaf
module, so `runs.py` (which re-exports these) and the readers share it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from makenomistakes.run_files import FileKind

# A child's place under its parent: one of four candidates, an edit of a stem or a range,
# a mix rendered from stems, an export folder, a training job.
RunRole = Literal["candidate", "edit", "mix", "master", "export", "training"]


class ActsOn(BaseModel):
    """What an edit changed: a stem, a range in seconds, a section of the lyrics."""

    stem: str | None = None
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)
    section: str | None = None


class ModelCredit(BaseModel):
    """One model that made the run: an instance with its files, or a workflow's name
    and packs, or a tool's pin."""

    instance: str | None = None
    repo: str | None = None
    file: str | None = None
    quant: str | None = None
    build: str | None = None  # the engine image's tag the instance ran
    workflow: str | None = None
    packs: list[str] = Field(default_factory=list)
    tool: str | None = None
    style: str | None = None  # The style an ACE-Step instance sang with
    sha256: str | None = None  # the style's adapter, as it was then
    trained_on: str | None = None  # what the style learned from, in the disclosure's words
    # A song's own credits: the item that made what the credit is for (the
    # stems' second package, the voice), so it leaves when that item is made again
    # (`pipeline/song_credits`). None on a child run's credits and on credits saved before.
    stage: str | None = None
    item: int = 0


class InputRecord(BaseModel):
    """Every input a run used: its role, its name, its hash and where it came from."""

    role: str
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = "owner"  # `owner`, `run:<id>`, `reference`
    stage: str | None = None  # the stage that used it


class RunKind(StrEnum):
    INSTANCE = "instance"  # one request to an instance: the Run form, `mnm run`
    WORKFLOW = "workflow"  # an imported ComfyUI workflow
    PIPELINE = "pipeline"  # mnm's own steps
    # A voice trained by the router's trainer (an older mnm cannot read the record:
    # it says so rather than drop it). Target: the voice's name.
    TRAINING = "training"


class RunState(StrEnum):
    RUNNING = "running"
    WAITING = "waiting"  # a pipeline at a review point: the user continues it
    DONE = "done"
    FAILED = "failed"


class StepState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"  # nothing to make (a shot with no line)


class StepRecord(BaseModel):
    """One item of a pipeline stage: the third still, the second clip."""

    stage: str
    index: int = 0
    state: StepState = StepState.PENDING
    file: str | None = None  # in the run's folder
    text: str = ""  # what it was asked, or what it answered
    error: str = ""
    seconds: float | None = None


class OutputFile(BaseModel):
    name: str  # inside the run's folder
    kind: FileKind
    size: int
    role: str = "output"  # "output", or "input" for what the user gave it


@dataclass(frozen=True)
class Removed:
    """What `RunStore.remove` took: a run, or a damaged record and its id's folder."""

    run_id: str
    size: int  # the files it freed (0 with `keep_files`, or for a damaged record)
    damaged: bool = False
    folder: Path | None = None  # a damaged record's folder removed; a run's says `size`
