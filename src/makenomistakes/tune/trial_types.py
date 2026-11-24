"""What a trial is and what it measured, typed: the JSON lines of
`<state>/tuning/<name>.jsonl` are these. A line of another `shape` is skipped when read, as a
reading of another shape is (`measurements`)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from statistics import median
from threading import Event
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from makenomistakes.quality.figures import QualityFigures
from makenomistakes.recipes.tuned import TUNED_KEYS, DraftChange, Score, Value

REQUEST_CAME = "a request came"
CANCELLED = "cancelled"  # the cut's why when the user cancels the job
# A quality tune's stages as its lines name them: the reference's run is its base run
QUALITY_STAGE_WORDS = {"reference": "base run", "speed": "speed", "quality": "quality"}


class Cut(Event):
    """The event that cuts a running job, with why (a request, the nightly holding the card,
    the user's cancel). A cut only stops the job's work sooner: whether a tune was cancelled
    is its queue entry's mark (`cancelled`), which a later cut cannot overwrite."""

    def __init__(self) -> None:
        super().__init__()
        self.why = REQUEST_CAME

    def cut(self, why: str = REQUEST_CAME) -> None:
        self.why = why
        self.set()


class Candidate(BaseModel):
    """The settings one trial runs: `changes` (engine args field -> value) and `draft`, both
    over mnm's plan (the seed: no changes); `stage` is the search stage that proposed it."""

    model_config = ConfigDict(frozen=True)

    stage: str = "seed"
    changes: dict[str, Value] = Field(default_factory=dict)
    draft: DraftChange | None = None
    quant: str | None = None  # A quality tune's quant (None: the recipe's own)

    @property
    def key(self) -> str:
        """The same key is the same settings, whichever stage proposed them."""
        shown: dict[str, object] = {"changes": self.changes, "draft": self.draft}
        if self.quant is not None:
            shown["quant"] = self.quant
        return json.dumps(shown, sort_keys=True)

    @property
    def label(self) -> str:
        """`ubatch 1024, threads 4` (the plan: "mnm's plan"); a quality tune's `Q5_K_M, speed`."""
        parts = [f"{TUNED_KEYS.get(f, f)} {_shown(v)}" for f, v in sorted(self.changes.items())]
        if self.draft is not None:
            parts.append("no draft" if self.draft == "off" else f"draft {self.draft}")
        if self.quant is not None:  # a quality tune's step: the quant and the stage
            return ", ".join([self.quant, QUALITY_STAGE_WORDS.get(self.stage, self.stage), *parts])
        return ", ".join(parts) or "mnm's plan"

    def changed(self, field: str, value: Value, planned: Value, stage: str) -> Candidate:
        """This candidate with `field` at `value`, proposed by `stage`; a value equal to the
        plan's leaves the field out (the same settings, the same key)."""
        changes = {k: v for k, v in self.changes.items() if k != field}
        if value != planned:
            changes[field] = value
        return Candidate(stage=stage, changes=changes, draft=self.draft)

    def drafting(self, draft: DraftChange | None, stage: str) -> Candidate:
        return Candidate(stage=stage, changes=dict(self.changes), draft=draft)


class Outcome(StrEnum):
    MEASURED = "measured"
    CRASHED = "crashed"  # did not start, or died while answering: `cause` says why
    NOISY = "not alone"  # the readings spread too far, or something else touched the card
    CUT = "cut"  # an outside request took the card back; never counted, run again
    REFUSED = "refused"  # the settings could not be run (mnm refused them before a start)
    HELD = "held"  # no room could be made for the copy now; never counted, run again


class TrialResult(BaseModel):
    shape: Literal[1] = 1
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session: str  # the tune it belongs to (`TuneState.session`)
    instance: str
    number: int = Field(ge=1)
    candidate: Candidate
    outcome: Outcome
    build: str | None = None  # the engine image's short id the copy ran
    decode: list[float] = Field(default_factory=list)  # tok/s of each fresh prompt
    prompt_2k: float | None = None  # tok/s, the median over the three ~2k-token prompts
    prompt_tokens: int | None = None  # the engine's count of one of them
    prompt_16k: float | None = None  # tok/s over the long prompt (only when the score reads it)
    long_tokens: int | None = None  # its length as the engine counted it
    ttft_seconds: float | None = None  # llama-server's prompt time for a ~2k prompt (median)
    vram_peak_bytes: list[int] = Field(default_factory=list)  # per card, nvidia-smi's order
    vram_added_bytes: int | None = None  # what the copy added to the first card, when alone
    ram_rss_bytes: int | None = None  # the engine process's VmRSS (`/proc/<pid>/status`)
    draft_n: int | None = None  # drafted tokens over the three prompts
    draft_accepted: int | None = None  # of them kept
    cause: str | None = None  # `diagnose`'s code for a crash (`out-of-memory`, …)
    quality: QualityFigures | None = None  # A quality run's scores (`candidate.stage`)
    why: str | None = None  # one line for the user
    seconds: float = 0.0  # the trial's wall clock, start to clean-up
    score: Score | None = None

    @property
    def out_of_memory(self) -> bool:
        return self.cause == "out-of-memory"

    @property
    def counted(self) -> bool:
        """It used the budget: every trial but a cut or held one."""
        return self.outcome not in {Outcome.CUT, Outcome.HELD}

    @property
    def line(self) -> str:
        """One progress line: `trial 3 (ubatch 1024): 41.2 tok/s, score 58.3`; a quality run's
        `trial 4 (Q4_K_M, quality): KLD 0.0723 ± 0.0015 against Q8_0, …`."""
        head = f"trial {self.number} ({self.candidate.label})"
        if self.candidate.quant is not None:  # a quality tune's step
            if self.outcome is Outcome.MEASURED and self.quality is not None:
                return f"{head}: {self.quality.words}"
            if self.outcome in {Outcome.MEASURED, Outcome.NOISY} and self.decode:
                prompt = f", prompt {self.prompt_2k:.0f} tok/s" if self.prompt_2k else ""
                spread = (
                    f"; {self.outcome.value} ({self.why})" if self.outcome is Outcome.NOISY else ""
                )
                return f"{head}: decode {median(self.decode):.1f} tok/s{prompt}{spread}"
        if self.outcome is Outcome.MEASURED and self.score is not None:
            drafted = ""
            if self.draft_n:
                drafted = f", the draft kept {self.draft_accepted or 0} of {self.draft_n}"
            prompt = f", prompt {self.prompt_2k:.0f} tok/s" if self.prompt_2k else ""
            return (
                f"{head}: decode {self.score.decode:.1f} tok/s{prompt}{drafted}, "
                f"score {self.score.value:.1f}"
            )
        return f"{head}: {self.outcome.value}" + (f" ({self.why})" if self.why else "")


def _shown(value: Value) -> str:
    return "auto" if value is None else str(value)
