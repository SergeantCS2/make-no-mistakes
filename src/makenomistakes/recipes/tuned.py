"""What the tuner saves on a recipe: the settings it chose, the score they
reached against mnm's own plan as measured, and the key of what they were measured on.

A tuned setting is a measurement: it holds for the engine build,
the cards, the driver, the RAM and its measured bandwidth, the model files and the tuner
that measured it. Any change to the key makes the tune stale (`TuneKey.differs`), said in
`mnm ls` and re-queued. Files are keyed by size and modification time, not sha256: hashing
100 GB of weights would take longer than the tune (a re-download changes both anyway).
Kept apart from `recipes.model` so the tuner and the recipe share it without a cycle."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# The tuner's own version: a change to the trial or the score bumps it, so every tune taken
# by an older tuner reads "tuned by an older tuner" and is queued again.
TUNER_VERSION = 1

ScoreMode = Literal["default", "chat", "agents"]
DraftChange = Literal["off", "mtp"]

# The engine args a tune may choose, and the `mnm set` key that changes each: a set of one
# of these is the user's override, and drops the tune.
TUNED_KEYS: dict[str, str] = {
    "cpu_moe_layers": "cpu-moe-layers",
    "moe_cache_mib": "moe-cache",
    "ubatch": "ubatch",
    "batch": "batch",
    "threads": "threads",
    "kv_type": "kv",
    "draft_max": "draft-max",
    "gpu_layers": "gpu-layers",
    # A MoE across cards, placed again from the files (`instance_settings.placed_again`):
    # the value is the expert blocks per card, `3,4,3,3`
    "placement": "card-experts",
    # llama.cpp's ROCm or Vulkan build on an AMD card (`backend_setting`): the recipe's
    # image, never an engine arg
    "backend": "backend",
    # Where the threads and RAM-held weights sit on a box of several NUMA nodes
    # (`numa_setting`): `distribute`, `isolate` or `node:N` (its CPUs read from the box)
    "numa": "numa",
}
PLACEMENT_FIELDS = frozenset({"placement"})  # set through the placer, never `with_args`
BACKEND_FIELDS = frozenset({"backend"})  # set on the image (`backend_setting.with_backend`)
NUMA_FIELDS = frozenset({"numa"})  # set through `numa_setting.with_numa` (a node's CPUs)
DRAFT_KEY = "draft"

Value = int | float | str | None


class Score(BaseModel):
    """One trial's score and the figures it was made of (`tune.score`)."""

    mode: ScoreMode
    value: float = Field(ge=0)
    decode: float = Field(ge=0)  # tok/s, the median of three fresh prompts
    prompt_2k: float | None = None  # tok/s over a ~2k-token prompt
    prompt_16k: float | None = None  # tok/s over the long prompt (up to 16k tokens)


class TuneKey(BaseModel):
    """What a tune was measured on; the same key means its settings still hold."""

    build: str | None  # the engine image's short id the trials ran
    cards: list[str] = Field(default_factory=list)  # "<uuid> <name>" of each card, in order
    driver: str | None = None
    ram_bytes: int | None = None  # what the kernel reports (the VM's under WSL)
    ram_gbps: float | None = None  # `mnm doctor --deep`'s bandwidth (`machine.json`)
    files: dict[str, str] = Field(default_factory=dict)  # relative path -> "size:mtime_ns"
    tuner: int = TUNER_VERSION
    # The box (its fingerprint id), its CPU's model and the engine backend; a key an older
    # mnm saved may lack them, and the cards and RAM checks below still tell a change
    machine: str | None = None
    cpu: str | None = None
    backend: str | None = None

    def differs(self, now: TuneKey) -> str | None:
        """What changed since, as `mnm ls` words it ("another build"); None when the key
        still holds. A box is compared only when both keys name one."""
        checks = (
            (self.tuner != now.tuner, "an older tuner"),
            (_both_differ(self.machine, now.machine) or _both_differ(self.cpu, now.cpu), BOX),
            (_both_differ(self.backend, now.backend), "another backend"),
            (self.build != now.build, "another build"),
            (self.cards != now.cards or self.driver != now.driver, "another card"),
            (self.files != now.files, "other files"),
            (
                _gib(self.ram_bytes) != _gib(now.ram_bytes)
                or not _near(self.ram_gbps, now.ram_gbps),
                "another machine",
            ),
        )
        return next((words for changed, words in checks if changed), None)


BOX = "another box"  # The fingerprint id or the CPU changed
GIB = 2**30
SAME_BANDWIDTH = 0.10  # `mnm doctor --deep` reads the same RAM a few percent apart


def _both_differ(before: str | None, now: str | None) -> bool:
    return before is not None and now is not None and before != now


def _gib(value: int | None) -> int | None:
    """RAM in whole GiB: MemTotal moves by a few kB from one boot to the next."""
    return round(value / GIB) if value is not None else None


def _near(before: float | None, now: float | None) -> bool:
    """The same bandwidth within 10%; a box that measured none since is not another one."""
    if before is None or now is None:
        return before == now or now is None
    return abs(now - before) <= SAME_BANDWIDTH * before


class Tuned(BaseModel):
    """The tune's result on a recipe (schema 18): `changes` is applied through
    `Recipe.with_args` (validated), the plan's estimate kept."""

    changes: dict[str, Value] = Field(default_factory=dict)  # args field -> value chosen
    draft: DraftChange | None = None  # the draft turned off, or the MTP head turned on
    score: Score
    seed: Score  # mnm's own plan, as the first trial measured it
    trials: int = Field(ge=1)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    key: TuneKey
    # The queue entry this result was saved for (its `queued_at`, written in the same recipe
    # write as the result): how a run that died mid-save is told landed (`tune.orphans`),
    # whatever the clock or a later mnm's serialization. None on a record saved before it
    # existed (an older mnm drops it, and the time rule stands in).
    made_for: datetime | None = None

    @property
    def gain(self) -> float:
        """The score over the plan's, less one: 0.18 reads "tuned +18%"."""
        return self.score.value / self.seed.value - 1 if self.seed.value > 0 else 0.0

    @property
    def keys(self) -> frozenset[str]:
        """The `mnm set` keys the tune chose: a set of one drops the tune."""
        chosen = {TUNED_KEYS[f] for f in self.changes if f in TUNED_KEYS}
        return frozenset(chosen | ({DRAFT_KEY} if self.draft is not None else set()))

    @property
    def summary(self) -> str:
        """`ubatch 1024, threads 6` (or "mnm's plan" when the plan won)."""
        parts = [f"{TUNED_KEYS.get(f, f)} {_shown(v)}" for f, v in self.changes.items()]
        if self.draft == "off":
            parts.append("no draft")
        elif self.draft == "mtp":
            parts.append("draft mtp")
        return ", ".join(parts) or "mnm's plan"


def _shown(value: Value) -> str:
    return "auto" if value is None else str(value)
