"""What the box measured: one JSON line per reading under
`<state>/measurements/<name>.jsonl`, appended as starts and requests happen.

A reading names the engine build it ran on (the container's image id) and what sizes its
memory (a hash of the args and the files the recipe loads, with the args themselves), so a
figure is only ever reused for the same build, settings and files. `calibration` turns
readings into figures; nothing here
decides anything. A file that cannot be read, or a line that does not parse (an older or
newer shape), is skipped: a reading is a hint, never a reason a command fails.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.backend import Backend
from makenomistakes.engine_args import PROMPT_CACHE_FIELDS, AudioCppArgs, LlamaCppArgs, SdCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.engine_args_whisper import WhisperArgs
from makenomistakes.engine_images import IMAGES
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.hub.kit_types import ImageFamily
from makenomistakes.locking import shared_lock
from makenomistakes.quality.figures import QualityFigures
from makenomistakes.queue_figures import QueueFigures
from makenomistakes.recipes.cards import recipe_cards
from makenomistakes.recipes.model import Recipe

FOLDER = "measurements"
# A reading's shape is 2 (the container's build, a key over the files,
# the type): a line of the first shape (no `shape`) means something else and is skipped.
# readings kept per instance, box and kind: the oldest go once one of them holds twice as many
KEEP = 200

Args = LlamaCppArgs | SdCppArgs | WhisperArgs | AudioCppArgs | VllmArgs


class Kind(StrEnum):
    LOAD = "load"  # a start: the VRAM it took from before `docker run` to ready
    REQUEST = "request"  # a request the router forwarded: its peak and its speed
    SCORE = "score"  # A candidate the studio's scorer heard (`studio/scores.py`)
    PREFIX = "prefix"  # A long prefix read cold, warm and after a park (`prefix_check`)
    QUEUE = "queue"  # A window of the router's queue for an instance (`router_queue`)
    QUALITY = "quality"  # One quant's scores against a reference (`quality.readings`)


class SpeedUnit(StrEnum):
    TOKENS = "tok/s"  # llama.cpp's timings, or vLLM's completion tokens over the wall clock
    SECONDS = "s/request"  # a transcription, speech: the wall clock of one request
    IMAGE = "s/image"  # sd.cpp: one picture's request, or its queued job
    CLIP = "s/clip"  # sd.cpp: a queued clip, from the POST to the poll that says completed
    AUDIO = "s/s"  # audio.cpp's `timing.rtf`, seconds of work a second of audio


class PrefixFigures(BaseModel):
    """What the prompt cache saves an agent's prefix (`prefix_check`): its tokens, the
    seconds llama.cpp read it cold and warm (the tokens it reused), and after a park, with the
    saved slot put back, the seconds read and reused and the load and restore's own."""

    tokens: int
    cold_seconds: float
    warm_seconds: float
    reused_tokens: int
    restored_seconds: float | None = None
    restored_reused: int | None = None
    reload_seconds: float | None = None


class Reading(BaseModel):
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: Kind
    engine: Engine
    modality: Modality  # the type: its fit model's correction (`calibration.fit_group`)
    shape: Literal[2]  # a line of another shape (an early build's) does not parse
    build: str | None  # the build its container ran (the image's short id); None: unknown
    settings: str  # `settings_key(recipe)`: the same key means the same memory-sizing recipe
    args: Args = Field(discriminator="engine")
    estimate_bytes: int | None = None  # the fit model's figure for these settings, then
    # What this instance holds on the GPU: at ready for an engine that allocates while it
    # loads, at its first request for one that loads its weights then (`EngineSpec.resident_at`)
    vram_bytes: int | None = None
    ready_bytes: int | None = None  # what the card grew by from `docker run` to ready
    container_started: str | None = None  # Docker's StartedAt: which start the reading is of
    load_seconds: float | None = None  # `docker run` to /health 200
    speed: float | None = None
    speed_unit: SpeedUnit | None = None
    path: str | None = None  # the request's route, for a request reading
    # A picture's or clip's time: the settings that made it, its family, and the fit
    # model's seconds for them then (None: no estimate, e.g. streamed weights). `cold`: the
    # first request after a start, which also moved the weights onto the card.
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    frames: int | None = None
    family: ImageFamily | None = None
    estimate_seconds: float | None = None
    cold: bool = False
    # The card it was taken on, nvidia-smi's name: `mnm restore` leaves out
    # another card's. None: an older line, or the card could not be read (this card's).
    gpu: str | None = None
    # The picture-speed model `estimate_seconds` came from (`fit.sdcpp.SPEED_MODEL` then); None
    # is model 1. `fit.sdcpp.speed_refit` rescales a
    # reading of an older model to today's estimate; one of a model this mnm does not know
    # corrects nothing.
    speed_model: int | None = None
    # Each card's peak memory in use during a `--measure` on several cards (nvidia-smi's
    # index and bytes, everything on that card): the per-card readings a placement's
    # estimate waits for; no instance figure is learned from them
    per_card: tuple[tuple[int, int], ...] = ()
    # The box it was taken on (`machine.json`'s current fingerprint id; None: an older
    # line, which belongs to the first box mnm saw, `readings_filter`) and the cards' stable
    # ids (nvidia-smi's uuids) in nvidia-smi's order. Every consumer reads this box's only.
    machine: str | None = None
    cards: list[str] = Field(default_factory=list)
    # What else held the card before the request or start: nvidia-smi's memory in
    # use less what mnm's running instances hold (a browser's video on Windows), and the
    # card's utilization percent then; None when it could not be read. A flag, never a filter.
    outside_bytes: int | None = None
    gpu_busy: int | None = None
    # The backend the instance's image is built for (`engine_images`: cuda12, rocm,
    # vulkan, cpu; None for an image mnm does not list, or an older line): a Vulkan reading
    # and a ROCm one differ in build too, and this says which without the ledger
    backend: str | None = None
    # A song candidate's score (the studio's scorer, `studio/tools_answers.ScoreAnswer`):
    # the one number, what scored it (its package and version), every axis it measured, and
    # the seconds of audio heard. A reading of kind `score` carries no memory or speed.
    score: float | None = None
    scorer: str | None = None
    prefix: PrefixFigures | None = None  # A `Kind.PREFIX` reading's figures
    queue: QueueFigures | None = None  # A `Kind.QUEUE` reading's window
    quality: QualityFigures | None = None  # A `Kind.QUALITY` reading's scores
    axes: dict[str, float] = Field(default_factory=dict)
    audio_seconds: float | None = None
    scored_run: str | None = None  # the candidate's run: scored again, the newest counts
    # A music model's file name beside its `s/s` speed (`music_speed`): turbo and base
    # are one family and differ by their files; and the task's route the request named
    # (ACE-Step's `text2music`, `repaint` …, a converter's singing route), each its own speed
    model_file: str | None = None
    task_route: str | None = None

    @property
    def ratio(self) -> float | None:
        """Measured over estimated: the engine's correction is the median of these."""
        if not self.vram_bytes or not self.estimate_bytes:
            return None
        return self.vram_bytes / self.estimate_bytes

    @property
    def speed_ratio(self) -> float | None:
        """A picture's or clip's seconds over the fit model's: a family's speed correction
        is the median of these."""
        if self.speed_unit not in (SpeedUnit.IMAGE, SpeedUnit.CLIP):
            return None
        if not self.speed or not self.estimate_seconds:
            return None
        from makenomistakes.fit.sdcpp import speed_refit  # noqa: PLC0415 - no import cycle

        refit = speed_refit(self.family, self.speed_model)
        return self.speed * refit / self.estimate_seconds if refit is not None else None


# What sizes an instance's memory besides its args: the files it loads (a replan's new quant,
# a LoRA added, another model under the same name after `mnm rm` and `mnm add`).
_SIZING = (
    "repo_id",
    "revision",
    "layout",
    "files",
    "projector",
    "companions",
    "loras",
    "high_noise",
    "upscaler",
    "quant",
)


# A draft model's own settings: part of the key only while a draft runs.
_DRAFT_ARGS = ("draft_max", "draft_p_min")


def settings_key(recipe: Recipe) -> str:
    """The same key for the same args and files, whatever order a file stored them in. A
    draft model is part of it (its file, `draft_max`, `draft_p_min`): else qwen3-8b read with a
    draft at 9.63 GiB lands under the key it has alone, and the figure learned from it (9.6)
    says "won't fit" for the model alone (7.1). Without a draft the draft's settings are left out,
    so a draft-max saved under a past default keys nothing, and the key is the one from before
    draft settings existed: readings keyed with a default `draft_max` are left behind, older
    ones count again."""
    sizing = recipe.model_dump(mode="json", include={"args", *_SIZING})
    args = sizing.get("args")
    if isinstance(args, dict):  # one card's key as before the card fields: unset ones go
        for name in _CARD_ARGS:
            if args.get(name) is None:
                args.pop(name, None)
        if not args.get("lazy_tensors"):  # the Engram flags, left out when there are none
            args.pop("lazy_tensors", None)
        if args.get("numa") is None:  # the CPU backend on several NUMA nodes only
            args.pop("numa", None)
        if args.get("numa_cpus") is None:  # `numa=node:N` only
            args.pop("numa_cpus", None)
        if args.get("options") is None:  # A music instance's options; a voice has none
            args.pop("options", None)
        for name in PROMPT_CACHE_FIELDS:  # RAM and disk beside the engine, never its speed
            args.pop(name, None)
    if recipe.gpus:  # the cards it is pinned to: a reading on others says nothing of these
        sizing["gpus"] = list(recipe.gpus)
    if isinstance(args, dict) and args.get("engine") == "llamacpp":
        drafting = {name: args.pop(name, None) for name in _DRAFT_ARGS}
        if recipe.draft is not None:
            sizing["draft"] = {
                "repo_id": recipe.draft.repo_id,
                "file": recipe.draft.file,
                "quant": recipe.draft.quant,
                **drafting,
            }
    text = json.dumps(sizing, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# The card fields and the placement, left out of the key while unset (one card)
_CARD_ARGS = ("tensor_split", "split_mode", "main_gpu", "tensor_parallel", "placement", "experts")
CARD_JOIN = " + "  # several cards in a reading's `gpu`, in nvidia-smi's order


def gpu_name(memory: Callable[[], MemorySnapshot], recipe: Recipe | None = None) -> str | None:
    """The card a reading is taken on; None when it cannot be read (never a failed reading).
    On several cards the names of the cards the instance runs on, joined in
    nvidia-smi's order: a string still, so an older mnm reading the file (a restore) keeps
    the line and leaves it out as another card's."""
    try:
        snapshot = memory()
    except Exception:
        return None
    return gpu_name_of(snapshot, recipe)


def first_card_only(memory: Callable[[], MemorySnapshot], recipe: Recipe) -> bool:
    """The instance runs on the first chosen card alone, the one card a VRAM reading watches
    (`MeasureWindow`): on several cards, or another one, its VRAM is not measured (split
    readings are not taken); its speed and load time still are."""
    try:
        snapshot = memory()
    except Exception:
        return False
    first = snapshot.gpu
    if first is None or (len(snapshot.cards) < 2 and not recipe.gpus):
        return True
    cards = recipe_cards(recipe, listed=[g.index for g in snapshot.cards])
    return cards == (first.index,)


def gpu_name_of(snapshot: MemorySnapshot, recipe: Recipe | None = None) -> str | None:
    """`gpu_name` of a snapshot already read."""
    gpu = snapshot.gpu
    if recipe is None or len(snapshot.cards) < 2:
        return gpu.name if gpu is not None else None
    cards = recipe_cards(recipe, listed=[g.index for g in snapshot.cards]) or ()
    names = [c.name for i in sorted(cards) if (c := snapshot.card(i)) is not None]
    return CARD_JOIN.join(names) or (gpu.name if gpu is not None else None)


def card_names(gpu: str | None) -> list[str]:
    """A reading's cards in order: `Reading.gpu` split; an older reading's one name is that
    card."""
    return gpu.split(CARD_JOIN) if gpu else []


@dataclass(frozen=True)
class Stamp:
    """Where and beside what a reading was taken: the card's name (`gpu`), the box
    (`machine`), the cards' stable ids, and what else held the card then. `reading_stamp`
    builds one; every reading is made through `reading_for`, which requires it, so no writer
    leaves a reading without its box."""

    gpu: str | None = None
    machine: str | None = None
    cards: tuple[str, ...] = ()
    outside_bytes: int | None = None
    gpu_busy: int | None = None


def reading_for(
    recipe: Recipe,
    kind: Kind,
    build: str | None,
    *,
    stamp: Stamp | None = None,
    started_with: Reading | None = None,
    **values: object,
) -> Reading:
    """The one place a reading is made: the recipe's key and the stamp's box. Every writer
    passes `stamp`; a value given beside it wins.
    `started_with`: the load reading of the container start the reading counts
    under, whose settings, args and backend it takes instead of the recipe's as saved since
    (a queue window of a container started before `mnm set` or `mnm replan`)."""
    stamped: dict[str, object] = {}
    if stamp is not None:
        stamped = {
            "gpu": stamp.gpu,
            "machine": stamp.machine,
            "cards": list(stamp.cards),
            "outside_bytes": stamp.outside_bytes,
            "gpu_busy": stamp.gpu_busy,
        }
    ran = started_with
    return Reading(
        kind=kind,
        engine=recipe.engine,
        modality=recipe.modality,
        shape=2,
        build=build,
        settings=ran.settings if ran is not None else settings_key(recipe),
        args=ran.args if ran is not None else recipe.args,
        backend=ran.backend if ran is not None else built_for(recipe),
        **(stamped | values),
    )


def start_reading(readings: list[Reading], started: str) -> Reading | None:
    """The load reading of one container start (Docker's StartedAt), newest first: the args
    and settings that container started with, which a recipe saved since does not change
    (the router's queue counts under them too). None when no reading names that start."""
    return next(
        (r for r in reversed(readings) if r.kind is Kind.LOAD and r.container_started == started),
        None,
    )


def built_for(recipe: Recipe) -> str | None:
    """The backend the instance's image is built for (`engine_images`), as a reading says it."""
    found = IMAGES[recipe.engine].backend_of(recipe.image)
    return found.value if found is not None else None


def readings_backend(engine: Engine, backend: Backend | None, arch: str | None) -> str | None:
    """The backend this box's readings of `engine` name when its plans are for `backend`: the
    one the image they run is listed under (`EngineImages.runs_as`), which is what
    `built_for` stamped on every reading, older ones included. Every reader that compares
    readings with a box's backend asks this first (the box's own, `cuda13`
    on an RTX 4070, matches no reading of its `server-cuda` or `full-cuda12` images). None
    (every reading counts) for a budget that names no backend."""
    if backend is None:
        return None
    return IMAGES[engine].runs_as(backend, arch or "x86_64").value


class _Box(BaseModel):
    machine: str | None = None  # a reading's box and kind, and nothing else of it
    kind: str | None = None


def trimmed(lines: list[str]) -> list[str]:
    """Once a box's readings of one kind pass `2 * KEEP`, the newest `KEEP` of each box and
    kind, in their order; else the lines as they are. Another box's readings are kept for
    when it returns (a trim of the oldest whatever box wrote them would leave a card
    swapped back with nothing), and the router's queue windows, one every
    burst of requests, never push out a start's or a request's figures. A line that does not
    parse goes with the readings of no box and no kind."""
    groups: list[tuple[str | None, str | None]] = []
    for line in lines:
        try:
            found = _Box.model_validate_json(line)
        except ValidationError:
            found = _Box()
        groups.append((found.machine, found.kind))
    sizes: dict[tuple[str | None, str | None], int] = {}
    for group in groups:
        sizes[group] = sizes.get(group, 0) + 1
    if all(size <= 2 * KEEP for size in sizes.values()):
        return lines
    seen: dict[tuple[str | None, str | None], int] = {}
    keep = [False] * len(lines)
    for at in range(len(lines) - 1, -1, -1):
        group = groups[at]
        seen[group] = seen.get(group, 0) + 1
        keep[at] = seen[group] <= KEEP
    return [line for line, kept in zip(lines, keep, strict=True) if kept]


class MeasurementStore:
    """Thread- and process-safe appends (the router and the CLI both write)."""

    def __init__(self, state_dir: Path) -> None:
        self.folder = state_dir / FOLDER
        # name -> (the file's mtime and size when parsed, its readings): the router asks on
        # every request it may measure, and a file changes only when a reading is added
        self._parsed: dict[str, tuple[tuple[int, int], list[Reading]]] = {}
        self._parse_lock = threading.Lock()

    def path(self, name: str) -> Path:
        return self.folder / f"{name}.jsonl"

    def add(self, name: str, reading: Reading) -> None:
        """Append one reading; past `2 * KEEP` of one box and kind the file is cut back to the
        newest `KEEP` of each (`trimmed`). A reading that cannot be saved is dropped:
        measuring never fails what it measures."""
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with shared_lock(self.folder)():
                path = self.path(name)
                with path.open("a", encoding="utf-8") as out:
                    out.write(reading.model_dump_json() + "\n")
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) > 2 * KEEP and len(kept := trimmed(lines)) < len(lines):
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
                    os.replace(tmp, path)
        except OSError:
            return

    def readings(self, name: str) -> list[Reading]:
        """Oldest first; lines that do not parse are skipped. Parsed again only when the
        file changed (its mtime or size)."""
        path = self.path(name)
        try:
            info = path.stat()
            stamp = (info.st_mtime_ns, info.st_size)
            with self._parse_lock:
                cached = self._parsed.get(name)
            if cached is not None and cached[0] == stamp:
                return list(cached[1])
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        found = []
        for line in text.splitlines():
            try:
                reading = Reading.model_validate_json(line)
            except ValidationError:
                continue
            found.append(reading)
        with self._parse_lock:
            self._parsed[name] = (stamp, found)
        return list(found)

    def forget(self, name: str) -> None:
        """`mnm rm`: another model saved under the name later starts with no readings."""
        with contextlib.suppress(OSError):
            self.path(name).unlink(missing_ok=True)
        with self._parse_lock:
            self._parsed.pop(name, None)

    def names(self) -> list[str]:
        try:
            return sorted(p.stem for p in self.folder.glob("*.jsonl"))
        except OSError:
            return []

    def every(self) -> dict[str, list[Reading]]:
        return {name: self.readings(name) for name in self.names()}

    def drop(self, gone: Callable[[Reading], bool]) -> int:
        """Remove every reading `gone` names, in every file (`mnm machine forget`);
        lines that do not parse stay. The readings removed are counted."""
        removed = 0
        with contextlib.suppress(OSError), shared_lock(self.folder)():
            for name in self.names():
                path = self.path(name)
                lines = path.read_text(encoding="utf-8").splitlines()
                kept = [line for line in lines if not _gone(line, gone)]
                if len(kept) == len(lines):
                    continue
                removed += len(lines) - len(kept)
                tmp = path.with_suffix(".tmp")
                tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
                os.replace(tmp, path)
        return removed


def _gone(line: str, gone: Callable[[Reading], bool]) -> bool:
    try:
        return gone(Reading.model_validate_json(line))
    except ValidationError:
        return False
