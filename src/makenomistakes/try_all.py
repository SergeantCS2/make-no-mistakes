"""`mnm try --all`: every saved instance, one after another, each its curl card's request
through the router, and one table of what came back (a canary for every model in one command).

Largest VRAM estimate first, so the big ones meet the emptiest card. Before each, every
running instance is parked (`mnm stop --park`: loadable on request, never off), so each
loads alone: its load is timed (the way `mnm start` loads it, under the box's start lock)
apart from its answer, and the VRAM it added is read from before its load to its request's
peak. What the router never parks (`workflows.held_back`: a pinned instance, the members of
a set in use) is not parked here either, and not tried: its row says so. What cannot be
tried becomes a worded row and the run goes on: one the user stopped (never loaded behind
their back), one not downloaded (never downloaded here), one the card cannot hold, one that
failed to start or answer, and whatever else any step raised (Docker, a park, a file of
mnm's): each instance's attempt is one guarded unit, so the table and its file are always
written. Every start and request leaves its readings in calibration as any other does. A
chat, coding or vision instance also gets the tool-call test after its card's request
(`mnm try --tools`): the Tools column, its verdict saved for the instance's page.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import httpx

from makenomistakes.capacity import FitNow, fit_now
from makenomistakes.held_back import held_back_in
from makenomistakes.hub.facts import CHAT_LIKE, Modality
from makenomistakes.instance_types import InstanceError, InstanceStatus
from makenomistakes.instances import InstanceService
from makenomistakes.media_bytes import tone_wav
from makenomistakes.owner_start import start_for_owner
from makenomistakes.recipes import Recipe
from makenomistakes.room_verdicts import comfy_room, lighter
from makenomistakes.smoke import TryError
from makenomistakes.smoke_replies import EmptyTranscriptError, TryResult
from makenomistakes.tool_check import ToolCheck
from makenomistakes.try_figures import Figure, try_figures
from makenomistakes.try_run import TryInputs, TryRun, UsedBytes, run_try
from makenomistakes.workflows import WorkflowStore

MIB = 2**20
NOTE_LIMIT = 160  # a refusal's start in the table; the file has the whole of it

STOPPED = "switched off by hand: left off (`mnm stop --park {name}` makes it loadable)"
NOT_DOWNLOADED = "not downloaded: `mnm start {name}` downloads it (try --all never does)"
NO_FIT = "won't fit with everything else parked: {note}"
NOT_STARTED = "did not start: {error}"
NO_ANSWER = "did not answer: {error}"
PINNED = "pinned (keep=always): left as it is, never parked for the next one; `mnm try {name}`"
IN_A_SET = (
    "a member of a set in use: left with its set, never parked for the next one; `mnm try {name}`"
)
BROKE = "not tried: {kind}: {error}"
HEADERS = (
    "name",
    "type",
    "load s",
    "answer s",
    "speed",
    "VRAM added",
    "tools",
    "figures / why not",
)
SPEED_PER = {
    Modality.IMAGE_GEN: "s/image",
    Modality.IMAGE_EDIT: "s/image",
    Modality.VIDEO: "s/clip",
}


class Refusal(StrEnum):
    """Why a row was not tried, typed (the nightly's rule tells the user's choices and
    what the run left alone from what went wrong)."""

    STOPPED = "stopped"  # by the user
    NOT_DOWNLOADED = "not downloaded"
    HELD = "held"  # pinned, or a member of a set in use
    NO_FIT = "no fit"
    NOT_STARTED = "not started"
    NO_ANSWER = "no answer"
    BROKE = "broke"  # anything a step raised

    @property
    def chosen(self) -> bool:
        """The user's doing or the run's rule, never a fault: not a regression."""
        return self in (Refusal.STOPPED, Refusal.NOT_DOWNLOADED, Refusal.HELD)


@dataclass(frozen=True)
class Row:
    """One instance's line: its figures, or why it was not tried (`why`, `refusal`)."""

    name: str
    modality: Modality
    load_seconds: float | None = None
    answer_seconds: float | None = None
    speed: str | None = None  # tok/s, s/image, s/clip, x realtime
    vram_added: int | None = None  # bytes, from before its load to its request's peak
    figures: str = ""
    why: str | None = None
    tools: str | None = None  # a chat-like one's tool-call verdict: `works 5/5`
    refusal: Refusal | None = None  # with `why`
    numbers: dict[Figure, float] = field(default_factory=dict)  # the answer's own figures

    @property
    def cells(self) -> tuple[str, ...]:
        def seconds(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        added = "-" if self.vram_added is None else f"{self.vram_added // MIB:+,} MiB"
        last = _cut(self.why) if self.why is not None else self.figures
        return (
            self.name,
            self.modality.label,
            seconds(self.load_seconds),
            seconds(self.answer_seconds),
            self.speed or "-",
            added,
            self.tools or "-",
            last,
        )


def _cut(text: str) -> str:
    return text if len(text) <= NOTE_LIMIT else text[: NOTE_LIMIT - 1] + "…"


@dataclass(frozen=True)
class Steps:
    """What a try of one instance does, injected so tests need no Docker or socket."""

    status: Callable[[Recipe], InstanceStatus]
    held: Callable[[], frozenset[str]]  # what the router never parks: never parked, not tried
    park_running: Callable[[], list[str]]  # the names it parked (never a held one)
    fits: Callable[[Recipe], str | None]  # None, or why it will not fit now
    load: Callable[[Recipe], Recipe]  # the recipe that ended up running; InstanceError
    ask: Callable[[Recipe], TryRun]  # TryError or InstanceError
    used: UsedBytes
    clock: Callable[[], float] = time.monotonic
    # a chat-like instance's tool-call test after its card's request; TryError or
    # InstanceError become the Tools column's "not tested"
    tools: Callable[[Recipe], ToolCheck] | None = None


def ordered(recipes: Iterable[Recipe], estimate: Callable[[Recipe], int | None]) -> list[Recipe]:
    """Largest VRAM estimate first; one without an estimate last; names break ties."""
    sized = [(estimate(r), r) for r in recipes]
    sized.sort(key=lambda item: (item[0] is None, -(item[0] or 0), item[1].name))
    return [recipe for _, recipe in sized]


def try_one(recipe: Recipe, steps: Steps) -> Row:
    """Park the rest, load it, send its card's request; a row whatever happens. The one
    boundary where anything any step raises becomes the row, worded with its class: a run
    over every instance never stops at one, and never loses its table."""
    loaded: list[float] = []
    try:
        return _attempt(recipe, steps, loaded)
    except Exception as exc:
        why = BROKE.format(kind=type(exc).__name__, error=_one_line(exc))
        first = next(iter(loaded), None)
        return Row(recipe.name, recipe.modality, load_seconds=first, why=why, refusal=Refusal.BROKE)


def _left_as_it_is(recipe: Recipe, steps: Steps) -> Row | None:
    """Why it is not tried at all, before anything is parked: held, stopped, not on disk."""
    name, modality = recipe.name, recipe.modality
    if name in steps.held():
        why = (PINNED if recipe.pinned else IN_A_SET).format(name=name)
        return Row(name, modality, why=why, refusal=Refusal.HELD)
    status = steps.status(recipe)
    if recipe.off and not status.running:
        return Row(name, modality, why=STOPPED.format(name=name), refusal=Refusal.STOPPED)
    if not status.files_present:
        why = NOT_DOWNLOADED.format(name=name)
        return Row(name, modality, why=why, refusal=Refusal.NOT_DOWNLOADED)
    return None


def _attempt(recipe: Recipe, steps: Steps, loaded_at: list[float]) -> Row:
    if (left := _left_as_it_is(recipe, steps)) is not None:
        return left
    name, modality = recipe.name, recipe.modality
    steps.park_running()
    if (short := steps.fits(recipe)) is not None:
        return Row(name, modality, why=NO_FIT.format(note=short), refusal=Refusal.NO_FIT)
    before = steps.used()
    began = steps.clock()
    try:
        running = steps.load(recipe)
    except InstanceError as exc:
        why = NOT_STARTED.format(error=_one_line(exc))
        return Row(name, modality, why=why, refusal=Refusal.NOT_STARTED)
    loaded = steps.clock() - began
    loaded_at.append(loaded)
    try:
        run = steps.ask(running)
    except (TryError, InstanceError) as exc:
        why = NO_ANSWER.format(error=_one_line(exc))
        return Row(name, modality, load_seconds=loaded, why=why, refusal=Refusal.NO_ANSWER)
    result = run.result
    readings = run.readings or []
    added = max(readings) - before if readings and before is not None else None
    tools, tools_note = _tools(running, steps)
    figures = _figures(modality, result.numbers)
    return Row(
        name,
        modality,
        load_seconds=loaded,
        answer_seconds=result.seconds,
        speed=_speed(modality, result.numbers, result.seconds),
        vram_added=added,
        figures=", ".join(part for part in (figures, tools_note) if part),
        tools=tools,
        numbers=dict(result.numbers),
    )


def _tools(recipe: Recipe, steps: Steps) -> tuple[str | None, str]:
    """The Tools column (`works 5/5`) and, when the test could not run, why."""
    if steps.tools is None or recipe.modality not in CHAT_LIKE:
        return None, ""
    try:
        check = steps.tools(recipe)
    except (TryError, InstanceError) as exc:
        return "not tested", f"tools: {_one_line(exc)}"
    return check.short, ""


def _figures(modality: Modality, numbers: dict[Figure, float]) -> str:
    size = _size_words(numbers)
    shown = {f: v for f, v in numbers.items() if size is None or f not in SIZE}
    words = [f"{value} {label}" for value, label in try_figures(shown)]
    words = [size, *words] if size is not None else words
    return ", ".join([TONE, *words] if modality is Modality.SPEECH_TO_TEXT else words)


SIZE = (Figure.WIDTH, Figure.HEIGHT)


def _size_words(numbers: dict[Figure, float]) -> str | None:
    """`256x256, the test request`: the picture the card's request asked for, which an editor's
    test keeps small (else kontext's row reads "256 width, 256 height" beside a 1024x1024
    instance)."""
    if Figure.WIDTH not in numbers or Figure.HEIGHT not in numbers:
        return None
    return f"{numbers[Figure.WIDTH]:.0f}x{numbers[Figure.HEIGHT]:.0f}, the test request"


def _speed(modality: Modality, numbers: dict[Figure, float], seconds: float) -> str | None:
    """The one figure to compare across runs: a picture's or a clip's seconds, a chat's tok/s, a
    voice's realtime factor."""
    if (per := SPEED_PER.get(modality)) is not None:
        return f"{seconds:.1f} {per}"
    if modality in CHAT_LIKE:
        for figure in (Figure.PREDICTED_PER_SECOND, Figure.REQUEST_TOKENS_PER_SECOND):
            if figure in numbers:
                return f"{numbers[figure]:.1f} tok/s"
    if Figure.REALTIME in numbers:
        return f"{numbers[Figure.REALTIME]:.1f}x realtime"
    return None


def _one_line(exc: Exception) -> str:
    return " ".join(str(exc).split())


def steps_for(
    service: InstanceService,
    http: httpx.Client,
    out_dir: Path,
    speed: bool,
    used: UsedBytes,
    report: Callable[[str], None] = lambda _line: None,
) -> Steps:
    """The real steps: the service's park and start, the router's request, nvidia-smi."""

    workflows = WorkflowStore(service.settings.expanded_state_dir())

    def held() -> frozenset[str]:
        """The router's rule; with the sets unreadable, the pinned ones alone."""
        statuses = service.statuses()
        found = held_back_in(workflows, statuses)
        return (
            found
            if found is not None
            else frozenset(s.recipe.name for s in statuses if s.recipe.pinned)
        )

    def park_running() -> list[str]:
        kept = held()
        running = [
            s.recipe.name for s in service.statuses() if s.running and s.recipe.name not in kept
        ]
        if running:  # by name: `stop_many(None, park=True)` would wake user-stopped ones too
            service.stop_many(running, park=True)
        return running

    def fits(recipe: Recipe) -> str | None:
        """Judged as a start judges it: what ComfyUI gives back counts as free, since the
        start frees it first (else a row reads "won't fit" for what the start would load)."""
        snapshot = lighter(service.memory(), comfy_room(service.settings).freeable)
        verdict = fit_now(service.vram_estimate(recipe), snapshot, running=False)
        return verdict.note if verdict.level is FitNow.NO_FIT else None

    def load(recipe: Recipe) -> Recipe:
        return start_for_owner(service, recipe, report).started.recipe

    def tools(recipe: Recipe) -> ToolCheck:
        run = run_try(service, recipe, http, out_dir, TryInputs(tools=True))
        if run.tools is None:  # run_try always returns the check it was asked for
            raise TryError(f"{recipe.name}: the tool-call test returned nothing.")
        return run.tools

    def ask(recipe: Recipe) -> TryRun:
        if recipe.modality is not Modality.SPEECH_TO_TEXT:
            return run_try(service, recipe, http, out_dir, TryInputs(speed=speed), used)
        return _ask_with_a_tone(service, recipe, http, out_dir, used)

    return Steps(service.status, held, park_running, fits, load, ask, used, tools=tools)


TONE = "a tone clip"  # what a speech-to-text row was tried with (`media_bytes.tone_wav`)


def _ask_with_a_tone(
    service: InstanceService, recipe: Recipe, http: httpx.Client, out_dir: Path, used: UsedBytes
) -> TryRun:
    """A speech-to-text instance gets a 1.5 s 440 Hz tone (else its row reads
    "not tried: needs a clip"): the point is that the request answers. A tone
    has no words, so an empty transcript counts as the answer."""
    with tempfile.TemporaryDirectory() as folder:
        clip = Path(folder) / "tone.wav"
        clip.write_bytes(tone_wav())
        started = time.monotonic()
        try:
            return run_try(service, recipe, http, out_dir, TryInputs(audio=clip), used)
        except EmptyTranscriptError:
            return TryRun(TryResult('"" (no words in a tone)', time.monotonic() - started))


class Stopped(Exception):  # noqa: N818 - read as "the run stopped", with what it did
    """`before` said to stop (the nightly's: the router got busy between rows)."""

    def __init__(self, at: str, why: str, rows: list[Row]) -> None:
        super().__init__(f"stopped at {at}: {why}")
        self.at = at
        self.why = why
        self.rows = rows


def try_all(
    service: InstanceService,
    steps: Steps,
    on_row: Callable[[Row], None] = lambda _row: None,
    before: Callable[[], str | None] | None = None,
) -> list[Row]:
    """Every saved instance, largest estimate first, each tried alone. `before` is asked
    ahead of every instance but the first: a reason stops the run (`Stopped`, its rows)."""
    recipes = ordered(service.all(), lambda r: service.vram_estimate(r).bytes)
    rows: list[Row] = []
    for listed in recipes:
        if rows and before is not None and (why := before()) is not None:
            raise Stopped(listed.name, why, rows)
        try:
            recipe = service.get(listed.name)  # the saved file now, not the listing's
        except Exception as exc:  # removed meanwhile, or unreadable: a row, never the end
            why = BROKE.format(kind=type(exc).__name__, error=_one_line(exc))
            row = Row(listed.name, listed.modality, why=why, refusal=Refusal.BROKE)
        else:
            row = try_one(recipe, steps)
        on_row(row)
        rows.append(row)
    return rows


def plain_table(rows: list[Row]) -> str:
    """The table as plain text, columns padded: what `try-all-<time>.txt` holds."""
    lines = [HEADERS, *(row.cells for row in rows)]
    widths = [max(len(line[i]) for line in lines) for i in range(len(HEADERS) - 1)]
    rule = (*("-" * width for width in widths), "-" * len(HEADERS[-1]))
    out = []
    for line in [lines[0], rule, *lines[1:]]:
        padded = [cell.ljust(width) for cell, width in zip(line, widths, strict=False)]
        out.append("  ".join([*padded, line[-1]]).rstrip())
    return "\n".join(out) + "\n"


def report_name(now: datetime) -> str:
    """`try-all-<local time>.txt`: one per run, in time order in one folder."""
    return f"try-all-{now:%Y-%m-%d-%H%M%S}.txt"


def write_report(folder: Path, rows: list[Row], now: datetime | None = None) -> Path:
    """The table in plain text, and every refusal the table cut, whole."""
    when = now or datetime.now()
    whole = [f"{row.name}: {row.why}" for row in rows if row.why and len(row.why) > NOTE_LIMIT]
    text = f"mnm try --all, {when:%Y-%m-%d %H:%M:%S}\n\n" + plain_table(rows)
    if whole:
        text += "\nIn full:\n" + "\n".join(whole) + "\n"
    path = folder / report_name(when)
    path.write_text(text, encoding="utf-8")
    return path
