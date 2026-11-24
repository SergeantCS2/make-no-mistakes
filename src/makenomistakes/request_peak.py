"""A measured llama.cpp request's peak as a reading.

coder38 at ctx 32,768 loaded at 10,362 MiB and peaked at 11,927 during `mnm try --speed
--measure` (+1.5 GiB), above the plan's 10.9-11.1 GiB: llama.cpp's compute buffer at a real
request is more than the load shows. `request_meter` keeps such a peak for sd-server's first
request only; this keeps `--measure`'s for a llama.cpp chat, coding or vision instance, and
a music model's task (its load holds the weights alone), that was running
before the request: what its start's load reading held, plus what the card grew
by during the request, keyed like that load reading (the settings, the build, the container's
start). `calibration`'s rule takes the larger at once, so the instance's figure and the chat
correction learn from it; the planner's own per-request term is left alone until two boxes
agree. Only while nothing else touched the card (`measure_window`); never fails
the try it measures.

A try that loads the model itself (parked or stopped before it, as when `mnm try coder38
--speed --measure` reads "843 MiB before, peak 11,922 MiB (+11,079) … Alone") is one too: the
whole start and request are the instance's, so its figure is the card's peak less its lowest
reading before that peak (the router may park others to make room first), counted only when
no other request or start touched the card meanwhile and no other mnm instance runs at the
end (the load and the parks are this try's own)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes import box_events
from makenomistakes.hub.facts import CHAT_LIKE, Engine
from makenomistakes.measure_window import Window
from makenomistakes.measurements import Kind, Reading, first_card_only, reading_for
from makenomistakes.music_speed import model_file
from makenomistakes.reading_stamp import CardBefore, stamp_for, window_before
from makenomistakes.supervisor import Health

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService
    from makenomistakes.recipes import Recipe

PATH = "mnm try --measure"  # a reading's `path`: which request it was


@dataclass(frozen=True)
class Before:
    """The card and the container as the try found them."""

    window: Window
    started: str
    build: str | None
    card: CardBefore | None = None  # what else held the card then
    cold: bool = False  # not running when asked; the try's request loads it


def measured(recipe: Recipe) -> bool:
    """An instance whose request grows past what its load held: a llama.cpp chat-like one,
    and a music model (a task's buffers come after ready, and its load
    reading keeps the weights alone, `instance_start._record_load`)."""
    llama = recipe.engine is Engine.LLAMACPP and recipe.modality in CHAT_LIKE
    return llama or model_file(recipe) is not None


def before(service: InstanceService, recipe: Recipe) -> Before | None:
    """The window a measured try opens, for an instance whose request is `measured`, already
    ready or loaded by the try; None otherwise."""
    if not measured(recipe):
        return None
    try:
        status = service.status(recipe)
        if not status.running and status.error is None:  # the request loads it
            window = service.card.open()
            return Before(window, "", None, window_before(window), cold=True)
        if not status.running or status.health is not Health.READY or not status.started_at:
            return None
        window = service.card.open()
        return Before(window, status.started_at, status.build, window_before(window))
    except Exception:  # a reading is a hint; the try goes on
        return None


def record(
    service: InstanceService,
    recipe: Recipe,
    began: Before | None,
    readings: list[int] | None,
    card_peaks: tuple[tuple[int, int], ...] = (),
    others: list[str] | None = None,
) -> Reading | None:
    """Save the request's peak, its start's load figure plus the card's growth; None when it
    cannot be the instance's own (no load reading for this start, another request or start
    meanwhile). On several cards each card's peak is kept as it read (`per_card`), with
    no instance figure: nothing tells this instance's share of a card from the others'.
    `others`: the other mnm instances running when the request ended (a cold try's rule)."""
    if began is None or not readings:
        return None
    if began.cold:
        return None if card_peaks else _cold(service, recipe, began, readings, others)
    if card_peaks:
        return _per_card(service, recipe, began, card_peaks)
    return _warm(service, recipe, began, readings)


def _warm(
    service: InstanceService, recipe: Recipe, began: Before, readings: list[int]
) -> Reading | None:
    """An instance that was running before the request: what its start's load held (a music
    model's ready figure: its weights) plus the growth."""
    try:
        loads = [
            r
            for r in service.measurements.readings(recipe.name)
            if r.kind is Kind.LOAD and r.container_started == began.started and _held(r)
        ]
        if not loads or not service.card.alone(recipe.name, began.window, own=1):
            return None
        if not first_card_only(service.memory, recipe):
            return None
        reading = reading_for(
            recipe,
            Kind.REQUEST,
            began.build,
            estimate_bytes=service.raw_estimate(recipe),
            vram_bytes=_held(loads[-1]) + max(max(readings) - readings[0], 0),
            container_started=began.started,
            path=PATH,
            stamp=stamp_for(service, recipe, began.card),
        )
        service.measurements.add(recipe.name, reading)
    except Exception:  # never fails the try
        return None
    return reading


def _held(load: Reading) -> int:
    """What a start's load reading says the instance held at ready."""
    return load.vram_bytes or load.ready_bytes or 0


def _cold(
    service: InstanceService,
    recipe: Recipe,
    began: Before,
    readings: list[int],
    others: list[str] | None,
) -> Reading | None:
    """A try that loaded the model. Its figure is the peak over the lowest reading before
    it; only alone (the try's own request and nothing else in flight, no other change at the
    end, no other mnm instance running after it)."""
    try:
        if not _cold_alone(service, recipe, began, others):
            return None
        status = service.status(recipe)
        if not status.running or not status.started_at:
            return None
        top = readings.index(max(readings))
        grown = readings[top] - min(readings[: top + 1])
        if grown <= 0:
            return None
        reading = reading_for(
            recipe,
            Kind.REQUEST,
            status.build,
            estimate_bytes=service.raw_estimate(recipe),
            vram_bytes=grown,
            container_started=status.started_at,
            path=PATH,
            stamp=stamp_for(service, recipe, began.card),
        )
        service.measurements.add(recipe.name, reading)
    except Exception:  # never fails the try
        return None
    return reading


def _cold_alone(
    service: InstanceService, recipe: Recipe, began: Before, others: list[str] | None
) -> bool:
    if others is None or others or began.window.traffic is None:
        return False
    now = service.card.traffic()
    return (
        now is not None
        and began.window.traffic.only(recipe.name)
        and now.only(recipe.name)
        and now.arrivals == began.window.traffic.arrivals + 1  # the try's own request only
        and not box_events.others_changing(service.settings)
        and first_card_only(service.memory, recipe)
    )


def _per_card(
    service: InstanceService, recipe: Recipe, began: Before, peaks: tuple[tuple[int, int], ...]
) -> Reading | None:
    try:
        if not service.card.alone(recipe.name, began.window, own=1):
            return None
        reading = reading_for(
            recipe,
            Kind.REQUEST,
            began.build,
            estimate_bytes=service.raw_estimate(recipe),
            container_started=began.started,
            path=PATH,
            stamp=stamp_for(service, recipe, began.card),
            per_card=peaks,
        )
        service.measurements.add(recipe.name, reading)
    except Exception:  # never fails the try
        return None
    return reading


def load_and_peak(readings: list[Reading]) -> tuple[int, int] | None:
    """The newest settings' newest load figure and its largest request peak, when both were
    measured (`mnm calibration` shows them side by side)."""
    if not readings:
        return None
    newest = readings[-1].settings
    same = [r for r in readings if r.settings == newest and r.vram_bytes]
    loads = [r.vram_bytes for r in same if r.kind is Kind.LOAD and r.vram_bytes]
    peaks = [r.vram_bytes for r in same if r.kind is Kind.REQUEST and r.vram_bytes]
    if not loads or not peaks:
        return None
    return loads[-1], max(peaks)
