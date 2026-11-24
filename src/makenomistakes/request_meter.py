"""Readings from the requests the router forwards: how fast an instance answered and,
for an engine that loads its weights at the first request, what it then holds.

Only generation POSTs are measured (`MEASURED_ROUTES`: a model listing or an engine page
asset says nothing about speed or memory), and only while they add something:
- **speed**, until `ENOUGH` readings exist for this build and recipe, or the newest is
  `STALE_DAYS` old: llama-server's own `predicted_per_second`, vLLM's completion tokens over
  the wall clock (queue and prefill included), seconds per request for the other engines;
  for sd-server seconds per image, and per clip for a queued job (`job_meter`), with
  the settings and family that made it and the fit model's seconds for them
  (`picture_speed`); its first request after a start is timed too, marked `cold` (it also
  moved the weights onto the card), and counted apart from the warm ones;
- **VRAM**, for an engine whose `resident_at` is `request` (sd-server): its first request
  after a start, on top of what the card grew by from `docker run` to ready, both while it
  was alone on the card (`measure_window`). An engine that allocates while it loads is
  measured by its start (`instance_start`); a request adds nothing to that.

Deciding costs a stat of the readings file (they are parsed again only when it changed); a
measured VRAM request reads the card once before it is forwarded and samples it while it
runs. The reading is written on a thread of its own, after the reply. Nothing here fails or
holds up the request it measures.
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from makenomistakes.fit.sdcpp import SPEED_MODEL
from makenomistakes.gpu_peak import PeakSampler
from makenomistakes.hub.facts import Engine
from makenomistakes.job_meter import JobClock, Queued, job_unit, polled_job
from makenomistakes.measure_window import Window
from makenomistakes.measurements import (
    Kind,
    Reading,
    SpeedUnit,
    first_card_only,
    reading_for,
    settings_key,
)
from makenomistakes.music_api import TASK_ROUTE  # audio.cpp's music tasks
from makenomistakes.music_speed import configured_route, model_file
from makenomistakes.picture_speed import PictureSettings, estimated_seconds, picture_settings
from makenomistakes.reading_stamp import CardBefore, card_before, stamp_for, window_before
from makenomistakes.readings_filter import instance_readings
from makenomistakes.recipes import Recipe, spec_for
from makenomistakes.reply_tail import last_number

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

ENOUGH = 3
OK, ACCEPTED = 200, 202  # a poll's answer, and a queued job's
STALE_DAYS = 7
# The routes that make something; anything else (listings, props, page assets) is not timed.
# sd-server's own `/sdcpp/v1/img_gen` and `vid_gen` only queue a job (202, then a poll):
# their time is the poll's, so they are not here.
MEASURED_ROUTES = (
    "/completions",  # and /chat/completions
    "/responses",
    "/messages",
    "/embeddings",
    "/rerank",
    "/reranking",
    "/infill",
    "/images/generations",
    "/images/edits",
    "/audio/transcriptions",
    "/audio/speech",
    TASK_ROUTE,  # audio.cpp's music tasks, timed by their own `rtf`
)
# a music task's route in the request (`request.options.route`): a key, never a lyric's text
# (a quote inside a JSON string is escaped, so `"route"` matches only a key)
_ROUTE = re.compile(rb'"route"\s*:\s*"([A-Za-z0-9_]+)"')

Container = Callable[[str], tuple[str | None, str | None]]  # name -> (build, started at)
Spawn = Callable[[Callable[[], None]], None]


@dataclass
class Tap:
    """What the router fills in once the reply is passed on."""

    status: int = 0
    tail: bytes = b""


def measured_route(method: str, path: str) -> bool:
    return method == "POST" and path.split("?", 1)[0].rstrip("/").endswith(MEASURED_ROUTES)


def speed_of(
    engine: Engine, tail: bytes, seconds: float, route: str = ""
) -> tuple[float, SpeedUnit] | None:
    """tok/s from llama.cpp's own timings, or vLLM's completion tokens over the wall clock;
    seconds per image for sd.cpp, per request for transcripts and speech; a music task's
    seconds of work a second of audio, from audio.cpp's own `rtf` (a separator's
    answer has none, and its time alone says nothing of its input's length)."""
    if route.rstrip("/").endswith(TASK_ROUTE):
        rtf = last_number(tail, "rtf")
        return (round(rtf, 3), SpeedUnit.AUDIO) if rtf is not None else None
    if engine is Engine.LLAMACPP:
        speed = last_number(tail, "predicted_per_second")  # llama-server's timings
        return (round(speed, 2), SpeedUnit.TOKENS) if speed is not None else None
    if engine is Engine.VLLM:
        tokens = last_number(tail, "completion_tokens")  # an OpenAI usage block
        if tokens is None or seconds <= 0:
            return None
        return round(int(tokens) / seconds, 2), SpeedUnit.TOKENS
    if engine is Engine.SDCPP:
        return round(seconds, 2), SpeedUnit.IMAGE
    return round(seconds, 2), SpeedUnit.SECONDS


def wanted(
    readings: list[Reading],
    settings: str,
    build: str | None,
    cold: bool = False,
    task: str | None = None,
) -> bool:
    """Whether another speed reading for this recipe on this build adds anything (a first
    request after a start, `cold`, is counted apart from the others; a music `task`'s route
    too: a repaint runs at another speed than a song)."""
    if build is None:
        return False
    same = [
        r
        for r in readings
        if r.kind is Kind.REQUEST
        and r.settings == settings
        and r.build == build
        and r.speed
        and r.cold == cold
        and r.task_route == task
    ]
    if len(same) < ENOUGH:
        return True
    return datetime.now(UTC) - same[-1].at > timedelta(days=STALE_DAYS)


def ready_figure(readings: list[Reading], started: str) -> int | None:
    """What this container start grew the card by up to ready, if that was measured."""
    loads = [
        r
        for r in readings
        if r.kind is Kind.LOAD and r.container_started == started and r.ready_bytes
    ]
    return loads[-1].ready_bytes if loads else None


def started_at(docker: str) -> datetime | None:
    """Docker's StartedAt (nanoseconds, `Z`) as a datetime; None when it does not parse."""
    head, _, fraction = docker.removesuffix("Z").partition(".")
    try:
        return datetime.fromisoformat(f"{head}.{(fraction + '000000')[:6]}+00:00")
    except ValueError:
        return None


@dataclass(frozen=True)
class _First:
    """sd-server's first generation request after a start: its peak is the figure."""

    held: int  # what the card grew by from `docker run` to ready
    window: Window
    sampler: PeakSampler  # started

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.sampler.__exit__(None, None, None)


@dataclass
class _Claim:
    """The turn one request claimed, if it did: only that request gives it back."""

    key: tuple[str, str] | None = None


@dataclass(frozen=True)
class _Plan:
    build: str
    started: str | None
    speed: bool  # time it
    first: _First | None  # set when this request's peak is the instance's figure
    cold: bool = False  # the first generation after its container started
    unit: SpeedUnit | None = None  # a queued job's: its seconds are the speed, in this unit
    card: CardBefore | None = None  # the card before the request (what else held it)
    task: str | None = None  # a music task's route, ACE-Step's `text2music` …


def task_route(recipe: Recipe, path: str, body: bytes) -> str | None:
    """The route a music task runs: the one its request names (`request.options.route`), else
    the instance's own default (a song request names none); None for any other request."""
    if not path.split("?", 1)[0].rstrip("/").endswith(TASK_ROUTE):
        return None
    found = _ROUTE.search(body)
    return found.group(1).decode() if found else configured_route(recipe)


def _counts_turns(recipe: Recipe) -> bool:
    """Whether the first request after a start is told apart (`_turn`): the engines that load
    at their first request, and a music model (its first task after a load ran at
    54 s of work a second of audio against about 5.5 warm, `ace-step-repaint-inside.json`)."""
    return spec_for(recipe).resident_at == "request" or model_file(recipe) is not None


def _spawn(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="mnm-reading", daemon=True).start()


class RequestMeter:
    def __init__(
        self,
        service: InstanceService,
        container: Container,
        clock: Callable[[], float] = time.monotonic,
        spawn: Spawn = _spawn,
    ) -> None:
        self._service = service
        self._container = container
        self._clock = clock
        self._spawn = spawn
        # Container starts a generation request already went to: only the first carries the
        # weights' load. One started before this meter may have served before it.
        self._served: set[tuple[str, str]] = set()
        self._claimed: set[tuple[str, str]] = set()  # first requests in flight
        self._since = datetime.now(UTC)
        self._lock = threading.Lock()
        self.jobs = JobClock()  # sd-server's queued clips and pictures

    def _turn(self, name: str, started: str | None, claim: _Claim) -> bool | None:
        """True for the first request of this container start (it claims the turn), False
        for one after the start was served, None when it cannot be told: a start before this
        meter, no start time, or a first request still in flight (sd-server answers one at a
        time, so this one's time would hold the weights' load and the first picture). A cold
        request timed as a warm one would read slow."""
        if started is None:
            return None
        key = (name, started)
        when = started_at(started)
        with self._lock:
            if key in self._served:
                return False
            if key in self._claimed:
                return None
            if when is None or when < self._since:
                return None
            self._claimed.add(key)
        claim.key = key
        return True

    def _mark(self, recipe: Recipe, method: str, tap: Tap, claim: _Claim) -> None:
        """When a request ends: a POST the engine took (any route: the engine page's
        `img_gen` job loads the weights too) serves this container start; the request that
        claimed the turn gives it back however it ended, and no other request touches it.
        Only for the engines that load at their first request: nothing else reads it, and
        the router's hot path stays a stat."""
        took = method == "POST" and 200 <= tap.status < 300
        if claim.key is not None:
            with self._lock:
                self._claimed.discard(claim.key)
                if took:
                    self._served.add(claim.key)
            return
        if not took or not _counts_turns(recipe):
            return
        try:
            _build, started = self._container(recipe.name)
        except Exception:
            return
        if started is not None:
            with self._lock:
                self._served.add((recipe.name, started))

    @contextmanager
    def measure(self, recipe: Recipe, method: str, path: str, body: bytes = b"") -> Iterator[Tap]:
        """Around one forwarded request (already counted in flight). A reading that cannot
        be taken is skipped; the request's own errors pass through untouched. `body` is read
        only for a music task's route."""
        tap, claim = Tap(), _Claim()
        unit, polled = job_unit(method, path), polled_job(method, path)
        if unit is not None or polled is not None:
            yield from self._job(recipe, method, unit, polled, tap, claim)
            return
        task = task_route(recipe, path, body)
        plan = self._plan(recipe, claim, task) if measured_route(method, path) else None
        if plan is None:
            try:
                yield tap
            finally:
                self._mark(recipe, method, tap, claim)
            return
        began = self._clock()
        try:
            yield tap
        finally:
            if plan.first is not None:
                plan.first.stop()
            self._mark(recipe, method, tap, claim)
        seconds = self._clock() - began
        route = path.split("?", 1)[0]
        self._spawn(lambda: self._record(recipe, route, plan, tap, seconds))

    def _job(
        self,
        recipe: Recipe,
        method: str,
        unit: SpeedUnit | None,
        polled: str | None,
        tap: Tap,
        claim: _Claim,
    ) -> Iterator[Tap]:
        """A POST that queues a job (timed from here), or a poll that may end one."""
        queued = self._queue_plan(recipe, unit) if unit is not None else None
        try:
            yield tap
        finally:
            self._mark(recipe, method, tap, claim)
        if queued is not None and tap.status == ACCEPTED:
            self.jobs.queued(recipe.name, tap.tail, queued)
        if polled is None or tap.status != OK:
            return
        done = self.jobs.done(recipe.name, polled, tap.tail)
        if done is None:
            return
        seconds = self._clock() - done.began
        plan = _Plan(
            done.build, done.started, True, None, cold=done.cold, unit=done.unit, card=done.card
        )
        route = f"/sdcpp/v1/jobs/{polled}"
        self._spawn(lambda: self._record(recipe, route, plan, tap, seconds))

    def _queue_plan(self, recipe: Recipe, unit: SpeedUnit) -> Queued | None:
        """Whether this job's time adds a reading: its container's start is known, and a
        first generation (cold) or a later one still wants one."""
        try:
            build, started = self._container(recipe.name)
            if build is None or started is None:
                return None
            when = started_at(started)
            with self._lock:
                served = (recipe.name, started) in self._served
            if not served and (when is None or when < self._since):
                return None  # started before this meter: whether it is the first is unknown
            readings = instance_readings(self._service.measurements, recipe.name)
            if not wanted(readings, settings_key(recipe), build, cold=not served):
                return None
            card = card_before(self._service)
            return Queued(self._clock(), unit, not served, build, started, card)
        except Exception:
            return None

    def _plan(self, recipe: Recipe, claim: _Claim, task: str | None = None) -> _Plan | None:
        """What to measure about this request, or None when it adds nothing."""
        try:
            service = self._service
            build, started = self._container(recipe.name)
            if build is None:
                return None
            readings = instance_readings(service.measurements, recipe.name)  # this box's
            key = settings_key(recipe)
            cold = False
            if _counts_turns(recipe):
                turn = self._turn(recipe.name, started, claim)
                if turn is None:
                    return None
                if turn and spec_for(recipe).resident_at == "request":  # its peak, cold time
                    return self._first_plan(readings, key, build, started)
                cold = turn  # a music model's first task: timed apart, never its rate
            if not wanted(readings, key, build, cold=cold, task=task):
                return None
            card = card_before(service)
            return _Plan(build, started, True, None, cold=cold, card=card, task=task)
        except Exception:  # a reading is a hint; the request goes on unmeasured
            return None

    def _first_plan(
        self, readings: list[Reading], key: str, build: str, started: str | None
    ) -> _Plan | None:
        """sd-server's first request after a start: its peak (when its ready state was
        measured) and its time, marked cold."""
        held = ready_figure(readings, started) if started else None
        timed = wanted(readings, key, build, cold=True)
        first = self._first(held)
        if first is None and not timed:
            return None
        card = window_before(first.window) if first is not None else card_before(self._service)
        return _Plan(build, started, speed=timed, first=first, cold=True, card=card)

    def _first(self, held: int | None) -> _First | None:
        """Without a figure for its ready state (that start was not alone) a first request
        cannot be priced, and the next one's growth (none: the weights are in) never stands
        in for it."""
        if held is None:
            return None
        window = self._service.card.open()
        if window.used is None:
            return None
        sampler = PeakSampler(self._service.used_vram)
        sampler.__enter__()
        return _First(held, window, sampler)

    def _record(self, recipe: Recipe, route: str, plan: _Plan, tap: Tap, seconds: float) -> None:
        try:
            self._write(recipe, route, plan, tap, seconds)
        except Exception:
            return

    def _write(self, recipe: Recipe, route: str, plan: _Plan, tap: Tap, seconds: float) -> None:
        if tap.status != 200:  # a failure, or a 202 that only queued a job
            return
        if self._service.find(recipe.name) is None:  # `mnm rm` meanwhile: its readings went
            return
        speed: tuple[float, SpeedUnit] | None = None
        if plan.unit is not None:
            speed = (round(seconds, 2), plan.unit)
        elif plan.speed:
            speed = speed_of(recipe.engine, tap.tail, seconds, route)
        vram = None
        first = plan.first
        peak = first.sampler.peak if first is not None else None
        # Another request, start or stop meanwhile would count its memory as this one's.
        if (
            first is not None
            and peak is not None
            and first.window.used is not None
            and self._service.card.alone(recipe.name, first.window)
            and first_card_only(self._service.memory, recipe)  # one card watched
        ):
            vram = first.held + max(peak - first.window.used, 0)
        if speed is None and vram is None:
            return
        stamp = stamp_for(self._service, recipe, plan.card)
        picture = self._picture(recipe, stamp.gpu) if speed is not None else None
        self._service.measurements.add(
            recipe.name,
            reading_for(
                recipe,
                Kind.REQUEST,
                plan.build,
                estimate_bytes=self._service.raw_estimate(recipe),
                vram_bytes=vram,
                container_started=plan.started,
                speed=speed[0] if speed else None,
                speed_unit=speed[1] if speed else None,
                path=route,
                cold=plan.cold,
                stamp=stamp,
                model_file=model_file(recipe) if speed and speed[1] is SpeedUnit.AUDIO else None,
                task_route=plan.task,
                **(picture.model_dump() if picture is not None else {}),
            ),
        )

    def _picture(self, recipe: Recipe, card: str | None) -> PictureSettings | None:
        """A picture's or clip's settings and the fit model's seconds for them."""
        found = picture_settings(recipe)
        if found is None:
            return None
        try:
            seconds = estimated_seconds(recipe, self._service.models_dir, card)
        except Exception:  # the seconds stand without an estimate
            return found
        return found.model_copy(update={"estimate_seconds": seconds, "speed_model": SPEED_MODEL})
