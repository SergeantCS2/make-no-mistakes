"""A new engine build tests itself before it is trusted.

After `mnm engines update` pulls a build that moved, the smallest downloaded instance on that
engine that fits beside what runs is copied (`selftest-<engine>`, a spare loopback port, its
recipe as a start would run it: `start_fixes.as_started`). The copy runs first on the build
the ledger kept, then on the new one, each answering `mnm try`'s request on the engine's own
port, so both builds do the same work. Nothing is downloaded or asked; what holds the room
the copy needs is parked as a start parks it (`selftest_room`).

- **Failed** (the update rolls back): the new build did not start, did not answer as the card
  expects, took more than `VRAM_OVER` more VRAM, or ran more than `SPEED_UNDER` slower than
  the previous build did.
- **Inconclusive** (the update stands, and says so): the previous build failed on the same
  work too, so the new one is not shown to be worse (the card was full, the recipe refused).
- **Skipped:** nothing to test with (no instance on the engine, none downloaded, nothing fits,
  the image not pulled, a recipe a start would refuse).
- **Passed.**

Without the previous build (pruned, or `mnm engines test`, where the ledger's may be the
build a rollback rejected), the instance's own VRAM readings are the bar: the least any
other build was measured holding. Speed is judged only against the same work, never against
clients' requests, whose lengths differ. The copies record no readings of their own.
"""

from __future__ import annotations

import contextlib
import socket
import tempfile
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from makenomistakes import box_events
from makenomistakes.capacity import FitNow, fit_now
from makenomistakes.consumers import engine_origin
from makenomistakes.diagnose import Cause, diagnose
from makenomistakes.gpu_peak import PeakSampler
from makenomistakes.hub.facts import CHAT_LIKE, Engine, Modality
from makenomistakes.instance_types import InstanceError
from makenomistakes.last_logs import forget_log, keep_log, log_path
from makenomistakes.measurements import SpeedUnit, settings_key
from makenomistakes.progress_words import parked_for_room
from makenomistakes.readings_filter import here_of, instance_readings
from makenomistakes.recipes import Recipe, spec_for
from makenomistakes.recipes.llamacpp_cache import without_slots
from makenomistakes.recipes.model import SELFTEST_PREFIX
from makenomistakes.selftest_judge import (
    SPEED_UNDER,
    VRAM_OVER,
    Figures,
    Speed,
    baseline,
    compared,
    judge,
    with_tuned_speed,
)
from makenomistakes.selftest_room import NOTHING_FITS, room_for_copy, smallest_present
from makenomistakes.smoke import EmptyTranscriptError, TryError, TryResult, try_instance
from makenomistakes.start_fixes import StartRefusedError, as_started
from makenomistakes.supervisor import Health, LocalImage, SupervisorError
from makenomistakes.swap import start_turn
from makenomistakes.try_figures import Figure

__all__ = ["SPEED_UNDER", "VRAM_OVER", "Figures", "Speed", "baseline", "judge"]  # moved

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

REQUEST_SECONDS = 900.0  # a clip on the card takes minutes
# `docker run` lost the race for the spare port: to another container, or to a host process
PORT_TAKEN = ("port is already allocated", "address already in use")

Progress = Callable[[str], None]


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SelfTest:
    engine: Engine
    image: str
    outcome: Outcome
    reason: str  # one line, worded for the user
    instance: str | None = None  # the instance the copy was made of
    build: str | None = None  # the build tested
    measured: Figures = field(default_factory=Figures)
    baseline: Figures = field(default_factory=Figures)
    log: Path | None = None  # the copy's log, kept when it failed


def copy_name(engine: Engine) -> str:
    return f"{SELFTEST_PREFIX}{engine.value}"  # reserved: `check_name` refuses it


def choose(service: InstanceService, image: str) -> tuple[Recipe | None, str]:
    """The smallest downloaded instance on this image that fits beside what runs now, or
    why there is none. One without an estimate is tried last."""
    every = service.all()
    if taken := sorted(r.name for r in every if r.name.startswith(SELFTEST_PREFIX)):
        return None, (
            f"an instance is named {taken[0]}, a name kept for the test's copies; add it again "
            "under another name (`mnm add … --name …`, which plans it afresh), then `mnm rm` it"
        )
    users = [r for r in every if r.image == image]
    if not users:
        return None, "no instance uses it"
    present = [r for r in users if service.files_present(r)]
    if not present:
        return None, "no instance using it is downloaded"
    snapshot = service.memory()
    priced = [(service.vram_estimate(r), r) for r in present]
    priced.sort(key=lambda pair: (pair[0].bytes is None, pair[0].bytes or 0, pair[1].name))
    for estimate, recipe in priced:
        level = fit_now(estimate, snapshot, running=False).level
        if level in {FitNow.FITS, FitNow.NEAR_LIMIT, FitNow.UNKNOWN}:
            return recipe, ""
    smallest = priced[0][1].name
    return None, (
        f"{NOTHING_FITS}{smallest} fits beside what runs now, and nothing mnm may park holds "
        "the room; `mnm engines test` runs it when more of the card is free"
    )


def spare_port(lost: int | None = None) -> int:
    """A loopback port nothing listens on now (the kernel's pick), never the one a start just
    lost the race for: the kernel may hand the same port back while the winner holds it."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        if port != lost:
            return port


def silent_clip(folder: Path) -> Path:
    """One second of silence: what a speech-to-text copy is sent (it answers with nothing)."""
    path = folder / "silence.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * 16000)
    return path


class Stage(IntEnum):
    """How far a copy got before it failed: a build that fails earlier than the previous
    build did on the same work is worse, whatever stopped the previous one."""

    START = 1  # did not start or get ready
    ANSWER = 2  # started, answered wrongly


@dataclass(frozen=True)
class _Trial:
    figures: Figures = field(default_factory=Figures)
    failure: str | None = None  # why the copy did not start or answer
    stage: Stage | None = None  # where it failed
    log: Path | None = None
    error: Exception | None = None


class EngineSelfTest:
    def __init__(
        self,
        service: InstanceService,
        http: Callable[[], httpx.Client] | None = None,
        clock: Callable[[], float] = time.monotonic,
        progress: Progress | None = None,
    ) -> None:
        self._service = service
        self._http = http or (lambda: httpx.Client(timeout=REQUEST_SECONDS, trust_env=False))
        self._clock = clock
        self._progress = progress
        self._parked: list[str] = []
        self._room_for = ""

    def run(self, engine: Engine, image: str, previous: str | None = None) -> SelfTest:
        """`_run`, and what it parked to make room for its copy said first."""
        self._parked = []
        result = self._run(engine, image, previous)
        if not self._parked:
            return result
        said = parked_for_room(self._parked, f"the test's copy of {self._room_for}")
        return replace(result, reason=f"{said} {result.reason}")

    def _run(self, engine: Engine, image: str, previous: str | None = None) -> SelfTest:
        """Test the build `image` names now against `previous` (the image id the ledger kept
        at this update), else against the readings. Never raises for the build's failures."""
        service = self._service
        ready = self._ready(engine, image)
        if isinstance(ready, SelfTest):
            return ready
        recipe, copy, found = ready
        result = SelfTest(engine, image, Outcome.PASSED, "", recipe.name, found.short_id)
        old = service.supervisor.image(previous) if previous else None
        before: _Trial | None = None
        if old is not None and old.id != found.id:
            self._say(f"Running the previous build {old.short_id} first, for figures to compare")
            before = self._trial(copy.model_copy(update={"image": old.id}), recipe)
            before = replace(before, figures=replace(before.figures, build=old.short_id))
        self._say(f"Testing {image} with a copy of {recipe.name}")
        trial = self._trial(copy, recipe)
        if before is not None and before.failure is not None and not _earlier(trial, before):
            return self._unjudged(result, trial, before, old.short_id if old else "")
        bar = before.figures if before is not None else self._readings_bar(recipe, found.short_id)
        bar = with_tuned_speed(
            bar, recipe, previous, here_of(service.settings.expanded_state_dir())
        )
        result = replace(result, measured=trial.figures, baseline=bar)
        if trial.failure is not None:
            return self._failed(result, trial)
        if (failure := judge(trial.figures, bar)) is not None:
            return replace(result, outcome=Outcome.FAILED, reason=f"Failed: the copy {failure}.")
        return replace(
            result,
            reason=f"Passed: a copy of {recipe.name} started and answered"
            f"{compared(trial.figures, bar)}.",
        )

    def _ready(self, engine: Engine, image: str) -> tuple[Recipe, Recipe, LocalImage] | SelfTest:
        """The instance, its copy and the build to test, or why nothing is tested."""
        service = self._service
        recipe, why = choose(service, image)
        if recipe is None and why.startswith(NOTHING_FITS):
            refused = self._make_room(engine, image)
            if refused is not None:
                return refused
            recipe, why = choose(service, image)
        if recipe is None:
            return SelfTest(engine, image, Outcome.SKIPPED, f"Not tested: {why}.")
        found = service.supervisor.image(image)
        if found is None:  # `docker run` would pull it: only `mnm engines update` may
            return SelfTest(engine, image, Outcome.SKIPPED, "Not tested: the image is not pulled.")
        try:
            started = as_started(recipe, service.models_dir, service.memory)
        except StartRefusedError as exc:
            return SelfTest(engine, image, Outcome.SKIPPED, f"Not tested: {exc}")
        copy = started.model_copy(update={"name": copy_name(engine), "port": spare_port()})
        return recipe, without_slots(copy), found  # The instance's saved prompts stay

    def _make_room(self, engine: Engine, image: str) -> SelfTest | None:
        """Nothing on the engine fits now: park for its smallest instance what a start may
        park (`selftest_room.room_for_copy`: never a pinned model or a member of a set in use,
        each after its replies), as `mnm sets start` and the router do (else a parkable model
        holding the card leaves the engine untested). A refusal naming what is held back is the
        test's own refusal."""
        service = self._service
        smallest = smallest_present(service, image)
        if smallest is None:
            return None
        self._room_for = smallest.name
        try:
            with start_turn(service.settings, smallest.name, self._progress):
                say = None if self._progress is None else self._say_step
                self._parked = room_for_copy(service, smallest, say)
        except InstanceError as exc:
            return SelfTest(
                engine, image, Outcome.SKIPPED, f"Not tested: a copy of {smallest.name}: {exc}"
            )
        return None

    def _readings_bar(self, recipe: Recipe, build: str) -> Figures:
        readings = instance_readings(self._service.measurements, recipe.name)  # this box's
        return baseline(readings, settings_key(recipe), build)

    def _unjudged(self, result: SelfTest, trial: _Trial, before: _Trial, old: str) -> SelfTest:
        """The previous build failed on the same work: nothing shows the new one is worse."""
        if trial.failure is None:
            reason = (
                f"Passed: the new build answered where the previous build {old} failed "
                f"({before.failure})."
            )
            return replace(result, measured=trial.figures, reason=reason)
        reason = (
            f"Not judged: the previous build {old} failed on the same work too "
            f"({before.failure}), so the new build stays; fix that, then `mnm engines test`."
        )
        return replace(result, outcome=Outcome.INCONCLUSIVE, reason=reason, log=trial.log)

    def _say_step(self, line: str) -> None:
        """A step from `swap` (which ends its own with "…") on this test's progress, which adds
        one."""
        self._say(line.removesuffix("…"))

    def _say(self, line: str) -> None:
        if self._progress:
            self._progress(line)

    def _trial(self, copy: Recipe, recipe: Recipe) -> _Trial:
        """Start the copy, send it the card's request, and leave nothing behind but the log
        of a failure."""
        settings = self._service.settings
        forget_log(settings.expanded_state_dir(), copy.name)  # never an earlier test's
        # The copy is a change on the box from its start to its removal: its load and its
        # request (straight to its port, unseen by the router) are never another's reading.
        with box_events.changing(settings):
            try:
                return self._run_copy(copy, recipe)
            finally:
                self._clean(copy)

    def _run_copy(self, copy: Recipe, recipe: Recipe) -> _Trial:
        try:
            loaded = self._load(copy)
        except (SupervisorError, OSError, ValueError) as exc:
            return self._failure(copy, Stage.START, f"the copy of {recipe.name} did not start", exc)
        if loaded is None:
            why = f"the copy of {recipe.name} was not ready in time"
            return self._failure(copy, Stage.START, why)
        copy, load_vram = loaded
        try:
            seconds, answer, peak = self._ask(copy)
        except TryError as exc:
            why = f"the copy of {recipe.name} answered wrongly"
            return self._failure(copy, Stage.ANSWER, why, exc)
        vram = load_vram + peak if load_vram is not None and peak is not None else None
        return _Trial(Figures(vram, _speed(copy, answer, seconds)))

    def _load(self, copy: Recipe) -> tuple[Recipe, int | None] | None:
        """Start the copy under the box's start lock: the copy as it ran (its port may have
        moved once, after a lost race for it) and the VRAM it took, None unless nothing else
        touched the card. None when it never got ready."""
        service = self._service
        supervisor = service.supervisor
        with start_turn(service.settings, copy.name, self._progress):
            supervisor.stop(copy.name)  # a copy an interrupted test left behind
            if (prepare := spec_for(copy).prepare) is not None:
                prepare(copy, service.models_dir)
            window = service.card.open()
            try:
                supervisor.start(copy, service.models_dir)
            except SupervisorError as exc:
                if not any(taken in str(exc) for taken in PORT_TAKEN):
                    raise
                copy = copy.model_copy(update={"port": spare_port(lost=copy.port)})
                supervisor.start(copy, service.models_dir)
            if supervisor.wait_ready(copy) is not Health.READY:
                return None
            return copy, service.card.grew(copy.name, window)

    def _ask(self, copy: Recipe) -> tuple[float, TryResult | None, int | None]:
        """`mnm try`'s request on the engine's own port: (seconds, answer, VRAM it added)."""
        service = self._service
        window = service.card.open()
        sampler = PeakSampler(service.used_vram) if window.used is not None else None
        began = self._clock()
        with tempfile.TemporaryDirectory(prefix="mnm-selftest-") as scratch, self._http() as http:
            folder = Path(scratch)
            clip = silent_clip(folder) if copy.modality is Modality.SPEECH_TO_TEXT else None
            try:
                with sampler or contextlib.nullcontext():
                    answer: TryResult | None = self._send(copy, http, folder, clip)
            except EmptyTranscriptError:  # silence, transcribed as nothing: it answered
                answer = None
        seconds = self._clock() - began
        peak = sampler.peak if sampler is not None else None
        if window.used is None or peak is None or not service.card.alone(copy.name, window):
            return seconds, answer, None
        return seconds, answer, max(peak - window.used, 0)

    def _send(self, copy: Recipe, http: httpx.Client, folder: Path, clip: Path | None) -> TryResult:
        return try_instance(
            copy,
            http,
            folder,
            audio=clip,
            speed=copy.modality in CHAT_LIKE,
            root=engine_origin(copy),
        )

    def _failure(
        self, copy: Recipe, stage: Stage, why: str, error: Exception | None = None
    ) -> _Trial:
        """Keep this copy's log (the earlier one was forgotten) for the failure's line."""
        state_dir = self._service.settings.expanded_state_dir()
        keep_log(self._service.supervisor, state_dir, copy.name)
        log = log_path(state_dir, copy.name)
        return _Trial(failure=why, stage=stage, log=log if log.is_file() else None, error=error)

    def _failed(self, result: SelfTest, trial: _Trial) -> SelfTest:
        """The new build's failure, in one line: an out-of-memory start says so; else the
        log's last line, else the error's first."""
        text = trial.log.read_text(encoding="utf-8") if trial.log is not None else ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        error_lines = str(trial.error).splitlines() if trial.error is not None else []
        if any(d.cause_code is Cause.OUT_OF_MEMORY for d in diagnose(text, result.engine)):
            detail = "it ran out of GPU memory"
        elif lines:
            detail = lines[-1]
        else:
            detail = error_lines[0] if error_lines else ""
        reason = f"Failed: {trial.failure}" + (f" ({detail})" if detail else "") + "."
        return replace(result, outcome=Outcome.FAILED, reason=reason, log=trial.log)

    def _clean(self, copy: Recipe) -> None:
        """The copy leaves nothing: no container, no link folders, no config."""
        with contextlib.suppress(SupervisorError):  # Docker is away: the next test removes it
            self._service.supervisor.stop(copy.name)
        if (cleanup := spec_for(copy).cleanup) is not None:
            with contextlib.suppress(OSError):
                cleanup(copy, self._service.models_dir)


def _earlier(trial: _Trial, before: _Trial) -> bool:
    """The new build failed at an earlier stage than the previous one did."""
    return trial.stage is not None and before.stage is not None and trial.stage < before.stage


def _speed(copy: Recipe, answer: TryResult | None, seconds: float) -> Speed | None:
    """llama-server's own decode rate, vLLM's tokens over the wall clock, else the seconds
    the request took (a picture, a clip, a transcript, speech)."""
    if copy.engine is Engine.LLAMACPP:
        rate = answer.numbers.get(Figure.PREDICTED_PER_SECOND) if answer else None
        return Speed(round(rate, 2), SpeedUnit.TOKENS) if rate else None
    if copy.engine is Engine.VLLM:
        tokens = answer.numbers.get(Figure.COMPLETION_TOKENS) if answer else None
        if not tokens or seconds <= 0:
            return None
        return Speed(round(tokens / seconds, 2), SpeedUnit.TOKENS)
    return Speed(round(seconds, 2), SpeedUnit.SECONDS)
