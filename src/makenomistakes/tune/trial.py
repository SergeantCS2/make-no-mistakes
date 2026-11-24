"""One trial: a real llama-server start of the instance with candidate
settings, measured, and gone again.

The trial runs a **copy** (`tune-<name>`, a spare loopback port, the recipe as a start would
run it: `start_fixes.as_started`), as the engines' self-test does (`engine_selftest`): the
instance's own container and recipe stay as they were until the tune's result is saved, and a trial
never writes the recipe. The copy starts under the box's start lock after the room the
caller makes (`room`), then answers, under one `measure_window`: a discarded warm-up, three
fresh ~2k-token prompts generating 256 tokens each (decode speed: their median; prompt
speed; llama-server's prompt time as the time to the first token), and the long prompt when
the score reads it. Each card's peak, the engine's RSS and the drafted tokens kept are kept
too. Three decode readings more than 10% apart, or anything else touching the card, make
the trial *not alone*.

A model with experts in RAM (`--n-cpu-moe`, or a placement's RAM-held blocks) decodes at
what the page cache and the memory bus give it at that moment: Qwen3.6-35B on an RTX 4070
read 19.4, 20.6, 22.2, 28.3 and 30.0 tok/s for the same settings, and its tune's first two
trials spread 55% and 20% and it gave up. Such a trial first sends
one discarded request at its own settings (a full decode prompt: the experts its tokens
touch are read into the page cache), then five decode prompts instead of three, their
spread judged over the middle three (`score.judged_spread`) and their median the decode.

A crash is `diagnose`d (an out-of-memory start ends a search direction). `cut` set (an
outside request, or the nightly holding the card) stops it where it is: the caller's cut
stops the container, and the trial reports itself cut."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from threading import Event

import httpx

from makenomistakes import box_events
from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.consumers import engine_origin
from makenomistakes.controls import SettingsError, apply_settings
from makenomistakes.diagnose import diagnose
from makenomistakes.draft import DraftError, with_draft
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.engine_selftest import spare_port
from makenomistakes.hub.facts import Engine
from makenomistakes.instance_settings import numa_facts, placed_again
from makenomistakes.instances import InstanceService
from makenomistakes.model_meta import serving_context
from makenomistakes.recipes import Recipe, spec_for
from makenomistakes.recipes.llamacpp_cache import without_slots
from makenomistakes.recipes.model import TUNE_PREFIX
from makenomistakes.recipes.tuned import TUNED_KEYS, ScoreMode
from makenomistakes.start_fixes import StartRefusedError, as_started
from makenomistakes.supervisor import (
    HEALTH_TIMEOUT_SECONDS,
    ContainerState,
    Health,
    SupervisorError,
)
from makenomistakes.swap import start_turn
from makenomistakes.tune.ask import (
    GENERATED,
    LONG_GENERATED,
    WARM_UP_TOKENS,
    CardPeaks,
    Timings,
    TrialAskError,
    complete,
    resident_bytes,
)
from makenomistakes.tune.prompts import LONG_TOKENS, SHORT_TOKENS, fresh_prompt
from makenomistakes.tune.room import NoRoomError
from makenomistakes.tune.score import (
    DECODE_PROMPTS_IN_RAM,
    judged_spread,
    needs_long_prompt,
    scored,
    spread_limit,
)
from makenomistakes.tune.trial_types import Candidate, Cut, Outcome, TrialResult

REQUEST_SECONDS = 1800.0  # a 16k prompt on a CPU build takes minutes
PROMPT_ROOM = 384  # tokens kept free in a request's context beyond the prompt
DECODE_PROMPTS = 3
READY_SECONDS = HEALTH_TIMEOUT_SECONDS
READY_POLL_SECONDS = 0.5

Room = Callable[[Recipe], list[str]]  # make room for the copy; the names parked
Progress = Callable[[str], None]


class CandidateError(ValueError):
    """mnm refused the candidate's settings before anything started."""


def copy_name(name: str) -> str:
    return f"{TUNE_PREFIX}{name}"  # reserved: `check_name` refuses it


def candidate_recipe(service: InstanceService, recipe: Recipe, candidate: Candidate) -> Recipe:
    """The saved recipe with the candidate's settings, validated as `mnm set` validates them
    (`apply_settings`), nothing saved."""
    updates = {TUNED_KEYS[f]: _raw(v) for f, v in candidate.changes.items()}
    if candidate.draft is not None:
        updates["draft"] = "" if candidate.draft == "off" else candidate.draft

    def draft(current: Recipe, raw: str) -> Recipe:
        return with_draft(current, raw, service.find, service.models_dir)

    def place(current: Recipe, wanted: dict[str, str]) -> Recipe:  # the per-card axis
        return placed_again(service, current, wanted)

    try:
        return apply_settings(
            recipe,
            updates,
            draft,
            lambda: service.memory().listed(),
            place,
            lambda current: numa_facts(service, current),  # the numa stage
        )
    except (SettingsError, DraftError) as exc:
        raise CandidateError(str(exc)) from exc


def _raw(value: object) -> str:
    return "auto" if value is None else str(value)


@dataclass(frozen=True)
class Asked:
    """What the copy's answers measured."""

    decode: list[float]
    prompt: list[float]
    ttft: list[float]
    prompt_tokens: int | None
    long: Timings | None
    draft_n: int | None
    draft_accepted: int | None


@dataclass
class _Started:
    """How the copy's start went."""

    copy: Recipe
    ready: bool = False
    build: str | None = None
    held: str | None = None  # no room could be made: why
    loaded: int | None = None  # what the card grew by from `docker run` to ready, if alone


class TrialRunner:
    def __init__(
        self,
        service: InstanceService,
        room: Room | None = None,
        http: Callable[[], httpx.Client] | None = None,
        clock: Callable[[], float] = time.monotonic,
        progress: Progress | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._service = service
        self._sleep = sleep
        self._room = room or (lambda _copy: [])
        self._http = http or (lambda: httpx.Client(timeout=REQUEST_SECONDS, trust_env=False))
        self._clock = clock
        self._progress = progress

    def run(
        self,
        recipe: Recipe,
        candidate: Candidate,
        number: int,
        mode: ScoreMode,
        session: str,
        cut: Cut | None = None,
    ) -> TrialResult:
        """Run the candidate on a copy of `recipe`; never raises for the copy's failures."""
        cut = cut or Cut()
        began = self._clock()
        base = TrialResult(
            session=session,
            instance=recipe.name,
            number=number,
            candidate=candidate,
            outcome=Outcome.MEASURED,
        )
        try:
            copy = self.copy_of(recipe, candidate)
        except (CandidateError, StartRefusedError) as exc:
            return base.model_copy(update={"outcome": Outcome.REFUSED, "why": str(exc)})
        settings = self._service.settings
        with box_events.changing(settings):  # its start and answers are no other's reading
            try:
                result = self._run_copy(copy, base, mode, cut)
            finally:
                self._clean(copy)
        if cut.is_set() and result.outcome not in {Outcome.MEASURED, Outcome.HELD}:
            result = result.model_copy(update={"outcome": Outcome.CUT, "why": cut.why})
        return result.model_copy(update={"seconds": round(self._clock() - began, 1)})

    def copy_of(self, recipe: Recipe, candidate: Candidate) -> Recipe:
        """The candidate's recipe as a start would run it, under the copy's name and port."""
        service = self._service
        tried = candidate_recipe(service, recipe, candidate)
        started = as_started(tried, service.models_dir, service.memory)
        copy = started.model_copy(update={"name": copy_name(recipe.name), "port": spare_port()})
        return without_slots(copy)  # The instance's saved prompts are its own

    def _run_copy(
        self, copy: Recipe, base: TrialResult, mode: ScoreMode, cut: Event
    ) -> TrialResult:
        service = self._service
        started = _Started(copy)
        try:
            self._start(started, cut)
        except (SupervisorError, OSError, ValueError) as exc:
            return self._crashed(started.copy, base, f"it did not start: {_first(exc)}")
        copy = started.copy
        if started.held is not None:
            return base.model_copy(update={"outcome": Outcome.HELD, "why": started.held})
        if not started.ready:
            return self._crashed(copy, base, "it was not ready in time")
        base = base.model_copy(update={"build": started.build})
        window = service.card.open()
        try:
            with CardPeaks(service.memory) as peaks, self._http() as http:
                asked = self._ask(http, copy, base, mode, cut)
                rss = resident_bytes(service.supervisor.status(copy.name).pid)
        except (httpx.HTTPError, TrialAskError, SupervisorError) as exc:
            return self._crashed(copy, base, f"it stopped answering: {_first(exc)}")
        alone = service.card.alone(copy.name, window)
        added = None
        if started.loaded is not None and alone and window.used is not None and peaks.peaks:
            added = started.loaded + max(peaks.peaks[0] - window.used, 0)  # load + answers
        result = self._measured(base, asked, mode, peaks.peaks, added, rss)
        return _judged(result, alone)

    def _start(self, started: _Started, cut: Event) -> None:
        """Start the copy under the box's start lock after the caller's room; `started` says
        how it ran (its port may move once, after a lost race), its build, and whether it
        got ready."""
        service = self._service
        supervisor = service.supervisor
        copy = started.copy
        with start_turn(service.settings, copy.name, self._progress):
            supervisor.stop(copy.name)  # a copy an interrupted tune left behind
            if cut.is_set():
                return
            try:
                parked = self._room(copy)
            except NoRoomError as exc:
                started.held = str(exc)
                return
            if parked and self._progress:
                self._progress(f"Parked {', '.join(parked)} for the trial.")
            if (prepare := spec_for(copy).prepare) is not None:
                prepare(copy, service.models_dir)
            if cut.is_set():  # a request came while room was made: the card is its
                return
            window = service.card.open()  # after the parks: their stops move the stamp
            try:
                supervisor.start(copy, service.models_dir)
            except SupervisorError as exc:
                if "already allocated" not in str(exc) and "address already" not in str(exc):
                    raise
                copy = copy.model_copy(update={"port": spare_port(lost=copy.port)})
                started.copy = copy
                supervisor.start(copy, service.models_dir)
            if not self._wait_ready(copy, cut):
                return
            started.ready = True
            started.build = supervisor.status(copy.name).build
            started.loaded = service.card.grew(copy.name, window)

    def _wait_ready(self, copy: Recipe, cut: Event) -> bool:
        """Ready before the start timeout, unless cut first; SupervisorError when the copy
        died loading (its log says why)."""
        supervisor = self._service.supervisor
        deadline = self._clock() + READY_SECONDS
        while not cut.is_set():
            if supervisor.health(copy) is Health.READY:
                return True
            if supervisor.status(copy.name).state is not ContainerState.RUNNING:
                raise SupervisorError(f"{copy.name} exited while loading")
            if self._clock() >= deadline:
                return False
            self._sleep(READY_POLL_SECONDS)
        return False

    def _ask(
        self, http: httpx.Client, copy: Recipe, base: TrialResult, mode: ScoreMode, cut: Event
    ) -> Asked:
        """The warm-up (discarded), three fresh prompts, then the long one if the score
        reads it; each prompt seeded by the instance, the session, the trial and its place."""
        context = serving_context(copy) or SHORT_TOKENS * 2
        short = max(min(SHORT_TOKENS, context - GENERATED - PROMPT_ROOM), 64)
        seed = f"{base.instance}/{base.session}/{base.number}"
        origin = engine_origin(copy)  # its own port: the router never sees a trial's requests
        complete(http, origin, fresh_prompt(f"{seed}/warm", WARM_UP_TOKENS), 8)
        prompts = DECODE_PROMPTS
        if isinstance(copy.args, LlamaCppArgs) and copy.args.experts_in_ram:
            complete(http, origin, fresh_prompt(f"{seed}/cache", short), GENERATED)  # discarded
            prompts = DECODE_PROMPTS_IN_RAM
        answers: list[Timings] = []
        for index in range(prompts):
            if cut.is_set():
                raise TrialAskError("cut")
            prompt = fresh_prompt(f"{seed}/{index}", short)
            answers.append(complete(http, origin, prompt, GENERATED))
        long = None
        room = context - LONG_GENERATED - PROMPT_ROOM
        if needs_long_prompt(mode) and room >= SHORT_TOKENS * 2 and not cut.is_set():
            prompt = fresh_prompt(f"{seed}/long", min(LONG_TOKENS, room))
            long = complete(http, origin, prompt, LONG_GENERATED)
        drafted = [a.draft_n for a in answers if a.draft_n is not None]
        kept = [a.draft_n_accepted or 0 for a in answers if a.draft_n is not None]
        return Asked(
            decode=[round(a.predicted_per_second or 0.0, 2) for a in answers],
            prompt=[a.prompt_per_second for a in answers if a.prompt_per_second],
            ttft=[a.prompt_ms / 1000 for a in answers if a.prompt_ms is not None],
            prompt_tokens=answers[0].prompt_n,
            long=long,
            draft_n=sum(drafted) if drafted else None,
            draft_accepted=sum(kept) if drafted else None,
        )

    def _measured(
        self,
        base: TrialResult,
        asked: Asked,
        mode: ScoreMode,
        peaks: list[int],
        added: int | None,
        rss: int | None,
    ) -> TrialResult:
        prompt_2k = round(median(asked.prompt), 2) if asked.prompt else None
        long = asked.long
        prompt_16k = round(long.prompt_per_second, 2) if long and long.prompt_per_second else None
        return base.model_copy(
            update={
                "decode": asked.decode,
                "prompt_2k": prompt_2k,
                "prompt_tokens": asked.prompt_tokens,
                "prompt_16k": prompt_16k,
                "long_tokens": long.prompt_n if long else None,
                "ttft_seconds": round(median(asked.ttft), 3) if asked.ttft else None,
                "vram_peak_bytes": peaks,
                "vram_added_bytes": added,
                "ram_rss_bytes": rss,
                "draft_n": asked.draft_n,
                "draft_accepted": asked.draft_accepted,
                "score": scored(mode, asked.decode, prompt_2k, prompt_16k),
            }
        )

    def _crashed(self, copy: Recipe, base: TrialResult, why: str) -> TrialResult:
        """The copy's log, diagnosed: an out-of-memory start says so in its cause."""
        try:
            text = self._service.supervisor.logs(copy.name, 0)
        except SupervisorError:
            text = ""
        found = diagnose(text, Engine.LLAMACPP, recipe_backend_or_none(copy))
        cause = found[0].cause_code.value if found else None
        if found:
            why = f"{why.split(':', maxsplit=1)[0]}: {found[0].cause}"
        return base.model_copy(update={"outcome": Outcome.CRASHED, "cause": cause, "why": why})

    def _clean(self, copy: Recipe) -> None:
        """The copy leaves nothing: no container, no link folders, no config."""
        with contextlib.suppress(SupervisorError):
            self._service.supervisor.stop(copy.name)
        if (cleanup := spec_for(copy).cleanup) is not None:
            with contextlib.suppress(OSError):
                cleanup(copy, self._service.models_dir)


def _judged(result: TrialResult, alone: bool) -> TrialResult:
    """Not alone when the decode readings spread past the limit or the card was shared."""
    noisy, limit = judged_spread(result.decode), spread_limit()
    if noisy > limit:
        why = (
            f"readings too far apart: {noisy:.0%} spread over {_over(result.decode)} "
            f"(the rule: {limit:.0%})"
        )
        return result.model_copy(update={"outcome": Outcome.NOISY, "why": why})
    if not alone:
        why = "something else used the card meanwhile"
        return result.model_copy(update={"outcome": Outcome.NOISY, "why": why})
    return result


def _over(decode: list[float]) -> str:
    """`3 readings`, or `the middle 3 of 5 readings`."""
    if len(decode) >= DECODE_PROMPTS_IN_RAM:
        return f"the middle {len(decode) - 2} of {len(decode)} readings"
    return f"{len(decode)} readings"


def _first(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__
