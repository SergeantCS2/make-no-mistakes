"""One `mnm try`, shared by the CLI and the web UI's Test it panel: the request (`smoke`), the
GPU readings around it (`gpu_peak`), and what to say when nothing answers (Docker and the
engine's log decide whether it died during the request).

Inputs arrive as bytes or a path chosen by the caller: the CLI only takes files inside the
current folder, the web only its own upload folder. Outputs go to the `out_dir` it is given.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from makenomistakes import request_peak
from makenomistakes.card_room import instance_cards
from makenomistakes.config import Settings
from makenomistakes.draft import alone_speed, verdict
from makenomistakes.engine_exits import exit_meaning
from makenomistakes.fit.budget import BUSY_CARD_BYTES
from makenomistakes.gpu_peak import PeakSampler, ReadCards
from makenomistakes.hardware.probe import probe_memory
from makenomistakes.hub.facts import CHAT_LIKE, Modality
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.readings_filter import instance_readings
from makenomistakes.recipes import Recipe
from makenomistakes.smoke import (
    TRY_TIMEOUT_SECONDS,
    NoAnswerError,
    Picture,
    TryError,
    TryResult,
    try_instance,
)
from makenomistakes.supervisor import ContainerState, SupervisorError
from makenomistakes.tool_check import ToolCheck, check_tools

MIB = 2**20
GIB = 2**30
LAST_LINES = 12  # of a dead engine's log, given under the error
UsedBytes = Callable[[], int | None]


def http_client(settings: Settings) -> httpx.Client:
    """Never through a proxy: the instance is on this machine.
    With the router's key set, every request carries it."""
    key = settings.api_key_value()
    headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
    return httpx.Client(timeout=TRY_TIMEOUT_SECONDS, trust_env=False, headers=headers)


def used_bytes() -> int | None:
    """nvidia-smi's memory in use right now; None when the GPU cannot be read."""
    gpu = probe_memory().gpu
    return gpu.vram_used_bytes if gpu is not None else None


@dataclass(frozen=True)
class TryInputs:
    """What the request carries beyond the card's own: a clip, the speed request, a
    sentence to read, a picture (and mask) to repaint, edit or animate."""

    audio: Path | None = None
    speed: bool = False
    say: str | None = None
    picture: Picture | None = None
    prompt: str | None = None  # a Run: the user's words instead of the card's
    tools: bool = False  # `--tools`: the tool-call test instead of the card's request
    loading: bool = False  # set by `run_try`: the model was not loaded when it was asked
    lyrics: str | None = None  # What a song instance sings to the prompt's tags


@dataclass(frozen=True)
class TryRun:
    result: TryResult
    readings: list[int] | None = None  # GPU memory in use, every 0.25 s; None: not measured
    card_peaks: tuple[tuple[int, int], ...] = ()  # several cards: each one's peak
    verdict: str | None = None  # with a draft model and --speed: whether it gains here
    tools: ToolCheck | None = None  # `--tools`: each step, the verdict saved for the page
    company: str | None = None  # `--speed`/`--measure`: who else held the card
    parked: tuple[str, ...] = ()  # `--measure`: loaded before the request, parked during it

    @property
    def vram_line(self) -> str | None:
        """`--measure` in words: before, the peak (and how much it added), after."""
        if self.readings is None:
            return None
        if not self.readings:
            return "VRAM: unreadable (nvidia-smi gave no reading)."
        before, peak, after = self.readings[0], max(self.readings), self.readings[-1]
        lowest = min(self.readings)
        each = ", ".join(f"GPU{i} {used // MIB:,} MiB" for i, used in self.card_peaks)
        count = f"({len(self.readings)} readings)" + (f"; each card's peak: {each}" if each else "")
        if lowest >= before:
            return (
                f"VRAM: {before // MIB:,} MiB before, peak {peak // MIB:,} MiB "
                f"(+{(peak - before) // MIB:,}), {after // MIB:,} MiB after {count}"
            )
        # The card emptied during the request (the router parked the model loaded before it
        # to make room): a peak over "before" is the difference of
        # two models, so the rise is said over the lowest reading.
        if self.parked:
            them = "it" if len(self.parked) == 1 else "them"
            why = (
                f" (with {', '.join(self.parked)} loaded; the router parked {them} for this "
                "request)"
            )
        else:
            why = " (something on the card was freed during the request)"
        return (
            f"VRAM: {before // MIB:,} MiB before{why}, lowest {lowest // MIB:,} MiB, peak "
            f"{peak // MIB:,} MiB (+{(peak - lowest) // MIB:,} over the lowest), "
            f"{after // MIB:,} MiB after {count}"
        )


def run_try(
    service: InstanceService,
    recipe: Recipe,
    http: httpx.Client,
    out_dir: Path,
    inputs: TryInputs,
    measure: UsedBytes | None = None,
) -> TryRun:
    """Send the card's request and time it (or, with `inputs.tools`, the tool-call test,
    whose verdict is saved for the instance's page); with `measure`, sample GPU memory
    meanwhile. Raises `TryError`, or `InstanceError` with the log's diagnoses when the
    engine died."""
    if inputs.tools:
        _check_tools_apply(recipe, inputs)
    if recipe.modality is Modality.TEXT_TO_SPEECH:  # a rate that counts a load says so
        inputs = replace(inputs, loading=_not_loaded(service, recipe))
    try:
        if measure is not None:
            began = None if inputs.tools else request_peak.before(service, recipe)
            loaded = _loaded_others(service, recipe)
            with PeakSampler(measure, read_cards=card_reader(service, recipe)) as sampler:
                run = _run(recipe, http, out_dir, inputs)
            still = _loaded_others(service, recipe)
            gone = tuple(n for n in loaded or [] if still is not None and n not in still)
            peaks = tuple(sorted(sampler.card_peaks.items()))
            run = replace(run, readings=sampler.readings, parked=gone, card_peaks=peaks)
            request_peak.record(service, recipe, began, sampler.readings, peaks, still)
        else:
            run = _run(recipe, http, out_dir, inputs)
    except NoAnswerError as exc:
        raise why_no_answer(service, recipe, exc) from exc
    if run.tools is not None:
        service.tools.save(recipe, run.tools)
    if inputs.speed or measure is not None:
        run = replace(run, company=company(service, recipe))
    if not inputs.speed or recipe.draft is None:
        return run
    alone = alone_speed(instance_readings(service.measurements, recipe.name), recipe)
    return replace(run, verdict=verdict(recipe, run.result.numbers, alone))


def card_reader(service: InstanceService, recipe: Recipe) -> ReadCards | None:
    """Every card the instance runs on, read at each sample, when it runs on several;
    None on one card (the one reading is that card's)."""
    if len(instance_cards(recipe, service.memory())) < 2:
        return None

    def read() -> dict[int, int]:
        try:
            cards = instance_cards(recipe, service.memory())
        except Exception:  # a reading is a hint: an unreadable card is no sample
            return {}
        return {c.index: c.vram_used_bytes for c in cards}

    return read


def company(service: InstanceService, recipe: Recipe) -> str | None:
    """Who else of mnm's ran while a speed or VRAM reading was taken, with what each holds by
    its estimate (qwen3-8b reads 62.3 tok/s beside tiny, against 75 alone): a reading says
    its company. None when the instances cannot be read (the answer stands; `mnm ls` says
    why). ComfyUI's share is `mnm ls`'s last line."""
    try:
        statuses = service.statuses()
    except InstanceError:
        return None
    beside, held_all = [], 0
    for status in statuses:
        if not status.running:
            continue
        held = service.held_estimate(status).bytes
        held_all += held or 0
        if status.recipe.name != recipe.name:
            size = f" ({held / GIB:.1f} GiB)" if held else ""
            beside.append(f"{status.recipe.name}{size}")
    outside = _outside_words(service, held_all)
    if not beside:
        alone = "Alone: no other mnm instance was loaded."
        return f"{alone} {outside}" if outside else alone
    said = f"Beside: {', '.join(beside)}. A reading beside others can be slower than alone."
    return f"{said} {outside}" if outside else said


def _outside_words(service: InstanceService, held: int) -> str | None:
    """`Beside: something outside mnm holds 3.1 GiB.` when the card's memory in use is more
    than `BUSY_CARD_BYTES` above what mnm's running instances hold (a browser playing a
    video on Windows, say)."""
    try:
        cards = service.memory().cards
    except Exception:
        return None
    if not cards:
        return None
    outside = sum(c.vram_used_bytes for c in cards) - held
    if outside <= BUSY_CARD_BYTES:
        return None
    return f"Beside: something outside mnm holds {outside / GIB:.1f} GiB."


def _not_loaded(service: InstanceService, recipe: Recipe) -> bool:
    """Whether the request will also load the model (parked or stopped now)."""
    try:
        status = service.status(recipe)
    except InstanceError:
        return False
    return not status.running and status.error is None  # Docker silent: not said


def _loaded_others(service: InstanceService, recipe: Recipe) -> list[str] | None:
    """The other instances running now; None when they cannot be read."""
    try:
        statuses = service.statuses()
    except InstanceError:
        return None
    return [s.recipe.name for s in statuses if s.running and s.recipe.name != recipe.name]


def _check_tools_apply(recipe: Recipe, inputs: TryInputs) -> None:
    if recipe.modality not in CHAT_LIKE:
        raise TryError(
            f"{recipe.name} is {recipe.modality.a_label} model: --tools tests the tool calls "
            "of a chat, coding or vision model."
        )
    if inputs.audio or inputs.speed or inputs.say or inputs.picture or inputs.prompt:
        raise TryError("--tools sends its own five requests: leave out the other options.")


def _run(recipe: Recipe, http: httpx.Client, out_dir: Path, inputs: TryInputs) -> TryRun:
    if inputs.tools:
        check = check_tools(recipe, http)
        return TryRun(check.result(), tools=check)
    return TryRun(_send(recipe, http, out_dir, inputs))


def _send(recipe: Recipe, http: httpx.Client, out_dir: Path, inputs: TryInputs) -> TryResult:
    return try_instance(
        recipe,
        http,
        out_dir,
        inputs.audio,
        inputs.speed,
        say=inputs.say,
        picture=inputs.picture,
        prompt=inputs.prompt,
        loading=inputs.loading,
        lyrics=inputs.lyrics,
    )


def why_no_answer(service: InstanceService, recipe: Recipe, exc: NoAnswerError) -> Exception:
    """Docker says whether the engine died during the request (else a LoRA crash
    reads as "Is it running?"); its log says why."""
    try:
        status = service.supervisor.status(recipe.name)
    except SupervisorError:
        return exc
    match status.state:
        case ContainerState.ABSENT:
            return TryError(f"{exc} It is not running: `mnm start {recipe.name}`.")
        case ContainerState.EXITED:
            try:
                text, review = service.logs(recipe)  # the diagnosis reads the whole tail
            except InstanceError:
                text, review = "", None
            lines = [line for line in text.splitlines() if line.strip()][-LAST_LINES:]
            last = "\n".join(f"  {line}" for line in lines)
            meaning = exit_meaning(status.exit_code, status.oom_killed)
            return InstanceError(
                f"{exc} Its engine exited ({meaning}), maybe during this request. The last "
                f"lines of its log (`mnm logs {recipe.name}` has all of it, kept after "
                f"`mnm stop`):\n{last}"
                if last
                else f"{exc} Its engine exited ({meaning}): `mnm logs {recipe.name}` has the log.",
                review.diagnoses if review else [],
            )
        case _:
            return TryError(f"{exc} Its container runs: still loading? `mnm logs {recipe.name}`.")
