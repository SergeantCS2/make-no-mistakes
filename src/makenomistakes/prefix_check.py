"""`mnm try <name> --agent`, what the prompt cache saves an agent, measured on this box.

A long prefix (an agent's system prompt and tools are 7k to 19k tokens: Claude Code's first
request ~19,300, OpenCode's ~7,300, on the CPU server) is sent three times through the router,
each with another question after it:

1. **cold**: a prefix nothing has seen (its first line is new each run), read whole;
2. **warm**: the same prefix again, which llama.cpp keeps in the slot (`cache_n`);
3. **after a park** (`cache=on` only): the second request parks the instance as it ends
   (`keep_alive: 0`, the router's own park, whose stop saves the slot); the third loads it
   again, the router puts the saved slot back, and llama.cpp reads only what is new.

Each leg's figures are llama.cpp's own (`timings`: tokens read, tokens reused, ms reading the
prompt); the third's wall clock less its prompt and reply is the load and the restore. The
result is a `Kind.PREFIX` reading of this box (`measurements.reading_for`).

The user's saved prompts are set aside while it measures and come back after it, and the
slot its prefix filled is erased (else its synthetic prefix is saved as the newest file,
pushing the user's conversation out under the cap, and put back at every start after). The
prefix fits one request's context (`prefix_tokens`)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from makenomistakes import slot_files
from makenomistakes.instances import InstanceService
from makenomistakes.measurements import Kind, PrefixFigures, reading_for
from makenomistakes.model_meta import serving_context
from makenomistakes.reading_stamp import stamp_for
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.llamacpp_cache import keeps_slots
from makenomistakes.slot_folder import SlotFolder
from makenomistakes.supervisor import SupervisorError
from makenomistakes.tune.ask import TimedAnswer

PREFIX_TOKENS = 20_000
CONTEXT_SHARE = 0.8  # of one request's context: the questions, the reply, another tokenizer
# The rule lines below as Qwen3's tokenizer reads them: 220 lines were 4,311 tokens on the CPU
# build (2026-11-02); the reading keeps the count llama.cpp reports, whatever this guesses.
TOKENS_PER_LINE = 19.6
_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
]
QUESTIONS = ("Which word follows rule 5?", "Which word follows rule 6?", "What does rule 7 say?")
PARK_WAIT_SECONDS = 60.0  # the router parks after the second reply: saving can take a while


class PrefixError(RuntimeError):
    """A leg the engine did not answer as the check needs."""


@dataclass(frozen=True)
class Leg:
    """One request's prompt figures: what llama.cpp read, what it reused, how long it read."""

    read: int
    reused: int
    prompt_seconds: float
    wall_seconds: float
    reply_seconds: float

    @property
    def tokens(self) -> int:
        return self.read + self.reused


def prefix_tokens(recipe: Recipe, asked: int | None) -> int:
    """The prefix's length: `asked`, or `PREFIX_TOKENS`, at most `CONTEXT_SHARE` of the
    context one request gets (`model_meta.serving_context`); a `PrefixError` in words for an
    asked length past it (20,000 tokens at an 8,192 context would fail on the cold leg's
    context error)."""
    serving = serving_context(recipe) or PREFIX_TOKENS
    most = int(serving * CONTEXT_SHARE)
    if asked is None:
        return min(PREFIX_TOKENS, most)
    if asked > most:
        raise PrefixError(
            f"--prefix {asked:,} is more than {recipe.name}'s request context ({serving:,} "
            f"tokens) holds beside the questions and the reply; at most {most:,} "
            f"(`mnm set {recipe.name} ctx=` raises the context)."
        )
    return asked


def prefix_text(tokens: int, first_line: str) -> str:
    """About `tokens` tokens of numbered rules, `first_line` first (new each run, so no cache
    holds the start of it)."""
    lines = [
        f"Rule {i}: when {_WORDS[i % 26]} meets {_WORDS[(i * 7) % 26]}, write "
        f"{_WORDS[(i * 3) % 26]} {i * 13 % 997}."
        for i in range(max(1, round(tokens / TOKENS_PER_LINE)))
    ]
    return "\n".join([first_line, "You are a careful assistant. Follow the rules.", *lines])


def ask(
    http: httpx.Client,
    origin: str,
    recipe: Recipe,
    system: str,
    question: str,
    park: bool,
    clock: Callable[[], float],
) -> Leg:
    body: dict[str, object] = {
        "model": recipe.name,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if park:
        body["keep_alive"] = 0  # the router parks it after this reply (`keep_alive`)
    began = clock()
    try:
        answer = http.post(f"{origin}/v1/chat/completions", json=body)
    except httpx.TimeoutException as exc:  # a long prefix on the processor
        waited = http.timeout.read or 0
        raise PrefixError(
            f"{recipe.name} did not finish reading the prefix in {waited:.0f} s: a smaller "
            f"`--prefix` (4000, say) measures the same thing faster on a slow engine."
        ) from exc
    except httpx.HTTPError as exc:
        raise PrefixError(f"{recipe.name} did not answer: {exc}") from exc
    wall = clock() - began
    if answer.status_code != 200:
        raise PrefixError(f"{recipe.name} answered {answer.status_code}: {answer.text[:300]}")
    try:
        timings = TimedAnswer.model_validate_json(answer.content).timings
    except ValidationError as exc:
        raise PrefixError(f"{recipe.name}'s answer was not llama-server's: {exc}") from exc
    if timings is None or timings.prompt_n is None or timings.prompt_ms is None:
        raise PrefixError(f"{recipe.name}'s answer carried no prompt timings")
    predicted = (timings.predicted_n or 0) / (timings.predicted_per_second or 1e9)
    return Leg(
        read=timings.prompt_n,
        reused=timings.cache_n or 0,
        prompt_seconds=timings.prompt_ms / 1000,
        wall_seconds=wall,
        reply_seconds=predicted,
    )


@dataclass(frozen=True)
class PrefixCheck:
    cold: Leg
    warm: Leg
    restored: Leg | None  # the leg after a park, `cache=on` only
    why_not: str | None  # why there is no third leg

    @property
    def reload_seconds(self) -> float | None:
        """The third request's wall clock less its prompt and reply: the load and restore."""
        if self.restored is None:
            return None
        leg = self.restored
        return max(0.0, leg.wall_seconds - leg.prompt_seconds - leg.reply_seconds)

    def figures(self) -> PrefixFigures:
        return PrefixFigures(
            tokens=self.cold.tokens,
            cold_seconds=round(self.cold.prompt_seconds, 2),
            warm_seconds=round(self.warm.prompt_seconds, 2),
            reused_tokens=self.warm.reused,
            restored_seconds=round(self.restored.prompt_seconds, 2) if self.restored else None,
            restored_reused=self.restored.reused if self.restored else None,
            reload_seconds=round(r, 2) if (r := self.reload_seconds) is not None else None,
        )


def _parked(
    service: InstanceService,
    recipe: Recipe,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    """Whether the router parked it after the second reply within `PARK_WAIT_SECONDS`."""
    deadline = clock() + PARK_WAIT_SECONDS
    while clock() < deadline:
        if not service.status(recipe).running:
            return True
        sleep(0.5)
    return False


def check_prefix(
    service: InstanceService,
    recipe: Recipe,
    http: httpx.Client,
    origin: str,
    tokens: int = PREFIX_TOKENS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PrefixCheck:
    """The three legs (two with the cache off), the user's saved prompts set aside while
    they run, and the slot the prefix filled erased after (`_forget`)."""
    system = prefix_text(tokens, f"Session {uuid.uuid4().hex}.")
    if not keeps_slots(recipe):
        cold = ask(http, origin, recipe, system, QUESTIONS[0], False, clock)
        warm = ask(http, origin, recipe, system, QUESTIONS[1], False, clock)
        why = f"its prompt is not kept across unloads (`mnm set {recipe.name} cache=on`)"
        return PrefixCheck(cold, warm, None, why)
    with SlotFolder.of(service.models_dir, recipe.name).set_aside():
        try:
            return _three_legs(service, recipe, http, origin, system, clock, sleep)
        finally:
            _forget(recipe)


def _three_legs(
    service: InstanceService,
    recipe: Recipe,
    http: httpx.Client,
    origin: str,
    system: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> PrefixCheck:
    cold = ask(http, origin, recipe, system, QUESTIONS[0], False, clock)
    warm = ask(http, origin, recipe, system, QUESTIONS[1], True, clock)
    if not _parked(service, recipe, clock, sleep):
        why = (
            f"the router did not unload it within {PARK_WAIT_SECONDS:.0f} s (pinned, a member "
            "of a set in use, or another request in flight): nothing to put back"
        )
        return PrefixCheck(cold, warm, None, why)
    restored = ask(http, origin, recipe, system, QUESTIONS[2], False, clock)
    return PrefixCheck(cold, warm, restored, None)


def _forget(recipe: Recipe) -> None:
    """The slot the check's prefix filled, erased, so the next stop saves nothing of it (an
    engine not running has none; any error leaves it, the next start's restore replaces it)."""
    try:
        with slot_files.engine_client(recipe, slot_files.LOOK_SECONDS) as http:
            used = slot_files.last_used(slot_files.slots(http))
            if used is not None:
                slot_files.act(http, used.id, "erase", "", slot_files.LOOK_SECONDS)
    except slot_files.SlotError:
        return


def prefix_lines(check: PrefixCheck) -> list[str]:
    """What `mnm try --agent` prints, one line a leg."""
    cold, warm, after = check.cold, check.warm, check.restored
    lines = [
        f"Prefix: {cold.tokens:,} tokens.",
        f"  cold: read in {cold.prompt_seconds:.1f} s ({cold.read:,} tokens read).",
        f"  warm: read in {warm.prompt_seconds:.1f} s ({warm.reused:,} tokens from the cache, "
        f"{warm.read:,} read).",
    ]
    if after is None:
        lines.append(f"  after a park: not measured, {check.why_not}.")
    else:
        lines.append(
            f"  after a park: loaded and its saved slot put back in {check.reload_seconds:.1f} s, "
            f"then read in {after.prompt_seconds:.1f} s ({after.reused:,} tokens from the saved "
            f"slot, {after.read:,} read)."
        )
    return lines


def record(service: InstanceService, recipe: Recipe, check: PrefixCheck) -> None:
    """The check as this box's reading of the instance's settings and build."""
    try:
        build = service.supervisor.status(recipe.name).build
    except SupervisorError:
        build = None
    service.measurements.add(
        recipe.name,
        reading_for(
            recipe, Kind.PREFIX, build, prefix=check.figures(), stamp=stamp_for(service, recipe)
        ),
    )
