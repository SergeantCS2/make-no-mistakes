"""What a trial asks the copy and reads back: llama-server's own `/completion` on the
copy's loopback port, never through the router (the router counts its requests as traffic,
and an outside request is what cuts a trial), with the prompt cache off and the engine's
`timings` typed; the card's memory per card while it answers; the engine's resident memory.

RAM is the engine's main process's `VmRSS` from `/proc/<pid>/status` (the pid from `docker
inspect`, read as anyone may read it), not `docker stats`: it counts the model file's pages
the process has mapped (`--mmap`), which is what a model kept in RAM costs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from makenomistakes.hardware.probe import MemorySnapshot

GENERATED = 256  # tokens each decode prompt generates
WARM_UP_TOKENS = 128  # the discarded first request's prompt
LONG_GENERATED = 16  # the long prompt is for its prompt speed: a few tokens are enough


class TrialAskError(RuntimeError):
    """The copy did not answer as a trial needs (an error status, no timings, a reused cache)."""


class Timings(BaseModel):
    """llama-server's `timings` (unknown fields ignored)."""

    model_config = ConfigDict(extra="ignore")

    cache_n: int | None = None  # prompt tokens taken from the cache: must be 0
    prompt_n: int | None = None
    prompt_ms: float | None = None
    prompt_per_second: float | None = None
    predicted_n: int | None = None
    predicted_per_second: float | None = None
    draft_n: int | None = None
    draft_n_accepted: int | None = None


class TimedAnswer(BaseModel):
    """An answer's llama-server `timings` (a `/completion`'s, a chat's), nothing else."""

    model_config = ConfigDict(extra="ignore")

    timings: Timings | None = None


def complete(http: httpx.Client, origin: str, prompt: str, generated: int) -> Timings:
    """One `/completion` with the prompt cache off and the end-of-text ignored (every trial
    decodes the same count); its timings, checked."""
    body = {
        "prompt": prompt,
        "n_predict": generated,
        "cache_prompt": False,
        "ignore_eos": True,
        "temperature": 0.7,
        "stream": False,
    }
    answer = http.post(f"{origin}/completion", json=body)
    if answer.status_code != 200:
        raise TrialAskError(f"the copy answered {answer.status_code}: {answer.text[:200]}")
    try:
        timings = TimedAnswer.model_validate_json(answer.content).timings
    except ValidationError as exc:
        raise TrialAskError(f"the copy's answer was not llama-server's: {exc}") from exc
    if timings is None or not timings.predicted_per_second:
        raise TrialAskError("the copy's answer carried no timings")
    if timings.cache_n:  # a reused prefix would overstate the prompt speed
        raise TrialAskError(f"the copy reused {timings.cache_n} cached prompt tokens")
    return timings


class CardPeaks:
    """Each card's highest memory in use while a trial answers, read every `every` s."""

    def __init__(self, memory: Callable[[], MemorySnapshot], every: float = 0.5) -> None:
        self._memory = memory
        self._every = every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peaks: list[int] = []

    def __enter__(self) -> CardPeaks:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            self._sample()

    def _sample(self) -> None:
        try:
            cards = self._memory().cards
        except Exception:  # an unreadable card is no reading, never a failed trial
            return
        used = [card.vram_used_bytes for card in cards]
        if len(self.peaks) != len(used):
            self.peaks = used
        else:
            self.peaks = [max(a, b) for a, b in zip(self.peaks, used, strict=True)]


def resident_bytes(pid: int | None, proc: Path = Path("/proc")) -> int | None:
    """`VmRSS` of the process (kB in the file); None when it cannot be read."""
    if pid is None:
        return None
    try:
        text = (proc / str(pid) / "status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None
