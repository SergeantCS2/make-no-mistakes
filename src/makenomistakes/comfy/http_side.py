"""ComfyUI's HTTP side: what it holds, whether it is running a workflow, freeing its card and
clearing its history, asked of the running container.

Whether ComfyUI is running a workflow has three answers, never two: yes, no, and unknown.
Unknown is `/queue` not answering (an HTTP error, the 10 s timeout: a run streaming its weights
through swap), or Docker not saying whether the container runs while ComfyUI does not answer
at its address either. `queue_busy` is the one question, and every caller says what it does
with unknown: a run keeps following, the idle sweep waits, a model's load is refused
(`release_vram`), and the router's card jobs, the nightly and a packs switch wait or refuse
with `busy_words`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import ValidationError

from makenomistakes.comfy.api import Queue, SystemStats
from makenomistakes.comfy.image import ComfyBusyError
from makenomistakes.comfy.observe import Observed

FREE_WAIT_SECONDS = 60.0  # a job about to finish is waited for; a longer one refuses the load
FREED_STAMP = "freed"  # beside the ledger: touched whenever its card was freed (`comfy.idle`)
Progress = Callable[[str], None]
BUSY = "ComfyUI is running a workflow"
SILENT = "ComfyUI did not answer whether it is running a workflow"
RUNNING_HOLDS = (
    f"{BUSY} and holds the card: ask again when it ends (Runs shows it), or stop it there."
)
SILENT_HOLDS = (
    f"{SILENT} (it may be streaming a run's weights): a model loaded beside a run would run "
    "out of memory, so nothing was loaded. Ask again in a moment; `mnm comfyui stop` gives its "
    "card back."
)


class HttpSide(Observed):
    """`ComfyEngine`'s questions to ComfyUI itself."""

    base: Path
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    _http: Callable[[], httpx.Client]

    def stats(self, address: str | None = None, timeout: float = 3.0) -> SystemStats | None:
        ledger = self._ledger_or_none()
        where = address or (ledger.address if ledger is not None else None)
        if not where:
            return None
        try:
            with self._http() as http:
                response = http.get(f"{where}/system_stats", timeout=timeout)
            return (
                SystemStats.model_validate_json(response.content) if response.is_success else None
            )
        except (httpx.HTTPError, ValidationError):
            return None

    def queue_busy(self, address: str | None = None) -> bool | None:
        """True while a workflow runs or waits in ComfyUI's queue, False when none does or
        ComfyUI does not run, None when that is unknown (never "finished"). `address`: where
        the caller knows it runs (no Docker asked), else `asked_address`."""
        address = address or self.asked_address()
        if address is None:
            return False
        try:
            with self._http() as http:
                response = http.get(f"{address}/queue")
            if response.is_error:
                return None
            return Queue.model_validate_json(response.content).busy
        except (httpx.HTTPError, ValidationError):
            return None

    def busy_words(self) -> str | None:
        """Why the card is not free of ComfyUI's work, for what waits for it (the router's
        tunes and trainings, the nightly, a packs switch): running a workflow, or not saying
        whether it is; None when it runs nothing or does not run."""
        busy = self.queue_busy()
        if busy is False:
            return None
        return BUSY if busy else SILENT

    def release_vram(self, progress: Progress | None = None) -> bool:
        """Give the card back before mnm loads a model (the router's start path): wait a
        little for the job in flight, then `POST /free`. True when ComfyUI freed it; False
        when it does not run (a box without it, or stopped by mnm, costs one file read; a
        stale address is never waited on) or did not take the free. A job still running, or a
        queue that does not answer, after `FREE_WAIT_SECONDS` raises ComfyBusyError: ComfyUI
        frees only between jobs, so a model loaded now could run out of memory beside it."""
        address = self.asked_address()
        if address is None:
            return False
        deadline = self.clock() + FREE_WAIT_SECONDS
        while (busy := self.queue_busy(address)) is not False:
            if self.clock() >= deadline:
                raise ComfyBusyError(RUNNING_HOLDS if busy else SILENT_HOLDS)
            if progress:
                progress("Waiting for ComfyUI's run to finish before loading…")
            self.sleep(2.0)
        try:
            with self._http() as http:
                response = http.post(
                    f"{address}/free", json={"unload_models": True, "free_memory": True}
                )
        except httpx.HTTPError:
            return False
        if response.is_error:  # nothing was freed: never stamped as freed
            return False
        self.sleep(2.0)  # the worker frees between jobs: give it a moment
        # whoever asked (the router's load, a start, the web's button): the Status panel then
        # reads "its card freed at …" and drops the buffers from what it holds
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / FREED_STAMP).touch()
        return True

    def clear_history(self) -> bool:
        """Forget ComfyUI's finished prompts (`mnm clean comfy`): their outputs were moved
        into runs, so the history only names files that are gone. True when it answered."""
        address = self.asked_address()
        if address is None:
            return False
        try:
            with self._http() as http:
                response = http.post(f"{address}/history", json={"clear": True})
        except httpx.HTTPError:
            return False
        return response.is_success
