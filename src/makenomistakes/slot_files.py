"""A llama.cpp instance's slot saved to disk when it unloads and put back when it starts
again (`cache=on`), through llama-server's own `/slots/<id>?action=save|restore`.

A reload otherwise reads an agent's whole prompt again: on the CPU build a 4,311-token prefix
took 21.3 s after a restart and 0.6 s once its 495 MB slot file was restored (0.19 s); on the
hybrid Qwen3.5-0.8B 15.8 s and 0.3 s, the file carrying its 3 context checkpoints, a different
question over the same system prompt reusing all 4,410 of its tokens (2026-11-02, llama.cpp
a91eab67d). So the newest file is put back whatever the next request asks: llama.cpp keeps the
longest common prefix itself, token by token, and a prefix that shares nothing costs the read.

- **Saved** (`before_stop`, `InstanceService.stop`'s first step): the slot that ran the last
  task (the highest task id: llama.cpp numbers them in order), keyed by the settings the
  container was started with (`slot_folder.StartStamp`), never the recipe as saved since.
- **Put back** (`after_start`, the last step of a start once `/health` answered): the newest
  file of the started settings and build, into an empty slot; the start's stamp written first.

The folder's files and rules are `slot_folder`. Neither ever fails a stop or a start: every
error becomes the line's words."""

from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from makenomistakes.consumers import engine_origin
from makenomistakes.engine_answers import ErrorAnswer, said_error
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.journal import Note, to_stderr
from makenomistakes.measurements import settings_key
from makenomistakes.prompt_cache import DISK_FLOOR_GIB, GIB
from makenomistakes.recipes.llamacpp_cache import keeps_slots
from makenomistakes.recipes.model import Recipe
from makenomistakes.slot_folder import SavedSlot, SlotFolder, StartStamp, gib
from makenomistakes.supervisor import ContainerState, ContainerStatus, SupervisorError

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

SAVE_MIN_TOKENS = 2048  # shorter prompts are read again faster than their file is written
# One slot a stop, under the box's start lock: a full 32k slot of an 8B model is ~2.5 GB,
# 2.8 s for 495 MB on a virtual machine's disk; a stop never waits longer than this for it.
SAVE_SECONDS = 120.0
RESTORE_SECONDS = 120.0
LOOK_SECONDS = 10.0


class SlotState(BaseModel):
    """One entry of llama-server's `GET /slots` (only what this module reads)."""

    model_config = ConfigDict(extra="ignore")

    id: int
    is_processing: bool = False
    n_prompt_tokens: int | None = None
    # the last task this slot ran in this process; absent while it has run none since the
    # engine started (empty, or holding what a restore put back). `/slots` gives no prompt
    # text unless the server is in its slots-debug mode, so this is what tells states apart.
    id_task: int | None = None

    @property
    def tokens(self) -> int:
        return self.n_prompt_tokens or 0

    @property
    def ran_a_task(self) -> bool:
        return self.id_task is not None and self.id_task >= 0

    def digest(self, started: str | None) -> str:
        """The state it holds, by the engine's start, its last task and its length: the same
        state saved twice is one file (task ids start again in each process)."""
        return hashlib.sha256(f"{started}|{self.id_task}|{self.tokens}".encode()).hexdigest()


class _Timings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    save_ms: float = 0.0
    restore_ms: float = 0.0


class SlotAnswer(BaseModel):
    """`POST /slots/<id>?action=save|restore`'s answer."""

    model_config = ConfigDict(extra="ignore")

    n_saved: int | None = None
    n_written: int | None = None
    n_restored: int | None = None
    n_read: int | None = None
    timings: _Timings = Field(default_factory=_Timings)


class SlotError(Exception):
    """The engine refused or did not answer: the words go into the line."""


def engine_client(recipe: Recipe, timeout: float) -> httpx.Client:
    """The instance's own engine on this machine (`consumers.engine_origin`), never through a
    proxy: its slot routes are mnm's, not a client's (the router refuses a client's)."""
    return httpx.Client(base_url=engine_origin(recipe), timeout=timeout, trust_env=False)


def slots(http: httpx.Client) -> list[SlotState]:
    try:
        answer = http.get("/slots", timeout=LOOK_SECONDS)
        answer.raise_for_status()
        return [SlotState.model_validate(entry) for entry in answer.json()]
    except (httpx.HTTPError, ValueError, ValidationError, TypeError) as exc:
        raise SlotError(f"its slots could not be read ({_reason(exc)})") from exc


def act(http: httpx.Client, slot: int, action: str, file: str, timeout: float) -> SlotAnswer:
    """One slot action, typed both ways; a refusal is a `SlotError` in the engine's words."""
    body = {"filename": file} if file else {}
    try:
        answer = http.post(f"/slots/{slot}", params={"action": action}, json=body, timeout=timeout)
    except httpx.HTTPError as exc:
        raise SlotError(f"the engine did not answer ({_reason(exc)})") from exc
    if answer.status_code != 200:
        raise SlotError(f"the engine answered {answer.status_code}: {_message(answer)}")
    try:
        return SlotAnswer.model_validate(answer.json())
    except (ValueError, ValidationError) as exc:
        raise SlotError(f"the engine's answer could not be read ({_reason(exc)})") from exc


def _message(answer: httpx.Response) -> str:
    try:
        said = said_error(ErrorAnswer.model_validate_json(answer.content).error)
    except ValidationError:
        said = ""
    return (said or answer.text or "no message")[:200]


def _reason(exc: Exception) -> str:
    return str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__


def cap_bytes(recipe: Recipe) -> int:
    return int((recipe.llama_args.cache_disk_gib or DISK_FLOOR_GIB) * GIB)


def last_used(states: list[SlotState]) -> SlotState | None:
    """The slot worth keeping: idle, long enough, and the one that ran the last task (task
    ids rise in order), so a server of several slots keeps the conversation used last."""
    worth = [
        s for s in states if not s.is_processing and s.ran_a_task and s.tokens >= SAVE_MIN_TOKENS
    ]
    return max(worth, key=lambda s: s.id_task or 0, default=None)


def save(
    recipe: Recipe,
    folder: SlotFolder,
    stamp: StartStamp,
    http: httpx.Client,
    now: datetime | None = None,
) -> str | None:
    """The slot used last saved before the engine stops, under the settings it was started
    with (`stamp`); the line, or None when there was nothing to save (short prompts, a slot
    holding only what a restore put back: that file is on disk already)."""
    folder.stale(stamp.key, stamp.build)
    try:
        state = last_used(slots(http))
    except SlotError as exc:
        return f"{recipe.name}'s prompt was not saved: {exc}"
    if state is None:
        return None
    digest = state.digest(stamp.started_at)
    at = now or datetime.now(UTC)
    same = next((s for s in folder.saved() if s.digest == digest), None)
    if same is not None:  # unchanged since it was saved: kept, now the newest
        folder.write(same.model_copy(update={"saved_at": at}))
        return f"{recipe.name}'s prompt ({state.tokens:,} tokens) was saved already"
    cap = cap_bytes(recipe)
    free = _free(folder)
    if free is not None and free < cap:
        return (
            f"{recipe.name}'s prompt was not saved: the models disk has {gib(free)} GiB free, "
            f"under its cache-disk={cap / GIB:g} GiB"
        )
    file = f"{stamp.key}-{stamp.build}-{at.strftime('%Y%m%d%H%M%S%f')}-s{state.id}.bin"
    try:
        answer = act(http, state.id, "save", file, SAVE_SECONDS)
    except SlotError as exc:
        folder.drop(file)
        return f"{recipe.name}'s prompt was not saved: {exc}"
    row = SavedSlot(
        file=file,
        key=stamp.key,
        build=stamp.build,
        tokens=answer.n_saved or state.tokens,
        bytes=answer.n_written or folder.size(file),
        digest=digest,
        saved_at=at,
    )
    return _kept(recipe, folder, row, cap, answer.timings.save_ms)


def _kept(recipe: Recipe, folder: SlotFolder, row: SavedSlot, cap: int, save_ms: float) -> str:
    """A file the engine wrote, kept with its sidecar when it fits the cap with it (the
    older ones going first, never this one); else dropped. The line either way."""
    folder.write(row)
    if (size := folder.size(row.file)) > cap:
        folder.drop(row.file)
        return (
            f"{recipe.name}'s prompt ({row.tokens:,} tokens, {gib(size)} GiB) was not kept: "
            f"over cache-disk={cap / GIB:g} (`mnm set {recipe.name} cache-disk=` raises it)"
        )
    gone = folder.trimmed(cap, keep=row.file)
    said = (
        f"Saved {recipe.name}'s prompt: {row.tokens:,} tokens, {gib(size)} GiB in "
        f"{save_ms / 1000:.1f} s"
    )
    return said + (f" ({len(gone)} older dropped for cache-disk)" if gone else "")


def _free(folder: SlotFolder) -> int | None:
    try:
        return shutil.disk_usage(folder.path).free
    except OSError:
        return None


def restore(
    recipe: Recipe, folder: SlotFolder, stamp: StartStamp, http: httpx.Client
) -> str | None:
    """The newest file of the started settings and build put back into an empty slot of a
    freshly started engine; the line, or None when there is none to put back. A file the
    engine refuses, or cannot read in time, goes: it would be the newest at every start."""
    folder.stale(stamp.key, stamp.build)
    newest = next((s for s in folder.saved() if (folder.path / s.file).is_file()), None)
    if newest is None:
        return None
    try:
        empty = next((s for s in slots(http) if not s.is_processing and not s.tokens), None)
        if empty is None:
            return None  # every slot holds a prompt already: none is overwritten
        answer = act(http, empty.id, "restore", newest.file, RESTORE_SECONDS)
    except SlotError as exc:
        folder.drop(newest.file)
        return f"{recipe.name}'s saved prompt was not put back, and was deleted: {exc}"
    tokens = answer.n_restored or newest.tokens
    return (
        f"Put back {recipe.name}'s saved prompt: {tokens:,} tokens in "
        f"{answer.timings.restore_ms / 1000:.1f} s"
    )


def after_start(service: InstanceService, recipe: Recipe) -> str | None:
    """A start's last step once its engine answered (`instance_start`): the start stamped as
    the one that may save into the folder, and the newest saved prompt of these settings put
    back. The line for the start's notes, or None. Never fails the start."""
    if not keeps_slots(recipe):
        return None
    try:
        status = service.supervisor.status(recipe.name)
        if status.build is None or status.started_at is None:
            return None
        folder = SlotFolder.of(service.models_dir, recipe.name)
        if not folder.path.is_dir():
            return f"{recipe.name}'s saved prompts are off this start: {folder.path} is missing"
        stamp = StartStamp(
            key=settings_key(recipe), build=status.build, started_at=status.started_at
        )
        folder.stamp(stamp)
        with engine_client(recipe, RESTORE_SECONDS) as http:
            return restore(recipe, folder, stamp, http)
    except Exception as exc:  # a cache: the start succeeded
        return f"{recipe.name}'s saved prompt was not put back: {_reason(exc)}"


def before_stop(service: InstanceService, recipe: Recipe, told: Note = to_stderr) -> None:
    """`InstanceService.stop`'s first step: the running engine's slot used last saved, under
    the settings its start stamped (a start of other settings, or of none, saves nothing);
    the line to `told` (the router's journal, the CLI's console). A save is never why a stop
    fails: anything it raises becomes the line."""
    try:
        line = _save_running(service, recipe)
    except Exception as exc:
        line = f"{recipe.name}'s prompt was not saved: {_reason(exc)}"
    if line:
        told(line + ".")


def _save_running(service: InstanceService, recipe: Recipe) -> str | None:
    if not isinstance(recipe.args, LlamaCppArgs):
        return None
    folder = SlotFolder.of(service.models_dir, recipe.name)
    if not folder.path.is_dir():
        return None  # never started with the cache on, or `cache=off` since
    try:
        status: ContainerStatus = service.supervisor.status(recipe.name)
    except SupervisorError:
        return None  # the stop says it
    if status.state is not ContainerState.RUNNING:
        return None
    stamp = folder.stamped(status.started_at)
    if stamp is None:
        return None  # not a start that mounted the folder (before this mnm, or cache off)
    with engine_client(recipe, SAVE_SECONDS) as http:
        return save(recipe, folder, stamp, http)
