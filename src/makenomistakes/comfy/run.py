"""Run an imported workflow: its form's values into the converted prompt, the
uploads, `POST /prompt`, progress from ComfyUI's websocket, and the outputs moved from its
output folder into the run's folder (one copy), kept as a run in the gallery.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError
from websockets.exceptions import WebSocketException

from makenomistakes.comfy import idle
from makenomistakes.comfy.answers import (
    QUIET,
    ComfyRunError,
    OutputFile,
    Queued,
    Uploaded,
    open_messages,
    refusal_words,
    settled,
)
from makenomistakes.comfy.convert import ConvertError, to_prompt
from makenomistakes.comfy.engine import ComfyEngine, ComfyError, State, unread_nodes
from makenomistakes.comfy.exits import why_stopped
from makenomistakes.comfy.form import Field as FormField
from makenomistakes.comfy.form import FormError, Kind, Role, apply, build
from makenomistakes.comfy.image import Variant
from makenomistakes.comfy.object_info import ObjectInfo
from makenomistakes.comfy.outputs import collect, history_outputs, own_prefixes, wired_words
from makenomistakes.comfy.packs import approved_names
from makenomistakes.comfy.readings import Readings, model_bytes, size_warning
from makenomistakes.comfy.report import MODEL_SUFFIXES, analyze
from makenomistakes.comfy.room import MakeRoom, run_need
from makenomistakes.comfy.store import WorkflowMeta, WorkflowStore, WorkflowStoreError
from makenomistakes.hardware.probe import MemorySnapshot, probe_memory
from makenomistakes.instance_types import InstanceError
from makenomistakes.progress_words import Whole, parked_for_room
from makenomistakes.runs import RunError, RunKind, RunRecord, RunStore

__all__ = ["UPLOAD_TYPES", "Again", "ComfyRunError", "ComfyRunner", "upload_role"]

Progress = Callable[[str], None]
RUN_SECONDS = 4 * 3600.0  # a long video on 12 GB, streamed: hours, not days
# What a run says in its own words (an unreadable recipe or a missing link included); anything
# else is an interruption.
WORDED = (ComfyRunError, ComfyError, FormError, InstanceError, httpx.HTTPError, OSError)
UPLOAD_TYPES = {Kind.IMAGE: "image/png", Kind.VIDEO: "video/mp4", Kind.AUDIO: "audio/wav"}


@dataclass(frozen=True)
class Again:
    """What "Run again" sends: the workflow, its settings (the seed as used, unless a new
    one is drawn) and the uploads by field key."""

    target: str
    values: dict[str, str]
    uploads: dict[str, tuple[str, bytes]]


class ComfyRunner:
    def __init__(  # noqa: PLR0913 - every outside dependency can be replaced
        self,
        engine: ComfyEngine,
        workflows: WorkflowStore,
        runs: RunStore,
        make_room: MakeRoom,
        *,
        http: Callable[[], httpx.Client] | None = None,
        socket: Callable[[str], AbstractContextManager[Iterator[str | bytes]]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        readings: Callable[[], Readings] = Readings,
        memory: Callable[[], MemorySnapshot] = probe_memory,
    ) -> None:
        self.engine = engine
        self.workflows = workflows
        self.runs = runs
        self._make_room = make_room
        self._http = http or (lambda: httpx.Client(timeout=120.0, trust_env=False))
        self._socket = socket or open_messages
        self._clock = clock
        self._readings = readings
        self._memory = memory

    # -- the form ------------------------------------------------------------------------

    def info(self) -> ObjectInfo:
        found = self.engine.object_info()
        if found is not None:
            return found
        try:  # an unreadable ledger says so, never "not installed"
            state = self.engine.checked_state()
        except ComfyError as exc:
            raise ComfyRunError(str(exc)) from exc
        raise ComfyRunError(unread_nodes(state))

    def prompt(self, name: str) -> tuple[dict[str, dict[str, Any]], list[FormField]]:
        """The converted prompt and its form."""
        try:
            workflow = self.workflows.workflow(name)
        except (WorkflowStoreError, OSError, ValueError) as exc:
            raise ComfyRunError(str(exc)) from exc
        info = self.info()
        report = analyze(workflow, info)
        if not report.runnable:
            approved = approved_names(self.engine.settings.expanded_state_dir())
            why = report.problems or [
                *(
                    f"the node pack {p.pack} is "
                    + (
                        "approved but not built into this image (`mnm comfyui packs install`)"
                        if p.pack.lower() in approved
                        else "not installed"
                    )
                    for p in report.packs
                    if not p.installed
                ),
                *(f"no pack is named for {m}" for m in report.missing),
            ]
            raise ComfyRunError(f"{name} cannot run here yet: {'; '.join(why)}.")
        try:
            prompt = to_prompt(workflow, info)
        except ConvertError as exc:
            raise ComfyRunError(f"{name} cannot be converted: {exc}.") from exc
        fields = build(prompt, info)
        try:
            picked = set(self.workflows.meta(name).shown)
        except WorkflowStoreError:
            picked = set()
        for found in fields:
            found.picked = found.key in picked
        return prompt, fields

    def set_shown(self, name: str, keys: list[str] | Callable[[list[str]], list[str]]) -> list[str]:
        """The settings shown on top of the form (out of More settings), in the user's order:
        the keys, or a change of the current ones (worked out under the store's lock, so a
        Save on the web meanwhile is never lost). A key not under More settings is refused.
        They are a workflow tool's arguments over MCP too (every shown field is one)."""
        _prompt, fields = self.prompt(name)
        more = {f.key for f in fields if f.role is Role.MORE}

        def change(meta: WorkflowMeta) -> None:
            # a saved pick no longer under More settings (shown by itself now) is dropped,
            # never a refusal of every later pick
            current = [k for k in meta.shown if k in more]
            wanted = keys(current) if callable(keys) else keys
            unknown = [k for k in wanted if k not in more]
            if unknown:
                raise ComfyRunError(f"Not under More settings: {', '.join(unknown)}.")
            meta.shown = list(dict.fromkeys(wanted))

        try:
            return self.workflows.update(name, change).shown
        except WorkflowStoreError as exc:
            raise ComfyRunError(str(exc)) from exc

    def again(self, run_id: str, new_seed: bool = False) -> Again:
        """A workflow run's settings and uploads ("Run again"), its seeds drawn afresh with
        `new_seed`; the caller makes the record (`new`) once it may start. An upload the run
        kept and cannot give back is a refusal, never the workflow's own picture instead."""
        try:
            old = self.runs.get(run_id)
        except RunError as exc:
            raise ComfyRunError(str(exc)) from exc
        if old.kind is not RunKind.WORKFLOW:
            raise ComfyRunError(f"{run_id} is not a workflow run.")
        _prompt, fields = self.prompt(old.target)
        by_key = {f.key: f for f in fields}
        values = {
            key: value
            for key, value in old.settings.items()
            if key in by_key
            and by_key[key].kind not in UPLOAD_TYPES
            and not (new_seed and by_key[key].kind is Kind.SEED)
        }
        roles = {upload_role(f.key): f.key for f in fields if f.kind in UPLOAD_TYPES}
        try:
            kept = self.runs.inputs(old, list(roles))
        except RunError as exc:
            raise ComfyRunError(str(exc)) from exc
        return Again(old.target, values, {roles[role]: item for role, item in kept.items()})

    # -- a run ---------------------------------------------------------------------------

    def new(self, name: str, values: dict[str, str]) -> RunRecord:
        """The run's record, before anything is sent (the gallery shows it running)."""
        _prompt, fields = self.prompt(name)
        prompts = [f for f in fields if f.role.value == "prompt"]
        # an emptied prompt is the run's words, never the workflow's own
        words = next((values.get(f.key, str(f.default)) for f in prompts), None)
        try:
            return self.runs.new(RunKind.WORKFLOW, name, words, dict(values))
        except RunError as exc:
            raise ComfyRunError(str(exc)) from exc

    def run(
        self,
        run: RunRecord,
        values: dict[str, str],
        uploads: dict[str, tuple[str, bytes]] | None = None,
        progress: Progress | None = None,
    ) -> RunRecord:
        """Start ComfyUI if it is off, make room, send the prompt, follow it, keep the files.
        A failure is kept as a failed run and raised."""
        say = progress or (lambda _m: None)
        queued = False  # `last-run` is stamped only for a run ComfyUI got
        try:
            prompt, fields = self.prompt(run.target)
            used = apply(prompt, fields, values)
            if self.engine.state() is not State.RUNNING:
                self.engine.start(say)
            files_size, weights = self._weights(run.target)
            with ExitStack() as room:  # held until ComfyUI has the prompt (`_send`)
                need = self._need(run.target, files_size, weights)
                parked = room.enter_context(self._make_room(say, need))
                if parked:
                    say(Whole(parked_for_room(parked, "ComfyUI")))
                address = self.engine.address()
                if address is None:
                    raise ComfyRunError("ComfyUI is not running.")
                self.engine.refresh_object_info(address)  # files fetched since its start count
                warning = size_warning(files_size, self._memory())
                if warning:
                    say(Whole(warning))
                _posix_model_names(prompt)
                wired = own_prefixes(prompt, run.id)
                if wired:
                    say(Whole(wired_words(wired)))
                queued = True
                sent = self._send(address, run, fields, prompt, uploads or {}, say, room.close)
            files, readings, cached = sent
            run.cached_nodes = cached
            made = self._collect(run, files)
        except WORDED as exc:
            self.runs.fail(run, str(exc))
            raise ComfyRunError(str(exc)) from exc
        except (ValidationError, ValueError, OverflowError) as exc:
            self.runs.fail(run, f"ComfyUI answered something mnm cannot read: {exc}")
            raise ComfyRunError(str(exc)) from exc
        except BaseException:
            self.runs.fail(run, "Interrupted before it finished.")  # never left running
            raise
        finally:
            if queued:
                idle.ran(self.engine)  # the router's sweep frees the card ten minutes on
        run.settings = {**run.settings, **{k: str(v) for k, v in used.items()}}
        answer = f"{len(made)} file{'s' if len(made) != 1 else ''}" if made else "no files"
        numbers = readings.numbers(files_size)
        return self.runs.finish(run, made, answer, readings.seconds, numbers)

    def _send(
        self,
        address: str,
        run: RunRecord,
        fields: list[FormField],
        prompt: dict[str, dict[str, Any]],
        uploads: dict[str, tuple[str, bytes]],
        say: Progress,
        queued: Callable[[], None],
    ) -> tuple[list[OutputFile], Readings, list[str]]:
        """Upload, queue (then `queued`: the room made for it goes with the start lock),
        follow to the end, read what it saved; an aborted run's prompt is taken back from
        ComfyUI (`_abandon`)."""
        client = uuid.uuid4().hex
        socket = address.replace("http://", "ws://") + f"/ws?clientId={client}"
        with self._http() as http:
            self._upload(http, address, run, fields, prompt, uploads)
            prompt_id = ""
            try:  # listening before it is queued: a quick run's messages are not missed
                with self._socket(socket) as messages, self._readings() as readings:
                    with self.engine.locked():  # never between a Free VRAM's check and free
                        prompt_id = self._queue(http, address, run.target, prompt, client)
                    run.prompt_id = prompt_id  # what the web's Stop interrupts, only this
                    self.runs.save(run)
                    queued()
                    cached = self._follow(messages, address, prompt_id, say)
            except (WebSocketException, TimeoutError) as exc:
                self._abandon(http, address, prompt_id)
                ended = why_stopped(self.engine)  # out of memory reads as such
                raise ComfyRunError(ended or f"Lost ComfyUI's progress messages: {exc}") from exc
            except BaseException:  # Ctrl-C, a restart, the deadline: never left running
                self._abandon(http, address, prompt_id)
                raise
            try:
                return self._outputs(http, address, prompt_id), readings, cached
            except httpx.TransportError as exc:  # it ended as the last message came
                ended = why_stopped(self.engine)
                raise ComfyRunError(ended or f"ComfyUI stopped answering: {exc}") from exc

    def _need(self, name: str, files: int, weights: bool) -> int | None:
        """The VRAM this run needs free (`room.run_need`); none on a CPU build."""
        ledger = self.engine.ledger()
        if ledger is not None and ledger.variant is Variant.CPU:
            return 0
        try:
            earlier = self.runs.all()
        except RunError:
            earlier = []
        return run_need(earlier, name, files, weights)

    def _weights(self, name: str) -> tuple[int, bool]:
        """The size of the model files a workflow loads, and whether it reads any weights at
        all (a workflow mnm cannot read counts as one that does)."""
        try:
            report = analyze(self.workflows.workflow(name))
        except (WorkflowStoreError, OSError, ValueError):
            return 0, True
        return model_bytes(self.engine.settings.expanded_models_dir(), report.models), (
            report.reads_weights
        )

    def _upload(
        self,
        http: httpx.Client,
        address: str,
        run: RunRecord,
        fields: list[FormField],
        prompt: dict[str, dict[str, Any]],
        uploads: dict[str, tuple[str, bytes]],
    ) -> None:
        by_key = {f.key: f for f in fields}
        for key, (filename, data) in uploads.items():
            field_ = by_key.get(key)
            if field_ is None or field_.kind not in UPLOAD_TYPES:
                raise ComfyRunError(f"The workflow takes no file at {key}.")
            kept = self.runs.add_input(run, filename, data, role=upload_role(field_.key))
            files = {"image": (kept.name, data, UPLOAD_TYPES[field_.kind])}
            response = http.post(f"{address}/upload/image", files=files, data={"overwrite": "true"})
            if response.is_error:
                raise ComfyRunError(f"ComfyUI refused the upload {filename}: {response.text[:200]}")
            stored = Uploaded.model_validate_json(response.content)
            name = stored.name or kept.name
            sub = stored.subfolder
            prompt[field_.node]["inputs"][field_.input] = f"{sub}/{name}" if sub else name

    def _queue(
        self,
        http: httpx.Client,
        address: str,
        name: str,
        prompt: dict[str, dict[str, Any]],
        client: str,
    ) -> str:
        try:
            workflow = json.loads((self.workflows.folder(name) / "workflow.json").read_bytes())
        except (OSError, ValueError):
            workflow = None
        body = {
            "prompt": prompt,
            "client_id": client,
            # what SaveImage writes into each PNG, so a picture carries its workflow
            "extra_data": {"extra_pnginfo": {"workflow": workflow}} if workflow else {},
        }
        response = http.post(f"{address}/prompt", json=body)
        if response.status_code == httpx.codes.BAD_REQUEST:
            raise ComfyRunError(refusal_words(response.content))
        if response.is_error:
            raise ComfyRunError(
                f"ComfyUI answered HTTP {response.status_code}: {response.text[:300]}"
            )
        return Queued.model_validate_json(response.content).prompt_id

    def _follow(
        self, messages: Iterator[str | bytes], address: str, prompt_id: str, say: Progress
    ) -> list[str]:
        """Until ComfyUI says the prompt is done, failed or was interrupted; the nodes its
        cache served. A quiet spell asks its queue at `address`, never Docker: a `docker
        inspect` that times out is no answer, never "finished"."""
        cached: list[str] = []
        deadline = self._clock() + RUN_SECONDS
        for message in messages:
            if self._clock() > deadline:
                raise ComfyRunError("The run took longer than mnm waits (4 h).")
            if isinstance(message, bytes):
                continue  # a preview picture
            if message == QUIET:
                if self.engine.queue_busy(address) is False:
                    return cached  # it ended while no message came: its history says the rest
                continue  # still at work, only quiet, or `/queue` did not answer: unknown
            if settled(message, prompt_id, say, cached.extend):
                return cached
        raise ComfyRunError("ComfyUI closed its progress messages before the run ended.")

    def _outputs(self, http: httpx.Client, address: str, prompt_id: str) -> list[OutputFile]:
        return history_outputs(http, address, prompt_id)

    def _collect(self, run: RunRecord, files: list[OutputFile]) -> list[Path]:
        """Each output moved from ComfyUI's folder into the run's (`outputs.collect`)."""
        return collect(self.engine.base, self.runs.folder(run), run.id, files)

    def _abandon(self, http: httpx.Client, address: str, prompt_id: str) -> None:
        """A run mnm stopped following: its prompt taken out of ComfyUI's queue, or stopped
        if it is the one running (only that one: `/interrupt` with its id), so ComfyUI does
        not hold the card for work nobody collects."""
        if not prompt_id:
            return
        try:
            http.post(f"{address}/queue", json={"delete": [prompt_id]})
            http.post(f"{address}/interrupt", json={"prompt_id": prompt_id})
        except httpx.HTTPError:
            pass  # ComfyUI gone too: nothing runs

    def interrupt(self, run: RunRecord) -> None:
        """Stop one run: its prompt taken out of ComfyUI's queue, or interrupted if it is
        the one running (`/interrupt` with its id: never another run's work)."""
        if not run.prompt_id:
            raise ComfyRunError(
                f"{run.id} has not reached ComfyUI's queue yet (it is starting ComfyUI or "
                "making room); stop it once it runs."
            )
        address = self.engine.asked_address()  # Docker silent: ComfyUI itself is asked
        if address is None:
            raise ComfyRunError("ComfyUI is not running: nothing to stop.")
        try:
            with self._http() as http:
                http.post(f"{address}/queue", json={"delete": [run.prompt_id]})
                http.post(f"{address}/interrupt", json={"prompt_id": run.prompt_id})
        except httpx.HTTPError as exc:
            raise ComfyRunError(f"Could not reach ComfyUI to stop {run.id}: {exc}") from exc


def upload_role(key: str) -> str:
    """A field's key (`57:27.image`) as the name its upload is kept under, so "Run again"
    gives each upload back to its own field."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", key)


def _posix_model_names(prompt: dict[str, dict[str, Any]]) -> None:
    """A workflow saved on Windows names `WanVideo\\x.safetensors`; ComfyUI on Linux lists
    `WanVideo/x.safetensors` (mnm's links keep the subfolders)."""
    for entry in prompt.values():
        inputs = entry["inputs"]
        for name, value in inputs.items():
            if isinstance(value, str) and "\\" in value and value.lower().endswith(MODEL_SUFFIXES):
                inputs[name] = value.replace("\\", "/")
