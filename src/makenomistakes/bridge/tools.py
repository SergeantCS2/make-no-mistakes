"""mnm's runs as MCP tools: one tool per imported ComfyUI workflow that runs here, one
for the Short film, and `run_status`.

A tool starts its run on a thread of the router's process and answers at once with the run's
id and its page in mnm's web UI: a clip takes minutes. `run_status` reads the run record (and
can wait for it), and hands back a finished run's pictures as image content, which Odysseus
keeps as images. A workflow tool's arguments are its form's shown fields, by a name made from
each field's label; a seed left out is random each run.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic import Field as PydField

from makenomistakes.bridge.protocol import ToolError, ToolResult, ToolSpec, UnknownToolError
from makenomistakes.comfy.form import Field as FormField
from makenomistakes.comfy.form import Kind
from makenomistakes.comfy.run import ComfyRunError, ComfyRunner
from makenomistakes.instance_types import InstanceError
from makenomistakes.media_bytes import image_mime
from makenomistakes.pipeline import short_film as film
from makenomistakes.pipeline.runner import FilmRunner, instances_for
from makenomistakes.pipeline.short_film import FilmInputs
from makenomistakes.runs import FileKind, RunError, RunKind, RunRecord, RunState, RunStore
from makenomistakes.validation_words import worded

STATUS = "run_status"
FILM = "short_film"
WORKFLOW_PREFIX = "workflow_"
MAX_WAIT = 600.0  # run_status's wait_seconds, at most
POLL = 2.0
MAX_IMAGE = 4 * 2**20  # a picture handed back inline, at most (larger ones: the link)
MAX_IMAGES = 4
_SCHEMA_TYPES = {
    Kind.TEXT: "string",
    Kind.LONG_TEXT: "string",
    Kind.CHOICE: "string",
    Kind.INT: "integer",
    Kind.SEED: "integer",
    Kind.FLOAT: "number",
    Kind.BOOL: "boolean",
}
Spawn = Callable[[Callable[[], object]], None]


def _thread(work: Callable[[], object]) -> None:
    threading.Thread(target=work, daemon=True).start()


def arg_names(fields: list[FormField]) -> dict[str, FormField]:
    """Each shown, typed field under a name a model writes easily: its label in snake case
    (`negative_prompt`), numbered when two share one."""
    found: dict[str, FormField] = {}
    for f in fields:
        if not f.shown or f.kind not in _SCHEMA_TYPES:
            continue
        base = re.sub(r"[^a-z0-9]+", "_", f.label.lower()).strip("_")[:40] or "value"
        name, n = base, 2
        while name in found:
            name, n = f"{base}_{n}", n + 1
        found[name] = f
    return found


def _property(f: FormField) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": _SCHEMA_TYPES[f.kind], "description": f"{f.group}: {f.label}"}
    if f.kind is Kind.SEED:
        schema["description"] += " (left out: random each run)"
    else:
        schema["description"] += f" (the workflow's own: {f.default!r})"
    if f.kind is Kind.CHOICE and f.options:
        schema["enum"] = [str(o) for o in f.options]
    if f.kind in (Kind.INT, Kind.FLOAT):
        if f.minimum is not None:
            schema["minimum"] = f.minimum
        if f.maximum is not None and f.maximum < 1e15:
            schema["maximum"] = f.maximum
    return schema


class MnmTools:
    def __init__(
        self,
        comfy: ComfyRunner,
        films: FilmRunner,
        runs: RunStore,
        web_origin: str,
        spawn: Spawn = _thread,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.comfy = comfy
        self.films = films
        self.runs = runs
        self.web = web_origin.rstrip("/")
        self._spawn = spawn
        self._sleep = sleep
        self._clock = clock

    # -- the list -------------------------------------------------------------------------

    def _workflows(self) -> dict[str, tuple[str, list[FormField]]]:
        """Tool name → (workflow, its fields), for each workflow that runs here now."""
        found = {}
        for meta in self.comfy.workflows.all():
            try:
                _prompt, fields = self.comfy.prompt(meta.name)
            except ComfyRunError:
                continue  # a pack or a node missing: not offered until it runs
            found[tool_name(meta.name)] = (meta.name, fields)
        return found

    def tools(self) -> list[ToolSpec]:
        specs = [self._status_spec()]
        for tool, (name, fields) in sorted(self._workflows().items()):
            args = arg_names(fields)
            uploads = [f.label for f in fields if f.kind in (Kind.IMAGE, Kind.VIDEO, Kind.AUDIO)]
            note = (
                f" It also takes {', '.join(uploads)} as a file, which only mnm's web UI can "
                "give it; here the workflow's own is used."
                if uploads
                else ""
            )
            specs.append(
                ToolSpec(
                    tool,
                    f"Run the ComfyUI workflow '{name}' on this machine's GPU (minutes). Returns "
                    f"a run id: call {STATUS} with it for the files.{note}",
                    {
                        "type": "object",
                        "properties": {a: _property(f) for a, f in args.items()},
                        "additionalProperties": False,
                    },
                )
            )
        workflows = [m.name for m in self.comfy.workflows.all()]
        try:
            choices = instances_for(self.films.service, workflows)
        except InstanceError:  # a recipe this version cannot read: the workflows still list
            return specs
        if choices["chat"] and choices["image"] and choices["video"]:
            specs.append(self._film_spec(choices))
        return specs

    def _status_spec(self) -> ToolSpec:
        return ToolSpec(
            STATUS,
            "How an mnm run is going: running, done (with its files; pictures come back as "
            "images) or failed (with why). wait_seconds waits up to that long for it to end.",
            {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "The id a run tool returned."},
                    "wait_seconds": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": int(MAX_WAIT),
                        "description": "Wait this long for the run to end (default 0).",
                    },
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
        )

    def _film_spec(self, choices: dict[str, list[str]]) -> ToolSpec:
        def pick(role: str, what: str) -> dict[str, Any]:
            return {"type": "string", "enum": choices[role], "description": what}

        return ToolSpec(
            FILM,
            "Make a short film from one idea with mnm's own models: a shot list, a still per "
            "shot, a narrator's lines, a clip per still, one film. Runs straight through (tens "
            f"of minutes); returns a run id for {STATUS}.",
            {
                "type": "object",
                "properties": {
                    "idea": {"type": "string", "description": "The film's idea, a sentence."},
                    "shots": {"type": "integer", "minimum": 1, "maximum": film.MAX_SHOTS},
                    "style": {"type": "string", "description": "The look every still shares."},
                    "chat": pick("chat", "The model that writes the shot list."),
                    "image": pick("image", "The model that draws each still."),
                    "video": pick("video", "What animates each still (comfy:<workflow> too)."),
                    "voice": pick("voice", "The narrator's voice; left out: a silent film."),
                },
                "required": ["idea", "chat", "image", "video"],
                "additionalProperties": False,
            },
        )

    # -- a call ---------------------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == STATUS:
            return self._status(arguments)
        if name == FILM:
            return self._film(arguments)
        # only the workflow called is converted: run_status polls never pay for the rest
        wanted = next(
            (m.name for m in self.comfy.workflows.all() if tool_name(m.name) == name), None
        )
        if wanted is None:
            raise UnknownToolError(f"There is no tool {name}: list the tools again.")
        try:
            _prompt, fields = self.comfy.prompt(wanted)
        except ComfyRunError as exc:
            raise ToolError(str(exc)) from exc
        return self._workflow(wanted, fields, arguments)

    def _workflow(
        self, name: str, fields: list[FormField], arguments: dict[str, Any]
    ) -> ToolResult:
        args = arg_names(fields)
        unknown = sorted(set(arguments) - set(args))
        if unknown:
            raise ToolError(f"{name} takes no {', '.join(unknown)}; it takes {', '.join(args)}.")
        values = {args[k].key: _word(v) for k, v in arguments.items() if v is not None}
        try:
            record = self.comfy.new(name, values)
        except ComfyRunError as exc:
            raise ToolError(str(exc)) from exc

        def work() -> None:
            # a failure is kept on the record, with why: run_status says it
            with contextlib.suppress(ComfyRunError):
                self.comfy.run(record, values)

        self._spawn(work)
        return self._started(record, f"{self.web}/comfy/{name}")

    def _film(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            inputs = FilmInputs.model_validate({**arguments, "auto": True})
        except ValidationError as exc:
            raise ToolError(f"The film cannot start: {worded(exc.errors(), ARGUMENT)}.") from exc
        try:
            run = self.films.start(inputs)
        except (ValueError, InstanceError) as exc:
            raise ToolError(f"The film cannot start: {exc}") from exc
        self._spawn(lambda: self.films.advance(run.id))
        return self._started(run, f"{self.web}/pipelines/{run.id}")

    def _started(self, run: RunRecord, page: str) -> ToolResult:
        return ToolResult.text(
            f"Started run {run.id} ({run.target}). It takes minutes: call {STATUS} with "
            f"run_id {run.id!r} (wait_seconds up to {int(MAX_WAIT)}) for its files. "
            f"Its page in mnm: {page}"
        )

    def _status(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            asked = _StatusArgs.model_validate(arguments)
        except ValidationError as exc:
            said = worded(exc.errors(), ARGUMENT)
            raise ToolError(
                f"run_status takes run_id and wait_seconds (a number): {said}."
            ) from exc
        run_id, wait = asked.run_id, float(asked.wait_seconds)
        deadline = self._clock() + wait
        try:
            run = self.runs.get(run_id)
            while run.state is RunState.RUNNING and self._clock() < deadline:
                self._sleep(POLL)
                run = self.runs.get(run_id)
        except RunError as exc:
            raise ToolError(str(exc)) from exc
        return self._said(run)

    def _said(self, run: RunRecord) -> ToolResult:
        page = (
            f"{self.web}/pipelines/{run.id}"
            if run.kind is RunKind.PIPELINE
            else f"{self.web}/runs#run-{run.id}"
        )
        if run.state is RunState.FAILED:
            return ToolResult.text(f"Run {run.id} failed: {run.error} ({page})", is_error=True)
        if run.state is not RunState.DONE:
            step = next((s for s in run.steps if s.state.value == "running"), None)
            doing = f", now at {film.STAGE_LABELS.get(step.stage, step.stage)}" if step else ""
            return ToolResult.text(f"Run {run.id} is still {run.state.value}{doing}. {page}")
        files = [f"{f.name} ({f.kind.value})" for f in run.outputs]
        lines = [
            f"Run {run.id} is done: {run.answer}.",
            f"Files: {', '.join(files) or 'none'}.",
            f"See and download them in mnm: {page}",
        ]
        result = ToolResult.text(" ".join(lines))
        result.content += self._pictures(run)
        return result

    def _pictures(self, run: RunRecord) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for item in run.outputs:
            if item.kind is not FileKind.IMAGE or len(found) >= MAX_IMAGES:
                continue
            try:
                path = self.runs.file(run, item.name)
                data = path.read_bytes() if path.stat().st_size <= MAX_IMAGE else b""
            except (RunError, OSError):
                continue
            mime = image_mime(data) if data else None
            if mime:
                found.append(
                    {"type": "image", "data": base64.b64encode(data).decode(), "mimeType": mime}
                )
        return found


ARGUMENT = "is not one of its arguments"  # an extra argument, in a refusal


class _StatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    wait_seconds: float = PydField(default=0, ge=0, le=MAX_WAIT)


def tool_name(workflow: str) -> str:
    """`workflow_<name>`, cut to MCP's 64 characters with a hash of the whole name so two
    long names never meet."""
    full = WORKFLOW_PREFIX + workflow
    if len(full) <= 64:
        return full
    return f"{full[:55]}_{hashlib.sha256(workflow.encode()).hexdigest()[:8]}"


def _word(value: object) -> str:
    """A tool argument as the form's value (a string, as a posted form has it)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
