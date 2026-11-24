"""What ComfyUI answers a run, typed at the edge: a refused prompt in plain words,
the queued prompt's id, `/history`'s saved files, an upload's stored name, and the websocket's
progress messages (with a quiet tick, so a silent hang is noticed)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from websockets.exceptions import ConnectionClosedOK
from websockets.sync.client import ClientConnection, connect

QUIET = ""  # what the message stream yields after a quiet spell (never a real message)
QUIET_SECONDS = 60.0
Progress = Callable[[str], None]


class ComfyRunError(RuntimeError):
    """A run that did not finish, worded for the user."""


class NodeError(BaseModel):
    type: str = ""
    message: str = ""
    details: str = ""
    extra_info: dict[str, Any] = Field(default_factory=dict)


class NodeErrors(BaseModel):
    errors: list[NodeError] = Field(default_factory=list)
    class_type: str = ""


class Refusal(BaseModel):
    error: NodeError | str | None = None
    node_errors: dict[str, NodeErrors] = Field(default_factory=dict)


class Queued(BaseModel):
    prompt_id: str


class OutputFile(BaseModel):
    filename: str
    subfolder: str = ""
    type: str = "output"


class NodeOutput(BaseModel):
    """One node's saved files in `/history` (VideoHelperSuite's clips are its `gifs`)."""

    images: list[OutputFile] = Field(default_factory=list)
    gifs: list[OutputFile] = Field(default_factory=list)
    videos: list[OutputFile] = Field(default_factory=list)
    audio: list[OutputFile] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    outputs: dict[str, NodeOutput] = Field(default_factory=dict)


HISTORY = TypeAdapter(dict[str, HistoryEntry])


class Uploaded(BaseModel):
    name: str | None = None
    subfolder: str = ""


def refusal_words(body: bytes) -> str:
    """ComfyUI's 400 for a prompt, in plain words: a missing model file names the file."""
    try:
        refusal = Refusal.model_validate_json(body)
    except ValidationError:
        return body.decode("utf-8", "replace")[:400]
    lines = []
    for node, found in refusal.node_errors.items():
        for error in found.errors:
            name = error.extra_info.get("input_name", "")
            value = error.extra_info.get("received_value")
            if error.type == "value_not_in_list" and isinstance(value, str) and "." in value:
                lines.append(
                    f"{found.class_type} {node}: the file {value} is not where ComfyUI looks "
                    f"({name}); `mnm comfyui workflow show` lists the files it needs"
                )
            else:
                lines.append(f"{found.class_type} {node}: {error.message} {error.details}".strip())
    top = refusal.error
    if not lines and isinstance(top, str):
        lines.append(top)
    elif not lines and isinstance(top, NodeError):
        lines.append(f"{top.message} {top.details}".strip())
    return "; ".join(lines) or "ComfyUI refused the prompt."


class Frame(BaseModel):
    """One websocket text message: `{"type": …, "data": …}`. Node packs send events of their
    own (`crystools.monitor`, …) whose data is anything: only `FOLLOWED`'s data is read."""

    type: str
    data: object = None


NodeId = str | int


class EventData(BaseModel):
    """What a followed event says (ComfyUI's `server.py` / `execution.py` send these)."""

    prompt_id: str | None = None
    node: NodeId | None = None
    display_node: NodeId | None = None
    node_type: str | None = None
    node_id: NodeId | None = None
    exception_message: str | None = None
    value: int | float | None = None
    max: int | float | None = None
    nodes: list[NodeId] | None = None  # execution_cached: the nodes its cache served


FOLLOWED = frozenset(
    {
        "executing",
        "execution_cached",
        "execution_success",
        "execution_error",
        "execution_interrupted",
        "progress",
    }
)


def _unreadable(what: str, message: str) -> ComfyRunError:
    return ComfyRunError(
        f"ComfyUI sent a progress message mnm cannot read ({what}): {message[:200]}"
    )


def settled(
    message: str,
    prompt_id: str,
    say: Progress,
    cached: Callable[[list[str]], None] | None = None,
) -> bool:
    """Whether this message ends the run (done); a failure, a stop or a message mnm cannot
    read ends it with a worded `ComfyRunError` (not an AttributeError, which the run would
    record as "Interrupted before it finished"). The nodes
    ComfyUI's cache served go to `cached`: such a run's VRAM reading is low (`room.run_need`)."""
    try:
        frame = Frame.model_validate_json(message)
    except ValidationError:
        raise _unreadable("not a JSON object with a type", message) from None
    if frame.type not in FOLLOWED:
        return False
    try:
        data = EventData.model_validate(frame.data if frame.data is not None else {})
    except ValidationError:
        raise _unreadable(f"its {frame.type} data", message) from None
    if data.prompt_id not in (None, prompt_id):
        return False  # another client's run
    match frame.type:
        case "executing" if data.node is None and data.prompt_id == prompt_id:
            return True
        case "execution_success":
            return True
        case "execution_error":
            what = (data.exception_message or "").strip() or "no message"
            raise ComfyRunError(f"{data.node_type or 'A node'} {data.node_id or ''} failed: {what}")
        case "execution_interrupted":
            raise ComfyRunError("The run was stopped.")
        case "execution_cached" if data.prompt_id == prompt_id and cached is not None:
            cached([str(n) for n in data.nodes or []])
        case "executing":
            say(f"Running node {data.display_node or data.node}")
        case "progress":
            say(f"Node {data.node}: step {data.value} of {data.max}")
    return False


class Messages(AbstractContextManager[Iterator[str | bytes]]):
    def __init__(self, url: str) -> None:
        self.url = url

    def __enter__(self) -> Iterator[str | bytes]:
        # ComfyUI is on this machine: never through a proxy the environment names
        self._connection = connect(self.url, open_timeout=15, max_size=None, proxy=None)
        return _with_quiet_ticks(self._connection)

    def __exit__(self, *exc: object) -> None:
        self._connection.close()


def _with_quiet_ticks(connection: ClientConnection) -> Iterator[str | bytes]:
    """Each message, and `QUIET` after every `QUIET_SECONDS` without one, so a node that
    hangs without a word is noticed (the deadline, and whether ComfyUI still runs it)."""
    while True:
        try:
            yield connection.recv(timeout=QUIET_SECONDS)
        except TimeoutError:
            yield QUIET
        except ConnectionClosedOK:
            return


def open_messages(url: str) -> AbstractContextManager[Iterator[str | bytes]]:
    return Messages(url)
