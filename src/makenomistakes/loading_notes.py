"""A loading note while a parked model loads.

A streamed chat request (`/v1/chat/completions` with `"stream": true`) to a model that is not
ready gets the stream's head once the wait has gone past `GRACE_SECONDS`, then one chunk every
`EVERY_SECONDS` whose `delta.reasoning_content` says what is happening ("Loading qwen3-8b:
12 s, about 5.1 GiB on the card …"), then the engine's own stream, unchanged. The point is the
connection: Odysseus gives a local endpoint's stream a first-event watchdog of at most 60 s
(`src/llm_core.py:214-231` in 0.20.19) and counts a non-empty `reasoning_content` as that first
event (`:3562-3599`), so a load of a minute no longer ends the request; the note shows in its
thinking panel. llama-swap's `sendLoadingState` is the same idea.

Never for a request that is not streamed, one that sends `X-Mnm-Quiet: 1`, or a model that is
ready (it answers inside the grace). A load that fails ends the stream with the router's
refusal as an `error` chunk, then `[DONE]`: Odysseus shows the message (`:3600-3605`); so does
a fault of the router's own once the notes have begun (the head is out: no 500 can follow).

Ids: the notes share one `id` (`chatcmpl-mnm-…`), `created` and `model` (the name the request
used); the engine's stream that follows keeps its own `id`, `created` and `model` (its served
id), as it sent them. The router never rewrites the engine's chunks: a client that keys a
reply on its first chunk's id sees two ids in one stream (Odysseus does not)."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import Message
from enum import Enum
from typing import TYPE_CHECKING

from makenomistakes.disk_settings import redownload_words

if TYPE_CHECKING:
    from makenomistakes.recipes import Recipe

QUIET_HEADER = "X-Mnm-Quiet"
CHAT_COMPLETIONS = "/v1/chat/completions"
GRACE_SECONDS = 1.0  # a loaded model is let through well inside this: no note for it
EVERY_SECONDS = 2.0
SSE_FIELDS = [
    ("Content-Type", "text/event-stream"),
    ("Cache-Control", "no-cache"),
    ("Access-Control-Allow-Origin", "*"),
]
GIB = 1 << 30


def noted_model(method: str, path: str, body: bytes, headers: Message) -> str | None:
    """The `model` a request names when it may get loading notes (a streamed chat request
    without `X-Mnm-Quiet`), else None."""
    if method != "POST" or path.split("?", 1)[0].rstrip("/") != CHAT_COMPLETIONS:
        return None
    if (headers.get(QUIET_HEADER) or "").strip() not in ("", "0"):
        return None
    try:
        request = json.loads(body)
    except ValueError:
        return None
    if not isinstance(request, dict) or request.get("stream") is not True:
        return None
    model = request.get("model")
    return model if isinstance(model, str) else ""


# The notes a stream began with, as a client keeps them in the assistant turn it sends back
# (OpenCode does: "Loading gemma: 1 s, …" comes back in the next request's history)
_NOTES = re.compile(
    r"\A(?:Loading [^\n:]+: \d+ s(?:, about \d+(?:\.\d+)? GiB on the card)?"
    r"(?:, re-downloading [^\n]+ by the disk policy)? …\n)+"
)


def without_notes(body: bytes) -> bytes:
    """A chat request whose assistant turns lose the loading notes their reasoning began
    with: the router's words, never the model's, so they must not reach its prompt (with
    llama.cpp's preserve_thinking an earlier turn's reasoning is in it). Anything else, and a
    body without a note, passes as it came."""
    if b"Loading " not in body:
        return body
    try:
        request = json.loads(body)
    except ValueError:
        return body
    messages = request.get("messages") if isinstance(request, dict) else None
    changed = False
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for key in ("reasoning_content", "reasoning"):
            text = message.get(key)
            if isinstance(text, str) and (left := _NOTES.sub("", text)) != text:
                if left:
                    message[key] = left
                else:  # only the router's words: no reasoning of the model's at all
                    del message[key]
                changed = True
    return json.dumps(request, ensure_ascii=False).encode() if changed else body


def note_text(name: str, seconds: float, vram_bytes: int | None, fetching: str = "") -> str:
    """One note's words; a line of its own in the client's thinking panel. `fetching`: what
    the load downloads first (an instance the disk policy evicted)."""
    size = f", about {vram_bytes / GIB:.1f} GiB on the card" if vram_bytes else ""
    first = f", {fetching}" if fetching else ""
    return f"Loading {name}: {seconds:.0f} s{size}{first} …\n"


def recipe_note(recipe: Recipe, seconds: float) -> str:
    """The note for loading `recipe`: its planned VRAM, and a re-download when it was evicted."""
    fetching = redownload_words(recipe) or ""
    return note_text(recipe.name, seconds, recipe.vram_estimate_bytes, fetching)


@dataclass
class Frames:
    """The stream's frames: every chunk with the same `id`, `created` and `model` (the one
    the request named), the first with the assistant's role."""

    model: str
    created: int = field(default_factory=lambda: int(time.time()))
    ident: str = field(default_factory=lambda: f"chatcmpl-mnm-{uuid.uuid4().hex[:24]}")
    _first: bool = True

    def note(self, text: str) -> bytes:
        delta: dict[str, str] = {"reasoning_content": text}
        if self._first:
            delta = {"role": "assistant", **delta}
            self._first = False
        chunk = {
            "id": self.ident,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        return _event(chunk)

    def error(self, message: str) -> bytes:
        """The last frames of a stream whose model will not answer: the reason, then done."""
        return _event({"error": {"message": message}}) + b"data: [DONE]\n\n"


def _event(value: object) -> bytes:
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


class Opened(Enum):
    NO = "no"  # the work ended inside the grace: nothing was sent
    YES = "yes"  # the head and notes went out: the reply continues the stream
    GONE = "gone"  # the client left while notes went out


def wait_with_notes[T](
    work: Callable[[], T],
    open_stream: Callable[[], None],
    send: Callable[[float], None],
    clock: Callable[[], float] = time.monotonic,
    grace: float | None = None,  # GRACE_SECONDS, read at the call
    every: float | None = None,  # EVERY_SECONDS
) -> tuple[T, Opened]:
    """`work`'s result, with a note `send`(seconds waited) every `every` seconds once it has
    taken longer than `grace` (the stream opened first). A client gone mid-notes stops them;
    the work is still waited for (a load is not cut halfway). `work`'s exception is raised
    here, after the notes."""
    grace = GRACE_SECONDS if grace is None else grace
    every = EVERY_SECONDS if every is None else every
    began = clock()
    outcome: list[T] = []
    failure: list[BaseException] = []
    done = threading.Event()

    def run() -> None:
        try:
            outcome.append(work())
        except BaseException as exc:  # handed to the caller's thread
            failure.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    opened = Opened.NO
    if not done.wait(grace):
        try:
            open_stream()
            opened = Opened.YES
            send(clock() - began)
            while not done.wait(every):
                send(clock() - began)
        except OSError:
            opened = Opened.GONE
            done.wait()
    if failure:
        raise failure[0]
    return outcome[0], opened
