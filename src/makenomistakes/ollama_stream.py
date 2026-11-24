"""An engine's OpenAI chat answer as Ollama's: a stream of server-sent events as
NDJSON lines, one per piece of text, then the closing line with the counts and timings; an
answer in one piece as one object. `/api/chat` carries `message`, `/api/generate` `response`.

Tool calls come in pieces in an OpenAI stream (the arguments as text, a fragment at a time);
Ollama sends each call whole, its arguments as an object, so they are gathered and sent in one
line before the closing one.

The closing line's counts and durations: llama-server's `timings` when it sends them
(`prompt_eval_*` and `eval_*` exactly as it measured). Otherwise (vLLM) the counts come from
`usage` and the durations are the router's own clock: a stream's `prompt_eval_duration` from
the request to the first piece and its `eval_duration` from there to the end; a whole answer's
`eval_duration` is the whole request and its `prompt_eval_duration` 0 (no first piece to tell
them apart). `total_duration` and `load_duration` are always the router's (the load is its
wait for the model).

The engine's chunks are read through `engine_answers`: a chunk mnm cannot read, an error
chunk, or a stream that ends without its last chunk ends the reply with Ollama's
`{"error": …}` line instead of `done: true`."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Literal

from makenomistakes.engine_answers import (
    ChatAnswer,
    ChatChunk,
    FunctionPart,
    MalformedAnswerError,
    Said,
    Timings,
    Usage,
    parse,
    said_error,
)
from makenomistakes.ollama_api import now

Kind = Literal["chat", "generate"]
MS = 1_000_000  # nanoseconds in a millisecond: Ollama's durations are nanoseconds
CUT = "{model}'s stream ended before its last chunk (`mnm logs` says why)"


class Reply:
    """One reply in Ollama's words. `load_ns`: how long the router waited for the model."""

    def __init__(
        self,
        model: str,
        kind: Kind,
        load_ns: int = 0,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.model = model
        self.kind = kind
        self.load_ns = load_ns
        self._clock = clock
        self._began = clock()
        self._first: int | None = None  # when the first piece came (the router's clock)
        self._pending = b""
        self._calls: dict[int, FunctionPart] = {}
        self._finish: str | None = None
        self._ended = False  # `[DONE]` or a finish_reason came: the stream was whole
        self._usage = Usage()
        self._timings: Timings | None = None
        self.error: str | None = None

    # --- a stream ---------------------------------------------------------------------

    def feed(self, data: bytes) -> list[bytes]:
        """The NDJSON lines for what the engine sent so far (whole events only); nothing
        more once an error line has ended the reply."""
        self._pending += data
        *lines, self._pending = self._pending.split(b"\n")
        out: list[bytes] = []
        for raw in lines:
            line = raw.strip()
            if self.error is not None or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self._ended = True
            elif payload:
                out.extend(self._event(payload))
        return out

    def _event(self, payload: bytes) -> list[bytes]:
        try:
            chunk = parse(ChatChunk, payload, "stream")
        except MalformedAnswerError as exc:
            return self._fail(f"{self.model}: {exc}")
        if chunk.error is not None:
            return self._fail(said_error(chunk.error) or f"{self.model} sent an empty error")
        self._keep_counts(chunk.usage, chunk.timings)
        if not chunk.choices:
            return []
        choice = chunk.choices[0]
        if choice.finish_reason:
            self._finish = choice.finish_reason
            self._ended = True
        said = choice.delta
        for call in said.tool_calls or []:
            self._gather(call.index, call.function)
        if said.content or said.thought or said.tool_calls:
            self._first = self._first if self._first is not None else self._clock()
        if not said.content and not said.thought:
            return []
        return [_line(self._piece(said.content or "", said.thought, done=False))]

    def _fail(self, message: str) -> list[bytes]:
        self.error = message
        return [_line({"error": message})]

    def _keep_counts(self, usage: Usage | None, timings: Timings | None) -> None:
        self._usage = usage or self._usage
        self._timings = timings or self._timings

    def _gather(self, index: int, function: FunctionPart) -> None:
        found = self._calls.setdefault(index, FunctionPart(name="", arguments=""))
        found.name = (found.name or "") + (function.name or "")
        more = function.arguments
        text = more if isinstance(more, str) else json.dumps(more) if more else ""
        found.arguments = str(found.arguments or "") + text

    def finish(self) -> list[bytes]:
        """The last lines: the tool calls gathered, if any, then the closing line; or the
        error line when the stream ended without its last chunk."""
        if self.error is not None:
            return []  # the error line ended it, as Ollama ends a failed stream
        if not self._ended:
            return self._fail(CUT.format(model=self.model))
        out = []
        if self._calls:
            calls = [_call(c) for _, c in sorted(self._calls.items())]
            out.append(_line(self._piece("", "", done=False, calls=calls)))
        out.append(_line(self._closing("", "")))
        return out

    # --- one piece ---------------------------------------------------------------------

    def whole(self, raw: bytes) -> dict[str, object]:
        """An answer in one piece (`"stream": false`) as Ollama's one object;
        `MalformedAnswerError` when it cannot be read."""
        answer = parse(ChatAnswer, raw, "answer")
        self._keep_counts(answer.usage, answer.timings)
        choice = answer.choices[0] if answer.choices else None
        self._finish = choice.finish_reason if choice is not None else None
        message = choice.message if choice is not None else Said()
        self._first = self._began  # no pieces: the whole request counts as generation
        calls = [_call(c.function) for c in message.tool_calls or []]
        return self._closing(message.content or "", message.thought, calls)

    # --- the shapes ------------------------------------------------------------------------

    def _piece(
        self, text: str, thought: str, done: bool, calls: list[dict[str, object]] | None = None
    ) -> dict[str, object]:
        found: dict[str, object] = {"model": self.model, "created_at": now()}
        if self.kind == "chat":
            message: dict[str, object] = {"role": "assistant", "content": text}
            if thought:
                message["thinking"] = thought
            if calls:
                message["tool_calls"] = calls
            found["message"] = message
        else:
            found["response"] = text
            if thought:
                found["thinking"] = thought
        found["done"] = done
        return found

    def _closing(
        self, text: str, thought: str, calls: list[dict[str, object]] | None = None
    ) -> dict[str, object]:
        found = self._piece(text, thought, done=True, calls=calls)
        found["done_reason"] = "length" if self._finish == "length" else "stop"
        end = self._clock()
        first = self._first if self._first is not None else end
        timings = self._timings
        if timings is not None and timings.predicted_ms is not None:  # llama-server's own
            prompt_ns = int((timings.prompt_ms or 0) * MS)
            eval_ns = int(timings.predicted_ms * MS)
        else:  # the router's clock: request → first piece → end
            prompt_ns, eval_ns = first - self._began, end - first
        found.update(
            {
                "total_duration": end - self._began + self.load_ns,
                "load_duration": self.load_ns,
                "prompt_eval_count": (timings and timings.prompt_n) or self._usage.prompt_tokens,
                "prompt_eval_duration": prompt_ns,
                "eval_count": (timings and timings.predicted_n) or self._usage.completion_tokens,
                "eval_duration": eval_ns,
            }
        )
        return found


def _call(function: FunctionPart) -> dict[str, object]:
    """An OpenAI call (its arguments JSON text) as Ollama's (its arguments an object; text
    that is not JSON stays as it came)."""
    arguments = function.arguments or "{}"
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        parsed = arguments
    return {"function": {"name": function.name or "", "arguments": parsed}}


def _line(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode() + b"\n"


def load_only(model: str, kind: Kind, reason: str, load_ns: int) -> dict[str, object]:
    """Ollama's answer to a request with nothing to say: the model loaded, or unloaded."""
    found: dict[str, object] = {"model": model, "created_at": now()}
    if kind == "chat":
        found["message"] = {"role": "assistant", "content": ""}
    else:
        found["response"] = ""
    found.update({"done": True, "done_reason": reason})
    if reason == "load":
        found.update({"total_duration": load_ns, "load_duration": load_ns})
    return found
