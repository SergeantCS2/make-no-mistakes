"""What `mnm try --tools` offers and reads back: the two tools, a call checked against
them, and a stream's tool-call pieces assembled into calls.

The engine's answers are read through `engine_answers` (the router's own typed shapes), so a
reply mnm cannot read is a `MalformedAnswerError` saying where. A call is *accepted* when it
names a tool that was offered and its arguments are a JSON object carrying that tool's
required keys, each a non-empty string: what an agent needs to run it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from makenomistakes.engine_answers import ChatAnswer, ChatChunk, FunctionPart, parse, said_error

WEATHER = "get_weather"
TIME = "get_time"


def _tool(name: str, what: str, unit: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {"city": {"type": "string", "description": "A city name"}}
    if unit:
        properties["unit"] = {"type": "string", "enum": ["celsius", "fahrenheit"]}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": what,
            "parameters": {"type": "object", "properties": properties, "required": ["city"]},
        },
    }


WEATHER_TOOL = _tool(WEATHER, "The current weather in a city.", unit=True)
TIME_TOOL = _tool(TIME, "The current local time in a city.")
REQUIRED = {WEATHER: ("city",), TIME: ("city",)}
# What the round trip's tool answers, and what the model's answer must then carry.
WEATHER_RESULT = {"city": "Paris", "temperature_c": 17, "conditions": "light rain"}
RESULT_MARKS = ("17", "rain")  # either one: the question named neither


@dataclass(frozen=True)
class Call:
    """One tool call as an agent would run it."""

    id: str | None
    name: str
    arguments: dict[str, Any] | None  # None: not a JSON object
    raw: str  # the arguments as they came, for the words when they do not parse

    @property
    def accepted(self) -> bool:
        if self.name not in REQUIRED or self.arguments is None:
            return False
        return all(
            isinstance(self.arguments.get(key), str) and self.arguments[key].strip()
            for key in REQUIRED[self.name]
        )

    @property
    def shown(self) -> str:
        """`get_weather(city="Paris")`, or what came instead."""
        if self.arguments is None:
            return f"{self.name}({self.raw[:80]!r}, not a JSON object)"
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in self.arguments.items())
        return f"{self.name}({args})"

    def as_message(self) -> dict[str, Any]:
        """The call as the assistant's turn carries it back with the tool's result."""
        return {
            "id": self.id or "call_mnm_0",
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments or {})},
        }


def _arguments(raw: str | dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if isinstance(raw, dict):
        return raw, json.dumps(raw)
    text = raw or ""
    try:
        value = json.loads(text) if text.strip() else {}
    except ValueError:
        return None, text
    return (value if isinstance(value, dict) else None), text


def _whole(call_id: str | None, function: FunctionPart) -> Call:
    arguments, raw = _arguments(function.arguments)
    return Call(call_id, function.name or "", arguments, raw)


@dataclass
class _Pieces:
    call_id: str | None = None
    name: str = ""
    text: str = ""


@dataclass(frozen=True)
class Turn:
    """One answer of the model's, whole or assembled from a stream: what a step judges."""

    calls: list[Call]
    content: str
    thought: str  # reasoning_content (or `reasoning`)
    finish: str | None
    pieces: int = 0  # a stream's: the `tool_calls` deltas the calls came in
    error: str | None = None  # the error line a stream ended with, the engine's words

    @property
    def accepted(self) -> list[Call]:
        return [call for call in self.calls if call.accepted]


def turn_of(answer: ChatAnswer) -> Turn:
    """A whole answer's first choice as a turn; an answer without one is empty."""
    if not answer.choices:
        return Turn([], "", "", None)
    choice = answer.choices[0]
    said = choice.message
    calls = [_whole(part.id, part.function) for part in said.tool_calls or []]
    return Turn(calls, (said.content or "").strip(), said.thought, choice.finish_reason)


def data_lines(lines: Iterable[str]) -> Iterator[str]:
    """A server-sent event stream's `data:` payloads, up to `[DONE]`."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield payload


def assemble(payloads: Iterable[str]) -> Turn:
    """Each call's id and name from its first piece, its arguments' text from every piece
    with its index, in order: how OpenAI's stream sends them and agents rebuild them."""
    gathered: dict[int, _Pieces] = {}
    pieces = 0
    content: list[str] = []
    reasoning: list[str] = []
    finish: str | None = None
    error: str | None = None
    for payload in payloads:
        chunk = parse(ChatChunk, payload, "stream chunk")
        if chunk.error is not None:
            error = said_error(chunk.error) or "an error without a message"
        for choice in chunk.choices:
            said = choice.delta
            content.append(said.content or "")
            reasoning.append(said.thought)
            finish = choice.finish_reason or finish
            for part in said.tool_calls or []:
                pieces += 1
                into = gathered.setdefault(part.index, _Pieces())
                into.call_id = into.call_id or part.id
                into.name = into.name or part.function.name or ""
                arguments = part.function.arguments
                into.text += (
                    json.dumps(arguments) if isinstance(arguments, dict) else arguments or ""
                )
    calls = [
        _whole(p.call_id, FunctionPart(name=p.name, arguments=p.text))
        for _, p in sorted(gathered.items())
    ]
    return Turn(calls, "".join(content).strip(), "".join(reasoning), finish, pieces, error)
