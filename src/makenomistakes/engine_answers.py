"""An engine's OpenAI answers as the router's Ollama side reads them: a chat
stream's chunks, a whole chat answer, an embeddings answer, an error answer. Parsed once at
the edge into loose models (unknown fields ignored); an answer that does not fit is a
`MalformedAnswerError` saying where, which the caller words as a 502 (or a stream's last
line), never a 500 or a stream cut short. llama-server and vLLM both answer in these shapes;
`timings` is llama-server's own."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class MalformedAnswerError(ValueError):
    """An engine's answer mnm could not read; the message says where and why."""


class Usage(_Loose):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Timings(_Loose):
    """llama-server's own counts and times (`timings` on its last chunk or whole answer)."""

    prompt_n: int | None = None
    prompt_ms: float | None = None
    predicted_n: int | None = None
    predicted_ms: float | None = None


class FunctionPart(_Loose):
    name: str | None = None
    arguments: str | dict[str, Any] | None = None  # a stream's are pieces of JSON text


class ToolCallPart(_Loose):
    index: int = 0
    id: str | None = None  # what a tool's result names (`tool_call_id`); a stream's first piece
    function: FunctionPart = FunctionPart()


class Said(_Loose):
    """What a delta or a message carries: text, thinking (under any of its names), calls."""

    content: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallPart] | None = None

    @property
    def thought(self) -> str:
        return self.reasoning_content or self.reasoning or ""


class StreamChoice(_Loose):
    delta: Said = Said()
    finish_reason: str | None = None


class EngineError(_Loose):
    message: str = ""


class ChatChunk(_Loose):
    """One `data:` event of a chat stream."""

    choices: list[StreamChoice] = []
    usage: Usage | None = None
    timings: Timings | None = None
    error: EngineError | str | None = None


class AnswerChoice(_Loose):
    message: Said = Said()
    finish_reason: str | None = None


class ChatAnswer(_Loose):
    choices: list[AnswerChoice] = []
    usage: Usage | None = None
    timings: Timings | None = None


class Embedding(_Loose):
    embedding: list[float]
    index: int = 0


class EmbeddingAnswer(_Loose):
    data: list[Embedding]
    usage: Usage | None = None


class ErrorAnswer(_Loose):
    error: EngineError | str | None = None


def said_error(error: EngineError | str | None) -> str:
    return error.message if isinstance(error, EngineError) else (error or "")


def parse[T: BaseModel](shape: type[T], raw: bytes | str, what: str) -> T:
    """`raw` as `shape`, or a `MalformedAnswerError`: "the engine's <what> could not be read
    (choices.0.delta: Input should be an object)"."""
    try:
        return shape.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        said = f"{where}: {first['msg']}" if where else first["msg"]
        raise MalformedAnswerError(f"the engine's {what} could not be read ({said})") from None
