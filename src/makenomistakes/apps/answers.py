"""The router's answers as `mnm doctor --app` reads them, typed at the edge: the
model list, MCP's `initialize`; a chat and an embedding are `engine_answers`' own shapes.
Unknown fields are ignored; an answer that does not fit is a `MalformedAnswerError` saying
where, which the doctor words as a FAIL line."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from makenomistakes.engine_answers import ErrorAnswer, MalformedAnswerError, parse, said_error


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ListedModel(_Loose):
    id: str
    context_length: int | None = None


class ModelList(_Loose):
    data: list[ListedModel] = []

    def context(self, model: str) -> int | None:
        return next((m.context_length for m in self.data if m.id == model), None)


class ServerInfo(_Loose):
    name: str = ""


class McpResult(_Loose):
    serverInfo: ServerInfo = ServerInfo()  # noqa: N815 - MCP's own field name


class McpHello(_Loose):
    result: McpResult = McpResult()


def error_text(body: str) -> str:
    """What an error answer says (its `error.message`, or `error`), else the body itself."""
    try:
        said = said_error(parse(ErrorAnswer, body, "error answer").error)
    except MalformedAnswerError:  # not an error answer (or not JSON): the body says it
        said = ""
    return " ".join((said or body).split())
