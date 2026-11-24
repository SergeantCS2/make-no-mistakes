"""The Model Context Protocol, as much of it as a tool server needs: JSON-RPC 2.0 over
streamable HTTP, every answer plain JSON (no event stream, no sessions), the methods
`initialize`, `ping`, `tools/list` and `tools/call`, notifications taken with a 202.

A tool's own failure is a result with `isError` (the model reads why); a malformed request
is a JSON-RPC error. Pure: the router's handler passes the body in and writes what comes out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from makenomistakes import __version__

PATH = "/mcp"
VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first: what we answer with
PARSE_ERROR, INVALID_REQUEST, NO_METHOD, INVALID_PARAMS = -32700, -32600, -32601, -32602
ACCEPTED = 202  # a notification or a response sent to us: nothing to answer


@dataclass(frozen=True)
class ToolSpec:
    name: str  # ^[A-Za-z0-9_-]{1,64}$
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False

    @classmethod
    def text(cls, words: str, is_error: bool = False) -> ToolResult:
        return cls([{"type": "text", "text": words}], is_error)

    def wire(self) -> dict[str, Any]:
        return {"content": self.content, "isError": self.is_error}


class ToolError(Exception):
    """A tool refused its call: worded for the model, sent back as `isError`."""


class UnknownToolError(ToolError):
    """No tool by that name: a JSON-RPC error, not a tool's result."""


class Toolbox(Protocol):
    def tools(self) -> list[ToolSpec]: ...

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


class _Request(BaseModel):
    jsonrpc: Literal["2.0"]
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class _CallParams(BaseModel):
    name: str
    arguments: dict[str, Any] | None = None


def _error(ident: int | str | None, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def _result(ident: int | str | None, value: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "result": value}


def handle(box: Toolbox, body: bytes) -> tuple[int, dict[str, Any] | None]:
    """One POST's body → (HTTP status, the JSON to answer, or None for an empty 202)."""
    try:
        raw = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 400, _error(None, PARSE_ERROR, "the body is not JSON")
    if isinstance(raw, dict) and "method" not in raw and ("result" in raw or "error" in raw):
        return ACCEPTED, None  # a client's answer to a request we never send
    try:
        request = _Request.model_validate(raw)
    except ValidationError:
        return 400, _error(None, INVALID_REQUEST, "not a single JSON-RPC 2.0 request")
    if request.id is None:
        return ACCEPTED, None  # notifications/initialized, cancelled, …
    return 200, _answer(box, request)


def _answer(box: Toolbox, request: _Request) -> dict[str, Any]:  # noqa: PLR0911 - per method
    params = request.params or {}
    match request.method:
        case "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in VERSIONS else VERSIONS[0]
            return _result(
                request.id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "makenomistakes", "version": __version__},
                    "instructions": (
                        "mnm runs ComfyUI workflows and pipelines on this machine's GPU. A run "
                        "takes minutes: start it with its tool, then call run_status with the "
                        "run id (wait_seconds lets you wait for it)."
                    ),
                },
            )
        case "ping":
            return _result(request.id, {})
        case "tools/list":
            tools = [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in box.tools()
            ]
            return _result(request.id, {"tools": tools})
        case "tools/call":
            try:
                call = _CallParams.model_validate(params)
            except ValidationError:
                return _error(request.id, INVALID_PARAMS, "tools/call needs a tool `name`")
            try:
                found = box.call(call.name, call.arguments or {})
            except UnknownToolError:
                return _error(request.id, INVALID_PARAMS, f"no tool named {call.name!r}")
            except ToolError as exc:
                found = ToolResult.text(str(exc), is_error=True)
            return _result(request.id, found.wire())
    return _error(request.id, NO_METHOD, f"mnm's MCP server has no method {request.method!r}")
