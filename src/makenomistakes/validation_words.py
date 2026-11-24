"""Pydantic's refusals in words, one way everywhere: a clause per problem, the field it is
about first, never pydantic's own text (it ends each error with a documentation link, which
the MCP tools would answer to a model)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

FORM_PARTS = ("body", "query", "path")  # FastAPI's own place in a form error's location
UNKNOWN = "is not one it takes"  # an extra field, by default


def clause(error: Mapping[str, Any], unknown: str = UNKNOWN, skip: Sequence[str] = ()) -> str:
    """One problem: `<field> is required`, `<field> <unknown>`, or `<field>: <message>`; a
    rule over the whole model (no field) is its message alone."""
    where = ".".join(str(part) for part in error["loc"] if part not in skip)
    kind = error["type"]
    if kind == "missing" and where:
        return f"{where} is required"
    if kind == "extra_forbidden" and where:
        return f"{where} {unknown}"
    message = str(error["msg"]).removeprefix("Value error, ")
    return f"{where}: {message}" if where else message


def worded(
    errors: Iterable[Mapping[str, Any]], unknown: str = UNKNOWN, skip: Sequence[str] = ()
) -> str:
    """Every problem of a `ValidationError.errors()` (or FastAPI's), joined with "; "."""
    return "; ".join(clause(error, unknown, skip) for error in errors)
