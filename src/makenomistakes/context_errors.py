"""llama-server's "request exceeds the available context size" told in mnm's words: the
router adds which instance it was, the context one request gets, what the request needed,
and the two ways out. llama-server c7fc509 answers it the same on `/v1/messages` and
`/v1/chat/completions`, a 400 before anything streams:

    {"error": {"code": 400, "message": "request (36009 tokens) exceeds the available context
    size (32768 tokens), try increasing it", "type": "exceed_context_size_error",
    "n_prompt_tokens": 36009, "n_ctx": 32768}}

Claude Code, given the card's `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, refuses such a prompt itself
("Prompt is too long") and sends nothing; a client that does not know the window gets this."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.recipes.model import Recipe

KIND = "exceed_context_size_error"
STEP = 8192  # a suggested context rounds up to this


class _Exceeded(BaseModel):
    message: str
    type: str
    n_prompt_tokens: int | None = None
    n_ctx: int | None = None


class _Answer(BaseModel):
    error: _Exceeded


def explained(
    status: int, data: bytes, recipe: Recipe, trained: Callable[[], int | None] = lambda: None
) -> bytes:
    """The engine's answer with one sentence more when it is the context error; any other
    answer as it came (nothing is guessed from a body that is not that error). `trained`:
    the model's trained context, asked only for that error; a suggested `ctx=` never goes
    above what `mnm set` takes (trained x slots)."""
    if status != 400 or KIND.encode() not in data:
        return data
    try:
        raw = json.loads(data)
        found = _Answer.model_validate(raw).error
    except (ValueError, ValidationError):
        return data
    if found.type != KIND:
        return data
    raw["error"]["message"] = f"{found.message.rstrip('. ')}. {_why(found, recipe, trained())}"
    return json.dumps(raw).encode()


def _why(found: _Exceeded, recipe: Recipe, trained: int | None) -> str:
    name = recipe.name
    parallel = max(recipe.args.parallel if isinstance(recipe.args, LlamaCppArgs) else 1, 1)
    if found.n_ctx and found.n_prompt_tokens:
        suggested = math.ceil((found.n_prompt_tokens + STEP // 2) / STEP) * STEP * parallel
        if trained is not None and suggested > trained * parallel:
            return _at_most(found, name, trained, parallel)
        return (
            f"{name}'s per-request context is {found.n_ctx:,} tokens; this request needed "
            f"{found.n_prompt_tokens:,}. `mnm set {name} ctx={suggested}` (at its next start, "
            "if the card has the room) or a smaller request: a coding agent's first one holds "
            "its project's CLAUDE.md or AGENTS.md and its memory whole."
        )
    return (
        f"The request is longer than the context {name} gives one request (`mnm show {name}`). "
        f"`mnm set {name} ctx=…` or a smaller request: a coding agent's first one holds its "
        "project's CLAUDE.md or AGENTS.md and its memory whole."
    )


def _at_most(found: _Exceeded, name: str, trained: int, parallel: int) -> str:
    """The request needs more than the model was trained for: the largest `ctx=` that
    `mnm set` takes, when it is above the context now, else only a smaller request."""
    most = trained * parallel
    needed = f"this request needed {found.n_prompt_tokens:,}"
    head = f"{name}'s per-request context is {found.n_ctx:,} tokens; {needed}"
    tail = (
        "a coding agent's first one holds its project's CLAUDE.md or AGENTS.md and its "
        "memory whole."
    )
    if found.n_ctx is not None and trained > found.n_ctx:
        return (
            f"{head}, more than the {trained:,} it was trained for. `mnm set {name} ctx={most}` "
            f"(the most it takes, at its next start, if the card has the room) or a smaller "
            f"request: {tail}"
        )
    return f"{head}, more than the {trained:,} it was trained for: a smaller request; {tail}"
