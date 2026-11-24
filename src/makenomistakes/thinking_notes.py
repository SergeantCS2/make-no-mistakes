"""A chat answer whose every token went to thinking, said once in the router's journal: a
first `curl` to `tiny` with `max_tokens` 32 can come back with an empty `content`, all 32
tokens in `reasoning_content` and `finish_reason` "length". Qwen3 thinks by default; mnm's
cards and `mnm try` send `chat_template_kwargs: {enable_thinking: false}`, a user's own first
request does not.

The router never changes a client's request for it (a client that wants thinking gets it);
it journals one line per model per start (its container's start, by hand or a load the
router makes; `Router.container`), the first time such an answer passes, and hands the
answer on as the engine gave it. Only whole replies
are read (a stream is passed on piece by piece); llama-server's shape, typed here:

    {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "",
    "reasoning_content": "Okay, the user …"}}], "usage": {"completion_tokens": 32, …}}
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, Protocol

from pydantic import BaseModel, ValidationError

from makenomistakes.context_errors import explained
from makenomistakes.instance_types import InstanceError
from makenomistakes.recipes.model import Recipe

if TYPE_CHECKING:
    from makenomistakes.model_meta import MetaCache


class _Message(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None


class _Usage(BaseModel):
    completion_tokens: int | None = None


class _Answer(BaseModel):
    choices: list[_Choice]
    usage: _Usage | None = None


class Thought(NamedTuple):
    """An answer that thought to its token limit and said nothing: how many tokens it spent
    (None when the answer does not count them)."""

    tokens: int | None


def spent_thinking(status: int, data: bytes) -> Thought | None:
    """The answer when every choice is empty, has reasoning, and stopped at the limit; None
    for any other answer (a picture's or an error's is never parsed: the bytes say first)."""
    if status != 200 or b'"reasoning_content"' not in data or b'"length"' not in data:
        return None
    try:
        answer = _Answer.model_validate(json.loads(data))
    except (ValueError, ValidationError):
        return None
    choices = answer.choices
    if not choices or not all(
        c.finish_reason == "length"
        and not (c.message.content or "").strip()
        and (c.message.reasoning_content or "").strip()
        for c in choices
    ):
        return None
    return Thought(answer.usage.completion_tokens if answer.usage else None)


def words(name: str, thought: Thought) -> str:
    spent = f"all {thought.tokens:,} tokens" if thought.tokens else "every token of its reply"
    return (
        f"{name} spent {spent} thinking and answered nothing: raise max_tokens, or send "
        '"chat_template_kwargs": {"enable_thinking": false} (the cards and `mnm try` do). '
        "Said once per model per start."
    )


class ThinkingNotes:
    """The (model, container start) pairs already told about since the router started."""

    def __init__(self) -> None:
        self._told: set[tuple[str, str | None]] = set()
        self._lock = threading.Lock()

    def line(self, name: str, started: str | None, status: int, data: bytes) -> str | None:
        """The journal line for this answer, the first time this start of `name` gives one
        (`started`: Docker's start time, None when unknown); else None."""
        thought = spent_thinking(status, data)
        if thought is None:
            return None
        with self._lock:
            if (name, started) in self._told:
                return None
            self._told.add((name, started))
        return words(name, thought)


class _Router(Protocol):
    thinking: ThinkingNotes
    meta: MetaCache

    def container(self, name: str) -> tuple[str | None, str | None]: ...


def answer_edit(
    recipe: Recipe, router: _Router, note: Callable[[str], None]
) -> Callable[[int, bytes], bytes]:
    """What the router does to an engine's whole reply before the client gets it: the context
    error in mnm's words (`context_errors`), and the thinking line journaled (the reply
    itself unchanged)."""

    def edit(status: int, data: bytes) -> bytes:
        data = explained(status, data, recipe, lambda: router.meta.get(recipe).trained_ctx)
        if spent_thinking(status, data) is None:  # nearly every reply: no Docker question
            return data
        try:
            started = router.container(recipe.name)[1]
        except InstanceError:  # its recipe unreadable now: told as one start
            started = None
        if (line := router.thinking.line(recipe.name, started, status, data)) is not None:
            note(line)
        return data

    return edit
