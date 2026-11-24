"""`mnm try --tools`: does this instance's chat template give an agent the tool
calls it needs? Five requests through the router, the way an agent sends them:

1. **one call**: the weather tool offered, the model must answer with `tool_calls`;
2. **the tool's result**: its call and the tool's answer sent back, the reply must use it;
3. **two calls at once**: two tools, two questions, `parallel_tool_calls`; a model that
   calls one tool per turn is "not supported" (an agent then waits a turn per call), never
   a failure;
4. **streamed**: step 1 streamed, the calls assembled from their pieces (`tool_replies`);
5. **with thinking on**: step 1 with `enable_thinking: true`; `reasoning_content` beside the
   call is what an agent keeps (llama-server puts earlier turns' reasoning back into the
   prompt when the client sends it: `--reasoning-preserve`, on by default). A model that
   does not think says so, never a failure.

Steps 1 to 4 turn thinking off (`chat_template_kwargs: {enable_thinking: false}`, as the
chat card's 16-token check does); templates without the switch ignore it. Each step is a
worded verdict; the whole is *works* (every step passed, or what it lacks is not needed),
*partial* (a call works but a step failed) or *no* (no step got a call an agent could
run). An engine's refusal (HTTP 400 or 500, vLLM without a tool parser) fails the step it
answered; the router's own refusals (a stopped instance, the key) end the test as any
`mnm try` ends.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx

from makenomistakes.consumers import own_origin
from makenomistakes.engine_answers import ChatAnswer, MalformedAnswerError, parse
from makenomistakes.loading_notes import QUIET_HEADER
from makenomistakes.recipes import Recipe
from makenomistakes.smoke import TryError, TryResult, answered, unanswered
from makenomistakes.smoke_replies import cut, refusal_text
from makenomistakes.tool_replies import (
    RESULT_MARKS,
    TIME_TOOL,
    WEATHER,
    WEATHER_RESULT,
    WEATHER_TOOL,
    Call,
    Turn,
    assemble,
    data_lines,
    turn_of,
)
from makenomistakes.try_figures import Figure

CALL_TOKENS = 512  # a call and a line of text, thinking off
THINK_TOKENS = 2048  # room to think before the call (a small Qwen3 thinks ~300-800 tokens)
ENGINE_REFUSALS = frozenset({httpx.codes.BAD_REQUEST, httpx.codes.INTERNAL_SERVER_ERROR})
ASK_WEATHER = "What is the weather in Paris right now? Use the tool."
ASK_BOTH = (
    "I need two things at once: the weather in Paris and the local time in Tokyo. Call both "
    "tools in this one turn."
)


class Step(StrEnum):
    ONE = "one call"
    RESULT = "the tool's result"
    PARALLEL = "two calls at once"
    STREAMED = "streamed"
    THINKING = "with thinking on"


class Outcome(StrEnum):
    PASSED = "ok"
    FAILED = "failed"
    NOT_SUPPORTED = "not supported"  # parallel calls one at a time; a model that never thinks


class Verdict(StrEnum):
    WORKS = "works"
    PARTIAL = "partial"
    NO = "no"


@dataclass(frozen=True)
class StepResult:
    step: Step
    outcome: Outcome
    words: str
    calls: list[Call] = field(default_factory=list)

    @property
    def line(self) -> str:
        return f"{self.step.value}: {self.outcome.value}, {self.words}"


@dataclass(frozen=True)
class ToolCheck:
    steps: list[StepResult]
    seconds: float

    @property
    def calls(self) -> list[Call]:
        return [call for step in self.steps for call in step.calls]

    @property
    def verdict(self) -> Verdict:
        if not any(call.accepted for call in self.calls):
            return Verdict.NO
        if any(step.outcome is Outcome.FAILED for step in self.steps):
            return Verdict.PARTIAL
        return Verdict.WORKS

    @property
    def passed(self) -> int:
        return sum(step.outcome is Outcome.PASSED for step in self.steps)

    @property
    def summary(self) -> str:
        """The one line `mnm try --tools` and Test it lead with."""
        return summary_of(self.verdict, self.detail)

    @property
    def short(self) -> str:
        """`mnm try --all`'s Tools column: `works 5/5`."""
        return f"{self.verdict.value} {self.passed}/{len(self.steps)}"

    @property
    def detail(self) -> str:
        """The steps counted, and those that failed or are not supported."""
        said = f"{self.passed} of {len(self.steps)} steps passed"
        for outcome, word in ((Outcome.FAILED, "failed"), (Outcome.NOT_SUPPORTED, "not supported")):
            if names := [s.step.value for s in self.steps if s.outcome is outcome]:
                said += f"; {word}: {', '.join(names)}"
        return said

    def result(self) -> TryResult:
        calls = self.calls
        numbers = {
            Figure.TOOL_CALLS: float(len(calls)),
            Figure.TOOL_CALLS_ACCEPTED: float(sum(c.accepted for c in calls)),
            Figure.TOOL_STEPS_PASSED: float(self.passed),
            Figure.TOOL_STEPS: float(len(self.steps)),
        }
        return TryResult(self.summary, self.seconds, numbers)


def summary_of(verdict: Verdict, detail: str) -> str:
    """`tools works: 5 of 5 steps passed`: a check's, and a saved verdict's."""
    return f"tools {verdict.value}: {detail}"


Ask = Callable[[dict[str, Any]], Turn | str]  # the model's turn, or the engine's refusal


def check_tools(
    recipe: Recipe, http: httpx.Client, clock: Callable[[], float] = time.monotonic
) -> ToolCheck:
    """The five steps against the instance, through the router. Raises `TryError` or
    `NoAnswerError` when the router refuses or nothing answers."""
    root = own_origin(recipe)
    url = root + "/v1/chat/completions"
    began = clock()

    def ask(body: dict[str, Any]) -> Turn | str:
        response = _send(recipe, root, lambda: http.post(url, json=_body(recipe, body)))
        if (refused := _refusal(recipe, response)) is not None:
            return refused
        return _read(recipe, root, lambda: turn_of(parse(ChatAnswer, response.content, "answer")))

    def stream(body: dict[str, Any]) -> Turn | str:
        whole = {**_body(recipe, body), "stream": True}
        headers = {QUIET_HEADER: "1"}  # no loading notes in the pieces read here
        with _streaming(recipe, root, http.stream("POST", url, json=whole, headers=headers)) as got:
            if got.status_code != httpx.codes.OK:
                got.read()
                if (refused := _refusal(recipe, got)) is not None:
                    return refused
            turn = _read(recipe, root, lambda: assemble(data_lines(got.iter_lines())))
        return turn.error or turn  # a stream the engine ended with its error line

    one = _call(Step.ONE, ask(_thinking_off(WEATHER_TURN)))
    steps = [
        one,
        _round_trip(ask, one.calls),
        _parallel(ask(_thinking_off(BOTH_TURN))),
        _streamed(stream(_thinking_off(WEATHER_TURN))),
        _thinking(ask({**WEATHER_TURN, "max_tokens": THINK_TOKENS, **_THINKING_ON})),
    ]
    return ToolCheck(steps, clock() - began)


WEATHER_TURN: dict[str, Any] = {
    "messages": [{"role": "user", "content": ASK_WEATHER}],
    "tools": [WEATHER_TOOL],
}
BOTH_TURN: dict[str, Any] = {
    "messages": [{"role": "user", "content": ASK_BOTH}],
    "tools": [WEATHER_TOOL, TIME_TOOL],
    "parallel_tool_calls": True,
}
_THINKING_ON = {"chat_template_kwargs": {"enable_thinking": True}}


def _body(recipe: Recipe, body: dict[str, Any]) -> dict[str, Any]:
    return {"model": recipe.served_name, "temperature": 0, "max_tokens": CALL_TOKENS, **body}


def _thinking_off(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "chat_template_kwargs": {"enable_thinking": False}}


def _call(step: Step, turn: Turn | str) -> StepResult:
    """A step whose turn must hold a call an agent can run."""
    if isinstance(turn, str):
        return StepResult(step, Outcome.FAILED, f"the engine refused it: {turn}")
    if not turn.calls:
        return StepResult(step, Outcome.FAILED, _no_call(turn))
    if not turn.accepted:
        return StepResult(step, Outcome.FAILED, _unrunnable(turn.calls[0]), turn.calls)
    return StepResult(step, Outcome.PASSED, f"called {turn.accepted[0].shown}", turn.calls)


def _round_trip(ask: Ask, earlier: list[Call]) -> StepResult:
    """The model's own call from step 1 (or one like it), the tool's answer, then its words."""
    call = next((c for c in earlier if c.accepted), None) or Call(
        "call_mnm_0", WEATHER, {"city": "Paris"}, ""
    )
    sent = call.as_message()
    messages = [
        {"role": "user", "content": ASK_WEATHER},
        {"role": "assistant", "content": "", "tool_calls": [sent]},
        {"role": "tool", "tool_call_id": sent["id"], "content": json.dumps(WEATHER_RESULT)},
    ]
    turn = ask(_thinking_off({"messages": messages, "tools": [WEATHER_TOOL]}))
    if isinstance(turn, str):
        return StepResult(Step.RESULT, Outcome.FAILED, f"the engine refused it: {turn}")
    if any(mark in turn.content.lower() for mark in RESULT_MARKS):
        said = f'the answer uses it: "{cut(turn.content, 80)}"'
        return StepResult(Step.RESULT, Outcome.PASSED, said)
    if turn.calls:
        again = f"it called {turn.calls[0].shown} again instead of answering with the result"
        return StepResult(Step.RESULT, Outcome.FAILED, again, turn.calls)
    if turn.content:
        said = f'it answered without the result: "{cut(turn.content, 80)}"'
        return StepResult(Step.RESULT, Outcome.FAILED, said)
    return StepResult(Step.RESULT, Outcome.FAILED, _nothing(turn.finish, "answered"))


def _parallel(turn: Turn | str) -> StepResult:
    judged = _call(Step.PARALLEL, turn)
    if isinstance(turn, str) or judged.outcome is not Outcome.PASSED:
        return judged
    if len(turn.accepted) >= 2:
        shown = ", ".join(c.shown for c in turn.accepted)
        return StepResult(Step.PARALLEL, Outcome.PASSED, f"called {shown}", turn.calls)
    return StepResult(
        Step.PARALLEL,
        Outcome.NOT_SUPPORTED,
        f"it called {turn.accepted[0].shown} alone: one call per turn, so an agent waits a "
        "turn for each (slower, not broken)",
        turn.calls,
    )


def _streamed(turn: Turn | str) -> StepResult:
    judged = _call(Step.STREAMED, turn)
    if isinstance(turn, str) or not turn.calls:
        return judged
    first = (turn.accepted or turn.calls)[0]
    said = f"{turn.pieces} pieces assembled to {first.shown}"
    if judged.outcome is Outcome.PASSED:
        return StepResult(Step.STREAMED, Outcome.PASSED, said, turn.calls)
    return StepResult(Step.STREAMED, Outcome.FAILED, f"its {said}: not runnable", turn.calls)


def _thinking(turn: Turn | str) -> StepResult:
    judged = _call(Step.THINKING, turn)
    if isinstance(turn, str):
        return judged
    if judged.outcome is not Outcome.PASSED:
        if not turn.thought:
            return judged
        said = f"it thought, then {judged.words}"
        return StepResult(Step.THINKING, Outcome.FAILED, said, judged.calls)
    if not turn.thought:
        return StepResult(
            Step.THINKING,
            Outcome.NOT_SUPPORTED,
            f"{judged.words}; no reasoning_content came back (the model does not think, or "
            "its template has no thinking switch): nothing to keep",
            turn.calls,
        )
    return StepResult(
        Step.THINKING,
        Outcome.PASSED,
        f"{judged.words}, with {len(turn.thought):,} characters of reasoning_content beside "
        "it (an agent sends it back with its next turn)",
        turn.calls,
    )


def _unrunnable(call: Call) -> str:
    return f"called {call.shown}: not a call an agent can run"


def _no_call(turn: Turn) -> str:
    if turn.content and turn.finish != "length":
        return f'it answered in text instead of calling: "{cut(turn.content, 80)}"'
    return _nothing(turn.finish, "called")


def _nothing(finish: str | None, did: str) -> str:
    if finish == "length":
        return f"it ran out of tokens before it {did} (finish_reason length)"
    return "it answered with neither a call nor text"


def _refusal(recipe: Recipe, response: httpx.Response) -> str | None:
    """The engine's refusal of a step as words, None for a 200; the router's own refusals
    (and a missing engine) raise as any `mnm try` does."""
    if response.status_code == httpx.codes.OK:
        return None
    if response.status_code in ENGINE_REFUSALS:
        return f"HTTP {response.status_code}: {refusal_text(response.text)}"
    answered(recipe, response)  # raises: the router's 409/503/401…, or 502
    return None


def _send(recipe: Recipe, root: str, request: Callable[[], httpx.Response]) -> httpx.Response:
    try:
        return request()
    except httpx.TransportError as exc:
        raise unanswered(recipe, exc, root) from exc


@contextmanager
def _streaming(
    recipe: Recipe, root: str, opened: AbstractContextManager[httpx.Response]
) -> Iterator[httpx.Response]:
    """httpx's `stream(...)`, a transport error worded like `mnm try`'s."""
    try:
        with opened as response:
            yield response
    except httpx.TransportError as exc:
        raise unanswered(recipe, exc, root) from exc


def _read[T](recipe: Recipe, root: str, read: Callable[[], T]) -> T:
    try:
        return read()
    except MalformedAnswerError as exc:
        raise TryError(f"{recipe.name}: {exc}.") from exc
    except httpx.TransportError as exc:
        raise unanswered(recipe, exc, root) from exc
