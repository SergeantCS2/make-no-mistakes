"""The keys of a test's numbers (`smoke.TryResult.numbers`), named once: the request that
measures them (`smoke`), the engine self-test that reads two of them (`engine_selftest`)
and the words for each one (`FIGURE_WORDS`, on the page and in the CLI) all use these
members, so a key cannot reach the page unworded (the raw `prompt_per_second` on a real CPU
build did)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class Figure(StrEnum):
    # llama-server's own timings (`smoke_replies._Timings`, whose field names these are)
    PREDICTED_PER_SECOND = "predicted_per_second"
    PREDICTED_N = "predicted_n"
    PROMPT_PER_SECOND = "prompt_per_second"
    PROMPT_N = "prompt_n"
    DRAFT_N = "draft_n"  # a draft model's tokens for the reply, and those kept
    DRAFT_N_ACCEPTED = "draft_n_accepted"
    DRAFT_ACCEPTED_PERCENT = "draft_accepted_percent"
    # an engine without timings (vLLM): its usage block, and those tokens over the request
    COMPLETION_TOKENS = "completion_tokens"
    REQUEST_TOKENS_PER_SECOND = "request_tokens_per_second"
    DIMS = "dims"  # an embedding's
    FRUIT = "fruit"  # the rerank test's two scores
    CAR = "car"
    WIDTH = "width"  # a picture's
    HEIGHT = "height"
    FRAMES = "frames"  # a clip's
    FPS = "fps"
    AUDIO_SECONDS = "audio_seconds"  # speech's
    REALTIME = "realtime"
    # a ComfyUI run's readings
    VRAM_PEAK_GIB = "vram_peak_gib"
    VRAM_ADDED_GIB = "vram_added_gib"
    RAM_PEAK_GIB = "ram_peak_gib"
    MODEL_FILES_GIB = "model_files_gib"
    # the tool-call test's (`tool_check`): calls made, those an agent can run, steps
    TOOL_CALLS = "tool_calls"
    TOOL_CALLS_ACCEPTED = "tool_calls_accepted"
    TOOL_STEPS_PASSED = "tool_steps_passed"
    TOOL_STEPS = "tool_steps"


# Each figure under its word and format, in this order; every member has one.
FIGURE_WORDS: dict[Figure, tuple[str, str]] = {  # figure -> (label, format spec)
    Figure.PREDICTED_PER_SECOND: ("tok/s", ".1f"),
    Figure.REQUEST_TOKENS_PER_SECOND: ("tok/s, whole request", ".1f"),
    Figure.PREDICTED_N: ("tokens", ".0f"),
    Figure.COMPLETION_TOKENS: ("tokens", ".0f"),
    Figure.PROMPT_PER_SECOND: ("prompt tok/s", ".0f"),
    Figure.PROMPT_N: ("prompt tokens", ".0f"),
    Figure.DRAFT_N: ("drafted tokens", ".0f"),
    Figure.DRAFT_N_ACCEPTED: ("drafted tokens kept", ".0f"),
    Figure.DRAFT_ACCEPTED_PERCENT: ("% of the draft kept", ".0f"),
    Figure.DIMS: ("dimensions", ".0f"),
    Figure.FRUIT: ("fruit score", ".2f"),
    Figure.CAR: ("car score", ".2f"),
    Figure.WIDTH: ("width", ".0f"),
    Figure.HEIGHT: ("height", ".0f"),
    Figure.FRAMES: ("frames", ".0f"),
    Figure.FPS: ("fps", "g"),
    Figure.AUDIO_SECONDS: ("seconds of audio", ".1f"),
    Figure.REALTIME: ("x realtime", ".1f"),
    Figure.VRAM_PEAK_GIB: ("GiB VRAM at the peak", ".1f"),
    Figure.VRAM_ADDED_GIB: ("GiB VRAM the run added", ".1f"),
    Figure.RAM_PEAK_GIB: ("GiB RAM at the peak", ".1f"),
    Figure.MODEL_FILES_GIB: ("GiB of model files", ".1f"),
    Figure.TOOL_STEPS_PASSED: ("tool steps passed", ".0f"),
    Figure.TOOL_STEPS: ("tool steps", ".0f"),
    Figure.TOOL_CALLS: ("tool calls", ".0f"),
    Figure.TOOL_CALLS_ACCEPTED: ("tool calls an agent can run", ".0f"),
}


def try_figures(numbers: Mapping[Figure, float]) -> list[tuple[str, str]]:
    """(value, label) for each figure the test reported, in `FIGURE_WORDS` order."""
    return [
        (format(numbers[f], spec), label)
        for f, (label, spec) in FIGURE_WORDS.items()
        if f in numbers
    ]


def run_figures(numbers: Mapping[str, float]) -> list[tuple[str, str]]:
    """A run record's numbers (saved by their figure's name), worded like a test's."""
    known = {f.value: f for f in Figure}
    return try_figures({known[k]: v for k, v in numbers.items() if k in known})
