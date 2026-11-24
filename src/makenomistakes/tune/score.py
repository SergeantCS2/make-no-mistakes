"""A trial's score: what "faster" means for this instance.

- `default` (chat models): the median decode speed x the prompt speed at 2k tokens ^ 0.25:
  decode is what a reader waits on, a slow prompt still costs every turn.
- `chat` (`--for chat`): decode only.
- `agents` (`--for agents`; a coding instance's default): decode x the prompt speed at 16k
  tokens ^ 0.5: an agent's first request is 7,000 to 19,000 tokens (GUIDE §11), and it waits
  for all of them before the first token.

A score is only compared with another of the same mode on the same instance.

`MNM_TUNE_SPREAD` (a fraction, e.g. 0.9) replaces `SPREAD_LIMIT` for the trial's spread rule
only. It is for processor builds on shared vCPUs, which spread a decode reading 60-90% from
one request to the next; a box keeps the rule."""

from __future__ import annotations

import os
from statistics import median

from makenomistakes.hub.facts import Modality
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.tuned import Score, ScoreMode

PROMPT_WEIGHT = {"default": 0.25, "chat": 0.0, "agents": 0.5}
# A trial whose three decode readings spread more than 10% is rejected, since something else
# used the card; an RTX 4070's own readings of one setting differ by a few percent.
SPREAD_LIMIT = 0.10
SPREAD_SEAM = "MNM_TUNE_SPREAD"
DECODE_PROMPTS_IN_RAM = 5  # decode readings a trial takes with experts in RAM (`tune.trial`)


def spread_limit() -> float:
    """`SPREAD_LIMIT`, or `SPREAD_SEAM`'s fraction when set (and a number above 0)."""
    raw = os.environ.get(SPREAD_SEAM, "").strip()
    try:
        value = float(raw) if raw else SPREAD_LIMIT
    except ValueError:
        return SPREAD_LIMIT
    return value if value > 0 else SPREAD_LIMIT


def default_mode(recipe: Recipe) -> ScoreMode:
    """A coding instance is an agent's: its long first requests count; any other, chat's."""
    return "agents" if recipe.modality is Modality.CODING else "default"


def needs_long_prompt(mode: ScoreMode) -> bool:
    """Only `agents` reads the 16k prompt: on a slow box it costs minutes a trial."""
    return mode == "agents"


def judged_spread(readings: list[float]) -> float:
    """The spread the trial's rule judges: of every reading for three, of the middle three
    for five or more (`tune.trial`: a model with experts in RAM, where one slow or fast
    outlier of the page cache must never sink a trial)."""
    middle = sorted(readings)[1:-1] if len(readings) >= DECODE_PROMPTS_IN_RAM else readings
    return spread(middle)


def spread(readings: list[float]) -> float:
    """(highest - lowest) / median; 0 for fewer than two readings."""
    if len(readings) < 2:
        return 0.0
    middle = median(readings)
    return (max(readings) - min(readings)) / middle if middle > 0 else 0.0


def scored(
    mode: ScoreMode, decode: list[float], prompt_2k: float | None, prompt_16k: float | None
) -> Score:
    """The score of three decode readings and the prompt speeds; a prompt speed the mode
    needs and the trial lacks scores 0 (never better than a measured one)."""
    rate = median(decode) if decode else 0.0
    weight = PROMPT_WEIGHT[mode]
    prompt = prompt_16k if mode == "agents" else prompt_2k
    if weight == 0:
        value = rate
    elif prompt is None or prompt <= 0:
        value = 0.0
    else:
        value = rate * prompt**weight
    return Score(
        mode=mode,
        value=round(value, 4),
        decode=round(rate, 3),
        prompt_2k=round(prompt_2k, 2) if prompt_2k is not None else None,
        prompt_16k=round(prompt_16k, 2) if prompt_16k is not None else None,
    )
