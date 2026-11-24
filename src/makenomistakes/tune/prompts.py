"""Fresh prompts for a trial: every request gets text no earlier request had.

A repeated prompt overstates a mixture-of-experts model's decode speed about 4x (the experts
it routes to are already hot in the card's cache and the CPU's), and llama-server reuses a
prompt's cached prefix: so each trial's prompts are generated paragraphs, seeded by the
instance, the trial and the request, and differ from their first word. About four characters
make a token of English prose in the tokenizers mnm serves; the trial records the count the
engine reports (`timings.prompt_n`), never this estimate."""

from __future__ import annotations

import random

CHARS_PER_TOKEN = 4  # English prose in Qwen, Llama and Gemma tokenizers, give or take a few
SHORT_TOKENS = 2048  # the trial's three decode prompts (and the prompt speed at 2k)
LONG_TOKENS = 16384  # the long first request an agent sends (the score `--for agents`)

_NOUNS = [
    "river",
    "harbour",
    "engine",
    "lantern",
    "orchard",
    "valley",
    "ledger",
    "compass",
    "meadow",
    "quarry",
    "tower",
    "bridge",
    "garden",
    "market",
    "signal",
    "thread",
    "kettle",
    "window",
    "mirror",
    "chapter",
    "island",
    "forest",
    "beacon",
    "canyon",
    "furnace",
    "glacier",
    "helmet",
    "journal",
    "kitchen",
    "ladder",
    "magnet",
    "needle",
    "office",
    "pillow",
    "rocket",
    "saddle",
    "tunnel",
    "umbrella",
    "violin",
    "wagon",
    "yacht",
    "anchor",
    "basket",
    "candle",
    "desert",
    "feather",
    "harvest",
]
_VERBS = [
    "carries",
    "follows",
    "measures",
    "repairs",
    "gathers",
    "watches",
    "records",
    "builds",
    "crosses",
    "warms",
    "shelters",
    "lifts",
    "paints",
    "counts",
    "weighs",
    "sorts",
    "guards",
    "opens",
    "closes",
    "turns",
    "polishes",
    "folds",
    "traces",
    "names",
]
_ADJECTIVES = [
    "quiet",
    "narrow",
    "golden",
    "heavy",
    "careful",
    "distant",
    "patient",
    "bright",
    "crooked",
    "hollow",
    "early",
    "gentle",
    "silver",
    "ancient",
    "sudden",
    "steady",
    "humble",
    "rapid",
    "frozen",
    "woven",
    "tidy",
    "restless",
    "amber",
    "modest",
]
_LINKS = [
    "after the storm",
    "before dawn",
    "by the old road",
    "in the long winter",
    "near the coast",
    "under a grey sky",
    "along the canal",
    "beside the mill",
    "across the square",
    "at the edge",
]
_ASK = (
    "\n\nContinue this account in the same plain style, adding new events, places and people; "
    "do not summarise it.\n\n"
)


def fresh_prompt(seed: str, tokens: int = SHORT_TOKENS) -> str:
    """About `tokens` tokens of prose nobody asked before, the same for the same `seed`
    (`<instance>/<trial>/<request>`), ending in an instruction to keep writing."""
    rng = random.Random(seed)
    target = tokens * CHARS_PER_TOKEN
    opening = f"Account {rng.randrange(10**9)} of the {rng.choice(_ADJECTIVES)} {seed_word(rng)}."
    parts = [opening]
    size = len(opening)
    while size < target:
        sentence = _sentence(rng)
        parts.append(sentence)
        size += len(sentence) + 1
    return " ".join(parts) + _ASK


def seed_word(rng: random.Random) -> str:
    return rng.choice(_NOUNS)


def _sentence(rng: random.Random) -> str:
    subject = f"The {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}"
    verb = rng.choice(_VERBS)
    thing = f"{rng.choice(('a', 'the', 'every', 'one'))} {rng.choice(_NOUNS)}"
    where = rng.choice(_LINKS)
    number = rng.randrange(2, 9999)
    return f"{subject} {verb} {thing} {where}, number {number}."


def estimated_tokens(text: str) -> int:
    """The estimate the generator sizes by (tests; the trial reads the engine's count)."""
    return len(text) // CHARS_PER_TOKEN
