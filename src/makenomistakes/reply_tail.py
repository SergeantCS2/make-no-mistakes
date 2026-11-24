"""What the end of an engine's reply says: the router keeps its last bytes (`forward.REPLY_TAIL`)
and its meter and queue read the figures there (llama.cpp's `timings`, an OpenAI `usage`
block, Anthropic's, audio.cpp's `timing.rtf`).

A key is read only where it stands bare: inside a JSON string a quote is escaped, so
`"prompt_ms"` with bare quotes is only ever a key, never a model's words about one. The last
one wins (a stream ends with the final answer's). A number that is not finite or is
negative is not read: every figure read here is a count, a time or a rate (`1e999` is a
float's infinity)."""

from __future__ import annotations

import math
import re
from functools import cache

_NUMBER = rb"(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)"


@cache
def _pattern(key: str) -> re.Pattern[bytes]:
    return re.compile(rb'"' + re.escape(key.encode()) + rb'"\s*:\s*' + _NUMBER)


def last_number(tail: bytes, key: str) -> float | None:
    """The last finite, non-negative number given for `key` in `tail`; None when there is
    none."""
    for found in reversed(_pattern(key).findall(tail)):
        try:
            value = float(found)
        except ValueError:  # `1.2.3`: the grammar's dots are loose
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return None
