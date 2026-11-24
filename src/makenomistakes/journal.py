"""The router's journal lines (stderr, read by `journalctl --user -u makenomistakes-router` and
collected by `mnm report`): every `/mcp` call, every load and the parks it made, and every
answer the router gives itself that is not a success (409 stopped, 503 no room, 401 key, 403
origin or sandbox, 502 an engine that did not answer). Never one per proxied success. A peer's
repeated refusal is collapsed (`Quieter`).

What a client sent goes through `shown`: `repr` for anything that is not printable, and cut,
so a path or method of "x\\n/mcp POST …" never forges a second line."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

Note = Callable[[str], None]
SHOWN = 64  # a method or path is short; a longer one is cut in the log
REASON_SHOWN = 400  # a refusal's words: the no-room one names every holder


def to_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def shown(text: str, limit: int = SHOWN) -> str:
    """`text` as one safe line: printable as it is, else its `repr`; cut at `limit`."""
    said = text if text.isprintable() else repr(text)
    return said if len(said) <= limit else said[: limit - 1] + "…"


def reason(text: str) -> str:
    """A refusal's own words (mnm's, with line breaks folded) for a journal line."""
    return shown(" ".join(text.split()), REASON_SHOWN)


class Lasting:
    """A failure (or a reason to wait) a loop meets again at every tick, said once while it
    lasts: again only when its words change, or once a run of that look worked. The router's
    ticking loops say theirs through one: the card jobs' tick failures and each job's reason
    to wait, the health sweep's looks, the idle sweep, the tuner's unlistable queue. Said at
    every tick (every 30 s for the idle sweep, every 5 s for the card jobs), the same line would
    bury the journal and the router's 200-line activity log. `what` keeps the looks of one loop
    apart, so two failing looks are each said once instead of taking turns. It keys on the
    words: a failure whose words change at every tick (a counter, a time in them) is said
    every tick; a set of problems said once each is the tuner sweep's own (`_sweep_said`)."""

    def __init__(self, note: Note) -> None:
        self._note = note
        self._said: dict[str, str] = {}

    def failed(self, words: str, what: str = "") -> None:
        if self._said.get(what) != words:
            self._note(words)
        self._said[what] = words

    def worked(self, what: str = "") -> None:
        self._said.pop(what, None)


class Quieter:
    """Repeated refusals from one peer, collapsed: after `limit` identical
    lines (same peer, status and path) within `window` seconds the rest are counted, not
    written, and one line says how many once the window has passed. A client that retries
    in a loop must not bury the loads and refusals the user reads."""

    def __init__(
        self,
        note: Note,
        clock: Callable[[], float] = time.monotonic,
        limit: int = 5,
        window: float = 60.0,
    ) -> None:
        self.note = note
        self.clock = clock
        self.limit = limit
        self.window = window
        self._seen: dict[tuple[str, int, str], _Seen] = {}
        self._lock = threading.Lock()

    def refusal(self, line: str, peer: str, status: int, path: str) -> None:
        """One refusal's line, written unless it is the sixth like it within the minute."""
        self.flush()
        with self._lock:
            key = (peer, status, path)
            seen = self._seen.setdefault(key, _Seen(self.clock()))
            seen.count += 1
            if seen.count > self.limit:
                seen.held += 1
                return
        self.note(line)

    def flush(self) -> None:
        """The count of what a passed window held back, one line per peer and refusal."""
        now = self.clock()
        said = []
        with self._lock:
            for key, seen in list(self._seen.items()):
                if now - seen.started < self.window:
                    continue
                del self._seen[key]
                if seen.held:
                    peer, status, path = key
                    said.append(
                        f"({seen.held} more from {peer} in the last minute: {status} on {path})"
                    )
        for line in said:
            self.note(line)


@dataclass
class _Seen:
    started: float
    count: int = 0
    held: int = 0


def mcp_method(body: bytes) -> str:
    """The JSON-RPC method a body asks for, for the log ("?" when it is not one). A string
    from the client goes through `repr` and is cut: a method of "x\n/mcp POST …" must not
    forge a second journal line."""
    try:
        found = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "?"
    if not isinstance(found, dict):
        return "a batch"
    method = found.get("method")
    if method is None:  # an answer to us, or a body that is neither
        return "a reply" if "result" in found or "error" in found else "a body without a method"
    return shown(method if isinstance(method, str) else repr(method))


def error_message(value: object) -> str:
    """The message of an `error_body` or a JSON-RPC error (`{"error": {"message": …}}`),
    else nothing: what a refusal's journal line says."""
    error = value.get("error") if isinstance(value, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message if isinstance(message, str) else ""
