"""What the router remembers of engines that died while loaded: each one's
cause, while the healer restarts it once, and a crash loop's park until the user starts it
again. In the router's memory only: a router that restarts forgets them (the journal keeps
the lines). A request for a dead one is answered 503 with the cause, never a 502 from an
engine that is not there."""

from __future__ import annotations

import threading
from dataclasses import dataclass

RESTARTING = (
    "{name} died: {cause}. The router is restarting it once: ask again in a minute "
    "(`mnm logs {name} --review` says why it died)."
)
LOOPED = (
    "{name} died twice in ten minutes ({cause}): parked; `mnm logs {name} --review`, then "
    "`mnm start {name}`."
)


@dataclass(frozen=True)
class Dead:
    cause: str
    looped: bool = False  # a crash loop's park: never loaded again until the user starts it

    def words(self, name: str) -> str:
        return (LOOPED if self.looped else RESTARTING).format(name=name, cause=self.cause)


class DeadBook:
    """Thread-safe: the healer writes, every request reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dead: dict[str, Dead] = {}

    def mark(self, name: str, dead: Dead) -> None:
        with self._lock:
            self._dead[name] = dead

    def clear(self, name: str) -> None:
        with self._lock:
            self._dead.pop(name, None)

    def get(self, name: str) -> Dead | None:
        with self._lock:
            return self._dead.get(name)

    def names(self) -> list[str]:
        """Every dead instance: restarting, or parked by a crash loop."""
        with self._lock:
            return sorted(self._dead)

    def looped(self) -> list[str]:
        with self._lock:
            return sorted(n for n, d in self._dead.items() if d.looped)

    def refusal(self, name: str) -> str | None:
        """Why a request for it is not served now, or None when it is not dead."""
        dead = self.get(name)
        return dead.words(name) if dead is not None else None
