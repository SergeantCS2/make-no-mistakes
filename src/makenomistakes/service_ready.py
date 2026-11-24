"""After `mnm service install` restarts the units: wait until the web UI and the router answer,
and say where the router listens (a restart is asynchronous: `ss` right after the install
finds nothing, so MNM_ROUTER_HOST=local would never be seen)."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.config import Settings
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.router_bind import EVERYWHERE, LOOPBACK, BindError, bind_addresses
from makenomistakes.routing import HEALTH, PROBE_HEADER

WAIT_SECONDS = 30.0
EVERY = 0.5
ASK_SECONDS = 2.0

Answers = Callable[[str, dict[str, str], float], bool]  # (url, headers, seconds) -> answered


@dataclass(frozen=True)
class Ready:
    web: bool
    router: dict[str, bool]  # each address the router listens on -> it answered there
    listening: list[str]  # those addresses as the setting gives them (0.0.0.0: every one)
    waited: float  # the bound it waited within, in seconds
    problem: str | None = None  # why the addresses could not be worked out

    @property
    def all_up(self) -> bool:
        return self.web and bool(self.router) and all(self.router.values())


def answers(url: str, headers: dict[str, str], seconds: float = ASK_SECONDS) -> bool:
    """Any HTTP answer below 500: the server is up (a refusal is an answer too)."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=max(seconds, 0.1)) as reply:
            return int(reply.status) < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except (urllib.error.URLError, OSError):
        return False


def wait_until_up(
    settings: Settings,
    runner: CommandRunner,
    asked: Answers = answers,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wait: float = WAIT_SECONDS,
) -> Ready:
    """Ask the web UI and the router at each address it listens on (0.0.0.0 is asked on the
    loopback) until all answer or `wait` seconds pass: each question gets what is left of
    them (at most ASK_SECONDS), so the bound holds. When the addresses cannot be worked out
    (`local` without Docker's answer), the router is not asked: the unit itself cannot bind."""
    try:
        listening = bind_addresses(settings.router_host, runner)
        problem = None
    except (BindError, ValueError) as exc:
        listening, problem = [], str(exc)
    asked_at = [LOOPBACK if a == EVERYWHERE else a for a in listening]
    headers = {PROBE_HEADER: "1"}
    if (key := settings.api_key_value()) is not None:
        headers["Authorization"] = f"Bearer {key}"
    web_url = f"http://127.0.0.1:{settings.port}/bar"
    deadline = clock() + wait

    def ask(url: str, sent: dict[str, str]) -> bool:
        left = deadline - clock()
        return left > 0 and asked(url, sent, min(ASK_SECONDS, left))

    while True:
        web = ask(web_url, {})
        router = {a: ask(f"http://{a}:{settings.router_port}{HEALTH}", headers) for a in asked_at}
        ready = Ready(web, router, listening, wait, problem)
        settled = ready.all_up or (problem is not None and web)  # nothing more to wait for
        if settled or clock() >= deadline:
            return ready
        sleep(min(EVERY, max(deadline - clock(), 0)))


def words(ready: Ready, settings: Settings) -> list[tuple[bool, str]]:
    """What the user reads: each line with whether it is good news."""
    port = settings.router_port
    lines = [
        (
            ready.web,
            f"Web UI: answers at http://127.0.0.1:{settings.port}"
            if ready.web
            else f"Web UI: not answering at http://127.0.0.1:{settings.port} after "
            f"{ready.waited:.0f} s (`mnm service logs`)",
        )
    ]
    if ready.problem:  # never a guessed address: the setting itself could not be read
        lines.append(
            (False, f"Router: where it listens could not be worked out ({ready.problem}).")
        )
        return lines
    if EVERYWHERE in ready.listening:
        where = f"every address of this machine (0.0.0.0:{port})"
    else:
        where = ", ".join(f"{a}:{port}" for a in ready.listening)
    up = [a for a, ok in ready.router.items() if ok]
    down = [a for a, ok in ready.router.items() if not ok]
    said = f"Router: listening on {where}"
    if up:
        said += f"; answers at {', '.join(f'{a}:{port}' for a in up)}"
    if down:
        said += (
            f"; not answering at {', '.join(f'{a}:{port}' for a in down)} after "
            f"{ready.waited:.0f} s (`mnm service logs`)"
        )
    lines.append((not down, said))
    return lines
