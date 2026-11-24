"""`GET /metrics` on the router: Prometheus' text format, one URL for the whole box.

The router's own figures (instances by state, the VRAM each loaded one is estimated to hold,
its loads, parks and refusals since it started) and, after them, each loaded llama.cpp and
vLLM instance's own `/metrics` (llama-server's needs `--metrics`, which every recipe passes),
every sample relabelled with `model="<instance>"`. Families of the same name from several
engines are merged under one HELP and TYPE, as the format asks; a line the parser does not
know passes through as it came. An engine that does not answer within its timeout is left out
and counted (`mnm_engine_metrics_up 0`), never waited for longer.
"""

from __future__ import annotations

import http.client
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

from makenomistakes.hub.facts import Engine
from makenomistakes.instance_types import InstanceStatus
from makenomistakes.queue_figures import Reason, Tally, Waiting

PATH = "/metrics"
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
ENGINE_TIMEOUT_SECONDS = 2.0
ENGINE_LIMIT = 4 << 20  # an engine's metrics are a few kilobytes; a runaway one is cut
SCRAPED = frozenset({Engine.LLAMACPP, Engine.VLLM})  # the engines that serve /metrics


class RouterCounts:
    """What the router did since it started, counted where it does it (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loads = 0
        self.parks = 0
        self.refusals: dict[int, int] = {}
        self.deaths = 0  # loaded engines found dead by the healer
        self.restarts = 0  # of those, restarted and ready again

    def loaded(self, parked: int = 0) -> None:
        """A parked instance loaded for a request, after parking `parked` others for room."""
        with self._lock:
            self.loads += 1
            self.parks += parked

    def unloaded(self) -> None:
        """One unloaded when idle past its limit."""
        with self._lock:
            self.parks += 1

    def refused(self, status: int) -> None:
        """A request answered with this status by the router itself (never an engine's)."""
        with self._lock:
            self.refusals[status] = self.refusals.get(status, 0) + 1

    def died(self) -> None:
        """A loaded engine found dead."""
        with self._lock:
            self.deaths += 1

    def restarted(self) -> None:
        """A dead engine restarted and ready again."""
        with self._lock:
            self.restarts += 1

    def totals(self) -> tuple[int, int, dict[int, int]]:
        with self._lock:
            return self.loads, self.parks, dict(self.refusals)

    def healing(self) -> tuple[int, int]:
        """Deaths and restarts since the router started."""
        with self._lock:
            return self.deaths, self.restarts


Fetch = Callable[[int], str | None]  # an engine's port -> its /metrics text, None: no answer


def engine_metrics(port: int) -> str | None:
    """An engine's own `/metrics` on 127.0.0.1, within ENGINE_TIMEOUT_SECONDS; None when it
    does not answer 200 in Prometheus' text (llama-server without `--metrics` answers 501)."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=ENGINE_TIMEOUT_SECONDS)
    try:
        connection.request("GET", PATH)
        answer = connection.getresponse()
        body = answer.read(ENGINE_LIMIT + 1)
    except (OSError, http.client.HTTPException):
        return None
    finally:
        connection.close()
    if answer.status != 200 or not answer.getheader("Content-Type", "").startswith("text/plain"):
        return None  # not Prometheus' text: nothing of it is passed on
    return within_limit(body).decode("utf-8", errors="replace")


def within_limit(body: bytes) -> bytes:
    """A body over ENGINE_LIMIT cut back to its last whole line: a half line would make the
    whole page unreadable to Prometheus."""
    if len(body) <= ENGINE_LIMIT:
        return body
    return body[: body.rfind(b"\n", 0, ENGINE_LIMIT) + 1]


@dataclass(frozen=True)
class QueueNow:
    """The router's queue (`router_queue`): what waits on each instance now, and per
    (instance, reason) the waits since the router started."""

    waiting: dict[str, Waiting] = field(default_factory=dict)
    waited: dict[tuple[str, Reason], Tally] = field(default_factory=dict)


@dataclass
class _Family:
    """One metric family: its HELP and TYPE lines, then its samples, from every engine."""

    comments: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)


def render(
    statuses: list[InstanceStatus],
    vram: dict[str, int],
    counts: RouterCounts,
    fetch: Fetch = engine_metrics,
    queue: QueueNow | None = None,
) -> str:
    """The whole page: the router's own families, then the engines' merged."""
    states = {"loaded": 0, "parked": 0, "stopped": 0}
    for status in statuses:
        states[_state(status)] += 1
    loads, parks, refusals = counts.totals()
    deaths, restarts = counts.healing()
    lines = [
        *_family("mnm_instances", "gauge", "Saved instances by state.", _by("state", states)),
        *_family(
            "mnm_vram_held_bytes",
            "gauge",
            "VRAM each loaded instance is estimated to hold (its plan or its readings).",
            [(f'{{model="{name}"}}', value) for name, value in sorted(vram.items())],
        ),
        *_family(
            "mnm_loads_total", "counter", "Parked instances loaded for a request.", [("", loads)]
        ),
        *_family(
            "mnm_parks_total",
            "counter",
            "Instances the router unloaded: for room, or idle past their limit.",
            [("", parks)],
        ),
        *_family(
            "mnm_refusals_total",
            "counter",
            "Requests the router answered itself with an error, by status.",
            [(f'{{status="{code}"}}', n) for code, n in sorted(_every(refusals).items())],
        ),
        *_family(
            "mnm_engine_deaths_total",
            "counter",
            "Loaded engines whose container died.",
            [("", deaths)],
        ),
        *_family(
            "mnm_engine_restarts_total",
            "counter",
            "Dead engines the router restarted and saw ready again.",
            [("", restarts)],
        ),
        *_queue_families([s.recipe.name for s in statuses], queue or QueueNow()),
    ]
    families: dict[str, _Family] = {}
    up: list[tuple[str, int]] = []
    for status in statuses:
        recipe = status.recipe
        if not status.running or recipe.engine not in SCRAPED:
            continue
        text = fetch(recipe.port)
        up.append((f'{{model="{recipe.name}"}}', int(text is not None)))
        if text is not None:
            merge(families, text, recipe.name)
    lines += _family(
        "mnm_engine_metrics_up",
        "gauge",
        f"1 when a loaded engine answered its /metrics within {ENGINE_TIMEOUT_SECONDS:g} s.",
        up,
    )
    for family in families.values():
        lines += family.comments + family.samples
    return "\n".join(lines) + "\n"


def _queue_families(names: list[str], queue: QueueNow) -> list[str]:
    """Each instance's requests waiting now, and those that waited since the router
    started, for a slot at the engine or for their model to load (every sample from the start,
    at 0 until it happens)."""
    now = [(name, reason) for name in sorted(names) for reason in Reason]

    def waiting(name: str, reason: Reason) -> int:
        found = queue.waiting.get(name, Waiting())
        return found.slot if reason is Reason.SLOT else found.load

    def waited(name: str, reason: Reason) -> Tally:
        return queue.waited.get((name, reason), Tally())

    def labels(name: str, reason: Reason) -> str:
        return f'{{model="{name}",for="{reason.value}"}}'

    return [
        *_family(
            "mnm_requests_waiting",
            "gauge",
            "Requests waiting now: for a slot at the engine, or for their model to load.",
            [(labels(n, r), waiting(n, r)) for n, r in now],
        ),
        *_family(
            "mnm_requests_waited_total",
            "counter",
            "Requests that waited (past a second for a load) since the router started.",
            [(labels(n, r), waited(n, r).n) for n, r in now],
        ),
        *_family(
            "mnm_request_wait_seconds_total",
            "counter",
            "Their seconds of waiting.",
            [(labels(n, r), round(waited(n, r).total, 3)) for n, r in now],
        ),
    ]


class _Instances(Protocol):
    def statuses(self) -> list[InstanceStatus]: ...
    def running_vram(self, in_ram: bool | None = None) -> dict[str, int]: ...


def page(instances: _Instances, counts: RouterCounts, queue: QueueNow | None = None) -> bytes:
    """The page the router serves, from its instances as they are now: the VRAM gauge counts
    the cards' instances only (a model in the RAM holds none)."""
    statuses = instances.statuses()
    vram = instances.running_vram(in_ram=False)
    return render(statuses, vram, counts, queue=queue).encode()


def _state(status: InstanceStatus) -> str:
    if status.running:
        return "loaded"
    return "parked" if status.parked else "stopped"


# The statuses the router answers itself (the journal's refusals): each one a sample from
# the start, at 0 until it happens, as Prometheus counters are (else the family is empty
# on a quiet box).
OWN_STATUSES = (400, 401, 403, 404, 405, 409, 413, 421, 500, 502, 503)


def _every(refusals: dict[int, int]) -> dict[int, int]:
    return {**dict.fromkeys(OWN_STATUSES, 0), **refusals}


def _by(label: str, values: dict[str, int]) -> list[tuple[str, int]]:
    return [(f'{{{label}="{key}"}}', value) for key, value in values.items()]


def _family(
    name: str, kind: str, words: str, samples: Iterable[tuple[str, int | float]]
) -> list[str]:
    return [
        f"# HELP {name} {words}",
        f"# TYPE {name} {kind}",
        *(f"{name}{labels} {value}" for labels, value in samples),
    ]


def merge(families: dict[str, _Family], text: str, model: str) -> None:
    """One engine's page into `families`, each sample relabelled with its instance. A HELP or
    TYPE line is kept once per family (the first engine's); a sample goes to the family the
    comments before it named, else its own name's."""
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(maxsplit=3)
            if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                current = parts[2]
                family = families.setdefault(current, _Family())
                if not any(c.split(maxsplit=2)[1] == parts[1] for c in family.comments):
                    family.comments.append(line)
            continue  # any other comment is the engine's own note
        labelled = relabel(line, model)
        name = _name(line)
        key = current if current is not None and name.startswith(current) else name
        families.setdefault(key, _Family()).samples.append(labelled)


def relabel(line: str, model: str) -> str:
    """`name{a="b"} 1` → `name{model="m",a="b"} 1`; `name 1` → `name{model="m"} 1`. A line
    that is not a sample is returned as it came."""
    name = _name(line)
    if not name:
        return line
    rest = line[len(name) :]
    label = f'model="{model}"'
    if rest.startswith("{"):
        inner = _own_model_renamed(rest[1:])
        return f"{name}{{{label}{',' if not inner.startswith('}') else ''}{inner}"
    if rest.startswith((" ", "\t")):
        return f"{name}{{{label}}}{rest}"
    return line


def _name(line: str) -> str:
    """A sample's metric name (letters, digits, `_` and `:`), or "" when the line has none."""
    end = 0
    while end < len(line) and (line[end].isalnum() or line[end] in "_:"):
        end += 1
    return line[:end] if end and not line[0].isdigit() else ""


def _own_model_renamed(inner: str) -> str:
    """The labels after `{`, an engine's own `model` label (if any) renamed `engine_model`:
    a label twice in one sample is invalid. Values are walked past whole (quotes, escapes),
    so a value that reads `model=` is never touched."""
    out, i, at_key = [], 0, True
    while i < len(inner) and inner[i] != "}":
        char = inner[i]
        if char == '"':  # a value: copied up to its closing quote
            end = i + 1
            while end < len(inner) and inner[end] != '"':
                end += 2 if inner[end] == "\\" else 1
            out.append(inner[i : end + 1])
            i, at_key = end + 1, False
            continue
        if at_key and inner.startswith("model=", i):
            out.append("engine_")
        if char == ",":
            at_key = True
        elif not char.isspace():
            at_key = False
        out.append(char)
        i += 1
    return "".join(out) + inner[i:]
