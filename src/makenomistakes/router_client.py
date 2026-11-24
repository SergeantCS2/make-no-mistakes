"""What mnm's other processes ask the router: is it up, and what is it serving.
mnm's own probes, so they carry `PROBE_HEADER` and never make it load anything. On the box
they ask 127.0.0.1; on a client install (`MNM_ROUTER_URL`) the box's router there."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from makenomistakes.adopt_words import AdoptCounts
from makenomistakes.config import Settings
from makenomistakes.health import HEALTH_PATH, HealthReport
from makenomistakes.keep_alive import SAVED_PATH
from makenomistakes.queue_figures import Waiting
from makenomistakes.recipes.model import Recipe
from makenomistakes.router_events import EVENTS_PATH, Event, Events
from makenomistakes.routing import (
    ADOPTED_PATH,
    BUSY_PATH,
    HEALTH,
    PROBE_HEADER,
    TRAIN_PATH,
    TUNE_HOLD_PATH,
    TUNE_PATH,
)

# The router answers both from memory; a park asks once a second until it says 0.
TIMEOUT_SECONDS = 2.0

Busy = Callable[[Recipe], int]  # requests an instance is serving now


class _Busy(BaseModel):
    busy: dict[str, int] = {}
    arrivals: int = 0  # requests taken since the router started
    queue: dict[str, Waiting] = {}  # what waits on each instance now (an older router: none)
    tuning: str | None = None  # the tune running in the router
    training: str | None = None  # the training running in the router


@dataclass(frozen=True)
class Traffic:
    """What the router serves now, and how many requests it has taken: a reading is the
    instance's own only while no other was busy and no request arrived meanwhile."""

    busy: dict[str, int]
    arrivals: int

    def only(self, name: str) -> bool:
        """Nothing but `name` (if anything) is in flight."""
        return all(n == name or count == 0 for n, count in self.busy.items())


def router_base(settings: Settings, port: int | None = None) -> str:
    """Where mnm asks the router: `MNM_ROUTER_URL` on a client install (its Ollama listener at
    that host), else this machine's 127.0.0.1 (the router binds IPv4)."""
    asked = port or settings.router_port  # the router, or its Ollama listener
    if settings.router_url:
        if asked == settings.router_port:
            return settings.router_url
        parts = urllib.parse.urlsplit(settings.router_url)
        return f"{parts.scheme}://{parts.hostname}:{asked}"
    return f"http://127.0.0.1:{asked}"


def _get(
    settings: Settings, path: str, port: int | None = None, timeout: float = TIMEOUT_SECONDS
) -> bytes | None:
    """The body of the router's 200; None when nothing listens on its port. OSError for
    something there that did not answer so (an error status, a timeout, broken HTTP)."""
    headers = {PROBE_HEADER: "1"}
    if (key := settings.api_key_value()) is not None:  # the router asks every request for it
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"{router_base(settings, port)}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            body: bytes = answer.read()
            return body
    except urllib.error.URLError as exc:  # HTTPError (a status) is one too: re-raised
        if isinstance(exc.reason, ConnectionRefusedError):  # urlopen wraps connect errors
            return None
        raise
    except http.client.HTTPException as exc:  # something on the port speaks broken HTTP
        raise OSError(f"no HTTP answer: {exc!r}") from exc


def settings_saved(settings: Settings, name: str) -> None:
    """Tell a running router the user saved `name`'s idle or keep, so a client's keep_alive
    gives way at once (`IdleClock.forget_keep`). Best effort: a router that is not running
    has nothing to forget."""
    try:
        _post(settings, f"{SAVED_PATH}{urllib.parse.quote(name)}")
    except (OSError, http.client.HTTPException):
        return


def _post(settings: Settings, path: str, body: bytes = b"") -> bytes:
    """A POST to the router on this machine (tests replace it); the answer's body."""
    headers = {PROBE_HEADER: "1", "Content-Type": "application/json"}
    if (key := settings.api_key_value()) is not None:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{router_base(settings)}{path}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as answer:
        found: bytes = answer.read()
        return found


def router_adopted(settings: Settings, counts: AdoptCounts) -> bool:
    """Tell a running router `mnm machine adopt` ended (its journal line); False when
    no router answered (nothing to tell)."""
    try:
        _post(settings, ADOPTED_PATH, counts.model_dump_json().encode())
    except (OSError, http.client.HTTPException):
        return False
    return True


# A router that took the ask and never answered (a timeout, a reset): the caller cannot tell
# whether it acted. The router's own routes never answer 503, so the status says it alone.
SILENT = 503


def router_tune(settings: Settings, name: str, body: bytes) -> tuple[int, str] | None:
    """Ask the router's tuner (`POST /__mnm/tune/<name>`): its status and words; None
    when no router answers on this machine."""
    return _ask(settings, f"{TUNE_PATH}{urllib.parse.quote(name)}", body)


def router_train(settings: Settings, run_id: str, body: bytes) -> tuple[int, str] | None:
    """Ask the router's trainer (`POST /__mnm/train/<run>`): its status and words; None
    when no router answers on this machine."""
    return _ask(settings, f"{TRAIN_PATH}{urllib.parse.quote(run_id)}", body)


def _ask(settings: Settings, path: str, body: bytes) -> tuple[int, str] | None:
    try:
        said = _post(settings, path, body)
    except urllib.error.HTTPError as exc:
        return exc.code, error_message(exc.read()) or str(exc.reason)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return None
        return SILENT, f"the router did not answer: {exc.reason}"
    except (OSError, http.client.HTTPException) as exc:
        return SILENT, f"the router did not answer: {exc}"
    try:
        return 202, _Said.model_validate_json(said).message
    except ValidationError:
        return 202, said.decode("utf-8", "replace")


def hold_tunes(settings: Settings, seconds: float, who: str = "the nightly") -> str | None:
    """Hold the router's tuner off the card for `seconds` (0 lets go): its words, None
    when no router (or an older one without the route) took it."""
    body = json.dumps({"seconds": seconds, "who": who}).encode()
    try:
        said = _post(settings, TUNE_HOLD_PATH, body)
    except (OSError, http.client.HTTPException):  # URLError and HTTPError are OSErrors
        return None
    try:
        return _Said.model_validate_json(said).message
    except ValidationError:
        return said.decode("utf-8", "replace")


class _Said(BaseModel):
    message: str


def router_tuning(settings: Settings) -> str | None:
    """The tune the router is running now (`/__mnm/busy`), None when none or no router."""
    now = router_now(settings)
    return now.tuning if now is not None else None


def router_training(settings: Settings) -> str | None:
    """The training the router is running now (`/__mnm/busy`), None when none or no router."""
    now = router_now(settings)
    return now.training if now is not None else None


def ollama_version(settings: Settings, port: int) -> str | None:
    """The version what listens on Ollama's port says (`GET /api/version`): mnm's own when the
    router serves it there; "" for an answer that is not Ollama's; None when nothing listens."""
    try:
        body = _get(settings, "/api/version", port)
    except OSError:
        return ""
    if body is None:
        return None
    try:
        return _Version.model_validate_json(body).version
    except ValidationError:
        return ""


class _Version(BaseModel):
    version: str


def router_answers(settings: Settings) -> bool:
    """Whether the router answers its `/health`: every client URL goes through it."""
    try:
        return _get(settings, HEALTH) is not None
    except OSError:
        return False


def router_busy(settings: Settings) -> Busy:
    """A start outside the router's own process (`mnm start`, the web UI) asks the router
    what it is serving before it stops an instance. No router means nothing in flight (no
    client reaches an engine without it: they publish on 127.0.0.1 only); a router that does
    not say (busy, an error) counts as serving one, so the park waits."""

    def busy(recipe: Recipe) -> int:
        try:
            body = _get(settings, BUSY_PATH)
        except OSError:
            return 1
        if body is None:
            return 0
        try:
            return _Busy.model_validate_json(body).busy.get(recipe.name, 0)
        except ValidationError:
            return 1

    return busy


def router_traffic(settings: Settings) -> Traffic | None:
    """The router's traffic: none when no router runs (nothing can reach an engine then);
    None when it is there and did not say."""
    now = router_now(settings)
    return Traffic(now.busy, now.arrivals) if now is not None else None


@dataclass(frozen=True)
class RouterNow:
    """One look at `/__mnm/busy`: what is in flight, the requests taken, the tune running."""

    busy: dict[str, int]
    arrivals: int
    tuning: str | None = None
    training: str | None = None
    queue: dict[str, Waiting] = field(default_factory=dict)  # what waits now

    @property
    def serving(self) -> bool:
        return any(self.busy.values())


def router_now(settings: Settings) -> RouterNow | None:
    """What the router is doing now (the nightly asks before it starts and between its
    rows); nothing at all when no router runs; None when one is there and did not say."""
    try:
        body = _get(settings, BUSY_PATH)
    except OSError:
        return None
    if body is None:
        return RouterNow({}, 0)
    try:
        found = _Busy.model_validate_json(body)
    except ValidationError:
        return None
    return RouterNow(found.busy, found.arrivals, found.tuning, found.training, found.queue)


def router_events(settings: Settings) -> list[Event] | None:
    """The router's newest journal lines, oldest first; None when no router answers
    or it did not say (an older router without them answers 404)."""
    try:
        body = _get(settings, EVENTS_PATH)
    except OSError:
        return None
    if body is None:
        return None
    try:
        return Events.model_validate_json(body).events
    except ValidationError:
        return None


def router_health(settings: Settings) -> HealthReport | None:
    """The router's `/__mnm/health`; None when no router answers it (or an older one
    without the route: `router_answers` tells those apart)."""
    try:
        body = _get(settings, HEALTH_PATH)
    except OSError:
        return None
    if body is None:
        return None
    try:
        return HealthReport.model_validate_json(body)
    except ValidationError:
        return None


def get(settings: Settings, path: str, timeout: float = 10.0) -> bytes:
    """A GET whose answer a command needs (`mnm router manifest` on a client install): the
    body of a 200, else OSError saying what answered."""
    try:
        body = _get(settings, path, timeout=timeout)
    except urllib.error.HTTPError as exc:
        said = error_message(exc.read()) or exc.reason
        raise OSError(f"{router_base(settings)}{path} answered {exc.code}: {said}") from None
    except OSError as exc:
        raise OSError(f"{router_base(settings)}{path}: {exc}") from None
    if body is None:
        raise OSError(f"nothing listens at {router_base(settings)}")
    return body


def not_answering_note(settings: Settings) -> str:
    return (
        f"The router (port {settings.router_port}) does not answer, so no client reaches any "
        "model yet: `mnm service install` runs it as a service (or `mnm router` in a terminal)."
    )


class _ErrorDetail(BaseModel):
    message: str = ""


class ErrorBody(BaseModel):
    """An error answer: OpenAI's `{"error": {"message": …}}` (the router's refusals,
    llama-server, vLLM), or the bare `{"error": "…"}` sd-server answers its own refusals with
    (`"invalid size"`)."""

    error: _ErrorDetail | str

    @property
    def message(self) -> str:
        return (self.error if isinstance(self.error, str) else self.error.message).strip()


def error_message(body: str | bytes) -> str | None:
    """An error answer's message; None for any other body (or an empty message)."""
    try:
        return ErrorBody.model_validate_json(body).message or None
    except ValidationError:
        return None
