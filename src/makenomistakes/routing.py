"""Which instance a request to the router is for, as pure functions.

- `/upstream/<name>/…` goes to that instance with the prefix removed, whatever the body
  says: for clients that need a URL of their own (an Odysseus image endpoint must not share
  the chat URL) and for requests without a `model` (video jobs, `/infill`, engine pages).
- Anything else names its instance in the request's `model`, from a JSON body or a
  multipart field. The instance name matches first, whatever its state (a stopped one is
  then answered for what it is); an engine's own id (`whisper-1`, `qwen3-8b-q4_k_m`)
  matches when exactly one instance that can load serves it, so a client's saved choice
  and Odysseus's fixed `whisper-1` keep working beside a stopped spare.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.aliases import NO_ALIASES, Aliases
from makenomistakes.forward import form_field
from makenomistakes.hub.facts import CHAT_LIKE
from makenomistakes.hub.modality import MUSIC_MODALITIES
from makenomistakes.music_api import TASK_PATH, is_task
from makenomistakes.recipes import Recipe

UPSTREAM = "/upstream/"
CHAT = "/chat"  # `/chat/v1/…`: as `/v1/…`, over the chat, coding and vision instances only
MODELS = "/v1/models"
HEALTH = "/health"  # the router's own: it is up (each instance's is under /upstream/<name>/)
TUNE_PATH = "/__mnm/tune/"  # `mnm tune <name>` asks the router's tuner
TUNE_HOLD_PATH = "/__mnm/tune-hold"  # the nightly holds the tuner off the card
TRAIN_PATH = "/__mnm/train/"  # `mnm studio train` asks the router's trainer
ADOPTED_PATH = "/__mnm/adopted"  # `mnm machine adopt` ended: one journal line
BUSY_PATH = "/__mnm/busy"  # the router's own answer: requests in flight (a park waits)
PROBE_HEADER = "X-Mnm-Probe"  # mnm's health checks: answered, never a reason to load
WAKING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})  # real work loads a model


class RouteError(Exception):
    """No instance for this request; `status` is the HTTP answer the router gives."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Route:
    recipe: Recipe
    path: str  # what the engine is asked for: the request's own path, or after /upstream/
    alias: str | None = None  # the desk's role the request named (`desk.Aliases`)


Loadable = Callable[[Recipe], bool]  # loaded or parked: what a client may be routed to


def scoped(path: str, recipes: list[Recipe]) -> tuple[str, list[Recipe]]:
    """`/chat/…` is the same request over the chat-like instances only (an Odysseus LLM
    endpoint lists no image or voice model); any other path, every instance."""
    if path.startswith(CHAT + "/"):
        return path.removeprefix(CHAT), [r for r in recipes if r.modality in CHAT_LIKE]
    return path, recipes


def route(
    path: str,
    body: bytes,
    content_type: str,
    recipes: list[Recipe],
    loadable: Loadable = lambda _r: True,
    aliases: Aliases = NO_ALIASES,
) -> Route:
    """The instance and path for one request. RouteError: 404 for an unknown name or id, 400
    for a request that names none, or an id two instances serve. `alias` is the desk role the
    request named, when it named one."""
    raw, mark, query = path.partition("?")
    if raw.startswith(UPSTREAM):
        quoted, _, rest = raw.removeprefix(UPSTREAM).partition("/")
        name = urllib.parse.unquote(quoted)
        found = named(name, recipes, aliases)
        if found is None:
            raise unknown(name, aliases) or RouteError(
                404, f"no instance named {name!r} (`mnm ls` lists them)"
            )
        return Route(found.recipe, f"/{rest}{mark}{query}", found.alias)
    model = requested_model(body, content_type)
    if model is None:
        raise RouteError(
            400,
            f"this request names no model: send `model`, or ask {UPSTREAM}<name>{raw} "
            "for one instance",
        )
    found = find(model, recipes, loadable, aliases)
    if found.recipe.modality in MUSIC_MODALITIES and not is_task(raw):  # never loaded for it
        name = found.recipe.name
        raise RouteError(
            400,
            f"{name} is a studio model: it answers tasks only, at {UPSTREAM}{name}{TASK_PATH} "
            f"(`mnm try {name}` sends one)",
        )
    return Route(found.recipe, path, found.alias)


def requested_model(body: bytes, content_type: str) -> str | None:
    """The `model` a request names: a JSON object's field, or a multipart form's."""
    if "multipart/form-data" in (content_type or "").lower():
        return form_field(body, content_type, "model") or None
    try:
        request = json.loads(body) if body else None
    except ValueError:
        return None
    model = request.get("model") if isinstance(request, dict) else None
    return model if isinstance(model, str) and model else None


@dataclass(frozen=True)
class Named:
    """The instance an id names, and the desk role it was named by (None: its own name or its
    engine's id)."""

    recipe: Recipe
    alias: str | None = None


def named(model: str, recipes: list[Recipe], aliases: Aliases = NO_ALIASES) -> Named | None:
    """The instance named `model`, else the desk's member for that role (`desk.Aliases`): an
    instance's own name always wins. None when it is neither."""
    by_name = next((r for r in recipes if r.name == model), None)
    if by_name is not None:
        return Named(by_name)
    member = aliases.instance(model)
    by_alias = next((r for r in recipes if r.name == member), None) if member else None
    return Named(by_alias, model) if by_alias is not None else None


def find(
    model: str,
    recipes: list[Recipe],
    loadable: Loadable = lambda _r: True,
    aliases: Aliases = NO_ALIASES,
) -> Named:
    """`named`, else the one loadable instance whose engine serves that id (asked only on this
    path: it costs a look at each candidate's container)."""
    found = named(model, recipes, aliases)
    if found is not None:
        return found
    serving = [r for r in recipes if r.served_name == model and loadable(r)]
    if len(serving) == 1:
        return Named(serving[0])
    if serving:
        names = ", ".join(sorted(r.name for r in serving))
        raise RouteError(400, f"{len(serving)} instances serve {model!r} ({names}): name one")
    raise unknown(model, aliases) or RouteError(
        404, f"no model named {model!r} (`GET {MODELS}` lists them)"
    )


class GoneMemberError(RouteError):
    """A desk role whose member instance no longer exists (removed out from under the set)."""


def unknown(model: str, aliases: Aliases) -> GoneMemberError | None:
    """The refusal for a role whose member is gone, saying which instance it names and how to
    point it at another; None when `model` is no role."""
    member = aliases.instance(model)
    if member is None:
        return None
    return GoneMemberError(
        404,
        f"{model!r} is the desk's {model}, which names {member}, and no instance {member} "
        f"exists any more: `mnm sets edit {aliases.desk} {model}=<instance>` points it at "
        "another (`mnm ls` lists them)",
    )
