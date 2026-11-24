"""What the router says when it does not pass a request on: the refusals by the instance's state,
and the way out for each kind of holder when the room a load needs is held. The OpenAI routes and
Ollama's API share them."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import unquote

from makenomistakes.routing import WAKING_METHODS

if TYPE_CHECKING:
    from makenomistakes.instance_types import InstanceStatus

NOT_ANSWERING = "{name} is not answering: still loading, or it stopped (`mnm logs {name}`)"
STOPPED = "{name} is stopped: `mnm start {name}` starts it (the router never loads it)"
NOT_DOWNLOADED = "{name} is not downloaded: `mnm start {name}` downloads and starts it"
PARKED = "{name} is not loaded: the next request that asks for work loads it"
FAILED = (
    "{error} It is marked stopped, so later requests do not try again: fix it, then "
    "`mnm start {name}`."
)
NO_DOCKER = "Docker could not be asked which instances run: {error}"
SLOTS_REFUSED = (
    "A model's slots are saved and put back by mnm alone (`cache=on`): the router passes no "
    "save, restore or erase on to the engine."
)


@dataclass(frozen=True)
class Refusal:
    status: int
    message: str


def not_loadable(status: InstanceStatus, method: str, probe: bool) -> Refusal | None:
    """Why a request does not load an instance with no container, or None when it does:
    stopped by the user, not downloaded, or a request that only looks."""
    name = status.recipe.name
    if status.recipe.off:
        return Refusal(409, STOPPED.format(name=name))
    if not status.files_present and not status.evicted:  # evicted: the load downloads first
        return Refusal(409, NOT_DOWNLOADED.format(name=name))
    if probe or method not in WAKING_METHODS:
        return Refusal(503, PARKED.format(name=name))
    return None


# What to do depends on why each holder stays: a pinned one never finishes.
PINNED = (
    " Pinned (`keep=always`), never unloaded for room: {names}. `mnm set {first} keep=auto` "
    "lets the router unload it when it needs the room; `mnm stop {first}` frees it now."
)
BUSY = (
    " Busy with a reply, or in a set in use: {names}. Ask again when done; `mnm stop` frees "
    "{them} for good."
)


def what_to_do(room: list[str], kept: list[str], pinned: list[str]) -> str:
    """The way out for each kind of holder: a pinned one (kept) is let go by hand; the others
    (busy, offered as room; or a member of a set in use, kept) finish by themselves."""
    held_pinned = [name for name in kept if name in pinned]
    others = [*room, *(name for name in kept if name not in pinned)]
    said = PINNED.format(names=", ".join(held_pinned), first=held_pinned[0]) if held_pinned else ""
    if others:
        said += BUSY.format(names=", ".join(others), them="it" if len(others) == 1 else "them")
    elif not held_pinned:  # the verdict named nobody: the old general way out
        said = " Ask again when they are done; `mnm stop` frees them for good."
    return said


def slot_action(method: str, path: str) -> bool:
    """A client's `POST /upstream/<name>/slots/<id>?action=…`: refused (`SLOTS_REFUSED`),
    as the engine runs it on files under the models folder; looking (`GET /slots`) passes.
    Judged on the path as the engine routes it: llama-server's HTTP library decodes `%XX`
    before matching (cpp-httplib's `decode_path_component`), so `/%73lots/0` is `/slots/0`;
    decoded until it stops changing, then normalised (`//`, `.`, `..`)."""
    if method in ("GET", "HEAD"):
        return False
    target = path.partition("?")[0]
    while (decoded := unquote(target)) != target:
        target = decoded
    target = posixpath.normpath("/" + target.lstrip("/"))
    return target == "/slots" or target.startswith("/slots/")
