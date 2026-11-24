"""The context one request must have for each workspace app, and the check every
card and `mnm doctor --app` word the same way the coding-agent cards word theirs (enough; too
few, with the `mnm set … ctx=` that fixes it, slots counted; or no floor). Kept here, keyed by
`App`, beside `agent_profile` (what `mnm add --for <client>` plans with, keyed by the client's
name): every number is read from there, so the card, `mnm doctor --app` and the plan agree
(else Goose could plan at 32k while its card said it had no floor)."""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes import agent_profile
from makenomistakes.apps.names import App


@dataclass(frozen=True)
class Floor:
    tokens: int
    why: str
    hard: bool  # the app refuses or breaks below it; False: its docs suggest it


def _tokens(client: str) -> int:
    return agent_profile.FLOORS[client].tokens  # the one number per client


FLOORS: dict[App, Floor | None] = {
    App.OPENHANDS: Floor(
        _tokens("openhands"),
        "its docs ask 32k for agent work (docs.openhands.dev, read 2026-10-22); Agent Canvas "
        "has no context field, so mnm's plan is what holds it",
        hard=True,
    ),
    App.HERMES: Floor(
        _tokens("hermes"),
        "Hermes Agent refuses to start below 64k (its docs, read 2026-10-22)",
        hard=True,
    ),
    App.OPENCODE: Floor(
        _tokens("opencode"),
        "its docs suggest 16k to 32k for reliable tool calls; its file sets limit.context to "
        "the instance's own",
        hard=False,
    ),
    App.GOOSE: Floor(  # its file's `context_limit` is the instance's own context
        _tokens("goose"),
        "its docs state no minimum; its prompt and its extensions' tools are long",
        hard=False,
    ),
    App.LIBRECHAT: None,  # a chat: whatever the conversation holds
    App.OPEN_WEBUI: None,  # a chat too; it reads no context of its own (its docs name none)
    App.ODYSSEUS: None,  # it reads the context from the router before every message
}

# Each app with a floor is a client `mnm add --for` plans for.
CLIENTS = {
    App.OPENHANDS: "openhands",
    App.HERMES: "hermes",
    App.OPENCODE: "opencode",
    App.GOOSE: "goose",
}


def floor_for(app: App) -> Floor | None:
    return FLOORS[app]


def floor_met(have: int, floor: Floor | None) -> bool:
    return floor is None or have >= floor.tokens


def floor_note(name: str, have: int, shared_by: int, label: str, floor: Floor | None) -> str:
    """The context check a card carries: `have` is the context one request gets, `shared_by`
    the slots that split the instance's."""
    if floor is None:
        return f"Context: {have:,} tokens per request ({label} sets no floor of its own)."
    if have >= floor.tokens:
        return f"Context: {have:,} tokens per request, enough ({label}: {floor.why})."
    total = floor.tokens * shared_by  # llama-server gives each of its slots an equal share
    need = "needs at least" if floor.hard else "suggests at least"
    return (
        f"Context: {have:,} tokens per request, too few: {label} {need} {floor.tokens:,} "
        f"({floor.why}). `mnm set {name} ctx={total}` raises it"
        + (f" ({shared_by} slots share it)" if shared_by > 1 else "")
        + "; `mnm ls` then says whether it still fits (Fits now)."
    )
