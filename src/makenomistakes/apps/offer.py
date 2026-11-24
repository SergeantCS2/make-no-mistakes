"""What `mnm setup`'s Workspace apps section and the web's Get started offer: one
paragraph per app (what it is, what it needs, its licence, its port), the picks parsed from
an answer, and the card for an app that runs on another machine (the router's LAN address,
the key required). Odysseus is offered only when its folder exists: mnm
connects it, never installs it.

Sizes read on 2026-10-23:
LibreChat's five pinned images are about 3 GB and want 2 GB of RAM (librechat.ai/docs/local/
docker: "at least 2 GB of memory available to Docker"); Open WebUI's one image is 6.4 GB on
disk once pulled (v0.12.3, measured) and its first account is made on its own page; OpenCode's
release binary is one file of about 50 MB; Goose's documented Windows route is its desktop zip
(goose-docs.ai/docs/getting-started/installation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from makenomistakes.apps.defs import DEFS, Kind
from makenomistakes.apps.floors import floor_for
from makenomistakes.apps.names import LABELS, App
from makenomistakes.client_view import ReachError, lan
from makenomistakes.manifest import Manifest

NEEDS: dict[App, str] = {
    App.OPENCODE: "Needs nothing more: one release binary (~50 MB) run as a systemd user unit, "
    "working in your projects folder with your own toolchain.",
    App.LIBRECHAT: "Needs Docker: five pinned images (~3 GB) and ~2 GB of RAM; setup makes "
    "its first account.",
    App.OPEN_WEBUI: "Needs Docker: one pinned image (~6.4 GB on disk); you create its first "
    "account (the admin) on its page right after the install.",
    App.OPENHANDS: "Needs Docker: one pinned image; it mounts your projects folder, and wants a "
    "coder with {floor:,} tokens of context.",
    App.GOOSE: "Needs Goose itself (its Windows desktop app, or its CLI here): mnm writes its "
    "provider file and prints the rest.",
    App.ODYSSEUS: "Found in {folder}: mnm writes its .env lines (a diff first) and restarts "
    "its compose only when you say yes.",
}
# The apps whose files or agent work in MNM_PROJECTS_DIR: asked once before either installs.
USES_PROJECTS = frozenset({App.OPENCODE, App.OPENHANDS})
KEY_NAME = "MNM_API_KEY"


@dataclass(frozen=True)
class Offered:
    app: App
    number: int

    @property
    def label(self) -> str:
        return LABELS[self.app]

    @property
    def installs(self) -> bool:
        return DEFS[self.app].kind is not Kind.CONNECT

    def paragraph(self, odysseus: Path) -> str:
        found = DEFS[self.app]
        floor = floor_for(self.app)
        needs = NEEDS[self.app].format(floor=floor.tokens if floor else 0, folder=odysseus)
        if found.kind is Kind.CONNECT and not found.port:  # its own licence, its own port
            tail = f" {found.licence}." if found.licence != "its own" else ""
        else:
            tail = f" {found.licence}; 127.0.0.1:{found.port}."
        return f"{found.blurb[0].upper()}{found.blurb[1:]}. {needs}{tail}"


def offered(odysseus_present: bool) -> list[Offered]:
    """The apps in their ranked order, Open WebUI beside LibreChat (a second chat app);
    Odysseus only when it is there."""
    order = [App.OPENCODE, App.LIBRECHAT, App.OPEN_WEBUI, App.OPENHANDS, App.GOOSE, App.ODYSSEUS]
    shown = [a for a in order if a is not App.ODYSSEUS or odysseus_present]
    return [Offered(app, n) for n, app in enumerate(shown, 1)]


def parse_picks(answer: str, offers: list[Offered]) -> list[Offered] | None:
    """Numbers or names, separated by spaces or commas; `none` or nothing is none; None when a
    word names no offered app."""
    words = answer.replace(",", " ").lower().split()
    if not words or words == ["none"]:
        return []
    by_word = {str(o.number): o for o in offers} | {o.app.value: o for o in offers}
    picked: list[Offered] = []
    for word in words:
        found = by_word.get(word)
        if found is None:
            return None
        if found not in picked:
            picked.append(found)
    return picked


def remote_lines(app: App, manifest: Manifest) -> list[str]:
    """The card for an app on another machine (Windows, another PC): where it reaches the
    router, and that the key is required there; no LAN URL at all without a key (a
    keyless LAN router is never printed). Nothing on this machine changes."""
    label = LABELS[app]
    hosts = tuple(u.split("://", 1)[-1].split("/", 1)[0] for u in manifest.openai.lan)
    try:
        lan(hosts, manifest.lan_note, manifest.key_required)
    except ReachError as exc:
        return [f"{label} on another machine cannot use the router yet. {exc}"]
    return [
        f"On that machine, point {label} at {', '.join(manifest.openai.lan)} "
        "(OpenAI-compatible; the model ids are the desk's roles, `coder` for agents).",
        f"mnm's tools (MCP, Streamable HTTP): {', '.join(manifest.mcp.lan)}.",
        f"`mnm url --for {app.value} --from lan` prints {label}'s whole file with that address.",
        *lan_notes(app, mcp_takes_key=app not in NO_MCP_KEY),
    ]


# The apps whose MCP client sends no header: from another machine (the key required) their
# MCP tools cannot be reached (Odysseus's form has no key field, Goose documents none).
NO_MCP_KEY = frozenset({App.ODYSSEUS, App.GOOSE})


def lan_notes(app: App, mcp_takes_key: bool) -> list[str]:
    """What a card for another machine adds (`mnm url --for <app> --from lan`, the card)."""
    label = LABELS[app]
    lines = [
        f"The key is required off this machine: {label} sends {KEY_NAME} (the box's "
        "secrets file holds it; mnm never prints it) as its API key."
    ]
    if not mcp_takes_key:
        lines.append(
            f"{label} cannot send a key to /mcp: its MCP tools work only from this machine and "
            "its containers, keyless."
        )
    return lines
