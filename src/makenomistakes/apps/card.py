"""What an app's card is: the lines that say which model id to pick and why, the
files `--write` writes (each with how it merges into one mnm did not write), the blocks to
paste, and what the app uses, for `mnm doctor --app`. Plus the models a file offers, read
from the router's manifest: the desk's roles first (an app's config then survives a model
swap), then every chat model by its own name."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from makenomistakes.apps.floors import floor_for, floor_note
from makenomistakes.apps.names import LABELS, App
from makenomistakes.hub.facts import CHAT_LIKE, Modality
from makenomistakes.manifest import Manifest, ModelEntry
from makenomistakes.roles import CHAT_ROLES

Format = Literal["json", "yaml", "env"]
Doc = dict[str, Any]
USABLE = frozenset({"loaded", "parked", "unknown"})  # what a request may reach (not off)
KEY_PLACEHOLDER = "mnm"  # where an app insists on a key and the router asks none
CHAT_TYPES = frozenset(m.value for m in CHAT_LIKE)


@dataclass(frozen=True)
class AppFile:
    """One file `--write` writes: whole when it is new or mnm's own, else merged by `merge`
    (None: the format allows no merge, and only `--force` replaces it, a copy kept)."""

    label: str  # where it goes, as the app's docs name it
    default: Path | None  # `--write` without `--to` (None: no place of its own yet)
    format: Format
    value: Doc
    merge: Callable[[Doc], Doc] | None = None

    def text(self) -> str:
        return render(self.format, self.value)


@dataclass(frozen=True)
class Uses:
    """What the app's file points at, as the app sends it: what the doctor checks."""

    base_url: str  # the OpenAI base URL in the file (`/v1` or `/chat/v1`)
    chat_model: str | None  # the id it chats and calls tools with
    embed_model: str | None = None  # the id it embeds with (RAG, memory)
    embed_base: str | None = None  # where its embeddings go, when not `base_url`
    mcp_url: str | None = None  # mnm's MCP tools, when the file adds them
    # Its MCP requests can carry MNM_API_KEY (its model requests always can: each file names the
    # key, from the app's environment, whenever one is set). Odysseus's and Goose's cannot.
    mcp_takes_key: bool = True


@dataclass(frozen=True)
class AppCard:
    app: App
    lines: list[str]
    uses: Uses
    files: list[AppFile] = field(default_factory=list)
    blocks: list[tuple[str, str]] = field(default_factory=list)  # shown, never written
    no_write: str | None = None  # why `--write` writes nothing for this app

    @property
    def label(self) -> str:
        return LABELS[self.app]


class _Plain(yaml.SafeDumper):
    """YAML as a person writes it: a value used twice is written twice, never `&id001`."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def render(fmt: Format, value: Doc) -> str:
    if fmt == "json":
        return json.dumps(value, indent=2) + "\n"
    if fmt == "yaml":
        return yaml.dump(value, Dumper=_Plain, sort_keys=False, default_flow_style=False, width=100)
    return "".join(f"{key}={text}\n" for key, text in value.items())


def with_keys(owned: dict[tuple[str, ...], Any]) -> Callable[[Doc], Doc]:
    """A merge that sets each owned key path (`("provider", "mnm")`) and keeps the rest."""

    def merge(doc: Doc) -> Doc:
        merged: Doc = copy.deepcopy(doc)
        for path, value in owned.items():
            node = merged
            for key in path[:-1]:
                child = node.get(key)
                if not isinstance(child, dict):
                    child = node[key] = {}
                node = child
            node[path[-1]] = value
        return merged

    return merge


# --- the models a file offers ---------------------------------------------------------


@dataclass(frozen=True)
class Offer:
    id: str  # what the app names: a desk role, or an instance's own name
    entry: ModelEntry  # the instance behind it

    @property
    def alias(self) -> bool:
        return self.id != self.entry.name

    @property
    def described(self) -> str:
        return f"{self.id} ({self.entry.name})" if self.alias else self.id


def _usable(manifest: Manifest) -> dict[str, ModelEntry]:
    return {m.name: m for m in manifest.models if m.state in USABLE}


def role(manifest: Manifest, name: str) -> Offer | None:
    """A desk role whose member a request may reach."""
    member = manifest.aliases.get(name)
    found = _usable(manifest).get(member) if member else None
    return Offer(name, found) if found is not None else None


def chat_offers(manifest: Manifest) -> list[Offer]:
    """The chat ids a file lists: the desk's chat roles, then each chat model by name."""
    roles = [o for o in (role(manifest, r) for r in CHAT_ROLES) if o is not None]
    names = [Offer(m.name, m) for m in _usable(manifest).values() if m.type in CHAT_TYPES]
    return [*roles, *names]


def pick(manifest: Manifest, roles: tuple[str, ...]) -> Offer | None:
    """The id an app should use: the first of `roles` the desk has, else the first chat
    model by name (and the card says to make a desk)."""
    for wanted in roles:
        if (found := role(manifest, wanted)) is not None:
            return found
    names = [o for o in chat_offers(manifest) if not o.alias]
    return names[0] if names else None


def one_of(manifest: Manifest, wanted: str, kind: Modality) -> Offer | None:
    """A desk role of another kind (`memory`, `ear`, `voice`, `image`), else the only usable
    instance of that kind by name."""
    if (found := role(manifest, wanted)) is not None:
        return found
    of_kind = [m for m in _usable(manifest).values() if m.type == kind.value]
    return Offer(of_kind[0].name, of_kind[0]) if len(of_kind) == 1 else None


# --- the lines every card shares --------------------------------------------------------


def pick_line(app: App, chosen: Offer | None, wanted: str, why: str) -> str:
    """Which id to pick and why (`wanted`: the role the app should name); with no desk role
    for it, how to make one."""
    label = LABELS[app]
    if chosen is None:
        return (
            f"No chat model can be reached: `mnm add <link>` one (`--for-agents` for a coder), "
            f"then write {label}'s file again."
        )
    if chosen.alias:
        return (
            f"Pick `{chosen.id}` in {label}: the desk's {chosen.id}, {chosen.entry.name} now "
            f"({why}). `mnm sets edit <desk> {chosen.id}=<another>` swaps the model behind it; "
            "this file keeps working."
        )
    return (
        f"Pick `{chosen.id}` in {label} ({why}). No desk names a {wanted} yet: "
        f"`mnm sets create desk {wanted}={chosen.id} …` then `mnm sets default desk` gives the "
        "file ids that survive a model swap (`coder`, `chat`, `memory` …); write it again then."
    )


def context_line(app: App, chosen: Offer | None) -> str | None:
    context = chosen.entry.context if chosen is not None else None
    if chosen is None or context is None:
        return None
    entry = chosen.entry
    return floor_note(entry.name, context, entry.slots, LABELS[app], floor_for(app))


def key_line(manifest: Manifest, how: str) -> str:
    """What the key means for this app: how its file sends it, or that none is needed."""
    if manifest.key_required:
        return f"The router asks for MNM_API_KEY: {how} (the file names it, never holds it)."
    return "No MNM_API_KEY is set: the router answers this machine and its containers keyless."
