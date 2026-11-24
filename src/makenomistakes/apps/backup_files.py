"""The workspace apps' config files in `mnm backup`: the files mnm wrote or merged
into for an app (each named in `<state>/apps/written.json`), never an app's data (LibreChat's
databases and uploads, OpenHands' home and its encrypted settings, OpenCode's sessions) and
never LibreChat's or Open WebUI's `.env` (their generated secrets would leave the box; `mnm
apps install` makes new ones). Open WebUI has no other file: its settings are the compose
file's environment, which `mnm apps install open-webui` writes again from the desk. Each
goes through `report.redact` like every member. A restore puts one back only where none is:
an app's file of this machine is the user's.

A file mnm merged into, or one the user changed since mnm wrote it whole, is the user's too:
its other providers carry their own keys, which `report.redact` does not know. Only mnm's own
part of it is backed up (the key paths the card's merge owns), so a restore writes a file
holding mnm's part; and every line of every app file goes through `files.hidden_text` (the
diff's rule) as a second guard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from makenomistakes.apps import librechat, opencode
from makenomistakes.apps.card import Doc, Format, render
from makenomistakes.apps.files import (
    AppFileError,
    Ledger,
    Written,
    hidden_text,
    parse,
    same_as_written,
)
from makenomistakes.apps.names import App, config_home
from makenomistakes.config import Settings

PREFIX = "apps/"
NAMES = (
    "apps/librechat/librechat.yaml",
    "apps/opencode/opencode.json",
    "apps/goose/mnm.json",
)


def places(settings: Settings) -> dict[str, Path]:
    """Each archive name and where the file lives on this machine."""
    home = config_home()
    return {
        NAMES[0]: settings.expanded_apps_dir() / App.LIBRECHAT.value / "librechat.yaml",
        NAMES[1]: home / "opencode" / "opencode.json",
        NAMES[2]: home / "goose" / "custom_providers" / "mnm.json",
    }


def _opencode_part(doc: Doc) -> Doc:
    """`provider.mnm` and `mcp.mnm`: what `opencode.card`'s merge sets."""
    part: Doc = {key: doc[key] for key in ("$schema",) if key in doc}
    for top in ("provider", "mcp"):
        block = doc.get(top)
        if isinstance(block, dict) and opencode.PROVIDER in block:
            part[top] = {opencode.PROVIDER: block[opencode.PROVIDER]}
    return part


def _host(url: object) -> str | None:
    return str(url).split("://", 1)[-1].split("/", 1)[0] if url else None


def _librechat_part(doc: Doc) -> Doc:
    """What `librechat.merge_into` sets: the `mnm` endpoint, the allowed addresses, the speech
    blocks that point at the same router, `mcpServers.mnm` and `mcpSettings`."""
    part: Doc = {key: doc[key] for key in ("version",) if key in doc}
    found_endpoints = doc.get("endpoints")
    endpoints: Doc = found_endpoints if isinstance(found_endpoints, dict) else {}
    ours = [
        e
        for e in endpoints.get("custom") or []
        if isinstance(e, dict) and e.get("name") == librechat.ENDPOINT
    ]
    if ours:
        part["endpoints"] = {
            "allowedAddresses": list(endpoints.get("allowedAddresses") or []),
            "custom": ours,
        }
        router = _host(ours[0].get("baseURL"))
        speech: Doc = {}
        for kind, block in (doc.get("speech") or {}).items():
            found: Any = block.get("openai") if isinstance(block, dict) else None
            if isinstance(found, dict) and _host(found.get("url")) == router:
                speech[kind] = {"openai": found}
        if speech:
            part["speech"] = speech
    servers = doc.get("mcpServers")
    if isinstance(servers, dict) and librechat.ENDPOINT in servers:
        part["mcpServers"] = {librechat.ENDPOINT: servers[librechat.ENDPOINT]}
    settings = doc.get("mcpSettings")
    if isinstance(settings, dict) and "allowedAddresses" in settings:
        part["mcpSettings"] = {"allowedAddresses": list(settings["allowedAddresses"])}
    return part


# Each file's format and mnm's own part of it (None: the whole file is mnm's: Goose's
# provider file is mnm's alone, never merged).
_PARTS: dict[str, tuple[Format, Callable[[Doc], Doc] | None]] = {
    NAMES[0]: ("yaml", _librechat_part),
    NAMES[1]: ("json", _opencode_part),
    NAMES[2]: ("json", None),
}


def backed_up_text(name: str, path: Path, text: str, known: Written) -> str | None:
    """What the backup holds for one app file: the file as mnm left it, else mnm's own part
    of it (None: it cannot be read in its format, and nothing of it is taken); every line
    through `hidden_text`."""
    fmt, part = _PARTS[name]
    if part is not None and not (known.whole and same_as_written(text, known)):
        try:
            text = render(fmt, part(parse(fmt, text, path)))
        except AppFileError:
            return None
    return hidden_text(text)


def to_back_up(settings: Settings) -> list[tuple[str, str]]:
    """Each app file that exists and that mnm wrote, with the text the backup holds (an
    unreadable record: none)."""
    try:
        written = Ledger(settings.expanded_state_dir()).read().files
    except AppFileError:
        return []
    found: list[tuple[str, str]] = []
    for name, path in places(settings).items():
        known = written.get(str(path.resolve())) if path.is_file() else None
        if known is None:
            continue
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        if (kept := backed_up_text(name, path, text, known)) is not None:
            found.append((name, kept))
    return found


@dataclass
class AppsRestored:
    written: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)  # this machine has its own


def restore_app_files(files: dict[str, bytes], settings: Settings) -> AppsRestored:
    done = AppsRestored()
    where = places(settings)
    for name, data in sorted(files.items()):
        path = where.get(name)
        if path is None:
            continue
        if path.exists():
            done.kept.append(str(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".restoring")
        partial.write_bytes(data)
        partial.replace(path)
        done.written.append(str(path))
    return done
