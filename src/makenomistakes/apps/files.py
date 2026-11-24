"""Writing an app's file (`mnm url --for <app> --write [--to PATH] [--force]`).

- A file that is not there is written whole.
- A file mnm wrote whole and nobody changed since (its sha256 in the ledger,
  `<state>/apps/written.json`) is replaced whole.
- Any other file is the user's: its own keys are merged (JSON and YAML through the card's
  `merge`, a `.env` line by line, comments and order kept), a copy of it kept first as
  `<file>.before-mnm` (once: the first copy is the one worth having), and the change shown as a
  diff. A file that cannot be read in its format, a YAML file with comments (they would be
  lost), or a format without a merge is refused; `--force` then replaces it, the copy kept.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from makenomistakes.apps.card import AppFile, Doc, render
from makenomistakes.apps.names import App
from makenomistakes.instance_types import InstanceError
from makenomistakes.links import secret_name
from makenomistakes.locking import shared_lock
from makenomistakes.report import redact

LEDGER = "written.json"
SCHEMA_VERSION = 1
BACKUP_SUFFIX = ".before-mnm"
Action = Literal["wrote", "replaced", "merged", "unchanged", "overwrote"]


class AppFileError(InstanceError):
    """A file mnm will not write, worded for the user."""


class Written(BaseModel):
    app: str
    sha256: str  # what mnm left there
    whole: bool  # mnm wrote all of it (False: merged into the user's)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LedgerFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    files: dict[str, Written] = Field(default_factory=dict)


class Ledger:
    """What mnm wrote where, so it never mistakes the user's file for its own."""

    def __init__(self, state_dir: Path) -> None:
        self.folder = state_dir / "apps"
        self.path = self.folder / LEDGER
        self.locked = shared_lock(self.folder)  # `mnm apps` takes the same one around it

    def read(self) -> LedgerFile:
        if not self.path.exists():
            return LedgerFile()
        try:
            found = LedgerFile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (ValidationError, OSError) as exc:
            raise AppFileError(
                f"mnm's record of the files it wrote, {self.path}, cannot be read ({exc}); "
                "delete it and every file is then treated as yours (merged, a copy kept)."
            ) from exc
        if found.schema_version > SCHEMA_VERSION:
            raise AppFileError(f"{self.path} is from a newer mnm: run `mnm update`.")
        return found

    def get(self, path: Path) -> Written | None:
        return self.read().files.get(_key(path))

    def record(self, path: Path, app: App, text: str, whole: bool) -> None:
        ledger = self.read()
        ledger.files[_key(path)] = Written(app=app.value, sha256=_sha(text), whole=whole)
        _atomic(self.path, ledger.model_dump_json(indent=2))


@dataclass(frozen=True)
class WriteResult:
    path: Path
    action: Action
    backup: Path | None = None
    diff: str = ""


def _key(path: Path) -> str:
    """The ledger names a file by its real, absolute path: a relative `--to` and a symlink
    both mean the file they reach."""
    return str(path.expanduser().resolve())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def same_as_written(text: str, known: Written) -> bool:
    """Whether `text` is still what mnm left (the ledger's sha256)."""
    return known.sha256 == _sha(text)


NEW_MODE = 0o644  # a new config file; a new `.env` gets 0600 (`write`)


def _atomic(path: Path, text: str, mode: int = NEW_MODE) -> None:
    """Written beside and moved into place. The temp file is born 0600 (a merged `.env` holds
    the user's secrets), then takes the file's own mode, or `mode` for a new one. `path` is
    the real file: a symlinked config (stow, chezmoi) is written through, never replaced by a
    plain file (`write` resolves it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.mnm-{os.getpid()}.tmp")
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    if path.exists():
        shutil.copymode(path, tmp)
    else:
        tmp.chmod(mode)
    tmp.replace(path)


def parse(fmt: str, text: str, path: Path) -> Doc:
    """The user's file in its format, or AppFileError saying why it cannot be merged."""
    try:
        found = json.loads(text) if fmt == "json" else yaml.safe_load(text) or {}
    except (ValueError, yaml.YAMLError) as exc:
        kind = "JSON (comments, as JSONC allows, are not read here)" if fmt == "json" else "YAML"
        raise AppFileError(f"{path} cannot be read as {kind}: {exc}") from exc
    if not isinstance(found, dict):
        raise AppFileError(f"{path} does not hold a {fmt.upper()} object at its top.")
    if fmt == "yaml" and any(line.lstrip().startswith("#") for line in text.splitlines()):
        raise AppFileError(f"{path} has comments, which a YAML merge would lose.")
    return found


def merge_env(text: str, ours: Doc) -> str:
    """`.env` lines: each of ours replaces its key's line where it is, the rest are added at
    the end; comments, order and every other line stay."""
    left = {key: str(value) for key, value in ours.items()}
    out: list[str] = []
    for line in text.splitlines():
        key = (
            line.split("=", 1)[0].strip().removeprefix("export ").strip()  # apps-setup 12
            if "=" in line and not line.lstrip().startswith("#")
            else ""
        )
        out.append(f"{key}={left.pop(key)}" if key in left else line)
    out += [f"{key}={value}" for key, value in left.items()]
    return "\n".join(out) + "\n"


def merged_text(file: AppFile, old: str, path: Path) -> str:
    """The user's file with mnm's part merged in; AppFileError when it cannot be."""
    if file.format == "env":
        return merge_env(old, file.value)
    if file.merge is None:
        raise AppFileError(f"{path} was not written by mnm, which writes {file.label} only whole.")
    return render(file.format, file.merge(parse(file.format, old, path)))


def up_to_date(file: AppFile, path: Path) -> bool | None:
    """Whether the file there already holds what mnm would write now (None: not there)."""
    if not path.exists():
        return None
    old = path.read_text(encoding="utf-8")
    if old == file.text():
        return True
    try:
        return merged_text(file, old, path) == old
    except AppFileError:
        return False


def write(
    file: AppFile,
    path: Path,
    app: App,
    ledger: Ledger,
    force: bool = False,
    secrets: list[tuple[str, str]] | None = None,
) -> WriteResult:
    """Write `file` at `path` by the rules above, under the ledger's lock. The path is made
    absolute and followed through a symlink first: the ledger names the real file, and a
    linked config is written where it lives. `secrets` (`report.secrets(...).known`) are
    hidden in the diff besides what only looks secret."""
    path = path.expanduser().resolve()
    new = file.text()
    shown = secrets or []
    with ledger.locked():
        if not path.exists():
            _atomic(path, new, 0o600 if file.format == "env" else NEW_MODE)
            ledger.record(path, app, new, whole=True)
            return WriteResult(path, "wrote")
        old = path.read_text(encoding="utf-8")
        known = ledger.get(path)
        if known is not None and known.whole and known.sha256 == _sha(old):
            if old == new:
                return WriteResult(path, "unchanged")
            _atomic(path, new)
            ledger.record(path, app, new, whole=True)
            return WriteResult(path, "replaced", diff=shown_diff(old, new, path, shown))
        try:
            text = merged_text(file, old, path)
        except AppFileError as exc:
            if not force:
                raise AppFileError(
                    f"{exc} mnm leaves it as it is: paste the card's part by hand, or "
                    "`--force` replaces the file (a copy kept as "
                    f"{path.name}{BACKUP_SUFFIX})."
                ) from exc
            backup = _backup(path)
            _atomic(path, new)
            ledger.record(path, app, new, whole=True)
            return WriteResult(path, "overwrote", backup)  # no diff: every old line is the user's
        if text == old:
            return WriteResult(path, "unchanged")
        backup = _backup(path)
        _atomic(path, text)
        ledger.record(path, app, text, whole=False)
        return WriteResult(path, "merged", backup, shown_diff(old, text, path, shown))


def _backup(path: Path) -> Path:
    """The user's file as it was before mnm first changed it, kept once (a later merge keeps
    that first copy: it is the one worth having)."""
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def shown_diff(old: str, new: str, path: Path, secrets: list[tuple[str, str]]) -> str:
    """The changed lines only (no context: the user's other lines may hold secrets), each
    secret-looking value hidden (`hidden`), then every known secret's value and every token
    or signed link `report.redact` knows."""
    lines = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True), str(path), str(path), n=0
    )
    return redact("".join(hidden(line) for line in lines), secrets)


_KEYED = re.compile(r"""^([+-]\s*-?\s*["']?)([A-Za-z0-9_.-]+)(["']?\s*[:=]\s*)(.*?)(,?\n?)$""")
_SECRET_IN = ("apikey", "password", "secret", "token", "key", "credential")
_REFERENCE = re.compile(
    r"""^["']?(\$\{[A-Z0-9_]+\}|\{env:[A-Z0-9_]+\}|<[^>]*>|mnm)["']?$"""  # mnm: KEY_PLACEHOLDER
)
_URL_LOGIN = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^/\s@:]+(?::[^/\s@]*)?@")


def hidden(line: str) -> str:
    """A diff line whose key names a secret (`api_key`, `apiKey`, `CREDS_KEY`, `password` …)
    with its value replaced, unless the value is a reference (`${MNM_API_KEY}`,
    `{env:MNM_API_KEY}`) or a placeholder (`<your MNM_API_KEY>`); and any URL's login
    (`mongodb://user:pass@host`) hidden whatever its key."""
    line = _URL_LOGIN.sub(r"\1<hidden>@", line)
    found = _KEYED.match(line)
    if found is None:
        return line
    lead, name, sep, value, tail = found.groups()
    secret = secret_name(name) or any(word in name.lower() for word in _SECRET_IN)
    if not secret or not value.strip() or _REFERENCE.match(value.strip()):
        return line
    return f"{lead}{name}{sep}<hidden>{tail}"


def hidden_text(text: str) -> str:
    """A whole file with `hidden`'s rule applied to every line (an app file in a backup),
    each line read as an added diff line."""
    return "".join(hidden("+" + line)[1:] for line in text.splitlines(keepends=True))
