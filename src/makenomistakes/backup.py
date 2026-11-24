"""`mnm backup` and `mnm restore`: what a fresh install needs to be this box again,
in one .tar.gz, and nothing it can fetch again.

In it: the recipes (every instance's tuned settings), the sets, the imported ComfyUI
workflows (each one's `workflow.json` and `meta.json`; earlier versions stay behind), the
workflows' model links (`comfy/models.json`), the node-pack approvals (`comfy/packs.json`,
not the zips: each is fetched again by `mnm comfyui packs add`), the readings calibration
learns from (every box's), the boxes mnm saw (`machine.json`), the engine ledger,
and `secrets.names`: the secrets file's variable NAMES, never
a value. Not in it: model files (a restored instance downloads its files at its first
`mnm start`), runs and outputs, logs, ComfyUI's own folders.

Every member goes through `report.redact` (the secrets file's and `Settings`' values, a
signed link's query) on its way in, and links are saved without a secret in the first place
(`links.without_secrets`). The cards' uuids stay (`hardware=False`): a backup is the
user's own file on the box, and a restored `machine.json` must still match its cards. The
restore is `restore_state.py`.
"""

from __future__ import annotations

import fnmatch
import io
import os
import re
import tarfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, Field, ValidationError

from makenomistakes import __version__
from makenomistakes import config as config_module
from makenomistakes.apps.backup_files import NAMES as APP_NAMES
from makenomistakes.comfy.models import LEDGER_SCHEMA
from makenomistakes.comfy.packs import APPROVALS_SCHEMA
from makenomistakes.comfy.store import SCHEMA_VERSION as COMFY_META_SCHEMA
from makenomistakes.desk import SCHEMA_VERSION as DESK_SCHEMA
from makenomistakes.json_folder import cause, sentence, unlisted
from makenomistakes.machine_facts import SCHEMA_VERSION as MACHINE_SCHEMA
from makenomistakes.readings_filter import here_of
from makenomistakes.recipes.model import SCHEMA_VERSION as RECIPE_SCHEMA
from makenomistakes.recipes.model import NewerRecipeError, load_recipe_json
from makenomistakes.report import redact
from makenomistakes.studio.stages import SCHEMA_VERSION as PICKS_SCHEMA
from makenomistakes.studio.style_library import SCHEMA_VERSION as STYLE_SCHEMA
from makenomistakes.studio.voice_library import SCHEMA_VERSION as VOICE_SCHEMA
from makenomistakes.workflows import SCHEMA_VERSION as SET_SCHEMA

# 2: the apps' config files under `apps/`; 3: `machine.json`, every box's;
# 4: the voice library's rows and the studio's stage picks (never `studio/tools.json`:
# that names an image built on one box, as the engine ledger is this box's own); 5:
# the style library's rows and the probes' readings (the adapters stay on disk, as voices do)
BACKUP_SCHEMA = 5
MANIFEST = "manifest.json"
SECRET_NAMES = "secrets.names"
STATE = "state/"  # every state file sits under this in the archive
MAX_FILE_BYTES = 64 * 2**20  # a recipe, a set, a workflow or a reading file is far smaller
MAX_FILES = 20_000
# What a backup takes from the state folder, and the only paths a restore writes there.
PATTERNS = (
    "instances/*.json",
    "workflows/*.json",
    "desk/desk.json",
    "measurements/*.jsonl",
    "tuning/*.jsonl",  # Every trial (the tuned block rides on its recipe)
    "comfy/workflows/*/workflow.json",
    "comfy/workflows/*/meta.json",
    "comfy/models.json",
    "comfy/packs.json",
    "engines.json",
    "machine.json",  # Every box mnm saw and what it measured on each
    "studio/voices/*.json",  # The voice library's rows (the recordings stay on disk)
    "studio/stages.json",  # Which instance plays each stage's tier
    "studio/styles/*.json",  # The style library's rows (the adapters stay on disk)
    "studio/style-probes.json",  # Each box's measured peak per style setting
)
NOT_WRITTEN = "Nothing was written, and no older backup was deleted."
# The newest schema this mnm reads, per kind of file that carries one.
SCHEMAS = {
    "workflows/*.json": SET_SCHEMA,
    "desk/desk.json": DESK_SCHEMA,
    "comfy/workflows/*/meta.json": COMFY_META_SCHEMA,
    "comfy/models.json": LEDGER_SCHEMA,
    "comfy/packs.json": APPROVALS_SCHEMA,
    "machine.json": MACHINE_SCHEMA,
    "studio/voices/*.json": VOICE_SCHEMA,
    "studio/stages.json": PICKS_SCHEMA,
    "studio/styles/*.json": STYLE_SCHEMA,
}
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
NO_MODELS = (
    "Model files are not in a backup: each restored instance downloads its files again at "
    "its first `mnm start`."
)


class BackupError(RuntimeError):
    """Why a backup or restore did nothing (or, after writing began, where it stopped)."""


class Manifest(BaseModel):
    backup_schema: int = Field(ge=1)
    written_at: datetime
    mnm_version: str
    recipe_schema: int
    files: list[str]
    secret_names: list[str] = Field(default_factory=list)
    # the card the backup was written beside (None: unreadable then); a restore by an older
    # mnm left out another card's readings by it
    gpu: str | None = None
    machine: str | None = None  # The box's fingerprint id (None: not read yet)
    machine_words: str | None = None  # that box in one line
    app_files: list[str] = Field(default_factory=list)  # `apps/<app>/<file>`


class _Versioned(BaseModel):
    schema_version: int = 1


class _BackupSchema(BaseModel):
    backup_schema: int


def _wanted(rel: str) -> str | None:
    """The pattern a state-relative path belongs to, if any; every part a plain name."""
    if not all(_PART.match(part) for part in rel.split("/")):
        return None
    return next((p for p in PATTERNS if fnmatch.fnmatchcase(rel, p) and _depth(rel, p)), None)


def _depth(rel: str, pattern: str) -> bool:
    return rel.count("/") == pattern.count("/")  # `*` never spans a folder


def secret_names(path: Path | None = None) -> list[str]:
    """The secrets file's variable names (never a value); none when it cannot be read."""
    found = path or config_module.SECRETS_FILE
    try:
        return sorted(name for name in dotenv_values(found) if name)
    except OSError:
        return []


def state_files(state_dir: Path) -> list[str]:
    """The state-relative paths a backup takes, sorted. A folder a pattern lists and the
    system will not (or a file it will not look at) is a BackupError, so no archive is
    written: `Path.glob` swallows that error, and an archive could lack the sets, the voices,
    the styles or the tuning folder and still pass its restore test, which compares recipes
    only; `backup --auto` would then rotate the older archives away. A folder that is not
    there holds nothing."""
    found = set()
    for pattern in PATTERNS:
        for rel in _matching(state_dir, pattern.split("/")):
            path = state_dir / rel
            try:
                wanted = path.is_file() and not path.is_symlink()
            except OSError as exc:
                why = f"The file {path} could not be read ({cause(exc)})."
                raise BackupError(f"{why} {NOT_WRITTEN}") from exc
            if wanted and _wanted(rel) is not None:
                found.add(rel)
    return sorted(found)


def _matching(state_dir: Path, parts: list[str], prefix: str = "") -> list[str]:
    """The state-relative paths `parts` (a pattern split on `/`) names, each folder with a `*`
    in it listed by `os.scandir`, which raises (the way `JsonFolder.names` lists)."""
    head, rest = parts[0], parts[1:]
    if "*" in head:
        folder = state_dir / prefix
        try:
            with os.scandir(folder) as entries:
                names = sorted(entry.name for entry in entries)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise BackupError(f"{sentence(unlisted('folder', folder, exc))} {NOT_WRITTEN}") from exc
        heads = [name for name in names if fnmatch.fnmatchcase(name, head)]
    else:
        heads = [head]
    rels = [f"{prefix}/{name}" if prefix else name for name in heads]
    return [found for rel in rels for found in (_matching(state_dir, rest, rel) if rest else [rel])]


def backup_name(now: datetime) -> str:
    return f"mnm-backup-{now:%Y-%m-%d-%H%M%S}.tar.gz"


def write_backup(
    state_dir: Path,
    out: Path,
    known: list[tuple[str, str]],
    secrets: Path | None = None,
    gpu: str | None = None,
    app_files: list[tuple[str, str]] | None = None,
) -> Manifest:
    """The archive at `out` (written whole, then moved into place). `known` are the secret
    values `report.secrets` finds: each member is scrubbed of them, and of whatever only looks
    like one, before it is written (`report.redact`, the one scrubber)."""
    files = state_files(state_dir)
    names = secret_names(secrets)
    box = here_of(state_dir)
    manifest = Manifest(
        backup_schema=BACKUP_SCHEMA,
        written_at=datetime.now(UTC),
        mnm_version=__version__,
        recipe_schema=RECIPE_SCHEMA,
        files=files,
        secret_names=names,
        gpu=gpu,
        machine=box.id,
        machine_words=box.words,
        app_files=[name for name, _text in app_files or []],
    )
    partial = out.with_name(out.name + ".part")
    try:
        with tarfile.open(partial, "w:gz") as tar:
            _add(tar, MANIFEST, manifest.model_dump_json(indent=2).encode())
            _add(tar, SECRET_NAMES, "".join(f"{n}\n" for n in names).encode())
            for rel in files:
                text = (state_dir / rel).read_text(encoding="utf-8", errors="surrogateescape")
                # tokens and keys go; the cards' ids stay: a backup is the user's own file,
                # and the boxes it restores must still match their cards
                scrubbed = redact(text, known, hardware=False)
                scrubbed_bytes = scrubbed.encode("utf-8", errors="surrogateescape")
                _add(tar, STATE + rel, scrubbed_bytes)
            for name, text in app_files or []:  # `apps.backup_files.to_back_up`'s texts
                _add(tar, name, redact(text, known).encode("utf-8", errors="surrogateescape"))
        os.chmod(partial, 0o600)  # it carries the cards' ids on purpose
        os.replace(partial, out)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise BackupError(f"Could not write {out}: {exc}") from exc
    return manifest


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = int(datetime.now(UTC).timestamp())
    tar.addfile(info, io.BytesIO(data))


@dataclass(frozen=True)
class Archive:
    manifest: Manifest
    files: dict[str, bytes]  # state-relative path -> content
    app_files: dict[str, bytes] = field(default_factory=dict)  # `apps/<app>/<file>` -> content


def read_backup(path: Path) -> Archive:
    """Every file of the archive, checked: only regular files at the paths a backup writes,
    each bounded; anything else refuses the whole archive."""
    found: dict[str, bytes] = {}
    apps: dict[str, bytes] = {}
    manifest: bytes | None = None
    try:
        with tarfile.open(path, "r:gz") as tar:
            for count, member in enumerate(tar):
                if count > MAX_FILES:
                    raise BackupError(f"{path.name} holds more files than a backup writes.")
                data = _member(tar, member, path)
                if member.name == MANIFEST:
                    manifest = data
                elif member.name.startswith(STATE):
                    found[member.name.removeprefix(STATE)] = data
                elif member.name in APP_NAMES:
                    apps[member.name] = data
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise BackupError(f"{path} is not a backup mnm can read: {exc}") from exc
    if manifest is None:
        raise BackupError(f"{path.name} has no {MANIFEST}: it was not written by `mnm backup`.")
    try:  # the schema first: a newer manifest may have fields this one does not know
        schema = _BackupSchema.model_validate_json(manifest).backup_schema
        if schema > BACKUP_SCHEMA:
            raise BackupError(
                f"{path.name} was written by a newer mnm (backup schema {schema}; this one "
                f"reads up to {BACKUP_SCHEMA}): run `mnm update`, then restore it."
            )
        parsed = Manifest.model_validate_json(manifest)
    except ValidationError as exc:
        why = exc.errors()[0]["msg"]
        raise BackupError(f"{path.name}'s {MANIFEST} cannot be read: {why}.") from exc
    return Archive(parsed, found, apps)


def _member(tar: tarfile.TarFile, member: tarfile.TarInfo, path: Path) -> bytes:
    name = member.name
    allowed = name in (MANIFEST, SECRET_NAMES, *APP_NAMES) or (
        name.startswith(STATE) and _wanted(name.removeprefix(STATE)) is not None
    )
    if not member.isfile() or not allowed:
        raise BackupError(f"{path.name} holds {name!r}, which no backup writes: nothing restored.")
    if member.size > MAX_FILE_BYTES:
        raise BackupError(f"{path.name}: {name} is larger than any file a backup writes.")
    handle = tar.extractfile(member)
    if handle is None:
        raise BackupError(f"{path.name}: {name} cannot be read.")
    return handle.read()


def check(archive: Archive) -> None:
    """Every file readable by this mnm, or the restore writes nothing."""
    for rel, data in archive.files.items():
        pattern = _wanted(rel)
        if pattern == "instances/*.json":
            try:
                load_recipe_json(data.decode())
            except NewerRecipeError as exc:
                raise BackupError(
                    f"{rel} was written by a newer mnm ({exc}): run `mnm update`, "
                    "then restore again. Nothing was restored."
                ) from exc
            except (ValueError, ValidationError) as exc:
                raise BackupError(f"{rel} is not a recipe this mnm reads ({exc}).") from exc
        newest = SCHEMAS.get(pattern or "")
        if newest is None:
            continue
        try:
            version = _Versioned.model_validate_json(data).schema_version
        except ValidationError as exc:
            raise BackupError(f"{rel} cannot be read: {exc.errors()[0]['msg']}.") from exc
        if version > newest:
            raise BackupError(
                f"{rel} was written by a newer mnm (schema {version}, this one reads up to "
                f"{newest}): run `mnm update`, then restore again. Nothing was restored."
            )


def many(count: int, one: str, more: str) -> str:
    return f"{count} {one if count == 1 else more}"


def tally(paths: list[str]) -> list[str]:
    """What a list of backed-up paths holds, in words: one line per kind present."""

    def under(prefix: str) -> list[str]:
        return [rel for rel in paths if rel.startswith(prefix)]

    flows = {rel.split("/")[2] for rel in under("comfy/workflows/")}
    lines = [
        many(len(under("instances/")), "recipe", "recipes"),
        many(len(under("workflows/")), "set", "sets"),
        many(len(flows), "ComfyUI workflow", "ComfyUI workflows"),
        many(len(under("measurements/")), "instance's readings", "instances' readings"),
        many(len(under("tuning/")), "instance's tuning trials", "instances' tuning trials"),
        many(len(under("studio/voices/")), "library voice", "library voices"),
        many(len(under("studio/styles/")), "library style", "library styles"),
    ]
    found = [line for line in lines if not line.startswith("0 ")]
    for rel, words in (
        ("desk/desk.json", "the desk (which set's roles are model ids)"),
        ("comfy/models.json", "the workflows' model links"),
        ("comfy/packs.json", "the node-pack approvals (the zips are fetched again)"),
        ("engines.json", "the engine ledger"),
        ("machine.json", "the boxes mnm saw and what each measured"),
        ("studio/stages.json", "the studio's stage picks"),
        ("studio/style-probes.json", "the style trainer's measured peaks"),
    ):
        if rel in paths:
            found.append(words)
    return found
