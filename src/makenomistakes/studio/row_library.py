"""What the studio's libraries share (voices, styles): one JSON row per name under the state
folder (`JsonFolder`), its files in a folder of the same name under the models folder, where the
engine's container reads them. A row is read through one rule (refused in words when it does not
parse, comes from a newer mnm, or names another row), its files are copied beside and moved into
place whole (`staged`), and a removal takes its folder unless the user keeps the files.
`Training` is what a row trained by mnm records, `sha256_of` how every file is named by content.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.engine_args_music import LIBRARY_NAME
from makenomistakes.json_folder import BadNameError, JsonFolder, cause, unlisted

NAME_PATTERN = LIBRARY_NAME  # a row's name: its file, its folder, what a setting names


class Training(BaseModel):
    """How mnm trained a row: its settings, the songs or recordings by sha256, the
    box that trained it."""

    settings: dict[str, str] = Field(default_factory=dict)
    epochs: int | None = None
    dataset: list[str] = Field(default_factory=list)  # each input's sha256
    machine: str | None = None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _Named(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def sha256(self) -> str: ...


class Row(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def schema_version(self) -> int: ...

    @property
    def files(self) -> Sequence[_Named]: ...


class RowLibrary[R: BaseModel]:
    """The rows (state) and their files (models folder) of one library."""

    noun: ClassVar[str]  # "voice", "style": in every sentence
    row_type: ClassVar[type[BaseModel]]
    error: ClassVar[type[ValueError]]
    schema_version: ClassVar[int]
    rm_command: ClassVar[str]  # what removes a row, `mnm voices rm`
    add_command: ClassVar[str]  # what adds one, `mnm voices add`

    def __init__(self, rows: Path, root: Path) -> None:
        self.rows = JsonFolder(rows)
        self.root = root

    def folder(self, name: str) -> Path:
        return self.root / name

    def all(self) -> list[R]:
        names, _unlisted = self._listed()
        return [row for name in names if (row := self._read(name)) is not None]

    def problems(self) -> list[str]:
        """Rows that cannot be read, worded (a newer mnm's, a hand edit), and a folder the
        system will not list."""
        names, unlisted = self._listed()
        found = [unlisted] if unlisted is not None else []
        for name in names:
            try:
                self._parse(name)
            except self.error as exc:
                found.append(str(exc))
        return found

    def _listed(self) -> tuple[list[str], str | None]:
        """The rows' names, and why their folder cannot be listed (then none): every page and
        command that lists the library shows its problems."""
        try:
            return self.rows.names(), None
        except OSError as exc:
            return [], unlisted(f"{self.noun}s' folder", self.rows.folder, exc)

    def get(self, name: str) -> R:
        try:
            exists = self.rows.exists(name)
        except BadNameError:
            exists = False
        if not exists:
            names, unlisted = self._listed()
            held = unlisted or f"the library holds: {', '.join(names) or 'none yet'}"
            raise self.error(f"No {self.noun} named '{name}' ({held}).")
        return self._parse(name)

    def _read(self, name: str) -> R | None:
        try:
            return self._parse(name)
        except self.error:
            return None

    def _parse(self, name: str) -> R:
        path = self.rows.path(name)
        try:
            row: Row = self.row_type.model_validate_json(self.rows.read(name))  # type: ignore[assignment]
        except (OSError, ValidationError, ValueError) as exc:
            raise self.error(f"{path} cannot be read: {exc}") from exc
        if row.schema_version > self.schema_version:
            raise self.error(f"{path} is from a newer mnm: run `mnm update`.")
        if row.name != name:
            raise self.error(f"{path} names the {self.noun} '{row.name}': fix or delete it.")
        return row  # type: ignore[return-value]

    def missing_files(self, row: R) -> list[str]:
        """Files the row names that are not in its folder (a restore brings rows, not files)."""
        named: Row = row  # type: ignore[assignment]
        return [f.name for f in named.files if not (self.folder(named.name) / f.name).is_file()]

    def changed_files(self, row: R) -> list[str]:
        """Files the row names whose bytes in its folder are not the ones it recorded."""
        named: Row = row  # type: ignore[assignment]
        folder = self.folder(named.name)
        changed = []
        for file in named.files:
            try:
                same = sha256_of(folder / file.name) == file.sha256
            except OSError:
                same = False
            if not same:
                changed.append(file.name)
        return changed

    def check_files(self, row: R) -> None:
        """Refused in words unless the row's files are in its folder as it recorded them:
        read where a file sings or is credited (else a row restored from an older backup over
        files kept from another voice would credit that row's source and consent to them),
        never in a listing (an RVC voice hashes in a tenth of a second)."""
        named: Row = row  # type: ignore[assignment]
        folder, name = self.folder(named.name), named.name
        if missing := self.missing_files(row):
            raise self.error(
                f"The {self.noun} {name}'s files are not in {folder}: {', '.join(missing)} "
                f"(a restore brings rows, not files; `{self.rm_command} {name}`, then add it "
                "again)."
            )
        if changed := self.changed_files(row):
            raise self.error(
                f"The {self.noun} {name}'s {', '.join(changed)} in {folder} is not what its row "
                "recorded (a row restored over other files, or a file replaced by hand), so "
                f"mnm cannot say whose it is: `{self.rm_command} {name} --keep-files`, move "
                f"the files out of {folder}, then `{self.add_command} {name}` from there."
            )

    def has(self, name: str) -> bool:
        """Whether a row of that name is in the library, asked by its name, never read from
        the listing: a folder the system will not list still answers (read from the listing, a
        training of a taken name would queue, train for hours and be refused at the end); one
        it will not look into is the library's error."""
        try:
            return self.rows.exists(name)
        except OSError as exc:
            folder = self.rows.folder
            raise self.error(
                f"The {self.noun}s' folder {folder} could not be read ({cause(exc)})."
            ) from exc

    def refuse_taken(self, name: str) -> None:
        """Under the rows' lock: a name already in the library, or a folder still holding
        files (a row removed with --keep-files), refused before anything is written."""
        if self.has(name):
            raise self.error(
                f"A {self.noun} named '{name}' is in the library: `{self.rm_command} {name}` first."
            )
        folder = self.folder(name)
        try:
            left = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
        except OSError as exc:
            raise self.error(f"{folder} cannot be read: {exc}") from exc
        if folder.exists() and not folder.is_dir():
            raise self.error(f"{folder} is a file, where the {self.noun}'s folder goes.")
        if left:  # the user's: never added over, never deleted
            raise self.error(
                f"{folder} still holds {', '.join(left)}, kept when the {self.noun}'s row went "
                f"(`{self.rm_command} --keep-files`, or the uninstall): `mnm restore` a backup "
                f"that holds its row, or move them out of {folder} and `{self.add_command} "
                f"{name}` from there."
            )

    @contextmanager
    def staged(self, name: str) -> Iterator[Path]:
        """An empty folder beside the library's to copy into, moved into place whole when
        the block ends; removed, and the copy's failure worded, otherwise."""
        folder = self.folder(name)
        staging = self.root / f".adding-{name}"
        try:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            yield staging
            if folder.is_dir():
                folder.rmdir()  # empty: `refuse_taken` checked it
            staging.rename(folder)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise self.error(
                f"The {self.noun}'s files could not be copied into {folder}: {exc}"
            ) from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def remove(self, name: str, *, files: bool = True) -> R:
        """The row, and its files unless `files` is False (they are the user's)."""
        with self.rows.locked():
            row = self.get(name)
            self.rows.unlink(name)
            if files:
                shutil.rmtree(self.folder(name), ignore_errors=True)
            return row
