"""The nightly's own settings: `<state>/nightly/settings.json`, `mnm nightly set`.
One for now: `clean`, the age past which `mnm clean --auto` removes run records (their files
with them) and saved engine logs; off unless the user turns it on."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

from makenomistakes.cleanup import parse_age
from makenomistakes.nightly.report import FOLDER

SCHEMA_VERSION = 1
FILE = "settings.json"
OFF = ("off", "none", "")


class NightlySettingsError(RuntimeError):
    """The file cannot be read, or a value is not one."""


class NightlySettings(BaseModel):
    schema_version: int = SCHEMA_VERSION
    clean: str | None = None  # `30d`: an age `cleanup.parse_age` reads; None: off

    @field_validator("clean")
    @classmethod
    def _an_age(cls, value: str | None) -> str | None:
        """A hand-edited value that is no age is the file's error at read, never a
        traceback at the night's start."""
        if value:
            parse_age(value)  # ValueError: pydantic's, then `NightlySettingsError`
        return value

    @property
    def clean_age(self) -> timedelta | None:
        return parse_age(self.clean) if self.clean else None

    @property
    def clean_words(self) -> str:
        if not self.clean:
            return "off (`mnm nightly set clean=30d` turns it on)"
        return f"run records, their files and saved logs older than {self.clean}"


def path(state_dir: Path) -> Path:
    return state_dir / FOLDER / FILE


def read(state_dir: Path) -> NightlySettings:
    try:
        text = path(state_dir).read_text(encoding="utf-8")
    except FileNotFoundError:
        return NightlySettings()
    except OSError as exc:
        raise NightlySettingsError(f"{path(state_dir)} cannot be read: {exc}") from exc
    try:
        found = NightlySettings.model_validate_json(text)
    except ValidationError as exc:
        raise NightlySettingsError(f"{path(state_dir)} does not parse: {exc}") from exc
    if found.schema_version > SCHEMA_VERSION:
        raise NightlySettingsError(f"{path(state_dir)} was written by a newer mnm: `mnm update`.")
    return found


def changed(current: NightlySettings, assignment: str) -> NightlySettings:
    """`clean=30d` or `clean=off` applied; NightlySettingsError for anything else."""
    key, sep, value = assignment.partition("=")
    if not sep or key.strip() != "clean":
        raise NightlySettingsError(
            f"{assignment!r}: the nightly has one setting, clean=<age> (30d, 2w) or clean=off."
        )
    value = value.strip().lower()
    if value in OFF:
        return current.model_copy(update={"clean": None})
    try:
        parse_age(value)
    except ValueError as exc:
        raise NightlySettingsError(f"clean: {exc}.") from exc
    return current.model_copy(update={"clean": value})


def save(state_dir: Path, settings: NightlySettings) -> Path:
    target = path(state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{FILE}.part")
    partial.write_text(settings.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(target)
    return target
