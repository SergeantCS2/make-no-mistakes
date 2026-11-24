"""`MNM_GPU` out of the services' units.

An older `mnm service install` wrote a non-default `MNM_GPU` into the web, router, nightly
and backup units as `Environment=`, which pydantic-settings ranks above the secrets file, and
`mnm update` carried it into the new units: the doctor's and `mnm machine adopt`'s advice
("set it in the secrets file") never reached the router. The units now read it from the
secrets file like every other setting. An installed unit that still carries it is migrated
at the next `mnm service install` (and so at `mnm update`, which runs the new code's
install): the value goes to the secrets file when the file has none, then leaves every
unit; each step said in the output."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from makenomistakes.service import unquoted

NAME = "MNM_GPU"
PATTERN = "makenomistakes-*.service"  # the web's, the router's, the nightly's, the backup's


def _carried(text: str) -> str | None:
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "Environment":
            name, _, setting = unquoted(value).partition("=")
            if name == NAME:
                return setting
    return None


def _without(text: str) -> str:
    kept = [
        line
        for line in text.splitlines(keepends=True)
        if not (line.startswith("Environment=") and _carried(line) is not None)
    ]
    return "".join(kept)


def move_out(unit_dir: Path, secrets: Path) -> list[str]:
    """Take `MNM_GPU` out of every installed mnm unit; what was done, worded (none: nothing
    carried it)."""
    found: dict[Path, str] = {}
    for unit in sorted(unit_dir.glob(PATTERN)) if unit_dir.is_dir() else []:
        try:
            value = _carried(unit.read_text(encoding="utf-8"))
        except OSError:
            continue
        if value is not None:
            found[unit] = value
    if not found:
        return []
    from makenomistakes.setup import write_setting  # noqa: PLC0415 - setup imports service

    said: list[str] = []
    value = next(iter(found.values()))
    saved = dotenv_values(secrets).get(NAME) if secrets.exists() else None
    if saved is None or not saved.strip():
        write_setting(secrets, NAME, value)
        said.append(
            f"{NAME}={value} moved from the services' units to {secrets}: the web UI, the "
            f"router and the timers read it there now, like every other setting."
        )
    elif saved.strip() != value:
        said.append(
            f"The services' units carried {NAME}={value}, which outranked {secrets}'s "
            f"{NAME}={saved.strip()}; the file's value is the one they read now."
        )
    for unit, _value in found.items():
        unit.write_text(_without(unit.read_text(encoding="utf-8")), encoding="utf-8")
    names = ", ".join(unit.name for unit in found)
    said.append(f"{NAME} left the units ({names}).")
    return said


def shell_note(secrets: Path) -> str | None:
    """This shell's `MNM_GPU`, when it differs from what the services read (the file's, or
    the default 0): a service never sees a shell's variables."""
    shell = os.environ.get(NAME, "").strip()
    if not shell:
        return None
    saved = (dotenv_values(secrets).get(NAME) if secrets.exists() else None) or "0"
    if shell == saved.strip():
        return None
    return (
        f"This shell's {NAME}={shell} does not reach the services: they read {secrets}'s "
        f"({NAME}={saved.strip()}). `mnm setup` or an edit of that file changes it, then "
        f"`mnm service install`."
    )
