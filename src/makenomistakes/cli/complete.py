"""Shell completion: `mnm --install-completion` installs Typer's for the shell; an
argument that names an instance completes from the saved instances' names (their files'
names, no recipe read). A completer never raises: its error would print into the prompt."""

from __future__ import annotations

from makenomistakes.config import Settings
from makenomistakes.store import InstanceStore


def instance_names(incomplete: str) -> list[str]:
    """The saved instances whose names start with what is typed; none when the state folder
    cannot be listed or a setting cannot be read (a bad `MNM_*` value is a ValueError)."""
    try:
        names = InstanceStore(Settings().expanded_state_dir()).names()
    except (OSError, ValueError):
        return []
    return sorted(name for name in names if name.startswith(incomplete))
