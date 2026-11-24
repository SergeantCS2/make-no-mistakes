"""How to make a folder mnm writes to, worded once.

`mnm setup` makes it with `sudo mkdir -p` and `sudo chown`; a run that cannot make its folder
and the doctor's warning print the same two commands, so nobody has to find them in setup's table.
"""

from __future__ import annotations

import getpass
import os
import shlex
from pathlib import Path


def folder_hint(folder: Path, setting: str) -> str | None:
    """What is wrong with a folder mnm must write into and how to put it right, whole; None
    when it is a writable folder."""
    where = shlex.quote(str(folder))
    who = shlex.quote(getpass.getuser())
    if folder.is_dir():
        if os.access(folder, os.W_OK):
            return None
        return (
            f"{folder} is not writable: make it yours once with `sudo chown {who} {where}` "
            f"(or run `mnm setup`), or set {setting} to a folder you own."
        )
    if folder.exists():
        return f"{folder} is not a folder: set {setting} to a folder you own."
    return (
        f"{folder} does not exist yet: make it once with `sudo mkdir -p {where} && sudo chown "
        f"{who} {where}` (or run `mnm setup`), or set {setting} to a folder you own."
    )
