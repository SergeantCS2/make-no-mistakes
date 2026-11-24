r"""A WSL path as Windows' Explorer opens it (`\\wsl.localhost\<distro>\…`), so a folder
such as `/data/outputs` can be found from Windows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import PurePath, PurePosixPath

from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.platform import Platform


def windows_path(
    path: PurePath | str,
    environ: Mapping[str, str] = os.environ,
    platform: Platform | None = None,
) -> str | None:
    """None off WSL, and when WSL did not say its distro's name (`WSL_DISTRO_NAME`, set in
    every WSL shell): the release's own name (/etc/os-release) is often not WSL's, and a
    wrong path is worse than none."""
    where = platform or platforms.current()
    distro = environ.get("WSL_DISTRO_NAME")
    if not where.is_wsl or not distro:
        return None
    parts = [part for part in PurePosixPath(path).parts if part != "/"]
    return "\\\\wsl.localhost\\" + "\\".join([distro, *parts])
