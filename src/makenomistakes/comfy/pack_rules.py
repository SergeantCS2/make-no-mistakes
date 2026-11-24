"""What a node pack's files may be, whichever source they come from (the Registry's zip or
mnm's zip of a GitHub commit): its size and file count as the build unpacks it, and its
pip requirements as the build installs them. One place, so the approval page refuses what
the build would."""

from __future__ import annotations

import re

from makenomistakes.comfy.registry import PackError

MAX_FILES = 5000
MAX_UNPACKED = 400 * 2**20  # WanVideoWrapper unpacks to 47 MB, TTS-Audio-Suite v5.6.0 to 103.5
_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._ -]+\])?\s*"
    r"((===?|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!_-]+\s*,?\s*)*"
    r"""(;[A-Za-z0-9_.\s'"<>=!~(),]*)?$"""  # an environment marker's characters only
)


def requirements(text: str, pack: str) -> list[str]:
    """A pack's pip requirements: names, versions and markers; never a URL, a file or an
    option (those reach past the pack's pin)."""
    found = []
    for raw in text.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line:
            continue
        if not _REQUIREMENT.match(line) or "://" in line or "--" in line:
            raise PackError(
                f"{pack} asks pip for {line!r}, which is not a plain package name: mnm installs "
                "only named packages from PyPI."
            )
        found.append(line)
    return found
