"""Which llama.cpp build an architecture needs: the merge that brought it, and what the
installed image says against it. The plan is made either way (the plan is the box's, the
engine a separate question); the line says whether the image is known to lack it,
or whether only a start will tell (an image built after the merge day may still predate it:
the official images follow master by hours to days).

The table is keyed by engine and architecture (`SUPPORT`): a merge holds for every
backend's build of that upstream (llama.cpp's CUDA, ROCm, Vulkan and CPU images build from the
same commit), and the day it is judged against is the box's own image's (`installed_image` is
given the tag of the box's backend: a ROCm image built later than the CUDA one is judged on
its own day)."""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.hub.facts import Engine
from makenomistakes.supervisor import DockerSupervisor, LocalImage, SupervisorError


@dataclass(frozen=True)
class Merge:
    commit: str
    day: str  # ISO date, compared with the image's `Created` day
    pr: int

    @property
    def words(self) -> str:
        return f"llama.cpp {self.commit} of {self.day} (PR #{self.pr})"


# Architectures newer than images a box may hold, with the merge that brought each one, per
# engine.
SUPPORT: dict[tuple[Engine, str], Merge] = {
    (Engine.LLAMACPP, "deepseek41"): Merge(commit="74bca39", day="2026-10-25", pr=31086)
}
# llama.cpp's, by architecture
MERGES: dict[str, Merge] = {
    arch: merge for (engine, arch), merge in SUPPORT.items() if engine is Engine.LLAMACPP
}
UPDATE = "`mnm engines update llamacpp` once an image carries it"


def architecture_line(architecture: str, image: LocalImage | None) -> str | None:
    """`deepseek41 needs llama.cpp 74bca39 of 2026-10-25 …`, or None for an older one."""
    merge = MERGES.get(architecture)
    if merge is None:
        return None
    needs = f"{architecture} needs {merge.words}"
    if image is None or not image.created_day:
        return (
            f"{needs}: unverified until started (no llama.cpp image here says when it was built)."
        )
    if image.created_day < merge.day:
        return (
            f"{needs}; the installed llama.cpp image was built {image.created_day}, before "
            f"it: planned anyway, {UPDATE}."
        )
    built = image.created_day
    return f"{needs}; the installed llama.cpp image was built {built}: unverified until started."


def installed_image(architecture: str, image: str) -> LocalImage | None:
    """The local llama.cpp image, asked only for an architecture in `MERGES`; None when
    Docker cannot say (not installed, not running, not pulled)."""
    if architecture not in MERGES:
        return None
    try:
        return DockerSupervisor().image(image)
    except (SupervisorError, OSError):
        return None
