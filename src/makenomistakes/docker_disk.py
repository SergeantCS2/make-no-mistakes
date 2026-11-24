"""Where Docker keeps its images and how much room is left there: what a build of several GB
checks before it starts (ComfyUI's preflight, the studio's built images). A build that runs
out of room dies minutes in on a containerd path; said first, it is one
sentence with the figures."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.hardware.probe import CommandRunner

GIB = 2**30
SPARE_GIB = 3.0  # the build's layers and pip's unpacking, beyond the finished image
FREE_SOME = (
    "Free some space (`docker system df` shows what Docker holds; `mnm clean` and "
    "`mnm rm --files` give mnm's back)"
)


@dataclass(frozen=True)
class ImageRoom:
    """Docker's root folder and the GiB free there; either None when it could not be read."""

    root: Path | None
    free_gib: float | None

    def short_of(self, image_gib: float) -> float | None:
        """The GiB a build of an `image_gib` image needs when this room is short of it; None
        when it fits or the room could not be read (the build itself says if it runs out)."""
        need = image_gib + SPARE_GIB
        if self.free_gib is None or self.free_gib >= need:
            return None
        return need


def image_room(runner: CommandRunner) -> ImageRoom:
    code, out, _err = runner.run(["docker", "info", "--format", "{{.DockerRootDir}}"], timeout=30)
    root = Path(out.strip()) if code == 0 and out.strip() else None
    try:
        free = shutil.disk_usage(root).free / GIB if root is not None else None
    except OSError:
        free = None
    return ImageRoom(root, free)
