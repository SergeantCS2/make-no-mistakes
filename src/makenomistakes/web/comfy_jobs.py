"""ComfyUI's jobs on the web, one at a time: System → ComfyUI's install, start, stop, rollback
and uninstall, and Node packs' approvals, install and rollback. A packs install building for
minutes beside a ComfyUI rollback would save a ledger from before the rollback and undo it, so
one check refuses any of them, at once and in words, while another runs here or another mnm
holds ComfyUI's lock or builds its packs."""

from __future__ import annotations

from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.comfy.pack_build import packs_building
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import Job, JobKind, JobRegistry, Work

COMFY_JOB = "comfyui"
PACKS_JOB = "comfy-packs"
ONE_AT_A_TIME = (COMFY_JOB, PACKS_JOB)


def start_comfy_job(
    jobs: JobRegistry,
    engine: ComfyEngine,
    key: str,
    title: str,
    work: Work,
    keep_lines: int = 0,
) -> Job:
    """Start `work` as ComfyUI's job under `key` (`COMFY_JOB` or `PACKS_JOB`), refused while
    any of either kind runs here, or another mnm starts, stops or changes ComfyUI (its lock)
    or builds its packs (`packs_building`): a click never hangs behind minutes of work."""

    def elsewhere() -> None:
        with engine.try_locked() as free, packs_building(engine).attempt() as idle:
            if not (free and idle):
                raise InstanceError(
                    "ComfyUI is busy in another mnm (a start, a stop or a packs build in a "
                    f"terminal); {title} once that ends."
                )

    return jobs.start_alone(
        JobKind.ENGINES,
        key,
        work,
        among=ONE_AT_A_TIME,
        then=title,
        check=elsewhere,
        label=title,
        keep_lines=keep_lines,
    )
