"""The doctor's *Machine* line: this box in one line and when mnm first saw it (OK), or
that the hardware changed since saved plans were made, naming what changed card by card
(WARN, ending "`mnm machine adopt` re-fits them"). Taking it records the box
(`machine.observe`); a box that could not be read is a WARN that says why, and nothing is
recorded. When the card at an index `MNM_GPU` names is another card now than on the box the
plans were made on, the line says so ("MNM_GPU=1 now selects RTX 3090, was RTX 4070")."""

from __future__ import annotations

from makenomistakes.backend import Backend, not_taken
from makenomistakes.backend import words as backend_words
from makenomistakes.config import Settings
from makenomistakes.doctor_types import Check, Status
from makenomistakes.gpu_offer import gpu_note
from makenomistakes.hardware.box_facts import BoxFacts
from makenomistakes.hardware.fingerprint import MachineFingerprint
from makenomistakes.hardware.probe import HardwareReport
from makenomistakes.machine import SHOW, Observed, drift, observe
from makenomistakes.recipes import Recipe

NAME = "Machine"
REMEDY = (
    "`mnm machine adopt --dry-run` shows what it would do (nothing is saved); `mnm machine "
    f"adopt` plans every instance again for this box, asking before any download; {SHOW} says "
    "what changed and which instances"
)


def gpu_setting(settings: Settings) -> str | None:
    """`MNM_GPU` as the user set it; None when unset (card 0)."""
    return settings.gpu if "gpu" in settings.model_fields_set else None


def machine_check(
    settings: Settings, report: HardwareReport, box: BoxFacts | None, recipes: list[Recipe]
) -> Check:
    observed = observe(settings, report.box_cards, report.memory, box)
    return judged(observed, recipes, gpu_setting(settings), settings.backend)


def judged(
    observed: Observed, recipes: list[Recipe], gpu: str | None = None, chosen: str | None = None
) -> Check:
    """`chosen`: `MNM_BACKEND` as set (the line says when an APU does not take it yet)."""
    if observed.unread is not None:
        return Check(name=NAME, status=Status.WARN, detail=f"not read: {observed.unread}")
    file = observed.file
    if file is None:
        return Check(name=NAME, status=Status.WARN, detail=f"not recorded: {observed.error}")
    found = drift(file, recipes)
    if found is None:  # recorded just now: never
        return Check(name=NAME, status=Status.WARN, detail="not recorded")
    changed = found.words()
    selected = gpu_note(gpu, found.planned_box(file), found.here).changed
    backend = backend_line(found.here, chosen, gpu)
    if changed is not None:
        detail = "; ".join([changed, *selected, *backend])
        return Check(name=NAME, status=Status.WARN, detail=detail, remedy=REMEDY)
    return Check(name=NAME, status=Status.OK, detail="; ".join([found.here_words(), *backend]))


def backend_line(
    here: MachineFingerprint, chosen: str | None = None, gpu: str | None = None
) -> list[str]:
    """`backend: CUDA 13`, or `backend: Vulkan on Mesa's software device: slow`; none
    when it was not decided (`MNM_BACKEND` refused: its own FAIL line says why). An APU's
    `MNM_BACKEND=rocm` not taken yet says why (`backend.not_taken`)."""
    if here.backend is None:
        return []
    decided = Backend(here.backend)
    line = f"backend: {backend_words(decided, here)}"
    if (why := not_taken(here, chosen, gpu, decided)) is not None:
        line += f" ({why})"
    return [line]
