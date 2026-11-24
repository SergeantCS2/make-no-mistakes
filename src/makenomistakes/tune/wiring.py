"""The tuner as one process runs it: the router's loop (`for_router`) and the CLI's
foreground run when no router answers (`foreground`). Both build the same `Tuner`."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.card_jobs import CardGate
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.hardware.runner import SubprocessRunner
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Note
from makenomistakes.quality.perplexity import PerplexityRunner
from makenomistakes.quality.session import QualityTuner
from makenomistakes.recipes.model import TUNE_PREFIX
from makenomistakes.router_client import Traffic, router_busy
from makenomistakes.supervisor import SupervisorError
from makenomistakes.tune.room import copies_room
from makenomistakes.tune.runner import TuneRunner
from makenomistakes.tune.session import Tuner
from makenomistakes.tune.store import TuneStore
from makenomistakes.tune.trial import TrialRunner
from makenomistakes.workflows import WorkflowStore


def tuner_for(
    service: InstanceService, busy: Callable[[str], int], progress: Note | None = None
) -> Tuner:
    state_dir = service.settings.expanded_state_dir()
    room = copies_room(service, WorkflowStore(state_dir), busy)
    trials = TrialRunner(service, room=room, progress=progress)
    store = TuneStore(state_dir)
    perplexity = PerplexityRunner(SubprocessRunner())  # The quality tune's runs
    quality = QualityTuner(service, store, trials, perplexity, room=room, progress=progress)
    return Tuner(service, store, trials, progress, quality=quality)


def for_router(
    service: InstanceService,
    traffic: Callable[[], Traffic],
    busy: Callable[[str], int],
    note: Note,
    gate: CardGate | None = None,
) -> TuneRunner:
    """The loop `mnm router` runs on a thread of its own. A tune's copy left running by a
    router that stopped mid-trial goes first: this router owns every copy, and nobody else
    lists or prices it."""
    clear_copies(service, note)
    return TuneRunner(
        tuner_for(service, busy, note), traffic, comfy_busy_for(service), note, gate=gate
    )


def comfy_busy_for(service: InstanceService) -> Callable[[], str | None]:
    """Why ComfyUI holds the card now (running a workflow, or not saying whether it is), or
    None: the router's card jobs wait for it, and unknown waits too."""
    return ComfyEngine(service.settings).busy_words


def clear_copies(service: InstanceService, note: Note) -> list[str]:
    """Remove every `mnm-tune-*` container a router left (no trial runs while a router
    starts), each one a journal line; one whose tune another live process runs (a terminal's
    `--now`: its trial copy or its quality step) stays, and what Docker cannot say is one line
    too, nothing removed."""
    try:
        found = service.supervisor.names_with(TUNE_PREFIX)
    except SupervisorError as exc:
        note(f"A tune's copy left by an earlier router could not be looked for: {exc}")
        return []
    store = TuneStore(service.settings.expanded_state_dir())
    removed = []
    for name in found:
        if store.run_beside(name.removeprefix(TUNE_PREFIX)) is not None:
            continue  # another process's run (none of this router's runs yet): its own to stop
        try:
            service.supervisor.stop(name)
        except SupervisorError as exc:
            note(f"{name}, a tune's copy nobody runs, could not be removed: {exc}")
            continue
        note(f"Removed {name}, a tune's copy left by a router that stopped mid-trial.")
        removed.append(name)
    return removed


def foreground(service: InstanceService, progress: Note) -> Tuner:
    """`mnm tune <name> --now` with no router: this process runs it (nothing to cut it)."""
    asked = router_busy(service.settings)
    return tuner_for(service, lambda name: asked(service.get(name)), progress)
