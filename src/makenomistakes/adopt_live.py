"""`mnm machine adopt`'s real questions, shared by the CLI and the web's Adopt button:
the router idle as the nightly asks it, this box read through the probe, the RAM probe, the
budget `mnm replan --all` plans on, downloads through the instance service, tunes through
the router's tuner (else its queue file), the router's journal line and the notice."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from makenomistakes import machine
from makenomistakes.doctor_machine import gpu_setting
from makenomistakes.hardware.fingerprint import MachineFingerprint
from makenomistakes.hardware.machine_reads import Unread
from makenomistakes.hardware.ram_bandwidth import measure_all
from makenomistakes.hub.facts import CHAT_LIKE, Engine
from makenomistakes.instances import InstanceService
from makenomistakes.machine_adopt import AdoptDeps, tune_body
from makenomistakes.nightly.gate import gate_for, wait
from makenomistakes.notify import Notice, notifier
from makenomistakes.plan_budget import planning_budget
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import router_adopted, router_tune
from makenomistakes.tune.apply import key_or_none, stale_word
from makenomistakes.tune.orphans import Refused, queue_speed
from makenomistakes.tune.session import untunable
from makenomistakes.tune.store import Queued, TuneStore, TuneStoreError

IDLE_LIMIT_SECONDS = 60.0  # how long adopt waits for the router to be idle
HOLDER = "`mnm machine adopt`"  # who the router's journal says holds the tuner off the card


def live_deps(
    service: InstanceService, say: Callable[[str], None], ask: Callable[[str], bool]
) -> AdoptDeps:
    settings = service.settings
    gate = gate_for(settings, who=HOLDER)
    held: list[bool] = []  # the wait took the tuner's hold: `release` lets it go

    def wait_idle() -> str | None:
        waited = wait(gate, 0.0, IDLE_LIMIT_SECONDS, say=say, who="Adopt")
        if waited.held:
            held.append(True)
        return waited.gave_up

    def release() -> None:
        if held:
            held.clear()
            gate.hold(0)

    def read_box() -> MachineFingerprint | Unread:
        snapshot = service.memory()
        return machine.read_now(snapshot.box_cards, snapshot.memory)

    def tell(text: str) -> None:
        with contextlib.suppress(Exception):  # best effort: the adoption stands
            notifier(settings).send(Notice("mnm machine adopt", text))

    return AdoptDeps(
        service=service,
        wait_idle=wait_idle,
        release=release,
        read_box=read_box,
        measure=measure_all,
        budget=lambda: planning_budget(service),
        ask=ask,
        download=lambda recipe: _download(service, recipe, say),
        queue_tune=queue_tune(service),
        tell_router=lambda counts: router_adopted(settings, counts),
        notify=tell,
        say=say,
        gpu=gpu_setting(settings),
    )


def _download(service: InstanceService, recipe: Recipe, say: Callable[[str], None]) -> None:
    said: list[str] = []

    def progress(line: str) -> None:
        if not said or said[-1] != line:
            said.append(line)
            say(f"  {line}")

    service.download(recipe, progress)


def queue_tune(service: InstanceService) -> Callable[[Recipe], str]:
    """A tune for a llama.cpp chat-like instance whose tune is stale or absent: through the
    router's tuner when it runs (`POST /__mnm/tune/<name>`), else its queue file, as `mnm
    tune` without a router does. The words say which, or why none."""
    store = TuneStore(service.settings.expanded_state_dir())

    def queue(recipe: Recipe) -> str:
        if recipe.engine is not Engine.LLAMACPP or recipe.modality not in CHAT_LIKE:
            return "nothing to tune"
        if untunable(recipe) is not None:
            return f"after its download (`mnm tune {recipe.name}`)"
        key = key_or_none(service, recipe, None) if recipe.tuned is not None else None
        if key is not None and stale_word(recipe, key) is None:
            return "tuned on this box already"
        asked = router_tune(service.settings, recipe.name, tune_body())
        if asked is not None:
            code, said = asked
            return "queued" if code < 400 else f"not queued: {said}"
        return _queued_here(service, store, Queued(name=recipe.name, reason="adopted"))

    return queue


def _queued_here(service: InstanceService, store: TuneStore, entry: Queued) -> str:
    """The queue file without a router; a quality tune waiting there is never replaced, and a
    mark a stopped run left is settled first (`tune.orphans.queue_speed`, its words kept)."""
    said: list[str] = []
    try:
        if isinstance(queued := queue_speed(service, store, entry, said.append), Refused):
            return f"not queued: {queued.why or queued.words}"  # the row names the instance
    except (TuneStoreError, OSError) as exc:
        return f"not queued: {exc}"
    settled = f" ({' '.join(said)})" if said else ""
    return f"queued (the router tunes it once it runs){settled}"
