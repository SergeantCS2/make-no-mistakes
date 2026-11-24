"""The user's start: `mnm start` and the web UI's Start, one sequence.

Under the box's start lock (the router's loads take it too), a swap member makes room in its
group first; then whatever of mnm's still holds the card is refused, stopped (`stop_others`)
or ignored (`anyway`) as asked. VRAM held by anything else is the start's own business.

`stop_others` makes room by the router's rules (`SwapService.to_park` and `park`): never a
pinned instance or a member of a set in use (`workflows.held_back`), which it names in a
refusal instead, and each park waits for the replies in flight.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from makenomistakes.comfy.engine import release_for
from makenomistakes.comfy.image import ComfyBusyError
from makenomistakes.instance_types import InstanceError, Started
from makenomistakes.instances import InstanceService
from makenomistakes.parkable import parkable
from makenomistakes.plan_budget import held_by_mnm
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import Busy
from makenomistakes.swap import HeldByComfyError, Scope, SwapService, start_turn
from makenomistakes.workflows import WorkflowStore


@dataclass(frozen=True)
class OwnerStart:
    started: Started
    parked: list[str] = field(default_factory=list)  # what was parked for room (never off)


def start_for_owner(
    service: InstanceService,
    recipe: Recipe,
    report: Callable[[str], None],
    stop_others: bool = False,
    anyway: bool = False,
    busy: Busy | None = None,
    release: Callable[[Callable[[str], None] | None], bool] | None = None,
) -> OwnerStart:
    """Start `recipe` as the user asked; raises `InstanceError` when mnm's own running
    instances hold the room it needs and neither flag says what to do about them. ComfyUI
    gives its card back first, as for the router's loads (`release`, `release_vram`): the
    verdicts count it free."""
    if stop_others and anyway:
        raise InstanceError("Pass --stop-others or --anyway, not both.")
    name = recipe.name
    stopped: list[str] = []

    def held() -> None:
        if stop_others:
            stopped.extend(make_room(service, name, report, busy))
        elif not anyway:
            refuse_if_held(service, name)

    if recipe.swap_group is None:
        with start_turn(service.settings, name, report):  # one start on the box at a time
            refuse_if_running(service, recipe)
            try:
                (release or release_for(service.settings))(report)
            except ComfyBusyError as exc:
                raise HeldByComfyError(f"{name} is not loaded: {exc}") from exc
            held()
            return OwnerStart(service.start(recipe, report), stopped)
    done = SwapService(service).start(recipe, report, held)
    return OwnerStart(done.started, [*done.parked, *stopped])


def refuse_if_running(service: InstanceService, recipe: Recipe) -> None:
    """A start of a running instance is refused before anything moves (else the start saves
    its start-time upgrades, such as VAE tiling judged against a card the instance itself
    holds, and frees ComfyUI, before Docker says it runs)."""
    if service.status(recipe).running:
        raise InstanceError(
            f"{recipe.name} is already running; its settings apply at the next start "
            f"(`mnm stop {recipe.name} --park`, then `mnm start {recipe.name}`)."
        )


def refuse_if_held(service: InstanceService, name: str) -> None:
    """A start that will not fit beside mnm's running instances is refused: it would run out
    of memory, and auto-fix would shrink the recipe to the space they leave, for good.
    VRAM held by anything else (Windows apps) is the start's own business."""
    verdict = held_by_mnm(service, name)
    if verdict is not None:
        them = "it" if len(verdict.stop_first) == 1 else "them"
        raise InstanceError(
            f"{verdict.note} `mnm start {name} --stop-others` parks {them} first (the next "
            f"request loads {them} again); `--anyway` starts it beside {them} (when it runs out "
            "of memory, auto-fix saves smaller settings)."
        )


def make_room(
    service: InstanceService,
    name: str,
    report: Callable[[str], None],
    busy: Busy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """`--stop-others` and the web's "Make room and start": park what the router's own rules
    let it (`parkable`, the scope `Router.scope` uses too; `SwapService.to_park` picks among
    them as the router does, worked out again here so a stale page or a hand-made form
    cannot stop anything else). When those cannot make the room, the start is refused naming
    what holds it. Returns the names it parked, each after its replies in flight."""
    verdict = held_by_mnm(service, name)
    if verdict is None:  # nothing of mnm's holds the room this start needs
        return []
    free = parkable(service, WorkflowStore(service.settings.expanded_state_dir()), name)
    swap = SwapService(
        service,
        busy=busy,
        sleep=sleep,
        scope=Scope(lambda _r: free or [], makes_way_unpriced=False),
    )
    recipe = service.get(name)
    chosen = swap.to_park(recipe)
    if not chosen:  # what holds the room is what the router never parks
        names = {r.name for r in free or []}
        held = [n for n in verdict.stop_first if n not in names]
        if not held:  # the snapshot moved: `held_by_mnm` and `to_park` each read the card,
            # and between the two reads something else took VRAM, so parking every name the
            # verdict gave no longer makes the room; there is no held-back name to give
            raise InstanceError(
                f"{verdict.note} The room could not be made now (`mnm ls` says what holds the "
                f"card); `mnm start {name} --anyway` starts it beside them."
            )
        pinned = {r.name for r in service.all() if r.pinned}
        raise InstanceError(_held_words(name, held, pinned, readable=free is not None))
    return swap.park_for(recipe, chosen, report)


def _held_words(name: str, held: list[str], pinned: set[str], readable: bool) -> str:
    if not readable:
        return (
            f"{name} needs the room {', '.join(held)} hold, and mnm cannot read the sets "
            "(`mnm sets ls` says why), so it parks nothing to make room. `mnm stop` them by "
            f"hand, or `mnm start {name} --anyway`."
        )
    why = [
        f"{n} (pinned: `mnm set {n} keep=auto` lets it go)"
        if n in pinned
        else f"{n} (in a set in use)"
        for n in held
    ]
    one = len(held) == 1  # "tiny … holds, … stops it", never "stops them"
    those, them = ("it", "it") if one else ("those", "them")
    return (
        f"{name} needs the room that {', '.join(why)} hold{'s' if one else ''}, and "
        f"mnm never parks {those} to make room. `mnm stop {' '.join(held)}` stops {them} by "
        f"hand; `mnm start {name} --anyway` starts {name} beside {them}."
    )
