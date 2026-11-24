"""Would this instance fit the VRAM that is free right now? Pure functions over a snapshot.

The plan decided the recipe against the budget at `add` time; this answers the question the
instances page asks every time it renders, with the GPU as it is at that moment, using the
same headroom rule the planner used (`fit.estimate.headroom_level`).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field

from makenomistakes.fit.estimate import (
    GIB,
    FitLevel,
    headroom_level,
)
from makenomistakes.fit.image_memory import restreamed_bytes
from makenomistakes.hardware.probe import RAM_INDEX, Gpu, MemorySnapshot

COMFYUI = "ComfyUI"  # in a budget's holders: every start frees it first (`release_vram`)


class FitNow(StrEnum):
    RUNNING = "running"
    FITS = "fits"
    NEAR_LIMIT = "near limit"
    NO_FIT = "won't fit"
    TAKES_TURNS = "takes turns"  # a parked swap member: its next request makes the room
    ON_REQUEST = "on request"  # parked and short of room: its first request makes it (router)
    UNKNOWN = "unknown"


_LEVELS = {
    FitLevel.COMFORTABLE: FitNow.FITS,
    FitLevel.TIGHT: FitNow.NEAR_LIMIT,
    FitLevel.NO_FIT: FitNow.NO_FIT,
}


class ImageFallbacks(BaseModel):
    """What an sd.cpp instance can fall back to when the whole estimate does not fit
    (engine-specific figures kept out of the shared estimate); empty for every other engine."""

    # Decoding the whole image at once: the same settings with a tiled decode. sd.cpp falls
    # back to making room (evicting weights, reloading them per image) rather than failing,
    # so a fit here means "starts, slower", not "won't fit".
    tiled_bytes: int | None = None
    # Streamed (`offload-to-cpu`): the compute and VAE buffers that must fit on their own;
    # the weights stream in as room allows and are re-read when it does not.
    streamed_floor: int | None = None


class VramEstimate(BaseModel):
    """What an instance puts on the GPU, and where that number came from (or why there is
    none): the verdict note says which, so "unknown" always comes with its real cause."""

    bytes: int | None
    basis: str
    image: ImageFallbacks = Field(default_factory=ImageFallbacks)  # sd.cpp's own
    # The bytes on each card it runs on, when `MNM_GPU` names
    # several or it runs elsewhere than the first; empty: all of it on the first
    on: tuple[tuple[int, int], ...] = ()


class Verdict(BaseModel):
    level: FitNow
    note: str
    # running instances whose stopping lets this one start (largest first), when it won't fit
    stop_first: tuple[str, ...] = ()
    # running instances never stopped for room (`workflows.held_back`: pinned, a member of a
    # set in use) whose VRAM this one needs: the others alone do not make the room
    kept: tuple[str, ...] = ()
    # judged with what ComfyUI gives back at a start counted free (`room_verdicts.counted`)
    comfy_freed: int = 0
    holds: int | None = None  # a running one's estimate: what it holds on the card now

    @property
    def words(self) -> str:
        """The *Fits now* cell (CLI, web, sets): a running instance's is the room it holds,
        "loaded, 5.6 GiB" ("running" would read as a state word, the State column's),
        every other its level."""
        if self.level is FitNow.RUNNING:
            return f"loaded, {_gib(self.holds)}" if self.holds else "loaded"
        return self.level.value

    @property
    def warns(self) -> bool:
        """Start should ask first: the estimate says it will not fit right now. A start by
        hand asks for an *on request* one too: only the router makes its room by itself."""
        return self.level in (FitNow.NO_FIT, FitNow.ON_REQUEST)


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def _before_the_numbers(
    estimate: VramEstimate, snapshot: MemorySnapshot, running: bool
) -> Verdict | None:
    """The verdicts that need no comparison: running, no estimate, no readable GPU."""
    needed = estimate.bytes
    if running:
        if needed is None:
            return Verdict(level=FitNow.RUNNING, note=f"Running; no estimate: {estimate.basis}.")
        return Verdict(
            level=FitNow.RUNNING,
            note=f"Using about {_gib(needed)} of {_of(estimate, snapshot)}.",
            holds=needed,
        )
    if needed is None:
        return Verdict(level=FitNow.UNKNOWN, note=f"No estimate: {estimate.basis}.")
    if snapshot.gpu is None and not estimate.on:  # placed: each place judged (`card_room`)
        return Verdict(
            level=FitNow.UNKNOWN, note=f"Cannot read the GPU right now: {snapshot.gpu_error}."
        )
    return None


RAM_KEPT = 2 * GIB  # left to WSL, Docker and the engines' own processes when weights stream


def ram_short(in_ram: int, snapshot: MemorySnapshot) -> Verdict | None:
    """Weights streamed from RAM need the RAM: a box that swaps them is not a slower start.
    None when they fit, or when RAM cannot be read (that never refuses)."""
    available = snapshot.memory.available_bytes if snapshot.memory else None
    if available is None or in_ram <= available - RAM_KEPT:
        return None
    return Verdict(
        level=FitNow.NO_FIT,
        note=f"Streams about {_gib(in_ram)} of weights from RAM and {_gib(available)} of RAM "
        "is available. Stop another instance, or wait until more RAM is free.",
    )


def _streamed_verdict(needed: int, floor: int, free: int, snapshot: MemorySnapshot) -> Verdict:
    """A streamed image instance whose whole estimate is above the free VRAM: it still runs
    when its buffers alone fit and RAM holds its weights (re-reading them each step), else
    nothing does."""
    short = ram_short(needed - floor, snapshot)
    if short is not None:
        return short
    if floor > free:
        return Verdict(
            level=FitNow.NO_FIT,
            note=f"Even with the weights streamed from RAM its buffers need about {_gib(floor)} "
            f"and only {_gib(free)} is free. Stop another instance first.",
        )
    return Verdict(
        level=FitNow.NEAR_LIMIT,
        note=f"Streams its weights from RAM: it takes the {_gib(free)} free and re-reads about "
        f"{_gib(restreamed_bytes(needed, free))} each step (slower); nothing else fits "
        "beside it while it runs.",
    )


def _of(estimate: VramEstimate, snapshot: MemorySnapshot) -> str:
    """What the instance's room is: the RAM's when it sits there (placed there, or judged
    on the RAM alone), else the card's."""
    placed = [i for i, _ in estimate.on]
    gpu = snapshot.gpu
    if placed:
        return "RAM" if all(i == RAM_INDEX for i in placed) else "VRAM"
    return "RAM" if gpu is not None and gpu.in_ram else "VRAM"


def fit_now(estimate: VramEstimate, snapshot: MemorySnapshot, running: bool) -> Verdict:
    """`estimate` is what the instance puts on the GPU (`InstanceService.vram_estimate`), or
    in the RAM for an instance on the processor's image (placed at `RAM_INDEX`)."""
    early = _before_the_numbers(estimate, snapshot, running)
    if early is not None:
        return early
    if estimate.on:  # on other or several cards: each card judged on its own
        from makenomistakes.card_room import fit_on_cards  # noqa: PLC0415 - it imports this

        return fit_on_cards(estimate, snapshot)
    needed = estimate.bytes
    assert needed is not None and snapshot.gpu is not None  # `_before_the_numbers` answered else
    free = snapshot.gpu.vram_free_bytes
    level = _LEVELS[headroom_level(needed, free)]
    if estimate.image.streamed_floor is not None and level is FitNow.NO_FIT:
        return _streamed_verdict(needed, estimate.image.streamed_floor, free, snapshot)
    tiled = estimate.image.tiled_bytes
    if (
        level is FitNow.NO_FIT
        and tiled is not None
        and _LEVELS[headroom_level(tiled, free)]
        in (
            FitNow.FITS,
            FitNow.NEAR_LIMIT,
        )
    ):
        return Verdict(
            level=FitNow.NEAR_LIMIT,
            note=f"Needs about {_gib(needed)} to decode a whole image at once and "
            f"{_gib(free)} is free: sd.cpp then frees weights for the decode and reloads them "
            f"for the next image (slower). With a tiled decode it needs about {_gib(tiled)} "
            "(`vae-tiling=on`).",
        )
    if level is FitNow.FITS:
        note = f"Needs about {_gib(needed)}; {_gib(free)} free right now."
    elif level is FitNow.NEAR_LIMIT:
        note = (
            f"Needs about {_gib(needed)} of the {_gib(free)} free: it may start, but long "
            "prompts or a second instance will run out of memory."
        )
    else:
        where = "the RAM" if snapshot.gpu.in_ram else "the card"
        note = (
            f"Needs about {_gib(needed)} but only {_gib(free)} is free. Stop another instance "
            f"first, or wait until more of {where} is free; starting anyway will likely fail with "
            "out-of-memory."
        )
    return Verdict(level=level, note=note)


def streaming_member(waiting: list[tuple[str, VramEstimate]]) -> str | None:
    """The one member of a set that streams its weights when the set does not fit whole: the
    one whose streaming moves the most weights off the card, ties broken by name, so the
    verdict and the start order (`WorkflowService.start` starts it last) name the same one."""
    can_stream = [
        (name, (e.bytes or 0) - (e.image.streamed_floor or 0))
        for name, e in waiting
        if e.image.streamed_floor is not None
    ]
    if not can_stream:
        return None
    return max(can_stream, key=lambda ne: (ne[1], ne[0]))[0]


def fit_together(
    waiting: list[tuple[str, VramEstimate]],
    snapshot: MemorySnapshot,
    beside: tuple[str, ...] = (),
    owner: str = "",
) -> Verdict:
    """`_together`'s verdict; one that does not fit names the smallest member to leave out.
    `beside` names `owner`'s (a set's) members already running, for the wording."""
    from makenomistakes.card_room import pooled, together_on_cards  # noqa: PLC0415 - it imports this

    def judge(here: list[tuple[str, VramEstimate]], card: MemorySnapshot) -> Verdict:
        return _together(here, card, _needs(here, beside, owner))

    each = together_on_cards(waiting, snapshot, judge)  # Card by card
    if each is not None:
        return each
    snapshot = pooled(snapshot)  # several cards, nothing placed: their free VRAM together
    verdict = _together(waiting, snapshot, _needs(waiting, beside, owner))
    drop = _leave_out(waiting, snapshot) if verdict.warns else None
    if drop is None:
        return verdict
    name, size = drop
    note = f"{verdict.note} Leaving out {name} (about {_gib(size)}) lets the rest fit."
    return verdict.model_copy(update={"note": note})


def _needs(waiting: list[tuple[str, VramEstimate]], beside: tuple[str, ...], owner: str) -> str:
    """`coder8b needs about 7.4 GiB beside the desk's running members (qwen3-8b, embed)`;
    `a, b need about 9.2 GiB together` (one member never reads "need … together")."""
    names = ", ".join(name for name, _ in waiting)
    size = _gib(sum(e.bytes or 0 for _, e in waiting))
    words = f"{names} needs about {size}" if len(waiting) == 1 else f"{names} need about {size}"
    if not beside:
        return words if len(waiting) == 1 else f"{words} together"
    mates = "running member" if len(beside) == 1 else "running members"
    return f"{words} beside {owner}'s {mates} ({', '.join(beside)})"


def _together(
    waiting: list[tuple[str, VramEstimate]], snapshot: MemorySnapshot, together: str = ""
) -> Verdict:
    """Whether instances that are not running fit the free VRAM together (`name`, estimate
    each, every estimate known, the GPU readable): their sum with the planner's headroom
    rule, else their floors as `fit_now` reads each one alone, a tiled decode or one member
    streaming its weights. A streamed member takes the VRAM that is left (`fit_now`), so only
    one streams (`streaming_member`), started last. Unknown when the GPU cannot be read."""
    if snapshot.gpu is None:
        return Verdict(
            level=FitNow.UNKNOWN, note=f"GPU not readable ({snapshot.gpu_error}): run `mnm doctor`."
        )
    free = snapshot.gpu.vram_free_bytes
    needed = sum(e.bytes or 0 for _, e in waiting)
    level = headroom_level(needed, free)
    together = together or _needs(waiting, (), "")
    if level is FitLevel.COMFORTABLE:
        return Verdict(level=FitNow.FITS, note=f"{together}; {_gib(free)} is free.")
    if level is FitLevel.TIGHT:
        return Verdict(
            level=FitNow.NEAR_LIMIT,
            note=f"{together} with {_gib(free)} free: it fits, with little room for a long "
            "prompt or a big image.",
        )
    return _at_the_floors(waiting, snapshot, together, free)


def _leave_out(
    waiting: list[tuple[str, VramEstimate]], snapshot: MemorySnapshot
) -> tuple[str, int] | None:
    """The smallest member whose absence lets the others start (else a set that won't fit
    says nothing of what would)."""
    if len(waiting) < 2:
        return None
    for name, estimate in sorted(waiting, key=lambda ne: (ne[1].bytes or 0, ne[0])):
        rest = [(n, e) for n, e in waiting if n != name]
        if not _together(rest, snapshot).warns:
            return name, estimate.bytes or 0
    return None


def _at_the_floors(
    waiting: list[tuple[str, VramEstimate]], snapshot: MemorySnapshot, together: str, free: int
) -> Verdict:
    """The sum does not fit: the floors may (`fit_together`). Only one member streams
    (`streaming_member`), priced at its floor and started last; the others count whole (an
    image one at its tiled decode)."""
    can_stream = [(n, e) for n, e in waiting if e.image.streamed_floor is not None]
    streams = streaming_member(waiting)
    last = next(((n, e) for n, e in waiting if n == streams), None)
    floor = sum(
        (e.image.streamed_floor if n == streams else e.image.tiled_bytes or e.bytes) or 0
        for n, e in waiting
    )
    if headroom_level(floor, free) is FitLevel.NO_FIT:
        one = " Only one member can stream its weights at a time." if len(can_stream) > 1 else ""
        return Verdict(
            level=FitNow.NO_FIT,
            note=f"{together} and only {_gib(free)} is free. Start fewer members, lower a "
            f"context (`mnm set <name> ctx=…`), or pick smaller quants.{one}",
        )
    if last is not None:
        name, estimate = last
        short = ram_short((estimate.bytes or 0) - (estimate.image.streamed_floor or 0), snapshot)
        if short is not None:
            return short.model_copy(update={"note": f"{name}: {short.note}"})
    return Verdict(
        level=FitNow.NEAR_LIMIT,
        note=f"{together} at full speed and {_gib(floor)} with weights streamed from RAM or a "
        f"tiled decode; {_gib(free)} is free, so they start, slower"
        + (f" ({last[0]} last: it takes what is left)." if last is not None else "."),
    )


def with_room(
    verdict: Verdict,
    estimate: VramEstimate,
    snapshot: MemorySnapshot,
    held: dict[str, int],
    kept: frozenset[str] = frozenset(),
    pinned: frozenset[str] = frozenset(),
    held_on: dict[str, dict[int, int]] | None = None,
) -> Verdict:
    """A won't-fit verdict names the fewest running instances (largest first) whose VRAM would
    let this one start, comfortably if stopping more gets there. Those in `kept` are never
    offered as room. A kept one is named (`Verdict.kept`) only when its VRAM is needed: the
    others alone do not make the room and it does; one holding VRAM that would not help
    anyway (a game holds the rest) is not the reason, and is not named."""
    gpu = snapshot.gpu
    if not verdict.warns or not held or (gpu is None and held_on is None):
        return verdict
    if held_on is not None:  # what the places the instance sits on use (cards, the RAM)
        places = [p for i, _ in estimate.on if (p := snapshot.card(i)) is not None]
        used = sum(p.vram_used_bytes for p in places)
    else:
        assert gpu is not None
        used = gpu.vram_used_bytes
    largest_first = sorted((n for n in held if held[n]), key=lambda n: (-held[n], n))
    if held_on is not None:  # Only holders on a card the instance is short on
        from makenomistakes.card_room import short_cards  # noqa: PLC0415 - it imports this

        short = short_cards(estimate, snapshot)
        largest_first = [n for n in largest_first if any(held_on[n].get(i) for i in short)]
    offered = [n for n in largest_first if n not in kept]
    trial = _trial(snapshot, gpu, held, held_on)
    starts = _fewest(offered, estimate, trial, comfortably=True)
    if starts:
        return verdict.model_copy(
            update={
                "stop_first": starts,
                "note": f"{verdict.note} Parking {', '.join(starts)} (about "
                f"{_gib(_freed(starts, held, used))}) makes room.",
            }
        )
    needed = _fewest(largest_first, estimate, trial, comfortably=False)
    keeps = tuple(n for n in needed if n in kept)
    if not keeps:  # every holder together is not enough: something else holds the card
        return verdict
    room = tuple(n for n in needed if n not in kept)
    words = ", ".join(
        f"pinned {n} (about {_gib(held[n])})"
        if n in pinned
        else f"{n} (about {_gib(held[n])}, in a set in use)"
        for n in keeps
    )
    held_by = f"the {_of(estimate, snapshot)} held by {words}, never parked for room"
    said = (
        f"Parking {', '.join(room)} (about {_gib(_freed(room, held, used))}) "
        f"is not enough: it also needs {held_by}."
        if room
        else f"It needs {held_by}."
    )
    return verdict.model_copy(
        update={"stop_first": room, "kept": keeps, "note": f"{verdict.note} {said}"}
    )


Trial = Callable[[list[str]], MemorySnapshot]  # the card(s) with these instances stopped


def _trial(
    snapshot: MemorySnapshot,
    gpu: Gpu | None,
    held: dict[str, int],
    on: dict[str, dict[int, int]] | None,
) -> Trial:
    """The snapshot as it would read with some holders stopped: one card less their bytes,
    or (`card_room.room_view`) each card less what each holder holds on it."""

    def stopped(chosen: list[str]) -> MemorySnapshot:
        if on is not None:
            from makenomistakes.card_room import lighter_on  # noqa: PLC0415 - it imports this

            return lighter_on(snapshot, [on.get(n, {}) for n in chosen])
        assert gpu is not None  # `with_room` judges one card only with one read
        used = max(gpu.vram_used_bytes - sum(held[n] for n in chosen), 0)
        return snapshot.model_copy(
            update={"gpus": [gpu.model_copy(update={"vram_used_bytes": used})]}
        )

    return stopped


def _fewest(
    names: list[str], estimate: VramEstimate, trial: Trial, comfortably: bool
) -> tuple[str, ...]:
    """The first of `names` (in order) whose stopping together lets the estimate start:
    `comfortably` walks on to a comfortable fit when more of them get there; else the first
    that fits at all. Empty when all of them together do not."""
    chosen: list[str] = []
    found: tuple[str, ...] = ()
    for name in names:
        chosen.append(name)
        level = fit_now(estimate, trial(chosen), running=False).level
        if level is FitNow.FITS:
            return tuple(chosen)
        if level is not FitNow.NO_FIT and not found:
            found = tuple(chosen)
            if not comfortably:
                return found
    return found


def _freed(names: tuple[str, ...], held: dict[str, int], used: int) -> int:
    return min(sum(held[n] for n in names), used)
