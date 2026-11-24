"""How a verdict reads when someone else makes the room: a swap group's mates (`taking_turns`),
the router (`on_request`), and ComfyUI giving its card back before a start (`ComfyRoom`).

After a film ComfyUI holds about 2.3 GiB (its Status panel's estimate), and without this
`mnm sets start pset` refuses "only 9.0 GiB is free" while `mnm ls` reads *near limit* for
models that fit once it frees its card, which every start does first
(`ComfyEngine.release_vram`: the router's, a start by hand, `mnm sets start`). Wherever room
is judged, what that release gives back counts as free, and the verdict says so."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.capacity import FitNow, Verdict, VramEstimate, fit_now, fit_together
from makenomistakes.hardware.probe import MemorySnapshot

if TYPE_CHECKING:
    from makenomistakes.comfy.engine import ComfyEngine
    from makenomistakes.config import Settings

GIB = 2**30


@dataclass(frozen=True)
class ComfyRoom:
    """What a running ComfyUI holds of the first card (`held`, its Status panel's figure) and
    what a start's release gives back of it (`freeable`: never its CUDA context)."""

    held: int = 0
    freeable: int = 0

    def words(self) -> str:
        """`That free figure counts the 1.8 GiB ComfyUI gives back …`: the figure before it
        is the card's after that free, not nvidia-smi's now."""
        return (
            f"That free figure counts the {self.freeable / GIB:.1f} GiB ComfyUI gives back when "
            f"a start frees its card first (it holds about {self.held / GIB:.1f} GiB; the "
            "router frees it before every load)."
        )

    def offer(self, name: str) -> str:
        """`--no-room`'s word: the release would make the room."""
        return (
            f"ComfyUI holds about {self.held / GIB:.1f} GiB; freeing its card (about "
            f"{self.freeable / GIB:.1f} GiB back) would make room: `mnm sets start {name}` "
            "without --no-room frees it first."
        )


NOTHING = ComfyRoom()


def comfy_room(settings: Settings, engine: ComfyEngine | None = None) -> ComfyRoom:
    """ComfyUI's room now: nothing when it is not installed, not running or not answering
    (a box without it costs one file read; never raises)."""
    from makenomistakes.comfy.engine import ComfyEngine, State  # noqa: PLC0415 - no cycle
    from makenomistakes.comfy.status import ran_since  # noqa: PLC0415

    engine = engine or ComfyEngine(settings)
    state = engine.state()  # no ledger: no Docker asked
    address = engine.address(state) if state is State.RUNNING else None
    stats = engine.stats(address) if address is not None else None
    if stats is None:
        return NOTHING
    ran = ran_since(engine)
    return ComfyRoom(stats.held_estimate(ran), stats.freeable(ran))


def lighter(snapshot: MemorySnapshot, freed: int) -> MemorySnapshot:
    """The snapshot with `freed` bytes given back on the first card (where ComfyUI runs)."""
    gpu = snapshot.gpu
    if gpu is None or freed <= 0 or not isinstance(snapshot.gpus, list):
        return snapshot
    used = max(gpu.vram_used_bytes - freed, 0)
    return snapshot.model_copy(
        update={"gpus": [gpu.model_copy(update={"vram_used_bytes": used}), *snapshot.gpus[1:]]}
    )


def counted(
    verdict: Verdict, estimate: VramEstimate, snapshot: MemorySnapshot, room: ComfyRoom
) -> Verdict:
    """`verdict` was judged with ComfyUI's freeable VRAM as free; when that changed how the
    instance fits alone (`snapshot` is the card as nvidia-smi reads it), it says so."""
    if not room.freeable or verdict.level is FitNow.RUNNING:
        return verdict
    before = fit_now(estimate, snapshot, running=False).level
    after = fit_now(estimate, lighter(snapshot, room.freeable), running=False).level
    if before is after:
        return verdict
    return verdict.model_copy(
        update={"note": f"{verdict.note} {room.words()}", "comfy_freed": room.freeable}
    )


def together(
    waiting: list[tuple[str, VramEstimate]],
    snapshot: MemorySnapshot,
    room: ComfyRoom,
    beside: tuple[str, ...] = (),
    owner: str = "",
) -> Verdict:
    """`capacity.fit_together` with ComfyUI's freeable VRAM as free, said when it changes
    the verdict."""
    raw = fit_together(waiting, snapshot, beside, owner)
    if not room.freeable:
        return raw
    after = fit_together(waiting, lighter(snapshot, room.freeable), beside, owner)
    if after.level is raw.level:
        return raw
    return after.model_copy(
        update={"note": f"{after.note} {room.words()}", "comfy_freed": room.freeable}
    )


def taking_turns(verdict: Verdict, group: str, members: set[str]) -> Verdict:
    """A parked swap member that lacks room only because members of its own group hold the
    card is not "won't fit": its next request parks them and loads it (else zimage
    reads "won't fit" beside qwen3-8b, and the swap looks broken although it works)."""
    if not verdict.warns or not verdict.stop_first or not set(verdict.stop_first) <= members:
        return verdict
    names = ", ".join(verdict.stop_first)
    return verdict.model_copy(
        update={
            "level": FitNow.TAKES_TURNS,
            "note": f"Takes turns in the swap workflow {group}: starting it parks {names}, and "
            "while the workflow is up a request to its URL does the same (20-60 s for most "
            f"models). {verdict.note}",
        }
    )


def on_request(verdict: Verdict, held_back: frozenset[str]) -> Verdict:
    """A parked instance that lacks room only because mnm's own instances hold the card
    loads at its first request: the router parks them, unless one is `held_back`
    (`workflows.held_back`: what the router never parks). Saying "won't fit" in red reads
    as broken (every parked model beside a MoE, each with its note). It
    still `warns`, so a start by hand asks for `--stop-others`."""
    if verdict.level is not FitNow.NO_FIT or not verdict.stop_first:
        return verdict
    if verdict.kept:  # what it would park is not enough: the router never parks the rest
        return verdict
    names = ", ".join(verdict.stop_first)
    return verdict.model_copy(
        update={
            "level": FitNow.ON_REQUEST,
            "note": f"Its first request parks {names} and loads it (the router); a start by "
            "hand needs `--stop-others`.",
        }
    )
