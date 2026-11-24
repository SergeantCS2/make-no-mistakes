"""Room on the card for a ComfyUI run: mnm's running models are parked, largest
first, under the box's one start lock, the way the router parks them for a model's first
request. Pinned models and the members of a running set stay (`workflows.held_back`).

Only as many as the run needs: its size is the larger of the most VRAM an earlier run of the
same workflow added (`readings.py`) and its model files, plus a margin for what files do not
show; a CPU build needs none. A size nobody knows (no done run read, no model files) parks
every model. An earlier run that added nothing is a known size only for a workflow that reads
no weights, and only when ComfyUI's cache served none of its nodes (a run that reads no
weights and added 0.0 GiB parks nothing; a warm or cached run of a workflow with weights also
reads 0.0, and its next run, cold beside a loaded model, would run out of memory)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from makenomistakes.hardware.probe import MemorySnapshot, probe_memory
from makenomistakes.held_back import held_back_in
from makenomistakes.instances import InstanceService
from makenomistakes.progress_words import parking_for_room
from makenomistakes.runs import RunKind, RunRecord, RunState
from makenomistakes.swap import SwapService, start_turn
from makenomistakes.try_figures import Figure
from makenomistakes.workflows import WorkflowStore

Progress = Callable[[str], None]
# (progress, VRAM the run needs) -> the models parked, the room held while the `with` lasts
MakeRoom = Callable[[Progress, int | None], AbstractContextManager[list[str]]]
GIB = 2**30
MARGIN = GIB  # beyond the size: ComfyUI's CUDA context and the sampler's activations


def run_need(
    earlier: list[RunRecord], workflow: str, files: int, weights: bool = True
) -> int | None:
    """VRAM a run of `workflow` needs free; None when neither a reading nor its files say.
    A reading of 0.0 counts only when it is a known nothing (`_known`)."""
    added = [
        reading
        for r in earlier
        if r.kind is RunKind.WORKFLOW
        and r.target == workflow
        and r.state is RunState.DONE
        and (reading := r.numbers.get(Figure.VRAM_ADDED_GIB.value)) is not None
        and (reading > 0 or _known(r, weights))
    ]
    if not added and not files:
        return None
    seen = round(max(added, default=0.0) * GIB)
    # A warm run reads low (ComfyUI already held the weights it shares with an earlier one),
    # so a reading never undercuts the files it loads
    return max(seen, files) + MARGIN


def _known(run: RunRecord, weights: bool) -> bool:
    """A run that added 0.0 GiB says the workflow needs nothing beyond the margin only when
    the workflow reads no weights and ComfyUI's cache served none of its nodes (a run
    recorded before the cache was, `cached_nodes` None, says nothing)."""
    return not weights and run.cached_nodes == []


def to_park(need: int | None, free: int | None, sizes: dict[str, int]) -> list[str]:
    """The fewest running models, largest first, whose VRAM makes `need` free; every one
    when the need or the free VRAM is unknown. A model without an estimate counts as 0."""
    order = sorted(sizes, key=lambda n: (-sizes[n], n))
    if need is None or free is None:
        return order
    chosen: list[str] = []
    for name in order:
        if free >= need:
            break
        chosen.append(name)
        free += sizes[name]
    return chosen


def park_for_comfy(
    instances: InstanceService, memory: Callable[[], MemorySnapshot] = probe_memory
) -> MakeRoom:
    """Room for a ComfyUI run, held: the box's start lock is taken before anything is read
    and kept until the caller's `with` ends, which the runner ends once its prompt is queued.
    A router load in between would take the room just made, and the run would then run out
    of memory beside it. A CPU build (`need` 0) parks and holds nothing."""

    @contextmanager
    def make_room(progress: Progress, need: int | None) -> Iterator[list[str]]:
        if need == 0:
            yield []
            return
        with start_turn(instances.settings, "comfyui", progress):
            yield _park(instances, memory, progress, need)

    return make_room


def _park(
    instances: InstanceService,
    memory: Callable[[], MemorySnapshot],
    progress: Progress,
    need: int | None,
) -> list[str]:
    """Under the start lock: what runs now (a load that finished meanwhile is parkable), and
    the fewest of it parked."""
    statuses = instances.statuses()
    held = held_back_in(WorkflowStore(instances.settings.expanded_state_dir()), statuses)
    keep = held if held is not None else frozenset(s.recipe.name for s in statuses)
    running = {s.recipe.name: s.recipe for s in statuses if s.running}
    estimates = instances.running_vram(in_ram=False)  # what holds its card
    sizes = {name: estimates.get(name, 0) for name in running if name not in keep}
    swap = SwapService(instances, release=lambda _progress: False)
    gpu = memory().gpu
    free = gpu.vram_total_bytes - gpu.vram_used_bytes if gpu is not None else None
    parked = []
    for name in to_park(need, free, sizes):
        progress(parking_for_room(name, "ComfyUI"))
        swap.park(running[name])
        parked.append(name)
    return parked
