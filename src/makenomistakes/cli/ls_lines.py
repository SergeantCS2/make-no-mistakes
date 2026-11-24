"""`mnm ls`'s lines around its table: the card's line above it,
ComfyUI's, what waits in the router's queue and the box's below it (the last
line says when the hardware changed since saved plans were made, the tuner's line before it);
the table's State cell, which `mnm sets show` shares. `mnm status` prints the same lines."""

from __future__ import annotations

from rich.markup import escape

from makenomistakes.backend_now import box_now
from makenomistakes.cli._app import console
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.comfy.status import comfy_status
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.instance_types import InstanceStatus
from makenomistakes.instances import InstanceService
from makenomistakes.machine import Drift, drift, observe_snapshot, sentence
from makenomistakes.queue_figures import waiting_words
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import RouterNow
from makenomistakes.runs import RunStore
from makenomistakes.supervisor import ContainerState, Health
from makenomistakes.tune.store import TuneStore

GIB = 2**30


def print_comfy(service: InstanceService) -> None:
    """ComfyUI's line: its VRAM counts in the card's line above the table, in no row."""
    comfy = comfy_status(ComfyEngine(service.settings), RunStore.for_settings(service.settings))
    if comfy.installed:
        console.print(f"ComfyUI: {escape(comfy.words())}")


def print_queue(now: RouterNow | None) -> None:
    """What waits in the router's queue now (`router_now`, read once by `mnm ls`);
    nothing when nothing does, or no router answers."""
    if now is not None and now.queue:
        lines = [waiting_words(name, waiting) for name, waiting in sorted(now.queue.items())]
        console.print(f"Router: {escape('; '.join(lines))}.")


def print_machine(snapshot: MemorySnapshot) -> None:
    gpu = snapshot.gpu
    box = box_now()
    if gpu is None and box is not None and not box.cards:  # A box without a card
        free = snapshot.memory.available_bytes if snapshot.memory is not None else None
        ram = f" ({free / GIB:.1f} GiB of RAM free)" if free else ""
        console.print(f"No card: the CPU backend runs every model in RAM{ram}.")
        return
    if gpu is None:
        console.print(
            f"[red]GPU not readable: {escape(snapshot.gpu_error or '?')}. Run `mnm doctor`.[/red]"
        )
        return
    console.print(
        f"{escape(gpu.name)}: {gpu.vram_used_bytes / GIB:.1f} of {gpu.vram_total_bytes / GIB:.1f} "
        f"GiB VRAM in use, {gpu.vram_free_bytes / GIB:.1f} GiB free"
    )


def box_drift(
    service: InstanceService, snapshot: MemorySnapshot, recipes: list[Recipe]
) -> Drift | None:
    """The box read and recorded (cheap reads: no bandwidth probe), and what moved since saved
    plans were made; None when nothing was recorded."""
    observed = observe_snapshot(service.settings, snapshot)
    return drift(observed.file, recipes) if observed.file is not None else None


def print_box(service: InstanceService, snapshot: MemorySnapshot, recipes: list[Recipe]) -> None:
    """One line when saved plans were made on another box, nothing otherwise."""
    said_drift(box_drift(service, snapshot, recipes))


def said_drift(moved: Drift | None) -> None:
    said = sentence(moved)
    if said is not None:
        console.print(f"[yellow]{escape(said)}[/yellow]")


def state_cell(status: InstanceStatus) -> str:
    """`mnm ls`'s State cell (and `mnm sets show`'s)."""
    if status.error:  # the full text is printed once above the table
        return "[red]Docker unavailable[/red]"
    if status.state is ContainerState.RUNNING:
        if status.health is Health.READY:
            return "[green]ready[/green]"
        return f"[yellow]{status.health.value if status.health else 'unknown'}[/yellow]"
    if status.state is ContainerState.EXITED:
        return f"[red]exited ({status.exit_code})[/red]"
    if not status.files_present:  # evicted: the disk policy's, downloaded at a start
        return f"[yellow]{'evicted' if status.evicted else 'not downloaded'}[/yellow]"
    # Parked loads at its next request; stopped (the user's `mnm stop`) never does.
    return "[dim]stopped[/dim]" if status.recipe.off else "[cyan]parked[/cyan]"


def print_tuning(service: InstanceService, now: RouterNow | None) -> None:
    """`mnm ls`'s last line while a tune runs: the router's (`now`, the one look `mnm ls`
    takes at it; None when the router did not say), else a terminal's."""
    line = now.tuning if now is not None else None
    if line is None:
        running = TuneStore(service.settings.expanded_state_dir()).running()
        line = running[0].line if running else None
    if line is not None:
        console.print(f"Tuner: {escape(line)}")
