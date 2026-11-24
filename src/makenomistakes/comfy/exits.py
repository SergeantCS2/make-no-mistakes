"""Why ComfyUI stopped under a run, in words. A run that loses ComfyUI's
progress messages asks Docker how the container ended: killed out of memory (the big video
models stream ~38 GB of files through RAM) is the likely one, and reads as such instead of
"lost the socket"."""

from __future__ import annotations

from makenomistakes.comfy.doctor import wsl_remedy
from makenomistakes.comfy.engine import ComfyEngine, InspectState
from makenomistakes.engine_exits import KILLED_BY_SOMEONE
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.windows_ram import windows_ram

WAIT_SECONDS = 5.0  # the socket can close a moment before Docker marks the container exited
KILLED = 137
CRASHED = 139


def why_stopped(engine: ComfyEngine, wait: float = WAIT_SECONDS) -> str | None:
    """How ComfyUI's container ended, once it has; None while it still runs (or is gone)."""
    state = engine.container_state()
    waited = 0.0
    while state is not None and state.Running and waited < wait:
        engine.sleep(0.5)
        waited += 0.5
        state = engine.container_state()
    if state is None or state.Running:
        return None
    wsl = platforms.current().is_wsl
    return words(state, windows_ram(engine.runner) if state.OOMKilled and wsl else None)


def words(state: InspectState, host_ram: int | None = None) -> str:
    """`host_ram`: the PC's own RAM under WSL, so `.wslconfig` is never told more than it has."""
    if state.OOMKilled:
        return (
            "ComfyUI was killed (Docker says out of memory): most likely the machine ran out "
            "of RAM while the workflow loaded its models. " + _ram_advice(host_ram)
        )
    if state.ExitCode == KILLED:  # without the flag: not out of memory (`engine_exits`)
        return (
            f"ComfyUI was killed (exit 137) by {KILLED_BY_SOMEONE}: `mnm comfyui logs` "
            "shows its last lines; `mnm comfyui start` brings it back."
        )
    if state.ExitCode == CRASHED:
        return (
            "ComfyUI crashed (exit 139, a segmentation fault): `mnm comfyui logs` shows its "
            "last lines; `mnm comfyui start` brings it back."
        )
    return (
        f"ComfyUI stopped (exit {state.ExitCode}): `mnm comfyui logs` shows why; "
        "`mnm comfyui start` brings it back."
    )


def _ram_advice(host_ram: int | None) -> str:
    """The doctor's own `.wslconfig` lines (sized to this PC), never a fixed `memory=24GB`."""
    if platforms.current().is_wsl:
        return f"{wsl_remedy(host_ram)} Or pick smaller (fp8 or GGUF) model files in the workflow."
    return "Add swap or RAM, or pick smaller (fp8 or GGUF) model files in the workflow."
