"""ComfyUI's panel on the Status page: Stop it, or free the card it holds (refused while it
makes a run). The panel's data is `comfy.status.comfy_status`."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from makenomistakes.comfy import idle
from makenomistakes.comfy.engine import ComfyBusyError, ComfyEngine
from makenomistakes.comfy.status import ComfyStatus
from makenomistakes.instance_types import InstanceError

CHANGING = "ComfyUI is starting or changing its node packs; try again in a moment."
BUSY = (
    "ComfyUI is making {what}: it frees its card between runs only. Stop the run in Runs (or "
    "Stop ComfyUI), or free it when the run ends."
)


def register_comfy_status_routes(
    app: FastAPI, engine: ComfyEngine, status: Callable[[], ComfyStatus]
) -> None:
    @app.post("/comfyui/stop")
    def stop() -> RedirectResponse:
        """`mnm comfyui stop`: its container goes, and the card with it."""
        engine.stop()
        return RedirectResponse("/?comfy=stopped#comfyui", status_code=303)

    @app.post("/comfyui/free")
    def free() -> RedirectResponse:
        """Unload ComfyUI's models now (`POST /free`), as the router does before a load."""
        # a run queues under the same lock, so none slips in between; a start or a packs
        # build holds it for minutes, and the click is refused at once instead of hanging
        with engine.try_locked() as mine:
            if not mine:
                raise InstanceError(CHANGING)
            now = status()
            if now.unknown:
                raise InstanceError(f"Docker could not be asked about ComfyUI: {now.docker}")
            if not now.running:
                raise InstanceError("ComfyUI is not running: it holds nothing.")
            if now.busy:
                what = f"a run of {now.run.target}" if now.run else "a job in its queue"
                raise InstanceError(BUSY.format(what=what))
            if now.queued is None:  # unknown is never idle: refused at once, never waited on
                raise InstanceError(
                    "ComfyUI did not answer whether it is running a job: nothing was freed; "
                    "try again in a moment."
                )
            try:
                freed = engine.release_vram()
            except ComfyBusyError as exc:  # its queue did not answer, or a job came meanwhile
                raise InstanceError(str(exc)) from exc
            if not freed:
                raise InstanceError(
                    "ComfyUI did not answer: nothing was freed (`mnm comfyui logs`)."
                )
            idle.freed(engine)
        return RedirectResponse("/?comfy=freed#comfyui", status_code=303)
