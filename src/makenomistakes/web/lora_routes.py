"""Web routes for an image instance's LoRAs: add from a link, change, remove.

Adding reads the LoRA's header first (a second or two), so a wrong base family is refused on
the page before anything downloads; the download itself is a background job like `add`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, Form
from fastapi.responses import RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.loras import LoraService


def weight_value(text: str, default: float = 1.0) -> float:
    """A weight typed into the page: `0.8`, `0,8`, or empty for the default. A typo is the
    error page, never the framework's raw 422."""
    cleaned = text.strip().replace(",", ".")
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError as exc:
        raise InstanceError(f"A LoRA weight is a number such as 0.8 (got {text!r}).") from exc


def register_lora_routes(app: FastAPI, loras: LoraService, jobs: JobRegistry) -> None:
    @app.post("/instances/{name}/loras")
    def add_lora(
        name: str,
        reference: Annotated[str, Form()],
        lora_name: Annotated[str, Form()] = "",
        weight: Annotated[str, Form()] = "",
        always_on: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        jobs.refuse_if_busy(name, "add the LoRA")  # its job would swallow this one
        planned = loras.plan(
            name,
            reference.strip(),
            lora_name.strip() or None,
            weight_value(weight),
            bool(always_on),
        )

        def download(report: Callable[[str], None]) -> list[str]:
            loras.add(name, planned, report)
            return [planned.note] if planned.note else []

        jobs.start_unless_active(JobKind.DOWNLOAD, name, download)
        return RedirectResponse(f"/instances/{name}#loras", status_code=303)

    @app.post("/instances/{name}/loras/{lora}")
    def update_lora(
        name: str,
        lora: str,
        weight: Annotated[str, Form()] = "",
        always_on: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        loras.update(name, lora, weight_value(weight) if weight.strip() else None, bool(always_on))
        return RedirectResponse(f"/instances/{name}#loras", status_code=303)

    @app.post("/instances/{name}/loras/{lora}/remove")
    def remove_lora(name: str, lora: str) -> RedirectResponse:
        loras.remove(name, lora)
        return RedirectResponse(f"/instances/{name}#loras", status_code=303)
