"""Web routes for adding a model: plan a reference (one of its versions, as a type, or for
another card's VRAM), then create and download, as `mnm add` does with its flags."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.add_intent import AddIntent
from makenomistakes.agent_profile import FLOORS, add_page_hints, agent_context, requested_type
from makenomistakes.busy_card import plan_note
from makenomistakes.config import Settings
from makenomistakes.device_words import backend_tail, device_lines, pool_words
from makenomistakes.fit import Budget, KvType, ProfileName
from makenomistakes.hub.facts import Modality
from makenomistakes.instance_vram import vram_holders
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.model_uses import parse_use
from makenomistakes.plan import Plan, PlanError, explicit_budget, open_hub
from makenomistakes.plan_budget import budget_now, planning_budget
from makenomistakes.web.common import Render
from makenomistakes.what import answer

PlanFn = Callable[[str, Settings, Budget | None, str | None, Modality | None], Plan]
DEFAULT_GPU = "RTX 4070"  # `mnm fit --gpu`'s default: the bandwidth table's row
GIB = 1024**3
WEB_LIMIT = 6  # `mnm what --limit`'s default: rows a source gives the Add page


def model_type(value: str) -> Modality | None:
    """The add form's type field: empty means the model's own; a bad value is a 400 page."""
    if not value:
        return None
    try:
        return Modality(value)
    except ValueError:
        raise PlanError(f"Unknown model type '{value}'.") from None


def register_add_routes(
    app: FastAPI,
    render: Render,
    settings: Settings,
    service: InstanceService,
    jobs: JobRegistry,
    planner: PlanFn,
) -> None:
    def budget_for(vram: float | None) -> Budget | None:
        """`mnm fit --vram` when asked (another card, or this one emptied); otherwise the live
        budget, which counts mnm's running instances as free (else a plan saved against the
        VRAM they held keeps half its layers off the GPU for good)."""
        if vram is None:
            return planning_budget(service)
        snapshot = service.memory()
        ram = snapshot.memory.available_bytes if snapshot.memory else None
        gpu = snapshot.gpu.name if snapshot.gpu else DEFAULT_GPU
        return explicit_budget(vram, (ram or 14 * GIB) / GIB, gpu)

    def plan(reference: str, version: str, wanted: Modality | None, vram: float | None) -> Plan:
        return planner(reference, settings, budget_for(vram), version or None, wanted)

    def plan_page(
        request: Request,
        reference: str,
        version: str,
        type: str,
        vram: float | None,
        preset: AddIntent | None = None,
    ) -> HTMLResponse:
        try:
            planned = plan(reference, version, model_type(type), vram)
        except PlanError as exc:
            return render(
                request,
                "add.html",
                reference=reference,
                plan=None,
                error=str(exc),
                guidance=exc.guidance,
                vram=vram,
            )
        other_types, agent_ctx = add_page_hints(planned)
        return render(
            request,
            "add.html",
            reference=reference,
            plan=planned,
            profiles=list(planned.profiles.values()),
            default_name=(preset and preset.name) or service.default_name(planned.facts),
            preset=preset or AddIntent(reference),  # A Find a model row's, as graded
            kv_types=[k.value for k in KvType],
            vram_holders=vram_holders(planned) if vram is None else [],
            busy_note=plan_note(planned.budget) if vram is None else None,
            device_notes=device_lines(planned.budget, planned.profiles.values()),
            pool=pool_words(planned.budget),  # A plan made in RAM
            backend=backend_tail(planned.budget),
            model_type=type,
            other_types=other_types,
            agent_ctx=agent_ctx,
            agent_floors=FLOORS,
            vram=vram,
            error="",
        )

    @app.get("/add", response_class=HTMLResponse)
    def add_form(  # noqa: PLR0913, PLR0917  # a link's fields: a plan preset from Find a model
        request: Request,
        reference: str = "",
        version: str = "",
        type: str = "",
        plan: str = "",
        vram: Annotated[float | None, Query(gt=0, le=1024, allow_inf_nan=False)] = None,
        quant: str = "",  # A Find a model row's, with `agents` and `name`
        agents: bool = False,
        name: str = "",
    ) -> HTMLResponse:
        if reference and (plan or version):  # a link from a plan (a version, a type): plan it
            preset = AddIntent.from_link(reference, agents, quant, name)
            return plan_page(request, reference, version, type, vram, preset)
        return render(request, "add.html", reference=reference, plan=None, error="", vram=None)

    @app.get("/add/what", response_class=HTMLResponse)
    def add_what(request: Request, q: str = "", use: str = "") -> HTMLResponse:
        """`mnm what [words] --type <use>` on the Add page, the same search and grades
        (a use with no words: its curated rows, as `--type` gives them); each row that fits
        links to its plan above, preset as graded."""
        try:
            picked = parse_use(use or None)
            if not q.strip() and picked is None:
                raise ValueError("Type what to look for, or pick a use for its curated models.")
            budget = budget_now(service)
            with open_hub(settings) as hub:
                found = answer(hub, settings, budget, q, picked, WEB_LIMIT)
        except (ValueError, PlanError) as exc:
            return render(
                request, "add.html", reference="", plan=None, error="", vram=None,
                what_q=q, what_use=use, what_error=str(exc),
            )  # fmt: skip
        return render(
            request, "add.html", reference="", plan=None, error="", vram=None,
            what_q=q, what_use=use, found=found,
        )  # fmt: skip

    @app.post("/add/plan", response_class=HTMLResponse)
    def add_plan(
        request: Request,
        reference: Annotated[str, Form()],
        version: Annotated[str, Form()] = "",
        type: Annotated[str, Form()] = "",
        vram: Annotated[float | None, Form(gt=0, le=1024, allow_inf_nan=False)] = None,
    ) -> HTMLResponse:
        return plan_page(request, reference.strip(), version, type, vram)

    @app.post("/add/create")
    def add_create(  # noqa: PLR0913, PLR0917  # a form handler: its fields are the UI
        reference: Annotated[str, Form()],
        name: Annotated[str, Form()],
        profile: Annotated[ProfileName, Form()],
        quant: Annotated[str | None, Form()] = None,
        ctx: Annotated[int | None, Form()] = None,
        kv: Annotated[KvType | None, Form()] = None,
        version: Annotated[str, Form()] = "",
        type: Annotated[str, Form()] = "",
        upscaler: Annotated[bool, Form()] = False,  # a checkbox: absent when off
        for_agents: Annotated[bool, Form()] = False,
        client: Annotated[str, Form()] = "",  # `--for <client>`: one client's floor
        no_download: Annotated[bool, Form()] = False,  # "Save without downloading"
        vram: Annotated[float | None, Form(gt=0, le=1024, allow_inf_nan=False)] = None,
    ) -> RedirectResponse:
        service.check_name(name)
        for_agents = for_agents or bool(client)
        planned = plan(reference, version, requested_type(model_type(type), for_agents), vram)
        if for_agents:  # `mnm add --for-agents`/`--for`: type coding, the agent context
            ctx = agent_context(planned, ctx, client or None)
        recipe = service.add(
            planned, planned.choose(profile, quant or None, ctx, kv), name, upscaler=upscaler
        )
        if not no_download:

            def fetch(report: Callable[[str], None]) -> list[str]:
                service.download(recipe, report)
                return []

            jobs.start_unless_active(JobKind.DOWNLOAD, name, fetch)
        return RedirectResponse(f"/instances/{name}", status_code=303)
