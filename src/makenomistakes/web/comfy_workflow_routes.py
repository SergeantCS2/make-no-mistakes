"""The Workflows tab: imported ComfyUI workflows. Import from a link or a file,
read what each needs, fill in its form and run it; runs land in the gallery. A run is a job;
the router frees ComfyUI's card ten minutes after the last run (`comfy.idle`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.comfy import sources
from makenomistakes.comfy.engine import ComfyError, unread_nodes
from makenomistakes.comfy.form import Field as FormField
from makenomistakes.comfy.form import Kind
from makenomistakes.comfy.models import Status
from makenomistakes.comfy.packs import PackStore, standing
from makenomistakes.comfy.registry import PackError
from makenomistakes.comfy.report import analyze
from makenomistakes.comfy.run import ComfyRunError, ComfyRunner
from makenomistakes.comfy.sources import Found, SourceError, fetch, unpack
from makenomistakes.comfy.store import WorkflowStore, WorkflowStoreError
from makenomistakes.hub.civitai import CivitaiClient
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.runs import RunKind, RunRecord
from makenomistakes.web.common import Render, local
from makenomistakes.web.try_routes import MAX_UPLOAD

UPLOAD_KINDS = (Kind.IMAGE, Kind.VIDEO, Kind.AUDIO)
Fetch = Callable[[str], list[Found]]


def _key(name: str) -> str:
    return f"comfy:{name}"


def default_fetch(civitai_token: Callable[[], str | None]) -> Fetch:
    """Imports from a link; the Civitai key is read at each import (one added later counts)."""

    def run(reference: str) -> list[Found]:
        with sources.default_http() as http:
            civitai = CivitaiClient(civitai_token())
            try:
                return fetch(reference, http, civitai)
            finally:
                civitai.close()

    return run


def register_comfy_workflow_routes(  # noqa: PLR0915 - the tab's pages, one place
    app: FastAPI,
    render: Render,
    runner: ComfyRunner,
    jobs: JobRegistry,
    fetcher: Fetch,
    statuses: Callable[[str], list[Status]],
) -> None:
    store = runner.workflows

    @app.get("/comfy", response_class=HTMLResponse)
    def workflows(
        request: Request, removed: str = "", freed: int = 0, kept: int = 0, before: int = 0
    ) -> HTMLResponse:
        info = runner.engine.object_info()
        state = runner.engine.state()  # an unreadable ledger: System → ComfyUI says why
        rows = []
        for meta in store.all():
            try:
                report = analyze(store.workflow(meta.name), info)
            except (WorkflowStoreError, ValueError, OSError):
                report = None
            rows.append((meta, report))
        return render(
            request,
            "comfy_list.html",
            tab="comfy",
            rows=rows,
            problems=store.problems(),
            checked=info is not None,
            state=state,
            unread=None if info is not None else unread_nodes(state),
            removed=removed if _a_name(store, removed) else "",  # it is shown as a command
            freed=max(freed, 0),
            kept=bool(kept),
            before=bool(before),
        )

    @app.post("/comfy/import")
    def import_workflow(
        link: Annotated[str, Form(max_length=2000)] = "",
        file: Annotated[UploadFile | None, File()] = None,
    ) -> RedirectResponse:
        try:
            if file is not None and file.filename:
                data = file.file.read(MAX_UPLOAD + 1)
                if len(data) > MAX_UPLOAD:
                    raise SourceError("That file is larger than a workflow is.")
                found = unpack(file.filename, data, file.filename)
            elif link.strip():
                found = fetcher(link.strip())
            else:
                raise SourceError("Paste a link or choose a file.")
            names = [
                store.add(store.free_name(item.name), item.data, item.source, item.name).name
                for item in found
            ]
        except (SourceError, WorkflowStoreError) as exc:
            raise InstanceError(str(exc)) from exc
        target = f"/comfy/{names[0]}" if len(names) == 1 else "/comfy"
        return RedirectResponse(target, status_code=303)

    @app.get("/comfy/{name}", response_class=HTMLResponse)
    def workflow(request: Request, name: str) -> HTMLResponse:
        try:
            meta = store.meta(name)
            report = analyze(store.workflow(name), runner.engine.object_info())
        except (WorkflowStoreError, ValueError, OSError) as exc:
            raise InstanceError(str(exc)) from exc
        fields: list[FormField] = []
        why = ""
        try:
            _prompt, fields = runner.prompt(name)
        except ComfyRunError as exc:
            why = str(exc)
        runs = [r for r in runner.runs.all() if r.kind is RunKind.WORKFLOW and r.target == name]
        try:
            approved = PackStore(runner.engine.settings.expanded_state_dir()).approvals()
        except PackError:
            approved = []
        try:
            ledger, ledger_error = runner.engine.ledger(), ""
        except ComfyError as exc:  # the workflow still shows; its packs' state is unknown
            ledger, ledger_error = None, str(exc)
        installed = ledger.packs if ledger is not None else []
        return render(
            request,
            "comfy_workflow.html",
            tab="comfy",
            meta=meta,
            report=report,
            fields=fields,
            why=why,
            uploads=UPLOAD_KINDS,
            runs=runs[:6],
            ledger_error=ledger_error,
            pack_state={
                p.pack: standing(p.pack, p.version, approved, installed) for p in report.packs
            },
            models=statuses(name),
            job=jobs.active_for(_key(name))
            or jobs.active_for(f"comfy-models:{name}")
            or jobs.latest_for(_key(name))
            or jobs.latest_for(f"comfy-models:{name}"),
            state=runner.engine.state(),
        )

    @app.post("/comfy/{name}/run")
    async def run(request: Request, name: str) -> RedirectResponse:
        busy = jobs.active_for(_key(name))
        if busy is not None:
            raise InstanceError(f"{name} is running already; wait for it, or stop it.")
        form = await request.form()
        try:
            _prompt, fields = runner.prompt(name)
        except ComfyRunError as exc:
            raise InstanceError(str(exc)) from exc
        # a text the user emptied is sent empty; any other blank keeps the workflow's own, or
        # a setting a second node holds takes the one shown (`Field.follows`)
        texts = {f.key for f in fields if f.kind in (Kind.TEXT, Kind.LONG_TEXT) and not f.follows}
        values: dict[str, str] = {}
        uploads: dict[str, tuple[str, bytes]] = {}
        for key, value in form.multi_items():
            if isinstance(value, str):
                if value.strip() or key in texts:
                    values[key] = value
            elif value.filename:
                data = await value.read(MAX_UPLOAD + 1)
                if len(data) > MAX_UPLOAD:
                    raise InstanceError(
                        f"{value.filename} is larger than {MAX_UPLOAD // 2**20} MiB."
                    )
                uploads[key] = (value.filename, data)
        try:
            record = runner.new(name, values)
        except ComfyRunError as exc:
            raise InstanceError(str(exc)) from exc
        return start(name, record, values, uploads)

    @app.post("/comfy/runs/{run_id}/again")
    def again(
        run_id: str,
        new_seed: Annotated[str, Form(max_length=8)] = "",
        back: Annotated[str, Form(max_length=200)] = "",
    ) -> RedirectResponse:
        """A workflow run's settings and uploads again; `new_seed` draws its seeds afresh."""
        try:
            plan = runner.again(run_id, new_seed == "yes")
            if jobs.active_for(_key(plan.target)) is not None:
                raise ComfyRunError(f"{plan.target} is running already; wait for it.")
            record = runner.new(plan.target, plan.values)
        except ComfyRunError as exc:
            raise InstanceError(str(exc)) from exc
        start(plan.target, record, plan.values, plan.uploads)
        return RedirectResponse(local(back, f"/comfy/{plan.target}"), status_code=303)

    @app.post("/comfy/{name}/shown")
    async def shown(request: Request, name: str) -> RedirectResponse:
        """The settings the user shows on top of the form (out of More settings)."""
        form = await request.form()
        try:
            runner.set_shown(name, [k for k in form.getlist("key") if isinstance(k, str)])
        except ComfyRunError as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse(f"/comfy/{name}", status_code=303)

    def start(
        name: str,
        record: RunRecord,
        values: dict[str, str],
        uploads: dict[str, tuple[str, bytes]],
    ) -> RedirectResponse:
        """The run as a job; a record that lost the race for its job is failed at once."""

        def work(report: Callable[[str], None]) -> list[str]:
            try:
                done = runner.run(record, values, uploads, report)
            except ComfyRunError as exc:
                raise InstanceError(str(exc)) from exc
            return [done.answer]

        label = f"run {name} ({record.id})"
        job = jobs.start_unless_active(JobKind.RUN, _key(name), work, label=label)
        if job.label != label:  # another run won the race: this record never starts
            runner.runs.fail(record, f"Not started: {job.label} was running.")
            raise InstanceError(f"{name} is running already; wait for it, or stop it.")
        return RedirectResponse(f"/comfy/{name}", status_code=303)

    @app.post("/comfy/{name}/stop")
    def stop(name: str) -> RedirectResponse:
        """Stop this workflow's run in ComfyUI, by its prompt id (never another's work)."""
        mine = [r for r in runner.runs.all() if r.kind is RunKind.WORKFLOW and r.target == name]
        record = next((r for r in mine if r.going), None)  # newest first
        if record is None:
            raise InstanceError(f"No run of {name} is going.")
        try:
            runner.interrupt(record)
        except ComfyRunError as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse(f"/comfy/{name}", status_code=303)


def _a_name(store: WorkflowStore, name: str) -> bool:
    """A workflow name, nothing else (the list prints `removed=` inside a command to paste)."""
    try:
        store.folder(name)
    except WorkflowStoreError:
        return False
    return True
