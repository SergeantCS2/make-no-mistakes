"""A workflow's model files on its page: paste a link, find the unlinked ones
in known repositories, fetch the missing ones (a job: downloads are minutes to hours)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, Form
from fastapi.responses import RedirectResponse

from makenomistakes.comfy.fetch import fetch_all, find_links, remove_workflow
from makenomistakes.comfy.models import ModelError, Models, Status, parse_source
from makenomistakes.comfy.report import analyze
from makenomistakes.comfy.store import WorkflowStore, WorkflowStoreError
from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.pipeline.workflow_users import WorkflowInUseError, refuse_if_used
from makenomistakes.recipes import Recipe
from makenomistakes.runs import RunError, RunStore

Recipes = Callable[[], list[Recipe]]


def statuses_for(models: Models, store: WorkflowStore, name: str, recipes: Recipes) -> list[Status]:
    meta = store.meta(name)
    report = analyze(store.workflow(name), None)
    return models.status(report.models, recipes(), meta.links)


def register_comfy_model_routes(
    app: FastAPI,
    settings: Settings,
    store: WorkflowStore,
    models: Models,
    recipes: Recipes,
    jobs: JobRegistry,
) -> None:
    def back(name: str) -> RedirectResponse:
        return RedirectResponse(f"/comfy/{name}#models", status_code=303)

    def job(name: str, title: str, work: Callable[[Callable[[str], None]], list[str]]) -> None:
        def run(report: Callable[[str], None]) -> list[str]:
            try:
                return work(report)
            except (ModelError, WorkflowStoreError, OSError) as exc:
                raise InstanceError(str(exc)) from exc

        jobs.start_unless_active(JobKind.DOWNLOAD, f"comfy-models:{name}", run, label=title)

    @app.post("/comfy/{name}/remove")
    def remove(name: str, files: Annotated[bool, Form()] = False) -> RedirectResponse:
        """The workflow goes; with `files`, the downloads only it used (`remove_workflow`)."""
        try:
            runs = RunStore.for_settings(settings).read_all()
            refuse_if_used(name, runs, settings.expanded_state_dir())  # a song's, a pick's …
            removed = remove_workflow(
                store, models, name, recipes(), settings.expanded_models_dir(), files
            )
        except (WorkflowStoreError, ModelError, OSError, WorkflowInUseError, RunError) as exc:
            raise InstanceError(str(exc)) from exc
        # The list says what happened: a kept file is news, not an error (the removal worked)
        freed = removed.cleanup.freed_bytes if removed.cleanup is not None else 0
        kept = ("&kept=1" if removed.kept else "") + ("" if removed.present else "&before=1")
        return RedirectResponse(
            f"/comfy?removed={quote(name)}&freed={freed}{kept}", status_code=303
        )

    @app.post("/comfy/{name}/models/link")
    def link(
        name: str,
        file: Annotated[str, Form(max_length=500)],
        url: Annotated[str, Form(max_length=2000)],
    ) -> RedirectResponse:
        try:
            parse_source(url)
            store.update(name, lambda meta: meta.link(file, url.strip()))
        except (ModelError, WorkflowStoreError) as exc:
            raise InstanceError(str(exc)) from exc
        return back(name)

    @app.post("/comfy/{name}/models/find")
    def find(name: str) -> RedirectResponse:
        def work(_report: Callable[[str], None]) -> list[str]:
            found = find_links(models, statuses_for(models, store, name, recipes))
            store.update(name, found.apply)  # minutes later: what was clicked meanwhile stays
            return found.said or ["every file has a link already"]

        job(name, f"find {name}'s files", work)
        return back(name)

    @app.post("/comfy/{name}/models/fetch")
    def fetch(name: str) -> RedirectResponse:
        def work(report: Callable[[str], None]) -> list[str]:
            statuses = statuses_for(models, store, name, recipes)
            placed = fetch_all(models, statuses, settings, recipes(), report, owner=name)
            return [f"linked {len(placed)} files where ComfyUI looks"]

        job(name, f"fetch {name}'s files", work)
        return back(name)
