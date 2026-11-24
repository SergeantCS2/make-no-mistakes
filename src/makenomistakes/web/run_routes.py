"""The Run tab on an instance page, the Runs gallery and its Clean up panel.

A Run is `make.make` as a job: its record is written before the job starts (so the gallery
shows it running), and the files land in the run's folder under the outputs folder. Files are
served by the name a record lists, never by a path from the request.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Annotated, Protocol

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from makenomistakes.cleanup import Cleaner, ClearHistory, Item, Target, parse_age
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.make import (
    Ask,
    again,
    check,
    close_parent,
    make,
    open_parent,
    settings_used,
)
from makenomistakes.pipeline import song
from makenomistakes.recipes import Recipe
from makenomistakes.run_lineage import nested
from makenomistakes.run_prune import select
from makenomistakes.runs import (
    MEDIA_TYPES,
    RunError,
    RunKind,
    RunRecord,
    RunState,
    RunStore,
    size_limit,
)
from makenomistakes.smoke import TryError
from makenomistakes.try_run import http_client
from makenomistakes.web.common import Render, local
from makenomistakes.web.instance_routes import refuse_if_busy
from makenomistakes.web.try_routes import upload

SHOWN_PER_INSTANCE = 6  # the newest runs under an instance's Run form
GALLERY_PAGE = 60


class RunMaker(Protocol):
    def __call__(
        self, service: InstanceService, store: RunStore, recipe: Recipe, ask: Ask, run: RunRecord
    ) -> RunRecord: ...


def default_maker(
    service: InstanceService, store: RunStore, recipe: Recipe, ask: Ask, run: RunRecord
) -> RunRecord:
    with http_client(service.settings) as http:
        return make(service, store, recipe, http, ask, record=run)


def _age(text: str) -> timedelta | None:
    if not text.strip():
        return None
    try:
        return parse_age(text.strip())
    except ValueError as exc:
        raise InstanceError(str(exc)) from exc


class Runs:
    """Starts a Run as a job (its record written first, so the gallery shows it running) and
    lists an instance's newest runs for its Run tab."""

    def __init__(
        self,
        service: InstanceService,
        store: RunStore,
        jobs: JobRegistry,
        maker: RunMaker | None = None,
    ) -> None:
        self.service = service
        self.store = store
        self.jobs = jobs
        self._make = maker or default_maker

    def runs_for(self, name: str) -> list[RunRecord]:
        found = [r for r in self.store.all() if r.kind is RunKind.INSTANCE and r.target == name]
        return found[:SHOWN_PER_INSTANCE]

    def start(self, recipe: Recipe, ask: Ask) -> RunRecord:
        check(recipe, ask)
        refuse_if_busy(self.jobs, recipe.name, "run it")
        words = (ask.prompt or "").strip() or None
        try:
            run = self.store.new(RunKind.INSTANCE, recipe.name, words, settings_used(recipe))
        except RunError as exc:
            raise InstanceError(str(exc)) from exc

        def work(report: Callable[[str], None]) -> list[str]:
            report("sending your prompt through the router…")
            try:
                done = self._make(self.service, self.store, recipe, ask, run)
            except TryError as exc:  # worded for the user already; the run is kept failed
                raise InstanceError(str(exc)) from exc
            return [f"{done.answer} ({done.seconds or 0:.1f} s)"]

        label = f"run {recipe.name} ({run.id})"
        job = self.jobs.start_unless_active(JobKind.RUN, recipe.name, work, label=label)
        if job.label != label:  # another job for this instance won the race: not ours to run
            self.store.fail(run, f"Not started: {job.label or 'another job'} was running.")
            raise InstanceError(f"{recipe.name} is busy ({job.label}); wait for it.")
        return run

    def start_candidates(self, recipe: Recipe, ask: Ask, count: int) -> RunRecord:
        """Run again on a parent of candidates: `count` new candidates under a new
        parent, as `mnm run <song> --repeat N` makes them, as one job."""
        check(recipe, ask)
        refuse_if_busy(self.jobs, recipe.name, "run it")
        try:
            parent = open_parent(self.store, recipe, ask)
        except RunError as exc:
            raise InstanceError(str(exc)) from exc
        words = (ask.prompt or "").strip() or None

        def work(report: Callable[[str], None]) -> list[str]:
            made: list[RunRecord] = []
            try:
                for turn in range(1, count + 1):
                    report(f"candidate {turn} of {count} through the router…")
                    child = self.store.new(
                        RunKind.INSTANCE, recipe.name, words, settings_used(recipe)
                    )
                    child.parent, child.role = parent.id, "candidate"
                    made.append(self._make(self.service, self.store, recipe, ask, child))
            except (TryError, InstanceError, RunError) as exc:
                close_parent(self.store, parent, made, count, error=str(exc))
                raise InstanceError(str(exc)) from exc
            except BaseException as exc:
                close_parent(self.store, parent, made, count, error=str(exc) or "interrupted")
                raise
            done = close_parent(self.store, parent, made, count)
            return [done.answer]

        label = f"run {recipe.name} ({parent.id})"
        job = self.jobs.start_unless_active(JobKind.RUN, recipe.name, work, label=label)
        if job.label != label:
            self.store.fail(parent, f"Not started: {job.label or 'another job'} was running.")
            raise InstanceError(f"{recipe.name} is busy ({job.label}); wait for it.")
        return parent


def register_run_routes(app: FastAPI, runs: Runs) -> None:
    """An instance's Run form, and "Run again" on a kept run."""
    service, store = runs.service, runs.store

    @app.post("/instances/{name}/run")
    def run_it(
        name: str,
        prompt: Annotated[str, Form(max_length=20_000)] = "",
        lyrics: Annotated[str, Form(max_length=20_000)] = "",
        image: Annotated[UploadFile | None, File()] = None,
        mask: Annotated[UploadFile | None, File()] = None,
        audio: Annotated[UploadFile | None, File()] = None,
    ) -> RedirectResponse:
        recipe = service.get(name)
        ask = Ask(
            prompt=prompt,
            picture=upload(image, "picture"),
            mask=upload(mask, "mask"),
            clip=upload(audio, "clip"),
            lyrics=lyrics or None,
        )
        runs.start(recipe, ask)
        return RedirectResponse(f"/instances/{name}#run", status_code=303)

    @app.post("/runs/{run_id}/again")
    def run_again(run_id: str) -> RedirectResponse:
        """Every run's Run again: an instance's here (a parent of candidates makes as many
        new candidates), a workflow's handed on with its form (307 keeps the POST: its
        New seed and the page to come back to)."""
        try:
            record = store.get(run_id)
            refused = song.again_refusal(record, _parent(store, record))
            if refused is not None:  # a song's candidate is made again in its song
                raise RunError(refused)
            if record.kind is RunKind.WORKFLOW:
                return RedirectResponse(f"/comfy/runs/{run_id}/again", status_code=307)
            if record.kind is not RunKind.INSTANCE:
                raise RunError("A film is made again from its page: Redo a step there.")
            ask = again(store, record)
            candidates = [c for c in store.children(record.id) if c.role == "candidate"]
        except (RunError, OSError) as exc:
            raise InstanceError(str(exc)) from exc
        recipe = service.get(record.target)
        if candidates:  # A parent of candidates makes as many again, under a new one
            runs.start_candidates(recipe, ask, len(candidates))
        else:
            runs.start(recipe, ask)
        return RedirectResponse(f"/instances/{recipe.name}#run", status_code=303)


def _parent(store: RunStore, record: RunRecord) -> RunRecord | None:
    if record.parent is None:
        return None
    try:
        return store.get(record.parent)
    except RunError:  # gone: the run stands alone now
        return None


def register_gallery_routes(
    app: FastAPI, render: Render, runs: Runs, history: ClearHistory | None = None
) -> None:
    """The Runs page, its files, deleting them, pruning, and the Clean up panel
    (`history`: ComfyUI's, cleared with its folders)."""
    store = runs.store
    cleaner = Cleaner(store, runs.service.settings.expanded_state_dir(), history)
    last_clean: list[Item] = []  # what the last Clean did, shown once it redirects

    @app.get("/runs", response_class=HTMLResponse)
    def gallery(
        request: Request, kind: str = "", target: str = "", cleaned: str = ""
    ) -> HTMLResponse:  # `cleaned` only asks to show this process's last clean, never text
        read = store.read_all()  # one parse of every record per load
        runs, problems = read.records, read.problems
        targets = sorted({r.target for r in runs})
        shown = [
            r
            for r in runs
            if (not kind or r.kind.value == kind) and (not target or r.target == target)
        ]
        return render(
            request,
            "runs.html",
            tab="runs",
            runs=nested(shown[:GALLERY_PAGE]),  # Candidates under their parent
            more=max(len(shown) - GALLERY_PAGE, 0),
            total=sum(r.size for r in runs),
            count=len(runs),
            kinds=[k.value for k in RunKind],
            kind=kind,
            targets=targets,
            target=target,
            survey=cleaner.survey(runs=read),
            cleaned=last_clean if cleaned else [],
            problems=problems,
            outputs_dir=store.outputs_dir,
            RunState=RunState,
        )

    @app.get("/runs/{run_id}/files/{name}")
    def run_file(run_id: str, name: str) -> FileResponse:
        try:
            record = store.get(run_id)
            path = store.file(record, name)
        except RunError as exc:
            raise InstanceError(str(exc)) from exc
        media = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media, headers={"X-Content-Type-Options": "nosniff"})

    @app.post("/runs/{run_id}/delete")
    def delete_run(run_id: str, back: Annotated[str, Form()] = "/runs") -> RedirectResponse:
        try:
            store.delete(run_id)
        except RunError as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse(local(back, "/runs"), status_code=303)

    @app.post("/runs/{run_id}/files/{name}/delete")
    def delete_file(
        run_id: str, name: str, back: Annotated[str, Form()] = "/runs"
    ) -> RedirectResponse:
        try:
            store.delete_file(run_id, name)
        except RunError as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse(local(back, "/runs"), status_code=303)

    @app.post("/runs/prune")
    def prune(
        older_than: Annotated[str, Form()] = "",
        max_gb: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        age = _age(older_than)
        until = None
        if max_gb.strip():
            try:
                until = size_limit(max_gb.strip())
            except RunError as exc:
                raise InstanceError(str(exc)) from None
        if age is None and until is None:
            raise InstanceError("Say how: an age (30d), a size (20), or both.")
        for record in select(store.read_all(), age, until):
            store.delete(record.id)
        return RedirectResponse("/runs", status_code=303)

    @app.post("/clean")
    def clean(
        targets: Annotated[list[Target] | None, Form()] = None,
        older_than: Annotated[str, Form()] = "",
        keep_files: Annotated[bool, Form()] = False,
    ) -> RedirectResponse:
        if not targets:  # nothing ticked: said so, not "the form sent a value"
            raise InstanceError("Tick what to clean up first: nothing was deleted.")
        last_clean[:] = cleaner.clean(targets, _age(older_than), keep_files)
        return RedirectResponse("/runs?cleaned=1#clean", status_code=303)
