"""Workflows → Node packs: the vetted list, a pack's approval page (what the
Registry says about it, the sandbox note), approving, withdrawing, and the install as a job
(the packs' own weights fetched, the layer built, ComfyUI started to check every pack)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.comfy.engine import ComfyEngine, ComfyError
from makenomistakes.comfy.fetch import fetch_all
from makenomistakes.comfy.github_packs import GitHub, GitHubRef, parse_github
from makenomistakes.comfy.image import Ledger
from makenomistakes.comfy.models import ModelError, Models
from makenomistakes.comfy.pack_build import PackBuilder, missing_weights
from makenomistakes.comfy.packs import (
    SANDBOX_NOTE,
    VETTED,
    Approval,
    PackStore,
    approve_assessed,
    assess,
    check_approvable,
    standing,
)
from makenomistakes.comfy.registry import PackError, Registry
from makenomistakes.comfy.report import analyze
from makenomistakes.comfy.store import WorkflowStore, WorkflowStoreError
from makenomistakes.config import Settings
from makenomistakes.hardware.runner import SubprocessRunner
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import BUILD_LINES, JobRegistry
from makenomistakes.recipes import Recipe
from makenomistakes.web.comfy_jobs import PACKS_JOB, start_comfy_job
from makenomistakes.web.common import Render

OWN_PAGES = ("same-origin", "none")  # Fetch Metadata: mnm's own page, or typed by hand
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _github_ref(text: str) -> GitHubRef:
    """The page's `ref`: a GitHub pack's reference, or a worded refusal (never a 500, never a
    Registry look-up from the GitHub route)."""
    try:
        asked = parse_github(text.strip()[:300]) if text.strip() else None
    except PackError as exc:
        raise InstanceError(str(exc)) from exc
    if asked is None:
        raise InstanceError(
            f"'{text.strip()[:80]}' is not a GitHub pack: `github:owner/repo@<tag or commit>`."
        )
    return asked


def register_comfy_pack_routes(  # noqa: PLR0913, PLR0915 - the tab's pages and seams
    app: FastAPI,
    render: Render,
    settings: Settings,
    engine: ComfyEngine,
    workflows: WorkflowStore,
    recipes: Callable[[], list[Recipe]],
    jobs: JobRegistry,
    *,
    registry: Registry | None = None,
    github: GitHub | None = None,
) -> None:
    store = PackStore(settings.expanded_state_dir())
    hub = registry or Registry()
    git = github or GitHub(SubprocessRunner())
    models = Models(settings.expanded_models_dir(), settings.expanded_state_dir())

    def approvals() -> list[Approval]:
        try:
            return store.approvals()
        except PackError as exc:
            raise InstanceError(str(exc)) from exc

    def installed_ledger() -> Ledger | None:
        try:
            return engine.ledger()
        except ComfyError as exc:  # a worded page, never a 500
            raise InstanceError(str(exc)) from exc

    @app.get("/comfy/packs", response_class=HTMLResponse, response_model=None)
    def packs_page(request: Request, pack: str = "") -> HTMLResponse | RedirectResponse:
        if pack.strip():  # the page's look-up box: a Registry id, or a GitHub pack
            try:
                github_pack = parse_github(pack.strip()[:300]) is not None
            except PackError as exc:  # a malformed GitHub reference: worded, never a 500
                raise InstanceError(str(exc)) from exc
            if github_pack:
                return RedirectResponse(f"/comfy/packs-github?ref={quote(pack.strip())}", 303)
            return RedirectResponse(f"/comfy/packs/{quote(pack.strip(), safe='')}", 303)
        ledger = installed_ledger()
        installed = ledger.packs if ledger else []
        approved = approvals()
        rows = [(row, standing(row.pack, row.version, approved, installed)) for row in VETTED]
        others = [a for a in approved if not a.vetted]
        return render(
            request,
            "comfy_packs.html",
            tab="comfy",
            rows=rows,
            others=others,
            installed=installed,
            ledger=ledger,
            weights=missing_weights(models, approved),
            job=jobs.latest_for(PACKS_JOB),
        )

    @app.get("/comfy/packs-github", response_class=HTMLResponse)
    def github_page(request: Request, ref: str = "") -> HTMLResponse:
        """A pack off the Registry at a release tag or commit: its code fetched with git (a
        few seconds to a minute) to show its commit, licence and pip packages, then the same
        Approve form. Fetching runs git, so only mnm's own pages ask it (never a link
        another site's page follows)."""
        site = request.headers.get("sec-fetch-site")
        if site is not None and site not in OWN_PAGES:
            raise InstanceError("A GitHub pack is looked up from mnm's own Node packs page.")
        asked = _github_ref(ref)
        try:
            found = assess(hub, asked.words, None, git)
        except PackError as exc:
            raise InstanceError(str(exc)) from exc
        return render(
            request, "comfy_pack.html", tab="comfy", found=found, sandbox_note=SANDBOX_NOTE
        )

    @app.post("/comfy/packs-github/approve")
    def approve_github(
        ref: Annotated[str, Form(max_length=300)],
        commit: Annotated[str, Form(max_length=40)],
        sha256: Annotated[str, Form(max_length=64)],
        confirm: Annotated[str, Form(max_length=8)] = "",
    ) -> RedirectResponse:
        """Approves the code the page showed: fetched again in a job (git, off the request),
        and refused when the tag now names another commit or the code packs differently."""
        asked = _github_ref(ref)
        if not (_COMMIT.fullmatch(commit) and _SHA256.fullmatch(sha256)):
            raise InstanceError("The page's commit and sha256 did not come with it: look again.")

        def work(report: Callable[[str], None]) -> list[str]:
            report(f"Fetching {asked.words} again, to approve the code the page showed")
            found = assess(hub, asked.words, None, git)
            check_approvable(found, confirm == "yes")
            assert found.fetched is not None  # `check_approvable` refused otherwise
            if found.fetched.commit != commit:
                raise PackError(
                    f"{asked.words} names {found.fetched.commit[:12]} now, not the "
                    f"{commit[:12]} the page showed: look it up again."
                )
            if found.fetched.sha256 != sha256:
                raise PackError(
                    f"{asked.words} packs to {found.fetched.sha256[:12]}… now, not the "
                    f"{sha256[:12]}… the page showed: look it up again."
                )
            return [f"approved {approve_assessed(hub, store, found, confirm == 'yes').pack}"]

        return job(f"approve {asked.repo}", work)

    @app.get("/comfy/packs/{pack}", response_class=HTMLResponse)
    def pack_page(request: Request, pack: str, version: str = "") -> HTMLResponse:
        try:
            found = assess(hub, pack, version or None)
        except PackError as exc:
            raise InstanceError(str(exc)) from exc
        return render(
            request, "comfy_pack.html", tab="comfy", found=found, sandbox_note=SANDBOX_NOTE
        )

    @app.post("/comfy/packs/{pack}/approve")
    def approve_pack(
        pack: str,
        version: Annotated[str, Form(max_length=64)] = "",
        confirm: Annotated[str, Form(max_length=8)] = "",
    ) -> RedirectResponse:
        """Refusals at once; the zip (up to 100 MiB) downloads in a job, off the request."""
        confirmed = confirm == "yes"
        try:
            found = assess(hub, pack, version or None)
            check_approvable(found, confirmed)
        except PackError as exc:
            raise InstanceError(str(exc)) from exc

        def work(report: Callable[[str], None]) -> list[str]:
            report(f"Downloading {found.pack} {found.version}")
            return [f"approved {approve_assessed(hub, store, found, confirmed).pack}"]

        return job(f"approve {found.pack}", work)

    @app.post("/comfy/packs/{pack}/remove")
    def remove_pack(pack: str) -> RedirectResponse:
        try:
            store.remove(pack)
        except PackError as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse("/comfy/packs", status_code=303)

    def job(title: str, work: Callable[[Callable[[str], None]], list[str]]) -> RedirectResponse:
        def run(report: Callable[[str], None]) -> list[str]:
            try:
                return work(report)
            except (PackError, ComfyError, ModelError, OSError) as exc:
                raise InstanceError(str(exc)) from exc

        start_comfy_job(jobs, engine, PACKS_JOB, title, run, BUILD_LINES)  # one at a time
        return RedirectResponse("/comfy/packs", status_code=303)

    @app.post("/comfy/packs/install")
    def install() -> RedirectResponse:
        def work(report: Callable[[str], None]) -> list[str]:
            approved = store.approvals()
            weights = missing_weights(models, approved)
            if weights:
                fetch_all(models, weights, settings, recipes(), report)
            ledger = PackBuilder(engine, store, hub).install(report)
            return [f"installed {', '.join(ledger.packs) or 'no packs'}"]

        return job("install node packs", work)

    @app.post("/comfy/packs/rollback")
    def rollback() -> RedirectResponse:
        def work(report: Callable[[str], None]) -> list[str]:
            ledger = PackBuilder(engine, store).rollback(report)
            return [f"back to {', '.join(ledger.packs) or 'no packs'}"]

        return job("roll back node packs", work)

    @app.post("/comfy/{name}/packs")
    def add_for(name: str) -> RedirectResponse:
        """A workflow's vetted packs approved at once (their zips in a job); any other opens
        its page."""
        try:
            report = analyze(workflows.workflow(name), engine.object_info())
            wanted = [p for p in report.packs if not p.installed]
            approved = {known for a in store.approvals() for known in a.names}
            todo = []
            for use in wanted:
                if use.pack.lower() in approved:
                    continue
                if "/" in use.pack:  # a GitHub pack: its own page, at the commit it saved
                    asked = quote(use.reference, safe="")
                    return RedirectResponse(f"/comfy/packs-github?ref={asked}", status_code=303)
                found = assess(hub, use.pack)
                if found.row is None:
                    return RedirectResponse(f"/comfy/packs/{found.pack}", status_code=303)
                check_approvable(found, confirmed=False)
                todo.append(found)
        except (PackError, WorkflowStoreError, ValueError, OSError) as exc:
            raise InstanceError(str(exc)) from exc
        if not todo:
            return RedirectResponse("/comfy/packs", status_code=303)

        def work(say: Callable[[str], None]) -> list[str]:
            done = []
            for found in todo:
                say(f"Downloading {found.pack} {found.version}")
                done.append(approve_assessed(hub, store, found, confirmed=False).pack)
            return [f"approved {', '.join(done)}"]

        return job(f"approve {name}'s packs", work)
