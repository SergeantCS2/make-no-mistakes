"""Web routes for workflows: the list with the fit-together verdict, create, the detail page
with every member's cards, start (a job), stop, remove."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from makenomistakes.consumers import CONSUMER_LABELS, Consumer
from makenomistakes.harness_agents import AGENT_CLIENTS
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.progress_words import parked_for_room
from makenomistakes.web.common import Render
from makenomistakes.web.instance_routes import known_consumer, refuse_if_busy, start_notes
from makenomistakes.workflows import ROLE_ORDER, ROLES, WorkflowService


def job_key(name: str) -> str:
    """Jobs are keyed by instance; a workflow's start job gets its own key."""
    return f"workflow:{name}"


def register_workflow_routes(
    app: FastAPI, workflows: WorkflowService, jobs: JobRegistry, render: Render
) -> None:
    instances = workflows.instances

    @app.get("/workflows", response_class=HTMLResponse)
    def workflows_page(request: Request) -> HTMLResponse:
        snapshot = instances.memory()
        rows = []
        for workflow in workflows.all():
            members = workflows.members(workflow, snapshot)
            rows.append((workflow, members, workflows.listed(workflow, members, snapshot)))
        return render(
            request,
            "workflows.html",
            rows=rows,
            by_role=by_role(),
            roles=ROLE_ORDER,
            snapshot=snapshot,
            jobs=[j for j in jobs.recent(10) if j.instance.startswith("workflow:")],
        )

    @app.post("/workflows")
    async def create(
        request: Request,
        name: Annotated[str, Form()],
        swap: Annotated[bool, Form()] = False,
        persistent: Annotated[bool, Form()] = False,
    ) -> RedirectResponse:
        """Async only to read the form; the stores' reads and writes run on a worker thread."""
        form = await request.form()
        members = {
            role: str(form[role]).strip() for role in ROLE_ORDER if str(form.get(role, "")).strip()
        }
        # `--swap`: members take turns
        await run_in_threadpool(
            workflows.create, name.strip(), members, swap=swap, persistent=persistent
        )
        return RedirectResponse(f"/workflows/{name.strip()}", status_code=303)

    def by_role() -> dict[str, list[str]]:
        """The instances each role can take, for the create and edit forms: from the saved
        recipes alone (no container is asked about)."""
        recipes = instances.all()
        return {
            role: [r.name for r in recipes if r.modality in types] for role, types in ROLES.items()
        }

    @app.post("/workflows/{name}/members")
    async def members(request: Request, name: str) -> RedirectResponse:
        """The set page's member form: a role picked is set, a role emptied is dropped."""
        form = await request.form()
        picked = {r: str(form.get(r, "")).strip() for r in ROLE_ORDER}
        current = (await run_in_threadpool(workflows.get, name)).members
        given = {r: i for r, i in picked.items() if i and current.get(r) != i}
        drop = [r for r in current if not picked.get(r)]
        await run_in_threadpool(workflows.edit, name, given, drop)
        return RedirectResponse(f"/workflows/{name}", status_code=303)

    @app.post("/workflows/{name}/persistent")
    def persistent(name: str, persistent: Annotated[bool, Form()]) -> RedirectResponse:
        """`mnm sets edit <name> --persistent/--no-persistent`."""
        workflows.edit(name, {}, [], persistent)
        return RedirectResponse(f"/workflows/{name}", status_code=303)

    @app.get("/workflows/{name}", response_class=HTMLResponse)
    def workflow_page(
        request: Request, name: str, consumer: str = Consumer.ODYSSEUS.value
    ) -> HTMLResponse:
        known_consumer(consumer)
        workflow = workflows.get(name)
        snapshot = instances.memory()
        members = workflows.members(workflow, snapshot)
        chosen = Consumer(consumer)
        return render(
            request,
            "workflow.html",
            workflow=workflow,
            members=members,
            verdict=workflows.fit_together(members, snapshot, owner=workflows.owner(workflow)),
            cards_by_member=workflows.cards(workflow, chosen),
            consumer=chosen,
            consumers={c: n for c, n in CONSUMER_LABELS.items() if c not in AGENT_CLIENTS},
            agents={c: n for c, n in CONSUMER_LABELS.items() if c in AGENT_CLIENTS},
            job=jobs.latest_for(job_key(name)),  # how the last start ended stays shown
            snapshot=snapshot,
            by_role=by_role(),
            roles=ROLE_ORDER,
        )

    @app.post("/workflows/{name}/start")
    def start(name: str) -> RedirectResponse:
        workflow = workflows.get(name)

        def run(report: Callable[[str], None]) -> list[str]:
            done = workflows.start(workflow, report, anyway=True)  # the page confirmed
            parked = [parked_for_room(done.parked, name)] if done.parked else []
            return parked + [note for s in done.started for note in start_notes(s)]

        jobs.start_unless_active(JobKind.START, job_key(name), run)
        return RedirectResponse(f"/workflows/{name}", status_code=303)

    @app.post("/workflows/{name}/stop")
    def stop(name: str) -> RedirectResponse:
        workflows.stop(workflows.get(name))
        return RedirectResponse(f"/workflows/{name}", status_code=303)

    @app.post("/workflows/{name}/remove")
    def remove(name: str) -> RedirectResponse:
        refuse_if_busy(jobs, job_key(name), "remove it")  # its start is still under way
        workflows.remove(name)
        jobs.forget(job_key(name))  # a workflow created again under this name starts clean
        return RedirectResponse("/workflows", status_code=303)
