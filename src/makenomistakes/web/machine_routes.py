"""System → Machine: this box as `machine.json` last recorded it (`mnm machine show`'s
lines, uuids masked: the web never shows them), the last adoption's table, and Adopt: `mnm
machine adopt` as a job, with its Dry run, and the download question as a checkbox (the
sizes are the last dry run's, when one ran)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.adopt_live import live_deps
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobFailed, JobKind, JobRegistry
from makenomistakes.machine import saved_recipes
from makenomistakes.machine_adopt import (
    AdoptError,
    Adoption,
    AdoptOptions,
    adopt,
    gib,
    table_text,
    tables,
)
from makenomistakes.machine_facts import MachineFactsError, load_file
from makenomistakes.machine_show import show_lines
from makenomistakes.measurements import MeasurementStore
from makenomistakes.readings_filter import per_box
from makenomistakes.web.common import Render

ADOPT_JOB = "system:adopt"


@dataclass(frozen=True)
class Planned:
    """What the last dry run would download, for the checkbox's words."""

    files: int
    known_bytes: int
    unknown: int

    @property
    def words(self) -> str:
        if not self.files:
            return "download what the new plans need (the last dry run listed nothing)"
        extra = f" and {self.unknown} of unknown size" if self.unknown else ""
        return (
            f"download {self.files} file{'s' * (self.files != 1)}, {gib(self.known_bytes)}{extra}"
        )


def planned_of(done: Adoption) -> Planned:
    fetch = [f for row in done.rows if row.kept is None for f in row.fetch]
    return Planned(
        len(fetch), sum(f.size or 0 for f in fetch), sum(1 for f in fetch if f.size is None)
    )


def register_machine_routes(
    app: FastAPI, render: Render, service: InstanceService, jobs: JobRegistry
) -> None:
    state_dir = service.settings.expanded_state_dir
    last_dry: list[Planned] = []  # the newest dry run's downloads (this process only)

    @app.get("/system/machine", response_class=HTMLResponse)
    def machine_page(request: Request) -> HTMLResponse:
        try:
            found = load_file(state_dir())
            error = None
        except MachineFactsError as exc:
            found, error = None, str(exc)
        lines: list[str] = []
        if found is not None:
            counts = per_box(MeasurementStore(state_dir()), found)
            lines = show_lines(found, saved_recipes(service.settings), False, counts)
        kept = tables(state_dir())
        return render(
            request,
            "machine.html",
            lines=lines,
            error=error,
            last_table=kept[-1].read_text(encoding="utf-8") if kept else None,
            planned=last_dry[-1] if last_dry else None,
            job=jobs.latest_for(ADOPT_JOB),
        )

    @app.post("/system/machine/adopt")
    def adopt_now(
        download: Annotated[str, Form()] = "",
        dry_run: Annotated[str, Form()] = "",
        tune: Annotated[str, Form()] = "",  # an unticked box is left out of the POST
    ) -> RedirectResponse:
        """`mnm machine adopt` (`--dry-run` from its own button; the checkbox is `--yes`)."""
        dry = dry_run == "on"
        options = AdoptOptions(yes=download == "on", tune=tune == "on", dry_run=dry)

        def run(report: Callable[[str], None]) -> list[str]:
            said: list[str] = []

            def say(line: str) -> None:
                said.append(line)
                report(line)

            deps = live_deps(service, say, lambda _question: options.yes)
            try:
                done = adopt(deps, options)
            except (AdoptError, InstanceError) as exc:
                raise JobFailed(str(exc), said) from exc
            if dry:
                last_dry[:] = [planned_of(done)]
            return table_text(done).splitlines()

        label = "machine adopt --dry-run" if dry else "machine adopt"
        jobs.start_unless_active(JobKind.ADOPT, ADOPT_JOB, run, label=label, keep_lines=40)
        return RedirectResponse("/system/machine", status_code=303)
