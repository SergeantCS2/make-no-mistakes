"""System → Backup: the backups folder's archives with their sizes, ages and what the
rotation keeps each as, the last backup and restore test, the nightly timer, and Back up now
(`mnm backup --auto` as a job: written, restore-tested, rotated; a failure notifies, as the
timer's does)."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes import backup_timer
from makenomistakes.backup_auto import Tone, auto_backup, read_last, run_words
from makenomistakes.backup_rotation import age, keep, listing, size_words
from makenomistakes.config import Settings
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobFailed, JobKind, JobRegistry
from makenomistakes.measurements import gpu_name
from makenomistakes.notify import Notice, notifier
from makenomistakes.web.common import Render
from makenomistakes.web.system_data import SystemProbe

BACKUP_JOB = "system:backup"


@dataclass(frozen=True)
class Row:
    name: str
    size: str
    age: str
    kept: str


def rows(settings: Settings) -> list[Row]:
    found = listing(settings.expanded_backups_dir())
    kept = keep(found)
    now = datetime.now()
    return [
        Row(
            b.name,
            size_words(b.size),
            age(b.taken, now),
            kept[b.path].value if b.path in kept else "removed at the next backup",
        )
        for b in found
    ]


def _quiet(settings: Settings) -> Callable[[Notice], None]:
    sender = notifier(settings)

    def tell(notice: Notice) -> None:
        with contextlib.suppress(Exception):  # best effort: the job's outcome stands
            sender.send(notice)

    return tell


def register_backup_routes(
    app: FastAPI,
    render: Render,
    settings: Settings,
    service: InstanceService,
    jobs: JobRegistry,
    system: SystemProbe,
) -> None:
    @app.get("/system/backup", response_class=HTMLResponse)
    def backup_page(request: Request) -> HTMLResponse:
        return render(
            request,
            "backup.html",
            folder=settings.expanded_backups_dir(),
            rows=rows(settings),
            last=read_last(settings.expanded_state_dir()),
            timer=backup_timer.timer_state(system.runner).line,
            job=jobs.latest_for(BACKUP_JOB),
        )

    @app.post("/system/backup/now")
    def backup_now() -> RedirectResponse:
        """`mnm backup --auto`: the timer's path, from a button."""

        def run(report: Callable[[str], None]) -> list[str]:
            report("writing the backup, then restoring it in a scratch state…")
            done = auto_backup(settings, gpu_name(service.memory), _quiet(settings))
            lines = [text.strip() for _tone, text in run_words(done)]
            if not done.ok:
                bad = [t for tone, t in run_words(done) if tone in (Tone.BAD, Tone.WARN)]
                raise JobFailed(bad[0] if bad else "the backup failed", lines)
            return lines

        jobs.start_unless_active(JobKind.BACKUP, BACKUP_JOB, run, label="backup --auto")
        return RedirectResponse("/system/backup", status_code=303)
