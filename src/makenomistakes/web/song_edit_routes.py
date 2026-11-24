"""The Song page's edits: an edit of the take in use posted from its form (the range
from the take's waveform: a click sets the start, the next the end), and the take the song
uses chosen from its edits (Undo). Each runs as the song's job (`song_routes.song_job`). An
edit's fields, the take it is made from and the ACE-Step that plays it, and a take asked
for, are checked before the job starts, so a wrong one is a 400 at once; what needs the take
itself (its length, its parts, a swap's stems) is the job's refusal, said on the page."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobRegistry
from makenomistakes.pipeline import song_changes, song_edits, song_takes
from makenomistakes.pipeline.song_runner import SongRunner
from makenomistakes.pipeline.stage_runner import refuse_if_going
from makenomistakes.studio.edit_base import EditError, EditHost
from makenomistakes.studio.edits import ask_of
from makenomistakes.web.song_routes import page, song_job, song_of


def register_song_edit_routes(app: FastAPI, runner: SongRunner, jobs: JobRegistry) -> None:
    store = runner.store

    @app.post("/songs/{run_id}/edit")
    async def edit(request: Request, run_id: str) -> RedirectResponse:
        run = song_of(store, run_id)
        form = await request.form()
        values = {}
        for key in form:  # every field posted: one the edit does not take is refused
            given = [str(v) for v in form.getlist(key) if isinstance(v, str)]
            if key != "edit" and given:  # a box's hidden `off` first, its `on` last if ticked
                values[key] = given[-1]
        try:
            ask = ask_of(str(form.get("edit", "")), values)
        except EditError as exc:
            raise InstanceError(str(exc)) from exc
        refuse_if_going(run, "edit it")  # each said now as a 400, not in the job
        song_takes.take(store, run)
        if ask.host is EditHost.ACE_STEP:  # a section tool needs no model
            song_edits.editor(runner.service, runner.state_dir, run)
        song_job(jobs, run.id, ask.words(), lambda report: runner.edit(run.id, ask, report))
        return page(run.id)

    @app.post("/songs/{run_id}/take")
    def take(run_id: str, take: Annotated[str, Form(max_length=80)] = "") -> RedirectResponse:
        run = song_of(store, run_id)
        refuse_if_going(run, "choose a take")  # said now as a 400, not in the job
        chosen = song_changes.chosen(store, run, take or None)
        words = f"the take of {chosen}" if chosen else "the candidate's take"
        song_job(jobs, run.id, words, lambda report: runner.use_take(run.id, chosen, report))
        return page(run.id)
