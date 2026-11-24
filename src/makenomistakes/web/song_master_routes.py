"""The Song page's master: the mix in use mastered to an uploaded reference track (a
FLAC or MP3 fits the page's upload limit; a long WAV is the CLI's `mnm song master
--reference`), an earlier master used again, or none. Each runs as the song's job
(`song_routes.song_job`); what can be refused at once (no studio tools, no mix, a master of
another mix) is a 400 before the job starts."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobRegistry
from makenomistakes.pipeline import song_master
from makenomistakes.pipeline.song_runner import SongRunner
from makenomistakes.pipeline.stage_runner import refuse_if_going
from makenomistakes.studio.tools_answers import ToolsError
from makenomistakes.web.song_routes import page, song_job, song_of
from makenomistakes.web.try_routes import upload


def register_song_master_routes(app: FastAPI, runner: SongRunner, jobs: JobRegistry) -> None:
    store = runner.store

    @app.post("/songs/{run_id}/master")
    def master(
        run_id: str, reference: Annotated[UploadFile | None, File()] = None
    ) -> RedirectResponse:
        run = song_of(store, run_id)
        given = upload(reference, "reference")
        if given is None:
            raise InstanceError("Choose a reference track to master the mix to.")
        refuse_if_going(run, "master it")  # each said now as a 400, not in the job
        unmixed = song_master.no_mix(run)
        if unmixed is not None:
            raise InstanceError(unmixed)
        try:
            runner.tools.installed()
        except ToolsError as exc:
            raise InstanceError(str(exc)) from exc
        name, data = given
        words = f"master to {name}"
        song_job(jobs, run.id, words, lambda report: runner.master(run.id, name, data, report))
        return page(run.id)

    @app.post("/songs/{run_id}/master/use")
    def use(run_id: str, master: Annotated[str, Form(max_length=80)] = "") -> RedirectResponse:
        """The master the export uses: one of this song's (empty: none)."""
        run = song_of(store, run_id)
        refuse_if_going(run, "choose its master")
        chosen = master or None
        why = song_master.refusal(store, run, chosen) if chosen is not None else None
        if why is not None:
            raise InstanceError(f"{why.words}: {song_master.remedy(why, run.id)}.")
        words = f"use master {chosen}" if chosen else "no master"
        song_job(jobs, run.id, words, lambda report: runner.use_master(run.id, chosen, report))
        return page(run.id)
