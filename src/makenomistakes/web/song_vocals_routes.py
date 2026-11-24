"""The Song page's vocals panel: what is done to the separator's vocals before a
voice sings them (`mnm song vocals`), run as the song's job (`song_routes.song_job`); what
can be refused at once (an unknown clean-up, the dereverb not added, the tools missing for
tuning, a key mnm does not read) is a 400 before the job starts."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Form
from fastapi.responses import RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobRegistry
from makenomistakes.pipeline import song_vocals
from makenomistakes.pipeline.song_runner import SongRunner
from makenomistakes.pipeline.stage_runner import refuse_if_going
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.web.song_routes import page, song_job, song_of


def register_song_vocals_routes(app: FastAPI, runner: SongRunner, jobs: JobRegistry) -> None:
    store = runner.store

    @app.post("/songs/{run_id}/vocals")
    def vocals(
        run_id: str,
        clean: Annotated[list[str] | None, Form(max_length=len(song_vocals.ORDER))] = None,
        key: Annotated[str, Form(max_length=20)] = "",
        strength: Annotated[int, Form(ge=0, le=100)] = song_vocals.DEFAULT_STRENGTH,
    ) -> RedirectResponse:
        run = song_of(store, run_id)
        refuse_if_going(run, "clean its vocals")
        asked = ",".join(clean or []) or "none"
        try:  # each said now as a 400, not in the job
            song_vocals.settings_with(
                run.settings, asked, key or None, strength, runner.service, runner.tools,
                store=store, run_id=run.id,
            )  # fmt: skip
        except PipelineError as exc:
            raise InstanceError(str(exc)) from exc
        words = f"vocals: {asked}"
        song_job(
            jobs,
            run.id,
            words,
            lambda report: runner.set_vocals(run.id, asked, key or None, strength, report),
        )
        return page(run.id)
