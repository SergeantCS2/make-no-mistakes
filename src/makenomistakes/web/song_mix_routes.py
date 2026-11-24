"""The Song page's mixer and export:
Render a mix at the mixer's levels, which mix the export uses, the waveforms an older run
lacks, and Export again with its options. Each move that sends work runs as the song's job
(`song_routes.song_job`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from starlette.datastructures import FormData

from makenomistakes.audio_format import Loudness
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobRegistry
from makenomistakes.pipeline import song_mix
from makenomistakes.pipeline.song_runner import SongRunner
from makenomistakes.runs import RunRecord
from makenomistakes.studio import export_mix
from makenomistakes.studio.export import ExportOptions
from makenomistakes.studio.mixer import MixSettings, StemLevel
from makenomistakes.web.song_routes import page, song_job, song_key, song_of

FIELD_WORDS = {"gain_db": "level", "fade_in": "fade in", "fade_out": "fade out"}


def register_song_mix_routes(app: FastAPI, runner: SongRunner, jobs: JobRegistry) -> None:
    store = runner.store

    @app.post("/songs/{run_id}/mix")
    async def mix(request: Request, run_id: str) -> RedirectResponse:
        """Render: the stems at the mixer's levels into a new mix (a child run)."""
        run = song_of(store, run_id)
        settings = _levels_from(await request.form(), song_mix.stem_names(run))

        def work(report: Callable[[str], None]) -> RunRecord:
            report(f"rendering a mix: {settings.words()}…")
            runner.render_mix(run_id, settings)
            return store.get(run_id)

        song_job(jobs, run_id, "mix", work)
        return page(run_id)

    @app.post("/songs/{run_id}/mix/use")
    def use_mix(run_id: str, mix: Annotated[str, Form(max_length=80)] = "") -> RedirectResponse:
        """Which mix the export uses: a render of this song, or (empty) the song's own."""
        busy = jobs.active_for(song_key(run_id))
        if busy is not None:
            raise InstanceError(f"{run_id} is busy ({busy.title}); choose the mix after it.")
        runner.use_mix(run_id, mix or None)  # a pass running elsewhere refuses it, worded
        return page(run_id)

    @app.post("/songs/{run_id}/draw")
    def draw(run_id: str) -> RedirectResponse:
        """Measure the waveforms an older run lacks."""
        song_of(store, run_id)

        def work(report: Callable[[str], None]) -> RunRecord:
            report("measuring the waveforms…")
            runner.draw(run_id)
            return store.get(run_id)

        song_job(jobs, run_id, "waveforms", work)
        return page(run_id)

    @app.post("/songs/{run_id}/export")
    def export(
        run_id: str,
        mp3: Annotated[str, Form()] = "",
        release: Annotated[str, Form()] = "",
        loudness: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        """Export again; an unticked box sends nothing, so the loudness line is on only when
        its box is (else an unticked "as summed" normalises)."""
        run = song_of(store, run_id)
        line = export_mix.line_of(run.settings) or Loudness()  # the user's own line, kept
        options = ExportOptions(
            mp3=mp3 == "true",
            release=release == "true",
            loudness=line if loudness == "on" else None,
        )

        def work(report: Callable[[str], None]) -> RunRecord:
            report("rendering the export folder…")
            runner.export(run_id, options)
            return store.get(run_id)

        song_job(jobs, run_id, "export", work)
        return page(run_id)


def _levels_from(form: FormData, stems: list[str]) -> MixSettings:
    """The mixer's form: per stem `gain_<stem>` (dB), `fade_in_<stem>`, `fade_out_<stem>`
    (seconds) and the `mute_<stem>`/`solo_<stem>` boxes."""
    if not stems:
        raise InstanceError("This song has no stems yet: the mixer follows the stems.")
    levels = []
    for stem in stems:
        try:
            levels.append(
                StemLevel(
                    stem=stem,
                    gain_db=float(str(form.get(f"gain_{stem}", "0")) or 0),
                    mute=form.get(f"mute_{stem}") == "true",
                    solo=form.get(f"solo_{stem}") == "true",
                    fade_in=float(str(form.get(f"fade_in_{stem}", "0")) or 0),
                    fade_out=float(str(form.get(f"fade_out_{stem}", "0")) or 0),
                )
            )
        except ValidationError as exc:
            first = exc.errors()[0]
            field = FIELD_WORDS.get(str(first["loc"][0]), str(first["loc"][0]))
            raise InstanceError(f"The mixer, {stem}'s {field}: {first['msg']}.") from exc
        except ValueError as exc:
            raise InstanceError(f"The mixer, {stem}: a number is needed ({exc}).") from exc
    return MixSettings(levels=levels)
