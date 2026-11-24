"""The Song page in the web UI: the start form, a song
run's page with the lyrics editor (a section at a time, or Replace all), the candidate cards
with a player, their scores and Pick, the take's structure map, the stems with their
waveforms; the mixer and Export are `song_mix_routes`'. Every move that sends
requests runs as a job keyed `pipeline:<run id>`, so a page never blocks; the page polls the
job as every page does."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.datastructures import FormData

from makenomistakes.audio_format import Loudness
from makenomistakes.hub.facts import CHAT_LIKE
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.pipeline import (
    song,
    song_changes,
    song_master,
    song_midi,
    song_mix,
    song_takes,
    song_vocals,
)
from makenomistakes.pipeline.song import Lyrics, SongError, SongInputs, lyrics_from_text
from makenomistakes.pipeline.song_runner import SongRunner
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.pipeline.song_tools import read_structure
from makenomistakes.pipeline.steps import StepError
from makenomistakes.runs import MEDIA_TYPES, RunError, RunRecord, RunState, RunStore
from makenomistakes.studio import export_mix, peaks, stages
from makenomistakes.studio.edits import ASKS
from makenomistakes.studio.export import listing
from makenomistakes.studio.mixer import MAX_DB, MAX_FADE, MIN_DB
from makenomistakes.studio.stage_instances import players
from makenomistakes.studio.stages import StudioError
from makenomistakes.studio.tools import NOT_INSTALLED
from makenomistakes.studio.tools_answers import AXIS_WORDS, ToolsError
from makenomistakes.web.common import Render
from makenomistakes.web.try_routes import upload

Work = Callable[[Callable[[str], None]], RunRecord]
# the edits' forms, and those a click on the take's waveform fills a range in
EDIT_FORMS = [(edit.value, ask, ask.form()) for edit, ask in ASKS.items()]
EDIT_RANGES = " ".join(
    f"edit-{name}" for name, _ask, form in EDIT_FORMS if any(f.name == "start" for f in form)
)


def song_key(run_id: str) -> str:
    return f"pipeline:{run_id}"


@dataclass(frozen=True)
class Wave:
    """A file's waveform as the page draws it: the SVG path, its width in bars, its length."""

    path: str
    bars: int
    seconds: float


def _wave(folder: Path, name: str) -> Wave | None:
    found = peaks.kept(folder, name)
    if found is None:
        return None
    path, bars = peaks.svg_path(found.peaks)
    return Wave(path, bars, found.seconds)


def song_job(jobs: JobRegistry, run_id: str, title: str, work: Work) -> None:
    """A move on a song as a job keyed `pipeline:<run id>`: one at a time per song."""
    busy = jobs.active_for(song_key(run_id))
    if busy is not None:
        raise InstanceError(f"{run_id} is busy ({busy.title}); wait for it.")

    def job(report: Callable[[str], None]) -> list[str]:
        run = work(report)
        if run.state is RunState.FAILED:
            raise InstanceError(run.error)
        return [*run.notes, _where(run)]

    jobs.start_unless_active(JobKind.PIPELINE, song_key(run_id), job, label=title)


def song_of(store: RunStore, run_id: str) -> RunRecord:
    """A song run by id, refused as a 400 page when it is none (a mix or an edit points at
    its song)."""
    try:
        run = store.get(run_id)
    except RunError as exc:
        raise InstanceError(str(exc)) from exc
    what = song.CHILD_ROLES.get(run.role or "") if run.parent else None
    if what is not None:
        raise InstanceError(f"{run_id} is {what[0]} of the song {run.parent}: its page has it.")
    if not song.is_song(run):
        raise InstanceError(f"{run_id} is not a song run (`Runs` has it).")
    return run


def page(run_id: str) -> RedirectResponse:
    return RedirectResponse(f"/songs/{run_id}", status_code=303)


def register_song_routes(  # noqa: PLR0915 - the page and its moves, one place
    app: FastAPI, render: Render, runner: SongRunner, jobs: JobRegistry
) -> None:
    store = runner.store

    @app.get("/songs", response_class=HTMLResponse)
    def songs(request: Request) -> HTMLResponse:
        runs = [r for r in store.all() if song.is_song(r)]
        recipes = runner.service.all()
        return render(
            request,
            "songs.html",
            tab="runs",
            runs=runs,
            chats=[r.name for r in recipes if r.modality in CHAT_LIKE],
            problem=_stage_problem(runner),
            max_candidates=song.MAX_CANDIDATES,
            default_candidates=song.DEFAULT_CANDIDATES,
            uses=list(song.ROUTES),
            voices=[r.name for r in runner.library.all()],
        )

    @app.post("/songs")
    def start(  # noqa: PLR0913, PLR0917 - the form's fields
        theme: Annotated[str, Form(max_length=4000)],
        length: Annotated[str, Form()] = "30s",
        candidates: Annotated[int, Form(ge=1, le=song.MAX_CANDIDATES)] = song.DEFAULT_CANDIDATES,
        tier: Annotated[str, Form()] = "fast",
        chat: Annotated[str, Form()] = song.DESK_CHAT,
        tags: Annotated[str, Form(max_length=600)] = "",
        lyrics: Annotated[str, Form(max_length=20_000)] = "",
        use: Annotated[str, Form()] = "",
        auto: Annotated[str, Form()] = "",
        voice: Annotated[str, Form(max_length=64)] = "",
        shift: Annotated[int, Form(ge=-12, le=12)] = 0,
        recording: Annotated[UploadFile | None, File()] = None,
    ) -> RedirectResponse:
        take = upload(recording, "recording")
        try:
            inputs = SongInputs.model_validate(
                {
                    "theme": theme.strip(), "seconds": song.seconds_of(length),
                    "candidates": candidates, "tier": tier, "chat": chat or song.DESK_CHAT,
                    "tags": tags.strip() or None, "recording": use or None, "auto": auto == "true",
                    "voice": voice or None, "voice_shift": shift,
                }
            )  # fmt: skip
        except (ValueError, SongError) as exc:
            raise InstanceError(f"The song's form: {exc}") from exc
        if take is not None and inputs.recording is None:
            raise InstanceError("Say how the recording is used: complete, vocal2bgm or reference.")
        if take is None and inputs.recording is not None:  # the page's words, not the CLI's
            raise InstanceError(
                f"A recording used as {inputs.recording} needs its file: choose one, or leave "
                "how it is used empty."
            )
        run = runner.start(inputs, lyrics.strip() or None, take if inputs.recording else None)
        song_job(jobs, run.id, "song", lambda report: runner.advance(run.id, report))
        return page(run.id)

    @app.get("/songs/{run_id}", response_class=HTMLResponse)
    def one(request: Request, run_id: str) -> HTMLResponse:
        with store.reading_once():  # its mixes, edits and masters from one read of the runs
            return _one(request, run_id)

    def _one(request: Request, run_id: str) -> HTMLResponse:
        run = song_of(store, run_id)
        lyrics = None
        try:
            lyrics = runner.lyrics(run)
        except StepError:
            lyrics = None  # not written yet, or its folder cleaned: the page says so
        folder = store.folder(run)
        candidates = runner.candidates(run)
        stems = song.stem_files([f.name for f in run.outputs])
        made = {f.name for f in run.outputs} if not run.files_removed else set()
        waves = {n: _wave(folder, n) for n in [*stems, song.MIX_FILE] if n in made}
        for _step, child in candidates:
            if child is not None and child.state is RunState.DONE and not child.files_removed:
                waves[child.id] = _wave(store.folder(child), CANDIDATE_FILE)
        mixes = song_mix.mixes(store, run)
        for mix in mixes:
            if mix.state is RunState.DONE and not mix.files_removed:
                waves[mix.id] = _wave(store.folder(mix), song.MIX_FILE)
        in_use = song_mix.in_use(store, run)
        edits = song_takes.edits(store, run)
        for edit in edits:
            if edit.state is RunState.DONE and not edit.files_removed:
                waves[edit.id] = _wave(store.folder(edit), CANDIDATE_FILE)
        takes = song_takes.shown(store, run)
        return render(
            request,
            "song.html",
            tab="runs",
            run=run,
            sung_workflow=stages.workflow_of(run.settings.get("song_instance")),
            edits=edits,
            take=takes.take,
            take_problem=takes.problem,
            take_lyrics=takes.lyrics,
            refused=takes.refused,
            edit_forms=EDIT_FORMS,
            edit_ranges=EDIT_RANGES,
            lyrics=lyrics,
            candidates=candidates,
            seeds={c.id: song.candidate_seed(c) for _s, c in candidates if c is not None},
            picked=song.picked_index(run.settings),
            pick_stage=runner.pick_stage(run),
            stems=stems,
            waves=waves,
            missing_waves=any(w is None for w in waves.values()),
            structure=read_structure(folder),
            mixes=mixes,
            mix_in_use=in_use.id if in_use else "",
            levels=song_mix.settings_of(store, run),
            export_behind=song_mix.export_behind(run),
            tools_missing=_tools_missing(runner) if _skipped_tools(run) else None,
            db_range=(MIN_DB, MAX_DB),
            max_fade=MAX_FADE,
            exported=listing(folder / song.EXPORT_FOLDER),
            job=jobs.latest_for(song_key(run_id)),
            labels=song.STAGE_LABELS,
            stages=song.STAGES,
            items=song.ITEMS,
            picks=song.PICKS,
            axis_words=AXIS_WORDS,
            voices=[r.name for r in runner.library.all()],
            voice_step=run.step(song.VOICE),
            vocals_step=run.step(song.VOCALS),
            vocals_asked=run.settings.get("vocals_clean", "").split(","),
            vocals_order=[c.value for c in song_vocals.ORDER],
            mastering=song_master.shown(store, run),
            midi=[] if run.files_removed else song_midi.midi_files([f.name for f in run.outputs]),
            midi_step=run.step(song.MIDI),
            RunState=RunState,
            loudness=export_mix.line_of(run.settings) or Loudness(),  # the line kept, or -14
            line_on=export_mix.line_of(run.settings) is not None,
        )

    @app.get("/songs/{run_id}/export/{name:path}")
    def export_file(run_id: str, name: str) -> FileResponse:
        run = song_of(store, run_id)
        folder = store.folder(run) / song.EXPORT_FOLDER
        if name not in listing(folder):
            raise InstanceError(f"The export of {run_id} has no file '{name}'.")
        path = folder / name
        media = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        headers = {"X-Content-Type-Options": "nosniff"}
        return FileResponse(path, media_type=media, filename=path.name, headers=headers)

    @app.post("/songs/{run_id}/go")
    def proceed(run_id: str) -> RedirectResponse:
        song_job(jobs, run_id, "song", lambda report: runner.proceed(run_id, report))
        return page(run_id)

    @app.post("/songs/{run_id}/resume")
    def resume(run_id: str) -> RedirectResponse:
        song_job(jobs, run_id, "song", lambda report: runner.advance(run_id, report))
        return page(run_id)

    @app.post("/songs/{run_id}/pick")
    def pick(
        run_id: str, number: Annotated[int, Form(ge=1, le=song.MAX_CANDIDATES)]
    ) -> RedirectResponse:
        song_of(store, run_id)
        song_job(jobs, run_id, f"pick {number}", lambda report: runner.pick(run_id, number, report))
        return page(run_id)

    @app.post("/songs/{run_id}/redo")
    def redo(
        run_id: str,
        stage: Annotated[str, Form()],
        index: Annotated[int, Form(ge=0, lt=song.MAX_CANDIDATES)] = 0,
    ) -> RedirectResponse:
        if stage not in song.STAGES:
            raise InstanceError(f"'{stage}' is not a step that can be made again.")
        song_job(
            jobs, run_id, f"redo {stage}", lambda report: runner.redo(run_id, stage, index, report)
        )
        return page(run_id)

    @app.post("/songs/{run_id}/voice")
    def voice(
        run_id: str,
        voice: Annotated[str, Form(max_length=64)] = "",
        shift: Annotated[int, Form(ge=-12, le=12)] = 0,
    ) -> RedirectResponse:
        """The vocals in a library voice, or (empty) the song model's own."""
        run = song_of(store, run_id)  # refused here as a 400, not later in the job
        song_changes.voice_settings(runner, run, voice or None, shift)
        title = f"voice {voice}" if voice else "the song model's voice"
        song_job(
            jobs,
            run_id,
            title,
            lambda report: runner.set_voice(run_id, voice or None, shift, report),
        )
        return page(run_id)

    @app.post("/songs/{run_id}/score")
    def score(run_id: str) -> RedirectResponse:
        """Score every candidate again (after the tools were installed)."""
        song_of(store, run_id)
        song_job(jobs, run_id, "score", lambda report: runner.rescore(run_id, report))
        return page(run_id)

    @app.post("/songs/{run_id}/lyrics")
    async def edit_lyrics(request: Request, run_id: str) -> RedirectResponse:
        busy = jobs.active_for(song_key(run_id))
        if busy is not None:
            raise InstanceError(f"{run_id} is busy ({busy.title}); edit the lyrics after it.")
        runner.edit_lyrics(run_id, _lyrics_from(await request.form()))
        return page(run_id)


def _skipped_tools(run: RunRecord) -> bool:
    """Whether a score or the structure map was skipped (the page then says why)."""
    tools = (song.SCORE, song.STRUCTURE)
    return any(s.stage in tools and s.state.value == "skipped" for s in run.steps)


def _tools_missing(runner: SongRunner) -> str | None:
    """Why the studio tools cannot score or map here, from their ledger alone (a page never
    waits on Docker; a step that found the image gone says so in its own error)."""
    try:
        if runner.tools.ledger() is None:
            return NOT_INSTALLED
    except ToolsError as exc:
        return str(exc)
    return None


def _lyrics_from(form: FormData) -> Lyrics:
    """The lyrics editor: `title`, `tags`, a section's `label_<n>` and `lines_<n>` (an empty
    section's lines drop it), or `whole` (Replace all) in place of the sections."""
    title = str(form.get("title", "")).strip()[:200]
    tags = str(form.get("tags", "")).strip()
    whole = str(form.get("whole", "")).strip()
    try:
        if whole:
            return lyrics_from_text(whole, tags, title)
        sections = []
        for number in range(song.MAX_SECTIONS):
            lines = [x for x in str(form.get(f"lines_{number}", "")).splitlines() if x.strip()]
            if lines:
                label = str(form.get(f"label_{number}", "")).strip() or "Verse"
                sections.append({"label": label, "lines": lines})
        return Lyrics.model_validate({"title": title, "tags": tags, "sections": sections})
    except (SongError, ValueError) as exc:
        raise InstanceError(f"The lyrics: {exc}") from exc


def _stage_problem(runner: SongRunner) -> str | None:
    """What a song needs that this box lacks (the fast tier's instances), before the form."""

    recipes = runner.service.all()
    try:  # an unreadable picks file too: said on the form, never a 500
        sung = stages.resolve(song.SONG, "fast", runner.state_dir).instance
        wanted = (song.STEMS,) if stages.workflow_of(sung) else (song.SONG, song.STEMS)
        for stage in wanted:
            players(stage, "fast", runner.state_dir, recipes)
    except StudioError as exc:
        return str(exc)
    return None


def _where(run: RunRecord) -> str:
    match run.state:
        case RunState.WAITING if run.waiting_at in song.PICKS:
            return "waiting for you to pick a candidate"
        case RunState.WAITING:
            return f"waiting for you at the {song.STAGE_LABELS[run.waiting_at or ''].lower()}"
        case RunState.DONE:
            return run.answer
        case _:
            return run.state.value
