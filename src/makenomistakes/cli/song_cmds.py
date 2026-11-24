"""`mnm song`: the Song pipeline from a terminal: a theme to lyrics, candidates, the
pick, the stems, the mix and an export folder with a REAPER project."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import Failed, app, console, fail_with, print_run_folder
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.config import Settings
from makenomistakes.instances import InstanceService
from makenomistakes.pipeline import song, song_takes, song_vocals
from makenomistakes.pipeline.song import SongError, SongInputs, seconds_of
from makenomistakes.pipeline.song_runner import SongRunner, song_runner
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.progress_words import with_ellipsis
from makenomistakes.runs import RunError, RunRecord, RunState, StepState
from makenomistakes.studio import export_mix
from makenomistakes.studio.export import ExportOptions, listing

song_app = typer.Typer(help="Songs: a theme to lyrics, candidates, stems, a mix and an export.")
app.add_typer(song_app, name="song")
STEPS = "lyrics, song, score, structure, stems, vocals, voice, midi, mix or export"


def _runner() -> SongRunner:
    """The runner for this box; tests replace it."""
    return song_runner(InstanceService(Settings()))


def _progress(message: str) -> None:
    console.print(escape(with_ellipsis(message)))


def _lyrics_lines(runner: SongRunner, run: RunRecord) -> None:
    try:
        lyrics = runner.lyrics(run)
    except StepError as exc:  # its folder cleaned away: said, never a traceback
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        return
    console.print(f"  {escape(lyrics.title or '(no title)')}: {escape(lyrics.tags)}")
    for number, section in enumerate(lyrics.sections, start=1):
        console.print(f"  {number}. [{escape(section.label)}]")
        for line in section.lines:
            console.print(f"     {escape(line)}")


def _said(run: RunRecord, runner: SongRunner, *, moved: bool = True) -> None:
    """Where the run stands, and what to do next; a run this command moved and that ended
    failed exits 1 (`mnm song show`, which moves nothing, says it and exits 0)."""
    for note in run.notes:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    match run.state:
        case RunState.DONE if run.files_removed:
            console.print(f"[green]Done[/green]: {escape(run.answer)}")
            console.print("Its files were deleted (`mnm clean outputs`); the record stays.")
        case RunState.DONE:
            console.print(f"[green]Done[/green]: {escape(run.answer)}")
            export = runner.store.folder(run) / song.EXPORT_FOLDER
            print_run_folder(run.id, export)
        case RunState.FAILED:
            console.print(f"[red]{escape(run.error)}[/red]")
            console.print(f"Fix it, then `mnm song resume {escape(run.id)}`.")
            if moved:
                exit_if_failed(run)
        case RunState.WAITING if run.waiting_at == song.LYRICS:
            console.print("Waiting for you at the lyrics:")
            _lyrics_lines(runner, run)
            console.print(
                f"`mnm song go {escape(run.id)}` sings them; `mnm song redo {escape(run.id)} "
                "lyrics` writes new ones; the web's Song page edits a section or replaces all."
            )
        case RunState.WAITING:
            console.print("Waiting for you to pick a candidate:")
            made = runner.candidates(run)
            scores = [c.score for _s, c in made if c is not None and c.score is not None]
            top = max(scores) if len(scores) > 1 else None
            for step, child in made:
                if child is None:
                    console.print(f"  {step.index + 1}. not made ({escape(step.error or '-')})")
                    continue
                path = runner.store.folder(child) / "song.wav"
                score = f", score {child.score:.2f}" if child.score is not None else ""
                score += " (the highest)" if top is not None and child.score == top else ""
                console.print(f"  {step.index + 1}. {escape(str(path))}{score}")
            console.print(
                f"`mnm song pick {escape(run.id)} <n>` takes one to its stems; `mnm song redo "
                f"{escape(run.id)} song <n>` makes one again."
            )
        case RunState.RUNNING:
            console.print(f"{escape(run.id)} is running.")


def exit_if_failed(run: RunRecord) -> None:
    """A run that ended failed exits 1 with its error, so a script and the notice see it
    (else a 40-minute song that failed exits 0)."""
    if run.state is RunState.FAILED:
        raise Failed(run.error or f"{run.id} failed", code=1)


def _recording(path: Path | None) -> tuple[str, bytes] | None:
    if path is None:
        return None
    try:
        return path.name, path.read_bytes()
    except OSError as exc:
        fail_with(OSError(f"Cannot read the recording {path}: {exc.strerror or exc}."))


@song_app.command("new")
@notified("song new")
def new(
    theme: Annotated[str, typer.Argument(help="What the song is about, in a sentence or two.")],
    length: Annotated[str, typer.Option(help="30s, 3m, 2m30s (10 s to 10 minutes).")] = "30s",
    lyrics: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Your own lyrics, a section header like Verse in square brackets above its "
            "lines: no lyrics are written for you.",
        ),
    ] = None,
    tags: Annotated[
        str | None, typer.Option(help="The style line the song model sings to (else the chat's).")
    ] = None,
    recording: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help="A recording of yours."),
    ] = None,
    use: Annotated[
        str | None,
        typer.Option("--as", help="How the recording is used: complete, vocal2bgm or reference."),
    ] = None,
    candidates: Annotated[
        int, typer.Option(min=1, max=song.MAX_CANDIDATES, help="Candidates to pick from.")
    ] = song.DEFAULT_CANDIDATES,
    tier: Annotated[str, typer.Option(help="fast to explore, full for the keeper.")] = "fast",
    chat: Annotated[
        str, typer.Option(help="Who writes the lyrics: the desk's chat, or an instance.")
    ] = song.DESK_CHAT,
    auto: Annotated[
        bool, typer.Option("--auto", help="Straight through: no stop at the lyrics or the pick.")
    ] = False,
    voice: Annotated[
        str | None,
        typer.Option(help="A library voice the vocals are sung in (`mnm voices ls`)."),
    ] = None,
    shift: Annotated[
        int, typer.Option(min=-12, max=12, help="Semitones the voice moves the vocals by.")
    ] = 0,
    clean: Annotated[
        str | None,
        typer.Option(
            help="What to do to the separator's vocals: debleed, dereverb, denoise, deess, "
            "gate, tune (comma-separated; `mnm song vocals` changes it later)."
        ),
    ] = None,
) -> None:
    """Make a song: lyrics from your theme, candidates to pick from, the picked one's stems, a
    mix and an export folder. It stops for you at the lyrics and at the pick unless --auto;
    --clean cleans the vocals, --voice sings them in a library voice."""
    runner = _runner()
    try:
        wanted = song_vocals.parse(clean or "")  # checked now, not after the candidates
        song_vocals.settings_with({}, clean or "", None, None, runner.service, runner.tools)
        if use is not None and use not in song.ROUTES:
            raise SongError(f"--as {use}: a recording is used as complete, vocal2bgm or reference.")
        if tier not in ("fast", "full"):
            raise SongError(f"--tier {tier}: the tiers are fast and full.")
        inputs = SongInputs.model_validate(
            {
                "theme": theme, "seconds": seconds_of(length), "candidates": candidates,
                "tier": tier, "chat": chat, "tags": tags, "recording": use, "auto": auto,
                "voice": voice, "voice_shift": shift,
                "vocals_clean": ",".join(c.value for c in wanted) or None,
            }
        )  # fmt: skip
        words = lyrics.read_text(encoding="utf-8") if lyrics is not None else None
        run = runner.start(inputs, words, _recording(recording))
        print_run_folder(run.id, runner.store.folder(run))
        run = runner.advance(run.id, _progress)
    except (PipelineError, ValueError, OSError) as exc:
        fail_with(exc)
    _said(run, runner)


@song_app.command("go")
@notified("song go", key="run_id")
def go(run_id: Annotated[str, typer.Argument()]) -> None:
    """Past the review point the run waits at (the lyrics), and on."""
    runner = _runner()
    try:
        run = runner.proceed(run_id, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@song_app.command("pick")
@notified("song pick", key="run_id")
def pick(
    run_id: Annotated[str, typer.Argument()],
    candidate: Annotated[int, typer.Argument(help="Which one (1 = the first).", min=1)],
) -> None:
    """Pick a candidate: its stems, the mix and the export follow."""
    runner = _runner()
    try:
        run = runner.pick(run_id, candidate, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@song_app.command("voice")
@notified("song voice", key="run_id")
def voice_cmd(
    run_id: Annotated[str, typer.Argument()],
    voice: Annotated[
        str,
        typer.Argument(help="A library voice (`mnm voices ls`), or `none` for the model's own."),
    ],
    shift: Annotated[
        int, typer.Option(min=-12, max=12, help="Semitones the voice moves the vocals by.")
    ] = 0,
) -> None:
    """Sing a song's vocals in another voice (or `none`): the vocals as sung are converted,
    then the mix and the export follow."""
    runner = _runner()
    try:
        run = runner.set_voice(run_id, None if voice == "none" else voice, shift, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@song_app.command("vocals")
@notified("song vocals", key="run_id")
def vocals_cmd(
    run_id: Annotated[str, typer.Argument()],
    clean: Annotated[
        str,
        typer.Argument(
            help="What to do to the separator's vocals, comma-separated: debleed, dereverb, "
            "denoise, deess, gate, tune (done in that order), or `none`."
        ),
    ],
    key: Annotated[
        str | None,
        typer.Option(help="Tuning's key, as `A minor` (else the structure map's)."),
    ] = None,
    strength: Annotated[
        int | None,
        typer.Option(min=0, max=100, help="How far tuning moves a note, 0 to 100 (80)."),
    ] = None,
) -> None:
    """Clean and tune a song's vocals before a voice sings them: made again from the
    separator's vocals, then the voice, the mix and the export follow."""
    runner = _runner()
    try:
        run = runner.set_vocals(run_id, clean, key, strength, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@song_app.command("resume")
@notified("song resume", key="run_id")
def resume(run_id: Annotated[str, typer.Argument()]) -> None:
    """Carry on a run that failed or was interrupted, from the first item not done."""
    runner = _runner()
    try:
        run = runner.advance(run_id, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@song_app.command("redo")
@notified("song redo", key="run_id")
def redo(
    run_id: Annotated[str, typer.Argument()],
    stage: Annotated[str, typer.Argument(help=f"{STEPS}.")],
    number: Annotated[
        int | None, typer.Argument(help="Which one (a candidate's number; 1 if not said).", min=1)
    ] = None,
) -> None:
    """Make one item again and what follows it (the picked candidate: its score, the
    structure, its stems, the mix, the export, and the run waits at the pick again; another
    candidate: its score and the export, the pick stands). `score` without a number scores
    every candidate again (after `mnm studio install`)."""
    if stage not in song.STAGES:
        fail_with(ValueError(f"'{stage}' is not one of: {STEPS}."))
    runner = _runner()
    try:
        if stage == song.SCORE and number is None:
            run = runner.rescore(run_id, _progress)
        else:
            run = runner.redo(run_id, stage, (number or 1) - 1, _progress)
    except PipelineError as exc:
        fail_with(exc)
    _said(run, runner)


@song_app.command("export")
def export_cmd(
    run_id: Annotated[str, typer.Argument()],
    mp3: Annotated[bool, typer.Option("--mp3", help="An MP3 of the mix too.")] = False,
    release: Annotated[
        bool,
        typer.Option("--release", help="Mark it for release (a note where consent is missing)."),
    ] = False,
    lufs: Annotated[
        float | None,
        typer.Option(
            help="The mix's loudness, LUFS integrated (else the line kept, -14 at first)."
        ),
    ] = None,
    true_peak: Annotated[
        float | None,
        typer.Option(help="Its true peak ceiling, dBTP (else the line kept, -1 at first)."),
    ] = None,
    no_loudness: Annotated[
        bool, typer.Option("--no-loudness", help="The mix as summed, not normalised.")
    ] = False,
) -> None:
    """The export folder again: the mix (WAV 24-bit 48 kHz, FLAC, MP3 on request), the stems,
    song.rpp, provenance.json and disclosure.txt. Never refused for a missing consent."""
    runner = _runner()
    try:
        kept = runner.store.get(run_id).settings
        loudness = export_mix.asked_line(kept, lufs, true_peak, no_loudness)
        done = runner.export(run_id, ExportOptions(mp3=mp3, release=release, loudness=loudness))
    except (PipelineError, RunError, ValueError) as exc:
        fail_with(exc)
    for note in done.notes:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    console.print(f"{len(done.files)} files:")
    for name in done.files:
        console.print(f"  {escape(name)}")
    print_run_folder(run_id, done.folder)


@song_app.command("ls")
def list_cmd() -> None:
    """The song runs, newest first."""
    runner = _runner()
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Run", "Theme", "State"):
        table.add_column(column)
    for run in runner.store.all():
        if song.is_song(run):
            state = run.state.value + (f" at {run.waiting_at}" if run.waiting_at else "")
            state += " (interrupted)" if run.interrupted else ""
            table.add_row(escape(run.id), escape((run.prompt or "")[:60]), state)
    console.print(table)


def _take_line(runner: SongRunner, run: RunRecord) -> None:
    """The take the song uses, when it is an edit or cannot be used."""
    if not run.settings.get(song.TAKE_RUN):
        return
    try:
        take = song_takes.take(runner.store, run)
    except song_takes.TakeError as exc:
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        return
    if take.edit is None:  # it names the candidate's own run
        return
    console.print(
        f"Take in use: {escape(song_takes.words(run, take))} ({escape(take.edit.answer)}); "
        f"`mnm song take {escape(run.id)}` lists the takes."
    )


@song_app.command("show")
def show(run_id: Annotated[str, typer.Argument()]) -> None:
    """One song run's steps: each item, its state and what it made."""
    runner = _runner()
    try:
        run = runner.song(run_id)  # a mix's, an edit's or a master's id says whose it is
    except PipelineError as exc:
        fail_with(exc)
    table = Table(box=box.SIMPLE_HEAD)
    for column in ("Step", "State", "Made", "Seconds"):
        table.add_column(column)
    picked = song.picked_index(run.settings)
    for step in run.steps:
        label = song.STAGE_LABELS.get(step.stage, step.stage)
        many = step.stage in song.ITEMS
        mark = " (picked)" if step.stage == song.SONG and step.index == picked else ""
        table.add_row(
            (f"{label} {step.index + 1}" if many else label) + mark,
            step.state.value + (f": {escape(step.error)}" if step.error else ""),
            escape(step.text or step.file or "-"),
            f"{step.seconds:.0f}" if step.seconds else "",
        )
    console.print(table)
    _take_line(runner, run)
    export = runner.store.folder(run) / song.EXPORT_FOLDER
    if files := listing(export):
        made = run.step(song.EXPORT)
        done = made is None or made.state is StepState.DONE
        old = "" if done else ", from before the last change"
        console.print(f"Export ({len(files)} files{old}): {escape(str(export))}")
    _said(run, runner, moved=False)
