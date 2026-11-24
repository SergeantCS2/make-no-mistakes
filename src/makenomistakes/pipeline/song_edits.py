"""A song's edits: each one a child run of the song (role `edit`) that holds a new take, made by
ACE-Step on audio.cpp from the take the song uses now (`song_takes`), with the lyrics that take
was sung with.

The song's take is the picked candidate until an edit is made: then the song's `take_run`
names the edit, and the take's map, stems, voice, mix and export are made again from it
(`song.MADE_FROM[song.SONG]`). Edits chain (an edit of an edit); Undo is the take before it
(`use_take`), and nothing an edit replaced is deleted. A new pick, or a candidate made again,
goes back to the candidates.

The range of a repaint or a re-sing is snapped out to whole bars by the structure map's
downbeats unless `snap=off`; a part of the map (`part=2`) is its own range. `lego` answers
the track it plays alone, so an Add mixes it over the take, and a Swap over the other stems
of the take in use (summed; the separator's stems stand in for ACE-Step's `extract`).
"""

from __future__ import annotations

import base64
import secrets
import shutil
from bisect import bisect_left, bisect_right
from pathlib import Path

import httpx
from pydantic import ValidationError

from makenomistakes.audio_format import (
    Conversion,
    HouseFormat,
    HouseFormatError,
    Mixdown,
)
from makenomistakes.engine_answers import MalformedAnswerError
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.hub.music import MusicFamily
from makenomistakes.instances import InstanceService
from makenomistakes.make import credit, settings_used
from makenomistakes.media_bytes import wav_file_seconds
from makenomistakes.music_api import TaskInput, TaskRequest, parse_answer
from makenomistakes.pipeline import (
    song,
    song_record,
    song_retime,
    song_sections,
    song_takes,
    take_audio,
    vocals_layers,
)
from makenomistakes.pipeline.song import Lyrics, SongInputs
from makenomistakes.pipeline.song_steps import (
    CANDIDATE_FILE,
    digest,
    engine_wav,
    post_task,
    song_rate,
)
from makenomistakes.pipeline.song_takes import EDIT, ROOT, SOURCE, Take
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.recipes.model import Recipe
from makenomistakes.run_parts import InputRecord
from makenomistakes.runs import RunError, RunKind, RunRecord, RunStore
from makenomistakes.studio import peaks, stages
from makenomistakes.studio.edit_base import EditError, EditHost, RangedAsk, Span
from makenomistakes.studio.edits import EditAsk, LegoAsk, ResingAsk, Styled, SwapAsk
from makenomistakes.studio.edits_retime import RetimeAsk, TempoAsk
from makenomistakes.studio.edits_sections import SectionAsk
from makenomistakes.studio.stage_instances import is_package
from makenomistakes.studio.tools_answers import StructureMap

STAGE = "edit"  # the stages registry's row: which ACE-Step plays the edits
BACKING = ".swap-backing.wav"  # the stems less the one a swap takes out, in the song's folder
BED = ".bed.wav"  # what `lego`'s track is mixed over, in the edit's folder while it is
ENGINE_FILE = ".engine.wav"  # the engine's answer, in the edit's folder until converted


def _ace(recipe: Recipe) -> bool:
    family = MusicFamily.ACE_STEP.value
    return isinstance(recipe.args, AudioCppArgs) and recipe.args.family == family


def editor(service: InstanceService, state_dir: Path, run: RunRecord) -> Recipe:
    """The ACE-Step instance an edit runs on: the user's `edit` stage pick for the song's
    tier, else the registry's package, else the song's own ACE-Step, else any; a refusal
    naming the `mnm add` without one."""
    tier = SongInputs.from_settings(run.settings).tier
    try:
        resolved = stages.resolve(STAGE, tier, state_dir)
    except stages.StudioError as exc:
        raise PipelineError(str(exc)) from exc
    ace = [r for r in service.all() if _ace(r)]
    if resolved.instance:
        picked = next((r for r in ace if r.name == resolved.instance), None)
        if picked is None:
            raise PipelineError(
                f"The edit stage's {tier} tier is {resolved.instance}, which is no ACE-Step "
                f"instance on audio.cpp: `mnm studio stage edit {tier}=<instance>` picks "
                f"another, `{tier}=` goes back to the registry's."
            )
        return picked
    named = run.settings.get("song_instance", "")
    chosen = (
        next((r for r in ace for p in resolved.packages if is_package(r, p)), None)
        or next((r for r in ace if r.name == named), None)
        or (ace[0] if ace else None)
    )
    if chosen is None:
        add = resolved.packages[0].reference if resolved.packages else "an ACE-Step package"
        raise PipelineError(
            f"Edits run on ACE-Step through audio.cpp, and none is added: `mnm add {add}` "
            "(the take itself may come from any song model)."
        )
    return chosen


def snapped(ask: EditAsk, structure: StructureMap | None, length: float) -> Span | None:
    """The range an edit acts on: a part of the map, or start and end (the whole take where
    one is missing), snapped out to whole bars when asked and a map exists; None for an edit
    of the whole take."""
    if not isinstance(ask, RangedAsk):
        return None
    if ask.part is not None:
        if structure is None:
            raise PipelineError(
                "No structure map gives the take's parts: give start= and end= in seconds "
                "(`mnm studio install`, then `mnm song redo <run> structure`, maps it)."
            )
        if not structure.sections:  # a 10 s take's map may find none
            raise PipelineError(
                f"The take's structure map found no parts in its {structure.duration:.0f} s: "
                "give start= and end= in seconds."
            )
        if ask.part > len(structure.sections):
            raise PipelineError(f"The take has parts 1 to {len(structure.sections)}.")
        part = structure.sections[ask.part - 1]
        return Span(part.start, min(part.end, length))
    if ask.start is None and ask.end is None:
        return None
    start = ask.start or 0.0
    end = min(ask.end if ask.end is not None else length, length)
    asked = f"{start:g}-{end:g} s"
    if start >= length:
        raise PipelineError(f"The take is {length:.1f} s long: {start:g} s is past its end.")
    bars = structure.downbeats if structure is not None and ask.snap else []
    barred = False
    if bars and ask.snap_out:  # sung again: out to whole bars, never less than asked
        moved = bars[max(bisect_right(bars, start + 0.05) - 1, 0)] if start > bars[0] else 0.0
        at = bisect_left(bars, end - 0.05)
        out = bars[at] if at < len(bars) else length
        barred = (moved, out) != (start, end)
        start, end = moved, out
    elif bars:  # a section tool: the nearest bar lines, so the joint keeps the beat
        moved, out = (
            song_sections.nearest_bar(structure, start, length),
            song_sections.nearest_bar(structure, end, length),
        )
        barred = (moved, out) != (start, end)
        start, end = moved, out
    if end - start < 0.5 and barred:  # both ends on one bar line
        raise PipelineError(
            f"The range {asked} lies inside one bar: on the nearest bar lines it is "
            f"{start:g}-{end:g} s, nothing to {ask.edit.value}. Give a range across a bar line, "
            "or snap=off for the range as given."
        )
    if end - start < 0.5:
        raise PipelineError(f"The range {start:g}-{end:g} s is too short to edit.")
    return Span(start, end, barred)


def _lyrics_for(ask: EditAsk, lyrics: Lyrics) -> Lyrics:
    """The lyrics the edit sings, kept as its take's: the take's, with the style line the edit
    names (else the next edit would sing the old one and the export name it) and one
    section's lines new for a re-sing."""
    if isinstance(ask, Styled) and ask.tags:
        lyrics = Lyrics.model_validate({**lyrics.model_dump(), "tags": ask.tags})
    if not isinstance(ask, ResingAsk):
        return lyrics
    if ask.section > len(lyrics.sections):
        raise PipelineError(f"The lyrics have sections 1 to {len(lyrics.sections)}.")
    sections = list(lyrics.sections)
    at = ask.section - 1
    sections[at] = sections[at].model_copy(update={"lines": ask.lines})
    return lyrics.model_copy(update={"sections": sections})


def swap_stems(run: RunRecord, folder: Path, take: Take, ask: SwapAsk) -> list[str]:
    """The stems a swap sums, by name in the song's folder: those split from the take in use,
    less the one it takes out (`take_audio.take_stems`)."""
    said = f"A swap takes a stem out of the take in use ({song_takes.words(run, take)}), and"
    problem = take_audio.stems_problem(run, take)
    if problem == take_audio.NOT_MADE:
        raise PipelineError(
            f"{said} its stems are not made yet: `mnm song resume {run.id}` makes them, then swap."
        )
    if problem == take_audio.ANOTHER_TAKE:
        raise PipelineError(
            f"{said} its stems are another take's: `mnm song redo {run.id} stems` splits it, "
            "then swap."
        )
    stems = song.stem_files([f.name for f in run.outputs])
    out = song.stem_file(ask.track)
    if out not in stems:
        have = ", ".join(s.removeprefix("stem-").removesuffix(".wav") for s in stems) or "none"
        raise PipelineError(
            f"A swap of {ask.track} needs the take's {ask.track} stem, and its stems are {have}."
        )
    gone = {out, *vocals_layers.ORIGINALS} if vocals_layers.is_vocals(out) else {out}
    return [s for s in take_audio.take_stems(run, folder) if s not in gone]


def _inputs(store: RunStore, run: RunRecord, take: Take, sources: list[str]) -> list[InputRecord]:
    """What the edit was made from, by hash: the take, or for a swap the stems it summed."""
    if not sources:
        path = song_takes.take_file(store, run, take)
        return [InputRecord(role="take", name=path.name, sha256=digest(path.read_bytes()),
                            source=f"run:{take.run.id}", stage=EDIT)]  # fmt: skip
    folder = store.folder(run)
    return [
        InputRecord(role="stem", name=name, sha256=digest((folder / name).read_bytes()),
                    source=f"run:{run.id}", stage=EDIT)
        for name in sources
    ]  # fmt: skip


def make_edit(  # noqa: PLR0913 - the song, its take's parts and the seams
    store: RunStore,
    run: RunRecord,
    recipe: Recipe | None,
    http: httpx.Client,
    house: HouseFormat,
    ask: EditAsk,
    *,
    take: Take,
    lyrics: Lyrics,
    structure: StructureMap | None,
) -> RunRecord:
    """The edit made: a child run of the song holding the new take (`song.wav`; an Add's or a
    Swap's track alone beside it), the lyrics it was sung with, what it acted on and what it
    was made from by hash; refused before the run is made when the edit cannot be, a failure
    after it kept as a failed child. ACE-Step makes a sung edit (`recipe`, its instance); the
    house format's ffmpeg a section tool's (`recipe` None)."""
    try:
        path = song_takes.take_file(store, run, take)
        length = wav_file_seconds(path) or float(run.settings.get("seconds", song.DEFAULT_SECONDS))
    except OSError as exc:
        raise PipelineError(f"The take {take.run.id} cannot be read: {exc}") from exc
    span = snapped(ask, structure, length)
    pieces, after, far = _after(ask, structure, length, span, run.id)
    if after > song.MAX_SECONDS:  # ACE-Step's ten minutes: what every later edit can sing
        raise PipelineError(
            f"The take would be {after:.0f} s long, past ACE-Step's {song.MAX_SECONDS:.0f} s"
            + ("." if ask.host is EditHost.ACE_STEP else ", which later sung edits need.")
        )
    try:
        sent = _lyrics_for(ask, lyrics)
    except ValidationError as exc:
        raise PipelineError(f"The {ask.edit}'s style line: {exc.errors()[0]['msg']}.") from exc
    stems = swap_stems(run, store.folder(run), take, ask) if isinstance(ask, SwapAsk) else []
    try:
        child = _child(store, run, recipe, ask, take, sent)
    except RunError as exc:  # the outputs folder takes no new folder: said, never a traceback
        raise PipelineError(str(exc)) from exc
    handing = song_retime.carried(ask, structure)  # the key handed on
    child.settings |= handing.settings() if handing is not None else {}
    child.acts_on = ask.acts_on(span, length)
    store.save(child)
    try:
        child.inputs_recorded = _inputs(store, run, take, stems)
        here = store.folder(child)
        song_record.save_lyrics(store, child, sent)  # the lyrics this take is sung with
        if ask.host is EditHost.HOUSE:  # the one dispatch: what makes the edit is its host
            made, note = _housed(house, here, path, ask, structure, pieces=pieces, run=run,
                                 folder=store.folder(run), take=take)  # fmt: skip
            took: float | None = None
        else:
            if recipe is None:  # `song_changes.edit` asks `editor` for every sung edit
                raise PipelineError(f"The {ask.edit} is sung by ACE-Step, and none was given.")
            made, took, note = _sung(store, run, recipe, http, house, ask, sent=sent,
                                     child=child, path=path, stems=stems, span=span,
                                     length=length, after=after)  # fmt: skip
    except (PipelineError, StepError, HouseFormatError, RunError, OSError) as exc:
        store.fail(child, str(exc))
        raise PipelineError(f"The edit failed: {exc}") from exc
    except BaseException as exc:
        store.fail(child, str(exc) or f"Interrupted ({type(exc).__name__}).")
        raise
    finally:
        (store.folder(child) / BED).unlink(missing_ok=True)
    peaks.measure(house, here, [CANDIDATE_FILE])  # the take's waveform (a range is clicked)
    return store.finish(child, made, ask.words(span) + _said(ask, took, note + far), took)


def _housed(  # noqa: PLR0913 - the take, the ask and where its parts are
    house: HouseFormat,
    here: Path,
    path: Path,
    ask: EditAsk,
    structure: StructureMap | None,
    *,
    pieces: list[tuple[float, float]],
    run: RunRecord,
    folder: Path,
    take: Take,
) -> tuple[list[Path], str]:
    """ffmpeg's half of an edit: a section tool's splice, or Rubber Band's tempo or key."""
    if isinstance(ask, SectionAsk):
        return [song_sections.spliced(house, here, path, ask, pieces)], ""
    if isinstance(ask, RetimeAsk):
        return song_retime.retimed(house, here, path, ask, structure, run=run, folder=folder,
                                   take=take)  # fmt: skip
    raise PipelineError(f"No ffmpeg edit makes '{ask.edit}'.")


def _said(ask: EditAsk, took: float | None, note: str) -> str:
    """What follows the edit's words on its line: ACE-Step's time and notes, or ffmpeg's."""
    if ask.host is EditHost.ACE_STEP:
        return f" ({took or 0:.0f} s of work{note})"
    return f" ({note.removeprefix('; ')})" if note else ""


def _after(
    ask: EditAsk, structure: StructureMap | None, length: float, span: Span | None, run_id: str
) -> tuple[list[tuple[float, float]], float, str]:
    """A section tool's pieces, the take's length once edited, and a tempo or key change's
    note when it goes far; every refusal read before the edit's run is made."""
    try:
        if isinstance(ask, SectionAsk):
            assert span is not None  # a section ask always names its part (`_a_part`)
            pieces = song_sections.pieces_of(ask, structure, length, span)
            return pieces, song_sections.new_length(ask, pieces), ""
        if isinstance(ask, RetimeAsk):  # a key's `to=` is read here against the map
            after = length / ask.ratio(structure) if isinstance(ask, TempoAsk) else length
            return [], after, ask.note(structure, run_id)
        return [], ask.seconds_after(length), ""
    except EditError as exc:
        raise PipelineError(f"The {ask.edit}: {exc}.") from exc


def _child(
    store: RunStore,
    run: RunRecord,
    recipe: Recipe | None,
    ask: EditAsk,
    take: Take,
    sent: Lyrics,
) -> RunRecord:
    """The edit's run, under the song: an instance run of the ACE-Step that sings it, or a
    pipeline run of the song (as the mixer's renders are) for a section tool."""
    used = {
        "edit": ask.edit.value, SOURCE: take.run.id, ROOT: take.candidate.id,
        "lyrics": sent.sung(),
    }  # fmt: skip
    if recipe is None:
        child = store.new(
            RunKind.PIPELINE, song.NAME, sent.tags, used | {"tool": "ffmpeg"},
            parent=run.id, role=EDIT,
        )  # fmt: skip
    else:
        used = settings_used(recipe) | used | {"seed": str(secrets.randbelow(2**32))}
        child = store.new(RunKind.INSTANCE, recipe.name, sent.tags, used, parent=run.id, role=EDIT)
        child.models = credit(recipe)
    return child


def _sung(  # noqa: PLR0913 - the edit's run and request beside the seams
    store: RunStore,
    run: RunRecord,
    recipe: Recipe,
    http: httpx.Client,
    house: HouseFormat,
    ask: EditAsk,
    *,
    sent: Lyrics,
    child: RunRecord,
    path: Path,
    stems: list[str],
    span: Span | None,
    length: float,
    after: float,
) -> tuple[list[Path], float | None, str]:
    """ACE-Step's half: the take (or a swap's other stems) sent, its answer made the new take."""
    assert isinstance(ask, Styled)
    folder, here = store.folder(run), store.folder(child)
    try:
        source = house.mix(folder, Mixdown(sources=stems, target=BACKING)) if stems else path
        wav = engine_wav(house, source.parent, source, song_rate(recipe))
        if isinstance(ask, LegoAsk):  # the track is mixed over this
            shutil.copyfile(source, here / BED)
    finally:
        (folder / BACKING).unlink(missing_ok=True)
    request = TaskRequest(
        model=recipe.served_name,
        request=TaskInput(
            text=sent.tags, lyrics=sent.sung(), seed=int(child.settings["seed"]),
            audio_base64=base64.b64encode(wav).decode("ascii"),
            options=ask.options(span, length),
        ),
    )  # fmt: skip
    return _made(here, recipe, http, house, request, ask, after)


def _made(
    here: Path,
    recipe: Recipe,
    http: httpx.Client,
    house: HouseFormat,
    request: TaskRequest,
    ask: EditAsk,
    seconds: float,
) -> tuple[list[Path], float | None, str]:
    """The engine's answer in the house format: the new take, or for `lego` its track and
    the take it makes over the bed (the take, or the swap's other stems); a note when the
    sum had to be turned down."""
    raw = post_task(recipe, http, request.body(), seconds)
    try:
        answer = parse_answer(raw)
        files = answer.files()
    except MalformedAnswerError as exc:
        raise StepError(f"{recipe.name}: {exc}.") from exc
    if not files:
        raise StepError(f"{recipe.name} answered with no audio.")
    (here / ENGINE_FILE).write_bytes(files[0][1])
    target = f"track-{ask.track}.wav" if isinstance(ask, LegoAsk) else CANDIDATE_FILE
    try:
        played = house.convert(here, Conversion(source=ENGINE_FILE, target=target))
    finally:
        (here / ENGINE_FILE).unlink(missing_ok=True)
    if not isinstance(ask, LegoAsk):
        return [played], answer.seconds, ""
    take, note = take_audio.summed(house, here, [BED, played.name])
    return [take, played], answer.seconds, note
