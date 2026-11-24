"""A song run's export folder: rendered from the run record alone into
`<outputs>/<day>/<id>/export/`, every time afresh:

- `mix.wav` (24-bit, 48 kHz) and `mix.flac`, `mix.mp3` on request: the mix in use (the
  mixer's render, else the stems summed) at the loudness line (ffmpeg `loudnorm`, two
  passes; -14 LUFS and -1 dBTP unless set, or off), or the master in use as it was made
  (mastered to a reference, which set its loudness), then measured as written
  (`studio/export_mix`);
- `stems/<name>.wav` and `.flac`: each stem as separated, untouched by the levels and the
  loudness line;
- `midi/<name>.mid`: each pitched stem's notes, when the studio tools read them;
- `song.rpp`: a REAPER project with a track per stem at the mixer's level and mute, and the
  structure map's tempo, meter and part markers when the take was mapped (`studio/reaper.py`);
- `provenance.json`: the models, voices, inputs by hash, the candidates and their scores, the
  mix's levels and the structure map;
- `disclosure.txt`: the box, the models, the voice and its consent, the Article 50 label,
  the licences. A release-marked export of a voice with no consent recorded is never
  refused: it gets a note here, in the record and on the page.

`mnm clean outputs` removes the export with the run's other files (it lives in its folder).
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from makenomistakes.audio_format import Codec, Conversion, HouseFormat, Loudness
from makenomistakes.hub.converted import kit_named_by
from makenomistakes.local_time import local
from makenomistakes.media_bytes import wav_file_seconds
from makenomistakes.pipeline import (
    song,
    song_credits,
    song_master,
    song_mix,
    song_takes,
    song_vocals,
)
from makenomistakes.pipeline.song_tools import read_structure
from makenomistakes.run_parts import ActsOn, InputRecord, ModelCredit
from makenomistakes.runs import RunRecord, RunStore
from makenomistakes.studio import disclosure, export_extras, export_mix, reaper
from makenomistakes.studio.mixer import MixSettings
from makenomistakes.studio.tools_answers import StructureMap
from makenomistakes.studio.voice_library import VoiceError, VoiceLibrary, VoiceTake, read_take

STEMS = "stems"
FILES = ("mix.wav", "mix.flac", "mix.mp3", "song.rpp", "provenance.json", "disclosure.txt")


class ExportError(RuntimeError):
    """An export that cannot be made, worded for the user."""


class ExportOptions(BaseModel):
    mp3: bool = False
    release: bool = False
    loudness: Loudness | None = Field(default_factory=Loudness)  # None: the mix as it is


class Candidate(BaseModel):
    run: str
    seed: str | None = None
    score: float | None = None
    scorer: str | None = None
    picked: bool = False


class EditStep(BaseModel):
    """One edit the take went through: its run, what it did, on what, from which take."""

    run: str
    edit: str
    did: str
    acts_on: ActsOn | None
    source: str


class Provenance(BaseModel):
    """What made the export, kept beside it (and only what the run record holds)."""

    # 2: `measured` is what the written mix.wav measures, never the
    # input's first loudness pass; `loudness` the line it was asked at, in words
    schema_version: int = 2
    run: str
    theme: str | None
    made_at: datetime
    box: str
    tags: str | None
    models: list[ModelCredit]
    voices: list[str]
    voice: VoiceTake | None = None  # What sang the vocals, the similarity reading on it
    inputs: list[InputRecord]
    candidates: list[Candidate]
    # The edits the take in use was made by, from the picked candidate's take on
    edits: list[EditStep] = Field(default_factory=list)
    stems: list[str]
    midi: list[str] = Field(default_factory=list)  # The stems' MIDI files
    master: export_extras.MasterLine | None = None  # The master the mix is
    vocals: list[str] = Field(default_factory=list)  # What was done to the vocals
    mix: str = "every stem as made"  # the mixer's levels
    structure: StructureMap | None = None  # the take's map
    loudness: str
    measured: str | None
    licences: list[str]
    release: bool
    notes: list[str]


class Exported(BaseModel):
    folder: Path
    files: list[str]
    notes: list[str]
    disclosure: str


def listing(folder: Path) -> list[str]:
    """The export's files, by their name in the folder (`stems/vocals.wav`), sorted; only
    regular files mnm writes, never a link."""
    if not folder.is_dir() or folder.is_symlink():
        return []
    found = [p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()]
    return sorted(p.relative_to(folder).as_posix() for p in found)


def _stems(run: RunRecord) -> list[str]:
    return song.stem_files([f.name for f in run.outputs])


def _voice_line(
    run: RunRecord, take: VoiceTake | None, library: VoiceLibrary | None
) -> tuple[str, str | None]:
    """Who sings, and the note a release would carry without a consent on record. A library
    voice that sang the vocals replaces the recording's voice in the song; it is
    described as it was when it sang, its consent read from the library while the row is
    still that voice."""
    if take is not None:
        row = None
        if library is not None:
            try:
                row = library.get(take.voice)
            except VoiceError:
                row = None
        return disclosure.take_words(take, row)
    if run.voices:
        unknown = (
            f"sung in the voice {run.voices[0]}, whose record is not readable here: its source "
            "and consent are not known"
        )
        return unknown, None
    recording = next((i for i in run.inputs_recorded if i.role == "recording"), None)
    sung, wants = disclosure.voice_words(recording, run.settings.get("recording"))
    return sung, disclosure.release_note(recording) if wants else None


def _distinct(models: list[ModelCredit]) -> list[ModelCredit]:
    """Each model once, in the order first named (an edit's model is often the song's, a
    separator plays the stems and the vocals' debleed), without the item it played."""
    seen: list[ModelCredit] = []
    for model in map(song_credits.same_model, models):
        if model not in seen:
            seen.append(model)
    return seen


def _candidates(store: RunStore, run: RunRecord) -> list[Candidate]:
    index = song.picked_index(run.settings)
    found = []
    for step in sorted((s for s in run.steps if s.stage == song.SONG), key=lambda s: s.index):
        child = song_takes.candidate(store, run, step.index)
        if child is None:
            continue
        found.append(
            Candidate(
                run=child.id,
                seed=song.candidate_seed(child),
                score=child.score,
                scorer=child.scorer,
                picked=step.index == index,
            )
        )
    return found


def _licences(models: list[ModelCredit], run: RunRecord) -> list[str]:
    """Each model that made the export, with the licence words audio.cpp's table gives for
    its family (said, never checked); a model mnm has no row for says where to read it."""
    found: list[str] = []
    chat = run.settings.get("chat_instance")
    for credit in models:
        kit = kit_named_by(credit.file or "")
        if kit is not None:
            found.append(f"{kit.label}: {kit.licence} (commercial use: {kit.commercial})")
        elif credit.instance:
            role = "lyrics" if credit.instance == chat else "model"
            found.append(f"{credit.instance} ({role}): its own licence (its repository says)")
        elif credit.workflow:  # A candidate sung by a ComfyUI workflow
            what = f"{credit.file} in " if credit.file else ""
            found.append(
                f"{what}the ComfyUI workflow {credit.workflow}: its own licence "
                "(its repository says)"
            )
        elif credit.tool:  # a tool that made the mix (the master's Matchering)
            found.append(f"{credit.tool}: {song_master.tool_licence(credit.tool)}")
    return list(dict.fromkeys(found))


def render(
    store: RunStore,
    run: RunRecord,
    house: HouseFormat,
    options: ExportOptions,
    state_dir: Path,
    now: datetime | None = None,
    library: VoiceLibrary | None = None,
) -> Exported:
    """The export folder from the run record; the record keeps the disclosure and notes.
    `library` says whether the voice that sang is still the library's, for its consent."""
    folder = store.folder(run)
    _refuse_unready(run)
    out = folder / song.EXPORT_FOLDER
    if out.is_symlink():
        raise ExportError(f"{out} is a link: mnm writes an export only into its own folder.")
    try:  # the take in use: the picked candidate's, or the edits made from it
        in_use = song_takes.take(store, run)
    except song_takes.TakeError as exc:
        raise ExportError(str(exc)) from exc
    shutil.rmtree(out, ignore_errors=True)
    (out / STEMS).mkdir(parents=True)
    when = now or datetime.now(UTC)
    levels = song_mix.settings_of(store, run)
    assert levels is not None  # the stems are checked above
    mastered = export_extras.master_line(store, run)
    source = mastered[0] if mastered else song_mix.mix_source(store, run)
    master = mastered[1].reference if mastered else None
    mix = export_mix.made(house, folder, options.loudness, options.mp3, source, master, run.id)
    tracks = _stem_files(house, folder, run, _stems(run), levels)
    length = max((t.length for t in tracks), default=0.0)
    midi_tracks, midi = export_extras.midi_tracks(folder, run, length)
    structure = read_structure(folder)
    notes = _notes(run, in_use, mix.note)
    take = read_take(folder) if run.voices else None
    sung, release_note = _voice_line(run, take, library)
    if options.release and release_note:
        notes.append(release_note)
    child, edits = in_use.candidate, list(in_use.edits)
    models = _distinct(
        [*run.models, *child.models, *(m for e in edits for m in e.models),
         *song_master.credits(store, run)]
    )  # fmt: skip
    box = disclosure.box_words(state_dir)
    licences = _licences(models, run)
    text = disclosure.text(box, models, sung, licences, notes, f"{local(when):%Y-%m-%d}")
    (out / "disclosure.txt").write_text(text, encoding="utf-8")
    project = reaper.ReaperProject(
        created=int(when.timestamp()),
        notes=_project_notes(_lyrics(store, run, in_use), run, mix, structure, levels),
        tracks=[*tracks, *midi_tracks],
        **_timing(structure),
    )
    (out / "song.rpp").write_text(reaper.render(project), encoding="utf-8")
    provenance = Provenance(
        run=run.id,
        theme=run.prompt,
        made_at=when,
        box=box,
        tags=in_use.run.prompt,  # the style the take in use was sung to (an edit's own)
        models=models,
        voices=run.voices,
        voice=take,
        inputs=[*run.inputs_recorded, *child.inputs_recorded],
        candidates=_candidates(store, run),
        edits=[_edit_step(e) for e in edits],
        stems=[t.name for t in tracks],
        midi=midi,
        master=mastered[1] if mastered else None,
        vocals=song_vocals.done_in(store.folder(run)),
        mix=levels.words(),
        structure=structure,
        loudness=mix.line,
        measured=mix.measured.words(),
        licences=licences,
        release=options.release,
        notes=notes,
    )
    (out / "provenance.json").write_text(provenance.model_dump_json(indent=2) + "\n", "utf-8")
    run.notes, run.disclosure = notes, text
    run.settings |= {
        "export_mp3": str(options.mp3).lower(),
        "export_release": str(options.release).lower(),
        "export_loudness": mix.line,
        song_mix.EXPORT_MIX: run.settings.get(song.MIX_RUN, ""),
    }
    export_mix.keep_line(run.settings, options.loudness)
    store.save(run)
    return Exported(folder=out, files=listing(out), notes=notes, disclosure=text)


def _refuse_unready(run: RunRecord) -> None:
    """Nothing to export yet, or any more (a cleaned run's export would make its folder again
    and fail on ffmpeg's words)."""
    if run.files_removed:
        raise ExportError(
            f"{run.id}'s files were cleaned (`mnm clean outputs`): there is no mix or stems to "
            "export; the record stays."
        )
    if not _stems(run) or song.MIX_FILE not in {f.name for f in run.outputs}:
        raise ExportError(
            f"{run.id} has no stems and mix yet: an export follows the mix "
            f"(`mnm song show {run.id}` says where the run stands)."
        )


def _notes(run: RunRecord, in_use: song_takes.Take, missed: str | None) -> list[str]:
    """The run's notes for this export: the ones an export makes are made again (a release
    without consent, the line missed), a gap in the take's history said once."""
    made_again = ("Marked for release", export_mix.MISSED)
    notes = [n for n in run.notes if not n.startswith(made_again)]
    for gone in in_use.gaps:
        note = (
            f"The take's history lacks the edit {gone} (removed): the edits listed start after it."
        )
        notes += [] if note in notes else [note]
    return notes + ([missed] if missed else [])


def _edit_step(edit: RunRecord) -> EditStep:
    return EditStep(
        run=edit.id,
        edit=edit.settings.get("edit", ""),
        did=edit.answer,
        acts_on=edit.acts_on,
        source=edit.settings.get(song_takes.SOURCE, ""),
    )


def _stem_files(
    house: HouseFormat, folder: Path, run: RunRecord, stems: list[str], levels: MixSettings
) -> list[reaper.Track]:
    """Each stem as WAV (copied: it is in the house format already) and FLAC, and its
    REAPER track at the mixer's level, muted where the mix leaves it out."""
    tracks = []
    for name in stems:
        stem = name.removeprefix("stem-").removesuffix(".wav")
        wav = f"{STEMS}/{stem}.wav"
        shutil.copyfile(folder / name, folder / song.EXPORT_FOLDER / wav)
        flac = f"{song.EXPORT_FOLDER}/{STEMS}/{stem}.flac"
        house.convert(folder, Conversion(source=name, target=flac, codec=Codec.FLAC))
        length = wav_file_seconds(folder / name) or 0.0
        guid = reaper.track_guid(run.id, stem)
        level = levels.level(stem)
        tracks.append(
            reaper.Track(
                name=stem,
                file=wav,
                length=round(length, 6),
                guid=guid,
                volume=round(level.gain, 6),
                mute=not levels.sounds(stem),
            )
        )
    return tracks


def _timing(structure: StructureMap | None) -> dict[str, object]:
    """The project's tempo, meter and a marker at each part's start, from the take's map;
    REAPER's defaults without one."""
    if structure is None:
        return {}
    markers = [
        reaper.Marker(position=part.start, name=f"Part {part.label}") for part in structure.sections
    ]
    return {
        "tempo": round(structure.tempo, 3),
        "beats": structure.beats_per_bar,
        "markers": markers,
    }


def _lyrics(store: RunStore, run: RunRecord, take: song_takes.Take) -> song.Lyrics | None:
    """The lyrics the take in use was sung with (a re-sing's), None when unreadable."""
    try:
        return song_takes.lyrics(store, run, take)
    except song_takes.TakeError:
        return None


def _project_notes(
    lyrics: song.Lyrics | None,
    run: RunRecord,
    mix: export_mix.MixMade,
    structure: StructureMap | None,
    levels: MixSettings,
) -> list[str]:
    """The project's notes: the run, the loudness line, the mix's levels, where the tempo
    and markers come from, and the lyrics' sections in order (the markers are the
    analyser's parts; a lyric section is never placed on a guess)."""
    lines = [f"mnm run {run.id}: {run.prompt or ''}".strip()]
    lines.append(f"Loudness of mix.*: {mix.line} (mix.wav measures {mix.measured.words()})")
    lines.append(
        "Stems as separated (not normalised); the tracks carry the mix's levels and mutes, "
        f"its fades are in mix.* only. Mix: {levels.words()}."
    )
    if structure is None:
        lines.append("Tempo unknown to mnm: REAPER's default 120 BPM 4/4 (no structure map).")
    else:
        lines.append(
            f"Tempo, meter and part markers from the take's structure map ({structure.words()};"
            f" {structure.analyser}): measured, so check them by ear."
        )
    if lyrics is None:
        return lines
    lines.append(f"Style: {lyrics.tags}")
    for section in lyrics.sections:
        lines.append(f"[{section.label}] " + " / ".join(section.lines))
    return lines
