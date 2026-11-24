"""A new song's record: its steps in the template's order, the user's own lyrics marked done,
a recording kept and hashed."""

from __future__ import annotations

from makenomistakes.pipeline import song, song_record
from makenomistakes.pipeline import song_steps as steps
from makenomistakes.pipeline.song import Lyrics, SongInputs
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.run_parts import InputRecord
from makenomistakes.runs import RunRecord, RunStore, StepRecord, StepState


def new_steps(inputs: SongInputs, stems: int) -> list[StepRecord]:
    """Every item the run makes, in order (the vocals and voice steps skip themselves when
    nothing is asked of them)."""
    return [
        StepRecord(stage=song.LYRICS),
        *(StepRecord(stage=song.SONG, index=i) for i in range(inputs.candidates)),
        *(StepRecord(stage=song.SCORE, index=i) for i in range(inputs.candidates)),
        StepRecord(stage=song.STRUCTURE),
        *(StepRecord(stage=song.STEMS, index=i) for i in range(stems)),
        StepRecord(stage=song.VOCALS),
        StepRecord(stage=song.VOICE),
        StepRecord(stage=song.MIDI),
        StepRecord(stage=song.MIX),
        StepRecord(stage=song.EXPORT),
    ]


def ensure_voice_step(run: RunRecord) -> StepRecord:
    """The voice step, added before the mix to a run made before the voice stage."""
    return ensure_step(run, song.VOICE, before=(song.MIDI, song.MIX))


def ensure_vocals_step(run: RunRecord) -> StepRecord:
    """The vocals step, added after the stems to a run made before the vocals stage."""
    return ensure_step(run, song.VOCALS, before=(song.VOICE, song.MIDI, song.MIX))


def ensure_step(run: RunRecord, stage: str, before: tuple[str, ...]) -> StepRecord:
    """The run's step of `stage`, added before the first of `before` it has when a run made
    before that stage lacks it (the voice step, the MIDI step)."""
    found = run.step(stage)
    if found is not None:
        return found
    step = StepRecord(stage=stage)
    at = next((i for i, s in enumerate(run.steps) if s.stage in before), len(run.steps))
    run.steps.insert(at, step)
    return step


def own_lyrics(store: RunStore, run: RunRecord, own: Lyrics, text: str) -> None:
    """The user's lyrics: saved, the stage done, the words recorded by their hash."""
    song_record.save_lyrics(store, run, own)
    lyrics_step = run.step(song.LYRICS)
    assert lyrics_step is not None
    lyrics_step.state, lyrics_step.text = StepState.DONE, own.title or "your lyrics"
    run.inputs_recorded.append(
        InputRecord(role="lyrics", name="lyrics", sha256=steps.digest(text.encode()))
    )


def keep_recording(store: RunStore, run: RunRecord, name: str, data: bytes) -> None:
    """The user's recording in the run's folder, recorded by its hash for the song stage."""
    path = store.add_input(run, name, data, role=song_record.RECORDING)
    run.inputs_recorded.append(
        InputRecord(
            role=song_record.RECORDING, name=path.name, sha256=steps.digest(data), stage=song.SONG
        )
    )


# A run made before a stage existed, by stage: the stages a redo adds (before which), or the
# words that say what to do instead
ADDED_BEFORE: dict[str, tuple[str, ...]] = {
    song.MIDI: (song.MIX,),
    song.VOCALS: (song.VOICE, song.MIDI, song.MIX),
}
REFUSED: dict[str, str] = {
    song.SCORE: "{run} was made by an older mnm, without the scores and the structure map: "
    "it has no {stage} step; a new song has both.",
    song.STRUCTURE: "{run} was made by an older mnm, without the scores and the structure map: "
    "it has no {stage} step; a new song has both.",
    song.VOICE: "{run} was made by an older mnm, without the voice stage: "
    "`mnm song voice {run} <voice>` adds it and sings the vocals in that voice.",
}


def refuse_missing(run: RunRecord, stage: str) -> None:
    """A redo of a stage the run was made before and cannot be given: what to do instead."""
    words = REFUSED.get(stage)
    if words is not None:
        raise PipelineError(words.format(run=run.id, stage=stage))


def add_missing(run: RunRecord, stage: str) -> None:
    """A redo of a stage the run was made before, which can be added (MIDI): its step,
    in its place."""
    if stage in ADDED_BEFORE:
        ensure_step(run, stage, before=ADDED_BEFORE[stage])
