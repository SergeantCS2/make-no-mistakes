"""Run a Song on the stage loop (`StageRunner`): lyrics → N candidates → their scores → the pick
→ the take's structure map → its stems → the mix → the export.

Each candidate is a child run of the song's run (`parent`, role `candidate`); the pick is the
run's `picked` setting. Redoing the lyrics makes everything again; redoing the picked
candidate makes it again and puts the run back at the pick (its score, the structure, stems,
mix and export with it), another candidate is made again with its score beside the pick (the
export follows: it lists them); a stem, the mix or the export redo what follows them; a score
or the structure redoes only itself (and the export, which reads the structure). The song's
own credits leave with the items that made what they are for (`song_credits`).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

import httpx

from makenomistakes.audio_format import (
    HouseFormat,
    HouseFormatError,
    default_house,
)
from makenomistakes.comfy.run import ComfyRunner
from makenomistakes.instances import InstanceService
from makenomistakes.make import credit
from makenomistakes.pipeline import (
    song,
    song_comfy,
    song_credits,
    song_items,
    song_midi,
    song_plan,
    song_players,
    song_record,
    song_takes,
    song_tools,
    song_vocals,
    song_voice,
)
from makenomistakes.pipeline import song_steps as steps
from makenomistakes.pipeline.runner import comfy_runner_for
from makenomistakes.pipeline.song import Lyrics, SongError, SongInputs
from makenomistakes.pipeline.song_changes import SongChanges
from makenomistakes.pipeline.stage_runner import Report, StageRunner, quiet, refuse_if_going
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.progress_words import Whole
from makenomistakes.recipes import Recipe
from makenomistakes.runs import (
    RunError,
    RunKind,
    RunRecord,
    RunState,
    RunStore,
    StepRecord,
    StepState,
)
from makenomistakes.studio import export as exports
from makenomistakes.studio import peaks
from makenomistakes.studio.stage_instances import Player
from makenomistakes.studio.tools import StudioTools, studio_tools
from makenomistakes.studio.voice_library import VoiceLibrary, VoiceRow
from makenomistakes.try_run import http_client


class SongRunner(SongChanges, StageRunner):
    name = song.NAME
    stages = song.STAGES
    labels = song.STAGE_LABELS

    def __init__(
        self,
        service: InstanceService,
        store: RunStore,
        http: Callable[[], httpx.Client],
        house: HouseFormat,
        clock: Callable[[], float] = time.monotonic,
        tools: StudioTools | None = None,
        comfy: ComfyRunner | None = None,
    ) -> None:
        super().__init__(store, http, clock)
        self.service = service
        self.house = house
        self.tools = tools or studio_tools(service.settings)
        self.comfy = comfy  # A song stage a ComfyUI workflow plays
        self.state_dir = service.settings.expanded_state_dir()
        self.library = VoiceLibrary(self.state_dir, service.settings.expanded_models_dir())

    # -- starting --------------------------------------------------------------------------

    def start(
        self,
        inputs: SongInputs,
        lyrics: str | None = None,
        recording: tuple[str, bytes] | None = None,
    ) -> RunRecord:
        """A new run, checked and saved; nothing is sent until `advance`. `lyrics` (the
        user's own) skip the lyrics stage; `recording` is kept and used as `inputs.recording`
        says."""
        if (recording is None) != (inputs.recording is None):
            raise PipelineError(
                "A recording goes with how to use it: --recording FILE --as "
                "complete|vocal2bgm|reference."
            )
        own = song.lyrics_from_text(lyrics, inputs.tags or inputs.theme) if lyrics else None
        inputs = inputs.model_copy(update={"own_lyrics": own is not None})
        chat = None if own else self._chat(inputs)
        sung_by = self._sung_by(inputs)
        if recording is not None and isinstance(sung_by, str):
            raise PipelineError(song_players.takes_no_recording(sung_by, inputs.tier))
        stems = self._players(song.STEMS, inputs)
        if inputs.voice is not None:  # The voice and its converter, before anything
            song_voice.converter(self.service, self.state_dir, self._voice(inputs), inputs.tier)
        settings = inputs.settings() | {"song_instance": song_players.name_of(sung_by)}
        settings |= {"chat_instance": chat.name} if chat is not None else {}
        try:  # its steps in its first save: a run on record always has them
            run = self.store.new(
                RunKind.PIPELINE, song.NAME, inputs.theme, settings,
                steps=song_plan.new_steps(inputs, len(stems)),
            )  # fmt: skip
        except RunError as exc:
            raise PipelineError(str(exc)) from exc
        try:
            if own is not None:
                song_plan.own_lyrics(self.store, run, own, lyrics or "")
            if recording is not None:
                song_plan.keep_recording(self.store, run, *recording)
            self.store.save(run)
        except BaseException as exc:  # a run that cannot keep what it was given is not kept
            with contextlib.suppress(RunError):
                self.store.delete(self.store.fail(run, str(exc) or "not started").id)
            if isinstance(exc, (OSError, RunError)):
                raise PipelineError(f"The song could not be started: {exc}") from exc
            raise
        return run

    def _chat(self, inputs: SongInputs) -> Recipe:
        return song_players.chat(self.service, self.state_dir, inputs)

    def _voice(self, inputs: SongInputs) -> VoiceRow:
        voice = song_voice.voice_of(self.library, inputs.settings())
        assert voice is not None  # asked only when the inputs name one
        return voice

    def _sung_by(self, inputs: SongInputs) -> Recipe | str:
        return song_players.sung_by(self.service, self.state_dir, self.comfy, inputs)

    def _separator(self, inputs: SongInputs) -> Recipe:
        """The stems stage's player that gives vocals: de-bleed runs it again."""
        players = self._players(song.STEMS, inputs)
        return next((p for p in players if "stem:vocals" in p.package.gives), players[0]).recipe

    def _players(self, stage: str, inputs: SongInputs) -> list[Player]:
        return song_players.playing(self.service, self.state_dir, stage, inputs)

    # -- the loop's hooks ------------------------------------------------------------------

    def _context(self, run: RunRecord, report: Report) -> SongInputs:
        return SongInputs.from_settings(run.settings)

    def pick_stage(self, run: RunRecord) -> str:
        """Where the pick waits: after the scores (a run made before the scores has none, and
        waits after the candidates)."""
        return song.SCORE if self.items(run, song.SCORE) else song.SONG

    def _waits(self, run: RunRecord, stage: str, context: SongInputs) -> bool:
        if stage not in song.REVIEWS or stage in run.reviewed:
            return False
        if stage == song.LYRICS:
            return not (context.own_lyrics or context.auto)
        if stage != self.pick_stage(run) or song.past_pick(run.reviewed):
            return False
        if context.auto:
            if song.picked_index(run.settings) is None:
                self._pick_best(run)
            return False
        return True

    def _pick_best(self, run: RunRecord) -> None:
        index, score = song_tools.best(self.store, run)
        run.settings[song.PICKED] = str(index)
        if score is None:
            note = (
                f"Picked candidate {index + 1} by itself (--auto): no candidate has a score; "
                f"`mnm song pick {run.id} <n>` picks another and makes its stems."
            )
        else:
            note = (
                f"Picked candidate {index + 1} by itself (--auto): the highest score, "
                f"{score:.2f}; `mnm song pick {run.id} <n>` picks another and makes its stems."
            )
        if note not in run.notes:
            run.notes.append(note)
        self.store.save(run)

    def _passing(self, run: RunRecord, stage: str) -> None:
        if stage in song.PICKS and song.picked_index(run.settings) is None:
            raise PipelineError(
                f"Pick a candidate first: `mnm song pick {run.id} <n>` (1 to "
                f"{len(self.items(run, song.SONG))}), or Pick on the song's page."
            )

    def _which(self, run: RunRecord, step: StepRecord, context: SongInputs) -> tuple[str, str]:
        which, _ = super()._which(run, step, context)
        if step.stage == song.SONG:
            return which, f" ({context.seconds:g} s)"
        if step.stage == song.MIDI:  # the user's MuScriptor plays it, when picked
            picked = song_midi.picked_name(self.state_dir, context.tier)
            return (
                which,
                f" ({picked} on audio.cpp)" if picked else " (the studio tools, on the CPU)",
            )
        if step.stage in (song.SCORE, song.STRUCTURE):
            return which, " (the studio tools, on the CPU)"
        return which, ""

    def _make(
        self, run: RunRecord, step: StepRecord, context: SongInputs, http: httpx.Client
    ) -> None:
        try:
            self._make_item(run, step, context, http)
        except (HouseFormatError, exports.ExportError, PipelineError, SongError, RunError) as exc:
            raise StepError(str(exc)) from exc
        self._draw(run, step)

    def _draw(self, run: RunRecord, step: StepRecord) -> None:
        """The waveforms of what the step made (best effort: a picture fails nothing)."""
        if step.stage == song.SONG and (child := self.candidate(run, step.index)) is not None:
            peaks.measure(self.house, self.store.folder(child), [steps.CANDIDATE_FILE])
        elif step.stage in (song.STEMS, song.VOCALS, song.VOICE, song.MIX):
            names = [f.name for f in run.outputs if f.name.endswith(".wav")]
            peaks.measure(self.house, self.store.folder(run), names)

    def _ours(self, run: RunRecord) -> bool:
        return song.is_song(run)

    def song(self, run_id: str) -> RunRecord:
        """A song run by id; a mix or an edit of one, or another run, refused naming it."""
        return self._get(run_id)

    def _get(self, run_id: str) -> RunRecord:
        """The song run; a mixer's render, an edit or a master named where a song is asked for
        says what it is."""
        try:
            found = self.store.get(run_id)
        except RunError as exc:
            raise PipelineError(str(exc)) from exc
        what = song.CHILD_ROLES.get(found.role or "") if found.parent is not None else None
        if what is not None:
            where = what[1].format(parent=found.parent)
            raise PipelineError(
                f"{found.id} is {what[0]} of the song {found.parent}, not a song: {where}."
            )
        return super()._get(run_id)

    def _make_item(
        self, run: RunRecord, step: StepRecord, context: SongInputs, http: httpx.Client
    ) -> None:
        match step.stage:
            case song.LYRICS:
                chat = self._chat(context)
                lyrics = steps.ask_lyrics(chat, http, context)
                song_record.save_lyrics(self.store, run, lyrics)
                song_credits.give(run, step, credit(chat))
                run.settings["chat_instance"] = chat.name  # who wrote them now (the export)
                step.text = lyrics.title or "lyrics"
            case song.SONG:
                sung_by = self._sung_by(context)
                self._drop_candidate(run, step)  # a failed try's child: replaced, never kept

                def named(child: RunRecord) -> None:
                    step.text, step.file = child.id, None
                    run.settings["song_instance"] = song_players.name_of(
                        sung_by
                    )  # who plays it now
                    self.store.save(run)

                if isinstance(sung_by, str):  # A ComfyUI workflow plays the stage
                    if context.recording is not None:  # the pick moved since the start
                        raise PipelineError(song_players.takes_no_recording(sung_by, context.tier))
                    song_comfy.make_candidate(
                        self.comfy, self.store, self.house, run, sung_by,
                        lyrics=self.lyrics(run), inputs=context, made=named,
                    )  # fmt: skip
                    return
                steps.make_candidate(
                    self.store, run, sung_by, http, self.house, lyrics=self.lyrics(run),
                    inputs=context, recording=song_record.recording(self.store, run), made=named,
                    style=song_players.style_for(self.service, sung_by),
                )  # fmt: skip
            case song.SCORE:
                child = self.candidate(run, step.index)
                song_tools.score(self.store, self.service, self.tools, run, step, child)
            case song.STRUCTURE:
                song_items.structure(self.store, self.tools, run, step)
            case song.STEMS:
                player = self._players(song.STEMS, context)[step.index]
                song_items.stems(self.store, run, step, player, http, self.house)
            case song.VOCALS:  # Cleaned, or as separated
                song_items.vocals(
                    self.store, run, step, http, self.house, service=self.service,
                    separator=lambda: self._separator(context), tools=self.tools,
                )  # fmt: skip
            case song.VOICE:  # The vocals in a library voice, or as sung
                song_items.voice(
                    self.store, run, step, context, http, self.house,
                    service=self.service, library=self.library, tools=self.tools,
                    state_dir=self.state_dir,
                )  # fmt: skip
            case song.MIDI:  # Each pitched stem's notes

                def pick() -> Recipe | None:
                    return song_midi.transcriber(self.service, self.state_dir, context.tier)

                song_midi.make(self.store, self.tools, run, step, pick=pick, http=http)
            case song.MIX:
                song_items.mix(self.store, run, step, self.house)
            case _:  # the export
                song_items.export(self.store, run, step, self.house, self.state_dir, self.library)

    def _finish(self, run: RunRecord) -> RunRecord:
        run.pid = None
        try:
            take = song_takes.take(self.store, run)
            which = song_takes.words(run, take) + ("'s take" if take.edit else "")
        except song_takes.TakeError:
            which = "a take"  # the steps that read the take would have failed: never here
        voiced = f", the vocals in {run.voices[0]}" if run.voices else ""
        answer = f"a song ({which}, its stems{voiced}, the mix and the export folder)"
        return self.store.finish(run, [], answer, None)

    # -- the record's parts (`song_record`) ----------------------------------------------

    def lyrics(self, run: RunRecord) -> Lyrics:
        return song_record.lyrics(self.store, run)

    def candidate(self, run: RunRecord, index: int) -> RunRecord | None:
        return song_takes.candidate(self.store, run, index)

    def candidates(self, run: RunRecord) -> list[tuple[StepRecord, RunRecord | None]]:
        return [(s, self.candidate(run, s.index)) for s in self.items(run, song.SONG)]

    # -- the user's moves ------------------------------------------------------------------

    def pick(self, run_id: str, number: int, report: Report = quiet) -> RunRecord:
        """Candidate `number` (from 1) is the one: past the pick, on to its stems. A pick
        after the run finished makes the new pick's stems, mix and export."""
        run = self._get(run_id)
        refuse_if_going(run, "pick")
        items = self.items(run, song.SONG)
        if not 1 <= number <= len(items):
            raise PipelineError(f"{run.id} has candidates 1 to {len(items)}; {number} is none.")
        step = items[number - 1]
        child = self.candidate(run, step.index)
        if child is None or child.state is not RunState.DONE:
            raise PipelineError(
                f"Candidate {number} is not made (`mnm song redo {run.id} song {number}`)."
            )
        changed = song.picked_index(run.settings) != step.index
        run.settings[song.PICKED] = str(step.index)
        run.notes = [n for n in run.notes if " by itself (--auto)" not in n]
        if changed:  # what was made from the old pick goes: its map, stems, voice, mixes, export
            after = song.MADE_FROM[song.SONG]
            self._reset(run, [s for s in run.steps if s.stage in after])
            run.settings.pop(song.TAKE_RUN, None)  # the old pick's edits stay, not in use
        if not song.past_pick(run.reviewed):
            run.reviewed.append(self.pick_stage(run))
        self.store.save(run)
        report(Whole(f"Picked candidate {number} ({child.id})."))
        return self.advance(run.id, report)

    def edit_lyrics(self, run_id: str, lyrics: Lyrics) -> RunRecord:
        """New lyrics from the user, while the run waits at the lyrics."""
        with self.store.locked():
            run = self._get(run_id)
            if run.state is not RunState.WAITING or run.waiting_at != song.LYRICS:
                raise PipelineError("The lyrics can be edited only while the run waits at them.")
            song_record.save_lyrics(self.store, run, lyrics)
            self.store.save(run)
        return run

    def _stale(self, run: RunRecord, step: StepRecord) -> list[StepRecord]:
        after = song.MADE_FROM[step.stage]
        if step.stage == song.SONG and song.picked_index(run.settings) not in (None, step.index):
            after = frozenset({song.EXPORT})  # not the pick: its take's parts stand
        stale = [step, *(s for s in run.steps if s.stage in after and s is not step)]
        if step.stage == song.STEMS:
            stale += [s for s in self.items(run, song.STEMS) if s.index > step.index]
        if step.stage == song.SONG:  # the candidate's own score goes with it
            stale += [s for s in self.items(run, song.SCORE) if s.index == step.index]
        if step.stage == song.LYRICS and SongInputs.from_settings(run.settings).own_lyrics:
            stale = [s for s in stale if s.stage != song.LYRICS]  # your lyrics stay
        stale = list({id(s): s for s in stale}.values())
        self._undo(run, stale, lyrics=step.stage == song.LYRICS)
        return stale

    def _reset(self, run: RunRecord, items: list[StepRecord]) -> None:
        """Make these items again at the next pass, with what they made undone."""
        self._undo(run, items)
        for item in items:
            item.state, item.error = StepState.PENDING, ""

    def _undo(self, run: RunRecord, items: list[StepRecord], lyrics: bool = False) -> None:
        """The one place a reset's effects live: each item's credits go with what it made;
        candidates made again lose their child runs, and the picked one the pick and the pick's
        review; new stems start without the old; new stems or a new mix drop the mixer's render
        from use (it summed other stems); a new structure map starts without the old take's;
        new lyrics reopen every review."""
        picked = song.picked_index(run.settings)
        stages = {i.stage for i in items if i.stage != song.SONG or i.index == picked}
        song_credits.drop(run, items)
        for item in items:
            if item.stage == song.SONG:
                self._drop_candidate(run, item)
            if item.stage == song.STRUCTURE:
                song_tools.forget_structure(self.store, run, item)
            if item.stage == song.MIDI:
                song_midi.forget(self.store, run, item)
        if song.SONG in stages:
            run.settings.pop(song.PICKED, None)
            run.settings.pop(song.TAKE_RUN, None)  # its edits were of another take
            run.reviewed = [r for r in run.reviewed if r not in song.PICKS]
        if lyrics:
            run.reviewed = []
        if stages & {song.SONG, song.STEMS, song.VOCALS}:  # the separator's vocals back, in
            song_voice.as_sung(self.store, run)  # turn: the voice's input, then the vocals
            song_vocals.as_separated(self.store, run)  # stage's: never done twice
        song_items.forget_stems(self.store, run, items)
        if stages & {song.SONG, song.STEMS}:  # a stems item without vocals keeps them
            run.settings.pop(song.STEMS_TAKE, None)  # the stems on file are no take's now
        if stages & {song.SONG, song.STEMS, song.VOCALS, song.VOICE, song.MIX}:
            run.settings.pop(song.MIX_RUN, None)

    def redo(self, run_id: str, stage: str, index: int, report: Report = quiet) -> RunRecord:
        if not self.items(self._get(run_id), stage):  # a run made before the stage
            with self.store.locked():
                run = self._get(run_id)
                song_plan.refuse_missing(run, stage)  # what to do instead, first
                refuse_if_going(run, f"add the {stage} step to")
                song_plan.add_missing(run, stage)
                self.store.save(run)
        return super().redo(run_id, stage, index, report)

    def rescore(self, run_id: str, report: Report = quiet) -> RunRecord:
        """Score every candidate again (after `mnm studio install`, or a new scorer); the pick
        stands, and a run waiting at it waits there again."""
        run = self._get(run_id)
        refuse_if_going(run, "score")
        items = self.items(run, song.SCORE)
        if not items:
            raise PipelineError(
                f"{run.id} was made by an older mnm, without the scores: it has none."
            )
        export = [s for s in run.steps if s.stage == song.EXPORT and s.state is StepState.DONE]
        self._reset(run, [*items, *export])  # the export's provenance lists the scores
        self.store.save(run)
        return self.advance(run.id, report)

    def _drop_candidate(self, run: RunRecord, step: StepRecord) -> None:
        """A candidate made again replaces the old one: its child run and files go."""
        if step.text:
            with contextlib.suppress(RunError):  # gone already
                self.store.delete(step.text)
        step.text, step.file = "", None


def song_runner(service: InstanceService, house: HouseFormat | None = None) -> SongRunner:
    """The runner for this box: the router through `mnm try`'s client, ffmpeg in Docker, and
    ComfyUI for a song stage a workflow plays."""

    settings = service.settings
    comfy = comfy_runner_for(service)
    return SongRunner(
        service, comfy.runs, lambda: http_client(settings), house or default_house(), comfy=comfy
    )
