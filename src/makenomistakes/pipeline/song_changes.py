"""What changes a finished song's take, voice or master: the voice the vocals are sung in, an
edit (a new take made from the one in use), the take the song uses, and the master its export
is made from. Each saves the change, makes again what was made from it (`song.MADE_FROM`) and
goes on as far as the song goes; the parent and every earlier take stay as they were."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

import httpx

from makenomistakes.audio_format import HouseFormat, HouseFormatError
from makenomistakes.instances import InstanceService
from makenomistakes.pipeline import (
    song,
    song_edits,
    song_master,
    song_mix,
    song_plan,
    song_takes,
    song_vocals,
    song_voice,
)
from makenomistakes.pipeline.song_takes import TakeError
from makenomistakes.pipeline.song_tools import read_structure
from makenomistakes.pipeline.stage_runner import Report, quiet, refuse_if_going
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.progress_words import Whole
from makenomistakes.runs import RunError, RunRecord, RunStore, StepRecord
from makenomistakes.studio import export as exports
from makenomistakes.studio.edit_base import EditHost
from makenomistakes.studio.edits import EditAsk
from makenomistakes.studio.mixer import MixError, MixSettings
from makenomistakes.studio.tools import StudioTools
from makenomistakes.studio.voice_library import VoiceLibrary


class _Runner(Protocol):
    """What the changes need of the song's runner."""

    store: RunStore
    service: InstanceService
    house: HouseFormat
    http: Callable[[], httpx.Client]
    state_dir: Path
    library: VoiceLibrary
    tools: StudioTools

    def _get(self, run_id: str) -> RunRecord: ...
    def _reset(self, run: RunRecord, items: list[StepRecord]) -> None: ...
    def claim(
        self, run_id: str, what: str, *, hand_back: bool = False
    ) -> AbstractContextManager[RunRecord]: ...
    def advance(self, run_id: str, report: Report = quiet) -> RunRecord: ...
    def make_again(self, run: RunRecord, stage: str, index: int = 0) -> None: ...


class SongChanges:
    """The song runner's moves after the take (`SongRunner` takes them in)."""

    def set_voice(
        self: _Runner, run_id: str, voice: str | None, shift: int = 0, report: Report = quiet
    ) -> RunRecord:
        """The song's voice: a library voice, or None for the song model's own;
        the vocals are sung again in it, then the mix and the export follow. Saved under the
        run's claim, as every move is (a pass in another process would write it back)."""
        with self.claim(run_id, "change its voice") as run:
            run.settings = song_voice.settings_with(
                run.settings, voice, shift, self.library, self.service, self.state_dir
            )
            song_plan.ensure_voice_step(run)
            self.make_again(run, song.VOICE)
        return self.advance(run_id, report)

    def set_vocals(
        self: _Runner,
        run_id: str,
        clean: str,
        key: str | None = None,
        strength: int | None = None,
        report: Report = quiet,
    ) -> RunRecord:
        """What is done to the separator's vocals: `dereverb,denoise,tune` and the
        like, or `none`; they are made again from the separator's, then the voice, mix and
        export. `key` and `strength` are tuning's (else the structure map's key, and 80)."""
        with self.claim(run_id, "clean its vocals") as run:
            run.settings = song_vocals.settings_with(
                run.settings, clean, key, strength, self.service, self.tools,
                store=self.store, run_id=run.id,
            )  # fmt: skip
            song_plan.ensure_vocals_step(run)
            self.make_again(run, song.VOCALS)
        return self.advance(run_id, report)

    def export(self: _Runner, run_id: str, options: exports.ExportOptions) -> exports.Exported:
        """The export again, with these options (MP3, release, the loudness line), under the
        run's claim: a pass in another process neither runs beside it nor writes over what it
        saved; the run is as it was afterwards."""
        with self.claim(run_id, "export it", hand_back=True) as run:
            try:
                return exports.render(
                    self.store, run, self.house, options, self.state_dir, library=self.library
                )
            except (exports.ExportError, HouseFormatError, RunError) as exc:
                raise PipelineError(str(exc)) from exc

    def render_mix(self: _Runner, run_id: str, settings: MixSettings) -> RunRecord:
        """The mixer's Render: a new mix of this song's stems, now the one the export uses."""
        run = self._get(run_id)
        refuse_if_going(run, "mix")
        try:
            return song_mix.render(self.store, self.house, run, settings)
        except (MixError, RunError) as exc:
            raise PipelineError(str(exc)) from exc

    def use_mix(self: _Runner, run_id: str, mix_id: str | None) -> RunRecord:
        """A render of this song (None: its own mix) as the one the export uses; refused while
        a pass runs (its next save would put its own choice back)."""
        with self.store.locked():
            run = self._get(run_id)
            refuse_if_going(run, "choose its mix")
            try:
                return song_mix.use(self.store, run, mix_id)
            except MixError as exc:
                raise PipelineError(str(exc)) from exc

    def draw(self: _Runner, run_id: str) -> int:
        """Measure the waveforms an older run lacks; how many the song has now."""
        return song_mix.draw(self.store, self.house, self._get(run_id))

    def edit(self: _Runner, run_id: str, ask: EditAsk, report: Report = quiet) -> RunRecord:
        """An edit of the take in use: a child run holding the new take, which the
        song uses from now on; its map, stems, voice, mix and export are made again."""
        with self.claim(run_id, "edit it") as run:
            take = song_takes.take(self.store, run)
            sung = ask.host is EditHost.ACE_STEP
            recipe = song_edits.editor(self.service, self.state_dir, run) if sung else None
            lyrics = song_takes.lyrics(self.store, run, take)
            by = f"on {recipe.name}" if recipe is not None else "with ffmpeg"
            report(f"{ask.words().capitalize()} of {song_takes.words(run, take)}'s take {by}")
            with self.http() as http:
                child = song_edits.make_edit(
                    self.store, run, recipe, http, self.house, ask, take=take, lyrics=lyrics,
                    structure=read_structure(self.store.folder(run)),
                )  # fmt: skip
            report(Whole(f"Edit {child.id}: {child.answer}."))
            _use(self, run_id, child.id, report)
        return self.advance(run_id, report)

    def use_take(self: _Runner, run_id: str, take: str | None, report: Report = quiet) -> RunRecord:
        """The take the song uses: one of its edits by id, or None (or the candidate's own run
        id) for the picked candidate; what was made from the take is made again."""
        run = self._get(run_id)
        refuse_if_going(run, "choose a take")
        take = chosen(self.store, run, take)
        if (run.settings.get(song.TAKE_RUN) or None) == take:
            report(Whole("That take is the one in use already."))
            return run
        with self.claim(run_id, "choose a take"):
            _use(self, run_id, take, report)
        return self.advance(run_id, report)

    def master(
        self: _Runner, run_id: str, name: str, data: bytes, report: Report = quiet
    ) -> RunRecord:
        """The mix in use mastered to a reference: a new master, now the one the
        export uses; the export is made again from it."""
        with self.claim(run_id, "master it") as run:
            report(f"Mastering {run.id}'s mix to {name} (Matchering, on the CPU)")
            child = song_master.make(self.store, self.tools, self.house, run, name, data)
            report(Whole(f"Master {child.id}: {child.answer}."))
            _export_again(self, run_id)
        return self.advance(run_id, report)

    def use_master(
        self: _Runner, run_id: str, master_id: str | None, report: Report = quiet
    ) -> RunRecord:
        """A master of this song as the one the export uses, or None for none (the export's
        loudness line again); the export is made again."""
        run = self._get(run_id)
        refuse_if_going(run, "choose its master")
        current = song_master.in_use(self.store, run)
        named = run.settings.get(song.MASTER_RUN) or None
        if (current.id if current else None) == master_id and named == master_id:
            report(
                Whole("That is the master in use already." if master_id else "No master is in use.")
            )
            return run
        with self.claim(run_id, "choose its master"):
            song_master.use(self.store, run_id, master_id)
            _export_again(self, run_id)
        which = f"the master {master_id}" if master_id else "no master (the loudness line)"
        report(Whole(f"The export uses {which} now."))
        return self.advance(run_id, report)


def _export_again(runner: _Runner, run_id: str) -> None:
    """The export reset for the pass that follows (inside the claim)."""
    with runner.store.locked():
        run = runner._get(run_id)
        runner._reset(run, [s for s in run.steps if s.stage == song.EXPORT])
        runner.store.save(run)


def chosen(store: RunStore, run: RunRecord, take: str | None) -> str | None:
    """The take asked for as the song records it (None for the picked candidate's), refused
    naming the takes it can use (the web asks before its job)."""
    root = song_takes.picked(store, run)  # nothing picked, or the candidate gone: said
    if take is None or take == root.id:
        return None
    why = song_takes.refusal(store, run, take)
    if why is not None:
        ok = [e for e, refused in song_takes.usable(store, run).items() if refused is None]
        raise TakeError(
            f"{take} cannot be used: {why}. The takes {run.id} can use: "
            f"{', '.join(['candidate', *ok])} (`candidate` is the picked candidate's)."
        )
    return take


def voice_settings(
    runner: _Runner, run: RunRecord, voice: str | None, shift: int
) -> dict[str, str]:
    """The run's settings with this voice, checked (the web asks before its job)."""
    refuse_if_going(run, "change its voice")
    return song_voice.settings_with(
        run.settings, voice, shift, runner.library, runner.service, runner.state_dir
    )


def _use(runner: _Runner, run_id: str, take: str | None, report: Report) -> None:
    """The song on this take (inside the claim): saved, with what was made from the take
    reset for the pass that follows."""
    with runner.store.locked():
        run = runner._get(run_id)
        if take is None:
            run.settings.pop(song.TAKE_RUN, None)
        else:
            run.settings[song.TAKE_RUN] = take
        after = song.MADE_FROM[song.SONG]
        runner._reset(run, [s for s in run.steps if s.stage in after])
        runner.store.save(run)
    which = f"the edit {take}" if take else "the picked candidate"
    report(Whole(f"The song uses {which}'s take now; its stems, mix and export follow."))
