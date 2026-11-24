"""What the Song pipeline's later stages make, one item each: the structure map, a stem
package's stems, the vocals in a library voice, the mix and the export. The runner picks the
item; these do its work and fill its step."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from makenomistakes.audio_format import HouseFormat
from makenomistakes.instances import InstanceService
from makenomistakes.make import credit
from makenomistakes.pipeline import (
    song,
    song_credits,
    song_record,
    song_retime,
    song_takes,
    song_tools,
    song_vocals,
    song_voice,
    take_audio,
)
from makenomistakes.pipeline import song_steps as steps
from makenomistakes.pipeline.song import SongInputs
from makenomistakes.recipes.model import Recipe
from makenomistakes.runs import RunRecord, RunStore, StepRecord, StepState
from makenomistakes.studio import export as exports
from makenomistakes.studio.stage_instances import Player
from makenomistakes.studio.tools import StudioTools
from makenomistakes.studio.voice_library import VoiceLibrary


def structure(store: RunStore, tools: StudioTools, run: RunRecord, step: StepRecord) -> None:
    folder = store.folder(run)
    take = song_takes.take(store, run)
    path = song_takes.take_file(store, run, take)
    song_tools.structure(tools, run, step, path, folder, carried=song_retime.handed(take))
    if step.file:
        store.keep(run, folder / step.file)


def stems(
    store: RunStore,
    run: RunRecord,
    step: StepRecord,
    player: Player,
    http: httpx.Client,
    house: HouseFormat,
) -> None:
    take = song_takes.take(store, run)
    source = song_takes.take_file(store, run, take)
    made = steps.split(store, run, player, source, http, house)
    for path in made:
        store.keep(run, path)
    run.settings[song.STEMS_TAKE] = take.run.id  # what a swap sums is this take's
    step.file = made[0].name
    step.text = ", ".join(p.stem.removeprefix("stem-") for p in made)
    song_credits.give(run, step, credit(player.recipe))


def forget_stems(store: RunStore, run: RunRecord, items: list[StepRecord]) -> None:
    """New stems start without the old ones once every stems item is made again (else a split
    by another separator leaves the first one's stems in the mix, the mixer and the export);
    a later item alone (the full tier's second package) makes the same stems again over its
    own."""
    every = [s for s in run.steps if s.stage == song.STEMS]
    if not every or not all(any(s is item for item in items) for s in every):
        return
    names = set(song.stem_files([f.name for f in run.files]))
    for name in names:
        (store.folder(run) / name).unlink(missing_ok=True)
    run.files = [f for f in run.files if f.name not in names]


def vocals(  # noqa: PLR0913 - the run and who plays each clean-up beside the seams
    store: RunStore,
    run: RunRecord,
    step: StepRecord,
    http: httpx.Client,
    house: HouseFormat,
    *,
    service: InstanceService,
    separator: Callable[[], Recipe],
    tools: StudioTools,
) -> None:
    """The separator's vocals cleaned as the run asks, or as separated when it asks
    nothing: the step skips itself, and a clean-up asked before is put back."""
    wanted = song_vocals.asked(run.settings)
    if not wanted:
        put_back = song_vocals.as_separated(store, run)
        song_credits.give(run, step, [])
        step.state = StepState.SKIPPED
        step.text = "as separated" + (" (put back)" if put_back else "")
        return
    debleed = separator() if song_vocals.Clean.DEBLEED in wanted else None
    dereverb = song_vocals.dereverber(service) if song_vocals.Clean.DEREVERB in wanted else None
    tuning = (
        song_vocals.tuning_for(tools, run.settings, store.folder(run), run.id)
        if song_vocals.Clean.TUNE in wanted
        else None
    )
    take = song_vocals.clean(
        store, run, wanted, http, house, separator=debleed, dereverb=dereverb, tuning=tuning
    )
    step.file = song_vocals.VOCALS
    step.text = take.words()
    # the separator again (debleed) is the stems' own, credited by them
    song_credits.give(run, step, credit(dereverb) if dereverb is not None else [])


def voice(  # noqa: PLR0913 - the song's voice and its converter beside the seams
    store: RunStore,
    run: RunRecord,
    step: StepRecord,
    context: SongInputs,
    http: httpx.Client,
    house: HouseFormat,
    *,
    service: InstanceService,
    library: VoiceLibrary,
    tools: StudioTools,
    state_dir: Path,
) -> None:
    """The vocals in the run's library voice, or as sung when it names none (by default the
    song model sings), and how alike they sound to the voice's reference."""
    chosen = song_voice.voice_of(library, run.settings)
    if chosen is None:
        put_back = song_voice.as_sung(store, run)
        song_credits.give(run, step, [])
        step.state = StepState.SKIPPED
        step.text = "the song model's own voice" + (" (put back)" if put_back else "")
        return
    recipe = song_voice.converter(service, state_dir, chosen, context.tier)
    take = song_voice.convert(
        store, run, chosen, recipe, http, house, shift=context.voice_shift, library=library
    )
    song_credits.give(run, step, credit(recipe))
    missed = song_voice.similarity(tools, store, run, chosen, library, take)
    step.file = song_voice.VOCALS
    step.text = song_voice.words(take) + (f"; {missed}" if missed else "")


def mix(store: RunStore, run: RunRecord, step: StepRecord, house: HouseFormat) -> None:
    """The stems summed as made, turned down where the sum would clip."""
    folder = store.folder(run)
    names = song.stem_files([f.name for f in run.outputs])
    path, note = take_audio.summed(house, folder, sorted(names), target=song.MIX_FILE)
    store.keep(run, path)
    step.file, step.text = path.name, f"{len(names)} stems summed{note}"


def export(
    store: RunStore,
    run: RunRecord,
    step: StepRecord,
    house: HouseFormat,
    state_dir: Path,
    library: VoiceLibrary,
) -> None:
    options = song_record.export_options(run)
    done = exports.render(store, run, house, options, state_dir, library=library)
    step.file = song.EXPORT_FOLDER
    step.text = f"{len(done.files)} files in {song.EXPORT_FOLDER}/"
