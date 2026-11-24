"""A song's mixes and waveforms: the mixer's Render as a child run (role `mix`) of the
song, which mix the export uses, and the waveforms of every file the song made.

The song's own mix (`mix.wav`, the stems summed as made) is never changed: a render is a new
child run, the song's `mix_run` setting names the one in use, and choosing the song's own
again is undo. A mix of one pick's stems is dropped from use when the pick changes.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from makenomistakes.audio_format import Codec, HouseFormat
from makenomistakes.pipeline import song, take_audio
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.runs import RunError, RunKind, RunRecord, RunState, RunStore, StepState
from makenomistakes.studio import peaks
from makenomistakes.studio.mixer import SETTING, MixError, MixSettings, render_args

RENDERING = ".render-mix.wav"  # made in the song's folder (beside the stems, float), then
RENDERED = ".render-mix-24.wav"  # the house format's, turned down past the ceiling, moved
EXPORT_MIX = "export_mix"  # the song's setting: the mix (a render's id, or "") its export used
STEMS_KEY = "stems"  # a render's setting: which stems it summed (`stems_key`)


def stem_names(run: RunRecord) -> list[str]:
    """`vocals`, `drums` …: the stems the song has, as its files name them."""
    return [
        n.removeprefix("stem-").removesuffix(".wav")
        for n in song.stem_files([f.name for f in run.outputs])
    ]


def mixes(store: RunStore, run: RunRecord) -> list[RunRecord]:
    """The mixer's renders of this song, oldest first."""
    return [c for c in store.children(run.id) if c.role == "mix"]


def in_use(store: RunStore, run: RunRecord) -> RunRecord | None:
    """The render the export uses; None for the song's own mix (or a render that is gone)."""
    chosen = run.settings.get(song.MIX_RUN)
    if not chosen:
        return None
    try:
        child = store.get(chosen)
    except RunError:
        return None
    usable = child.parent == run.id and child.state is RunState.DONE and not child.files_removed
    return child if usable else None


def settings_of(store: RunStore, run: RunRecord) -> MixSettings | None:
    """The levels of the mix in use (every stem as made for the song's own); None before
    the song has stems."""
    stems = stem_names(run)
    if not stems:
        return None
    child = in_use(store, run)
    if child is None or SETTING not in child.settings:
        return MixSettings.flat(stems)
    try:
        return MixSettings.from_setting(child.settings[SETTING])
    except MixError:
        return MixSettings.flat(stems)


def render(store: RunStore, house: HouseFormat, run: RunRecord, settings: MixSettings) -> RunRecord:
    """The stems through the mixer into a new mix, a child run now in use. Refused until
    the song's mix step is done: before it, the stems are not the ones the song keeps."""
    mixed = run.step(song.MIX)
    if mixed is None or mixed.state is not StepState.DONE:
        raise MixError(
            f"{run.id}'s stems are not made yet (the mix step is not done): the mixer follows "
            f"them (`mnm song show {run.id}` says where the song stands)."
        )
    folder = store.folder(run)
    stems: dict[str, float | None] = {}
    for name in stem_names(run):
        shape = peaks.wav_shape(folder / f"stem-{name}.wav")
        stems[f"stem-{name}.wav"] = shape.seconds if shape is not None else None
    args = render_args(settings, stems, RENDERING, Codec.FLOAT)
    made_from = {SETTING: settings.setting(), STEMS_KEY: stems_key(store, run)}
    child = store.new(
        RunKind.PIPELINE, song.NAME, settings.words(), made_from, parent=run.id, role="mix"
    )
    started = time.monotonic()
    try:
        house.run(folder, args)
        rendered, note = take_audio.unclipped(house, folder, RENDERING, RENDERED)
        made = store.folder(child) / song.MIX_FILE
        shutil.move(str(rendered), made)
    except BaseException as exc:  # an interrupted render ends failed, never "running" forever
        store.fail(child, str(exc) or f"Interrupted ({type(exc).__name__}).")
        if isinstance(exc, (OSError, RuntimeError)):
            raise MixError(f"The mix did not render: {exc}") from exc
        raise
    finally:
        for scratch in (RENDERING, RENDERED):
            (folder / scratch).unlink(missing_ok=True)
    took = time.monotonic() - started
    child = store.finish(child, [made], f"a mix: {settings.words()}{note}", took)
    peaks.measure(house, store.folder(child), [song.MIX_FILE])
    _choose(store, run.id, child.id)
    return child


def use(store: RunStore, run: RunRecord, mix_id: str | None) -> RunRecord:
    """Make a render of this song (or, with None, the song's own mix) the one in use."""
    if mix_id is not None:
        try:
            child = store.get(mix_id)
        except RunError as exc:
            raise MixError(str(exc)) from exc
        if child.parent != run.id or child.role != "mix" or child.state is not RunState.DONE:
            raise MixError(f"{mix_id} is not a finished mix of {run.id} (`mnm song mix {run.id}`).")
        if child.settings.get(STEMS_KEY, stems_key(store, run)) != stems_key(store, run):
            raise MixError(
                f"{mix_id} summed other stems or vocals than {run.id} has now (an earlier pick "
                f"or voice): Render again from these (`mnm song mix {run.id} …`)."
            )
    return _choose(store, run.id, mix_id)


def stems_key(store: RunStore, run: RunRecord) -> str:
    """Which stems the song has now: each stem file's name, size and time (a new pick, new
    stems or another voice changes it), so a render of other stems is never used."""
    folder = store.folder(run)
    digest = hashlib.sha256()
    for name in sorted(song.stem_files([f.name for f in run.outputs])):
        try:
            found = (folder / name).stat()
        except OSError:
            continue
        digest.update(f"{name}:{found.st_size}:{found.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:16]


def _choose(store: RunStore, run_id: str, mix_id: str | None) -> RunRecord:
    """The mix in use, saved on the song as it is now (a pick made meanwhile stands)."""
    with store.locked():
        run = store.get(run_id)
        if mix_id is None:
            run.settings.pop(song.MIX_RUN, None)
        else:
            run.settings[song.MIX_RUN] = mix_id
        store.save(run)
    return run


def export_behind(run: RunRecord) -> bool:
    """Whether the export folder was made from another mix than the one in use now."""
    if "export_loudness" not in run.settings:
        return False  # never exported
    return run.settings.get(EXPORT_MIX, "") != run.settings.get(song.MIX_RUN, "")


def mix_source(store: RunStore, run: RunRecord) -> Path:
    """The file the export's mix is made from: the render in use, else the song's own."""
    child = in_use(store, run)
    if child is not None:
        path = store.folder(child) / song.MIX_FILE
        if path.is_file():
            return path
    return store.folder(run) / song.MIX_FILE


def draw(store: RunStore, house: HouseFormat, run: RunRecord) -> int:
    """Measure every waveform the song's page shows that has none yet (its candidates, its
    stems and mix, its renders); how many it has now."""
    count = 0
    names = [f.name for f in run.outputs if f.name.endswith(".wav")]
    count += len(peaks.measure(house, store.folder(run), names))
    for child in store.children(run.id):
        if child.files_removed or child.state is not RunState.DONE:
            continue
        made = {"mix": song.MIX_FILE, "master": song.MASTER_FILE}
        name = made.get(child.role or "", CANDIDATE_FILE)  # a candidate, an edit
        count += len(peaks.measure(house, store.folder(child), [name]))
    return count
