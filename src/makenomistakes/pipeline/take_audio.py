"""The take's audio as the tool-side edits read it: its own stems, a file put beside an
edit's run for ffmpeg's one mount, and files summed into a new take that never clips. A
swap's backing, a key change's shifted stems and `lego`'s track over its bed all
come through here.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

from makenomistakes.audio_format import Codec, Conversion, HouseFormat, Loudness, Mixdown
from makenomistakes.pipeline import song, vocals_layers
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.pipeline.song_takes import Take
from makenomistakes.runs import RunRecord, StepState

CEILING = -1.0  # dBTP: a summed take louder than this is turned down to it
SUMMED = ".summed.wav"  # a sum (32-bit float) before it is checked for clipping
NOT_MADE = "not made yet"
ANOTHER_TAKE = "another take's"


def stems_problem(run: RunRecord, take: Take) -> str | None:
    """Why the song's stems are not the take in use's own (`NOT_MADE`, `ANOTHER_TAKE`), or
    None when every stems item is done since the take changed and the split names it."""
    items = [s for s in run.steps if s.stage == song.STEMS]
    if not items or any(s.state is not StepState.DONE for s in items):
        return NOT_MADE
    split = run.settings.get(song.STEMS_TAKE)
    if split not in (take.run.id, None) or (split is None and take.edit):
        return ANOTHER_TAKE  # a run split before stems named their take counts as the pick's
    return None


def take_stems(run: RunRecord, folder: Path) -> list[str]:
    """The take's stems by name in the song's folder, sorted: the vocals as the separator gave
    them (`vocals_layers.original`), never cleaned or sung in a library voice since: a new
    take made from them is split, cleaned and sung again."""
    names = [f.name for f in run.outputs]
    vocals = vocals_layers.original(names, folder)
    return [vocals if s == vocals_layers.VOCALS else s for s in sorted(song.stem_files(names))]


def is_vocals(name: str) -> bool:
    return vocals_layers.is_vocals(name)


def beside(source: Path, target: Path) -> Path:
    """`source` at `target` (a hard link where the folders share a disk, else a copy), for a
    job whose one mount is the target's folder."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def summed(
    house: HouseFormat, here: Path, sources: list[str], target: str = CANDIDATE_FILE
) -> tuple[Path, str]:
    """`sources` (in `here`) summed into `target` (a new take's `song.wav`, the song's own
    mix), turned down when the sum would pass `CEILING` (a clipped take would be split into
    clipped stems); a note when it was."""
    try:
        house.mix(here, Mixdown(sources=sources, target=SUMMED, codec=Codec.FLOAT))
        return unclipped(house, here, SUMMED, target)
    finally:
        (here / SUMMED).unlink(missing_ok=True)


def unclipped(house: HouseFormat, here: Path, made: str, target: str) -> tuple[Path, str]:
    """A sum written as float (`made`) in the house format at `target`, turned down by as much
    as its true peak passes `CEILING` (in 24-bit it would clip, without a word); the note that
    says so, or ""."""
    peak = house.measure(here, made, Loudness()).input_tp
    down = peak - CEILING if math.isfinite(peak) and peak > CEILING else 0.0
    job = Conversion(source=made, target=target, filters=[f"volume=-{down:.2f}dB"] if down else [])
    said = f"; turned down {down:.1f} dB so it does not clip" if down else ""
    return house.convert(here, job), said
