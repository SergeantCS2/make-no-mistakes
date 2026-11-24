"""Tempo and key's half of an edit: the take through Rubber Band in the house format's
ffmpeg, or for a key change its own stems one by one (the drums left as they are, the vocals
keeping their formants) and summed back. `studio/edits_retime.py` holds the asks.
"""

from __future__ import annotations

from pathlib import Path

from makenomistakes.audio_format import Conversion, HouseFormat, HouseFormatError
from makenomistakes.pipeline import take_audio
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.pipeline.song_takes import Take
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.runs import RunRecord
from makenomistakes.studio import keys
from makenomistakes.studio.edit_base import EditHost
from makenomistakes.studio.edits import EditAsk
from makenomistakes.studio.edits_retime import DRUMS, KeyAsk, RetimeAsk, TempoAsk
from makenomistakes.studio.tools_answers import StructureMap

SOURCE = ".source.wav"  # the take beside the edit's run while ffmpeg reads it
IN = ".in-"  # a stem beside the edit's run, before it is shifted
OUT = ".key-"  # a stem shifted, before the stems are summed
NO_FILTER = (
    "The house format's ffmpeg has no Rubber Band filter (`rubberband`): audio.cpp's images "
    "carry one (Ubuntu's ffmpeg); `docker pull` one of them, then make the edit again."
)


def retimed(  # noqa: PLR0913 - the take, its stems and the seams
    house: HouseFormat,
    here: Path,
    take_path: Path,
    ask: RetimeAsk,
    structure: StructureMap | None,
    *,
    run: RunRecord,
    folder: Path,
    take: Take,
) -> tuple[list[Path], str]:
    """The new take (`song.wav` in the edit's folder) and what its line adds."""
    try:
        if isinstance(ask, TempoAsk):
            job = Conversion(source=SOURCE, target=CANDIDATE_FILE,
                             filters=ask.filters(ask.ratio(structure)))  # fmt: skip
            return [_one(house, here, take_path, job)], ""
        assert isinstance(ask, KeyAsk)
        step = ask.shift(structure)
        problem = take_audio.stems_problem(run, take)
        if problem is None:
            return _by_stems(house, here, folder, take_audio.take_stems(run, folder), step)
        job = Conversion(source=SOURCE, target=CANDIDATE_FILE,
                         filters=KeyAsk.filters(step, vocals=False))  # fmt: skip
        made = _one(house, here, take_path, job)
        return [made], f"; the whole mix shifted, drums too: the song's stems are {problem}"
    except HouseFormatError as exc:
        if "No such filter" in str(exc) and "rubberband" in str(exc):
            raise PipelineError(NO_FILTER) from exc
        raise


def _one(house: HouseFormat, here: Path, take_path: Path, job: Conversion) -> Path:
    take_audio.beside(take_path, here / SOURCE)
    try:
        return house.convert(here, job)
    finally:
        (here / SOURCE).unlink(missing_ok=True)


def _by_stems(
    house: HouseFormat, here: Path, folder: Path, stems: list[str], step: int
) -> tuple[list[Path], str]:
    """Each stem but the drums shifted, then summed into the new take."""
    made: list[str] = []
    try:
        for name in stems:
            take_audio.beside(folder / name, here / f"{IN}{name}")
            if name == DRUMS:
                made.append(f"{IN}{name}")
                continue
            job = Conversion(
                source=f"{IN}{name}",
                target=f"{OUT}{name}",
                filters=KeyAsk.filters(step, vocals=take_audio.is_vocals(name)),
            )
            house.convert(here, job)  # fmt: skip
            made.append(f"{OUT}{name}")
        take, note = take_audio.summed(house, here, made)
    finally:
        for name in stems:
            (here / f"{IN}{name}").unlink(missing_ok=True)
            (here / f"{OUT}{name}").unlink(missing_ok=True)
    kept = " but the drums" if DRUMS in stems else ""
    return [take], f"; its {len(stems)} stems{kept} shifted, then summed{note}"


def carried(ask: EditAsk, structure: StructureMap | None) -> keys.Carried | None:
    """The key the edit's take is in, handed on: a section tool or a tempo change keeps the
    take's, a key change moves it by its shift; an edit that sings (ACE-Step) hands nothing
    on, and neither does a take whose map names no key. The structure map of the new take then
    says this key rather than hearing it again (a key heard again drifts)."""
    if structure is None or structure.key is None or ask.host is not EditHost.HOUSE:
        return None
    key = structure.key
    if isinstance(ask, KeyAsk):
        key = keys.parse(key).shifted(ask.shift(structure)).words()
    return keys.Carried(key, structure.key_clarity)


def handed(take: Take) -> keys.Carried | None:
    """The key the take in use was handed by the edit that made it, if one was."""
    return keys.Carried.from_settings(take.edit.settings) if take.edit is not None else None
