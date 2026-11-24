"""The section tools' half of an edit: where a moved or copied part goes, and the new
take ffmpeg splices from the take's pieces (`studio/edits_sections.py` says which pieces,
`audio_format.Splice` joins them). The take is copied beside the edit's run for ffmpeg's one
mount and removed after; nothing is sent to a model.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from makenomistakes.audio_format import HouseFormat, Splice
from makenomistakes.pipeline.song_steps import CANDIDATE_FILE
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.studio.edit_base import Span
from makenomistakes.studio.edits_sections import PlacedAsk, SectionAsk
from makenomistakes.studio.sections import SectionError, length_after
from makenomistakes.studio.tools_answers import StructureMap

SOURCE_COPY = ".source.wav"  # the take beside the edit's run while ffmpeg reads it


def nearest_bar(structure: StructureMap | None, at: float, length: float) -> float:
    """`at` on the nearest bar line of the map (the take's start and end count as ones)."""
    bars = structure.downbeats if structure is not None else []
    return min([0.0, *bars, length], key=lambda bar: abs(bar - at))


def placed(
    ask: SectionAsk, structure: StructureMap | None, length: float, span: Span
) -> float | None:
    """Where a moved or copied part goes, in seconds of the take as it is (None: right after
    itself, a duplicate's default)."""
    if not isinstance(ask, PlacedAsk):
        return None
    if ask.after == 0:
        return 0.0
    if ask.after is not None:
        parts = structure.sections if structure is not None else []
        if not parts:
            raise PipelineError(
                "after= names a part of the structure map, and the take has none: give at= in "
                "seconds instead."
            )
        if ask.after > len(parts):
            raise PipelineError(f"The take has parts 1 to {len(parts)}: after={ask.after} is none.")
        return min(parts[ask.after - 1].end, length)
    if ask.at is None:
        return None
    if ask.at > length:
        raise PipelineError(f"The take is {length:.1f} s long: at={ask.at:g} is past its end.")
    return nearest_bar(structure, ask.at, length) if ask.snap else ask.at


def pieces_of(
    ask: SectionAsk, structure: StructureMap | None, length: float, span: Span
) -> list[tuple[float, float]]:
    """The take's pieces in their new order; a refusal worded for the user."""
    try:
        return ask.pieces(span, length, placed(ask, structure, length, span))
    except SectionError as exc:
        raise PipelineError(f"The {ask.edit}: {exc}.") from exc


def new_length(ask: SectionAsk, pieces: list[tuple[float, float]]) -> float:
    return length_after(pieces, ask.fade)


def spliced(
    house: HouseFormat, here: Path, take: Path, ask: SectionAsk, pieces: list[tuple[float, float]]
) -> Path:
    """The new take (`song.wav` in the edit's folder): the pieces joined by ffmpeg."""
    shutil.copyfile(take, here / SOURCE_COPY)
    try:
        job = Splice(source=SOURCE_COPY, pieces=pieces, target=CANDIDATE_FILE, fade=ask.fade)
        return house.splice(here, job)
    finally:
        (here / SOURCE_COPY).unlink(missing_ok=True)
