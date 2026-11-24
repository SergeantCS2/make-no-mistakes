"""Whether a music family's figures are this box's yet: the plan says it is an estimate
until this box holds three VRAM readings of instances of the family, as the picture
corrections wait for three. The readings themselves price an instance through
`calibration`'s instance figure (Fits now, room); this only words the plan."""

from __future__ import annotations

from pathlib import Path

from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.fit.music import basis_words
from makenomistakes.hub.music import MusicFamily
from makenomistakes.measurements import MeasurementStore
from makenomistakes.readings_filter import every_here
from makenomistakes.store import InstanceStore, StoreError


def family_readings(state_dir: Path, family: MusicFamily) -> int:
    """This box's VRAM readings of every saved instance of the family."""
    try:
        recipes = InstanceStore(state_dir).list()
    except StoreError:
        return 0
    names = {
        r.name
        for r in recipes
        if isinstance(r.args, AudioCppArgs) and r.args.family == family.value
    }
    readings = every_here(MeasurementStore(state_dir))
    return sum(1 for n in names for reading in readings.get(n, []) if reading.vram_bytes)


def plan_basis(state_dir: Path, family: MusicFamily) -> str:
    return basis_words(family_readings(state_dir, family))
