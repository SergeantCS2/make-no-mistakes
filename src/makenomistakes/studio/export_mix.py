"""The export's mix: `mix.wav` (24-bit, 48 kHz), `mix.flac` and, on request, `mix.mp3`, from the
mix in use at the loudness line (ffmpeg's `loudnorm`, two passes), or from the master in use as
it was made (turned down only past the line's true-peak ceiling); then what the written
`mix.wav` measures, which is what the provenance and the REAPER notes record, never the line's
target or the input's figure: `loudnorm` falls back to its dynamic mode when the gain would pass
the ceiling, and a quiet mix with loud transients came out at -18.1 LUFS / -0.9 dBTP against a
line of -14 / -1.

The loudness line the user chose is kept on the run (`LINE`), so every export the pipeline
makes after `mnm song export --lufs … --true-peak …` (a pick, an edit, a voice, a master)
keeps it."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from pydantic import BaseModel, ValidationError

from makenomistakes.audio_format import Codec, Conversion, HouseFormat, Loudness, Measured
from makenomistakes.pipeline import song

LINE = "export_line"  # the run's setting: the loudness line asked (Loudness as JSON, or `off`)
OFF = "off"
SOURCE = ".export-mix.wav"  # a render's mix or a master, copied beside the stems for ffmpeg
MASTER_DOWN = ".export-master-down.wav"  # a master turned down to the true-peak ceiling
TARGET = f"{song.EXPORT_FOLDER}/mix.wav"
MISSED = "The export's mix measures"  # the note's start: made again at every export
OFF_WORDS = "off (the mix as summed)"
LU_MISS = 1.0  # LU from the target past which the line is said missed
TP_SLACK = 0.05  # dB over the ceiling a measure may read without a word


class MixMade(BaseModel):
    """`mix.*` as written: the line it was asked at, in words, and what `mix.wav` measures."""

    line: str
    measured: Measured
    note: str | None = None  # the line missed or its ceiling passed, worded


def line_of(settings: dict[str, str]) -> Loudness | None:
    """The loudness line the last export asked (None: off); the default line for a run never
    exported, or exported before the line was kept (its words said only on or off)."""
    kept = settings.get(LINE)
    if kept == OFF:
        return None
    if kept:
        try:
            return Loudness.model_validate_json(kept)
        except ValidationError:
            return Loudness()
    return None if settings.get("export_loudness", "").startswith(OFF) else Loudness()


def asked_line(
    settings: dict[str, str], lufs: float | None, true_peak: float | None, off: bool
) -> Loudness | None:
    """The line an export asks: off when told; else the line the last export kept, changed
    only where a figure is given (the CLI's `--lufs`/`--true-peak`; a plain `mnm song export
    --mp3` keeps the user's -9, as the web's Export box does)."""
    if off:
        return None
    kept = line_of(settings)
    if kept is None and lufs is None and true_peak is None:
        return None
    base = kept or Loudness()
    return Loudness(
        lufs=base.lufs if lufs is None else lufs,
        true_peak=base.true_peak if true_peak is None else true_peak,
    )


def keep_line(settings: dict[str, str], loudness: Loudness | None) -> None:
    settings[LINE] = loudness.model_dump_json() if loudness is not None else OFF


def made(
    house: HouseFormat,
    folder: Path,
    loudness: Loudness | None,
    mp3: bool,
    source: Path,
    master: str | None,
    run_id: str,
) -> MixMade:
    """`mix.*` from `source` (the song's own mix, a render, or a master named `master`)."""
    asked: Loudness | None = None
    ceiling: float | None = None  # none for the mix as summed: no line was asked
    if master is not None:
        ceiling = (loudness or Loudness()).true_peak  # the line's ceiling holds a master too
        line = f"as mastered to {master}" + _master(house, folder, source, ceiling, mp3)
    else:
        line = loudness.words() if loudness is not None else OFF_WORDS
        _from(house, folder, loudness, mp3, source)
        asked, ceiling = loudness, loudness.true_peak if loudness is not None else None
    measured = house.measure(folder, TARGET, Loudness())
    note = _missed(measured, asked, ceiling, run_id)
    return MixMade(line=line, measured=measured, note=note)


def _missed(
    measured: Measured, asked: Loudness | None, ceiling: float | None, run_id: str
) -> str | None:
    """Why the written mix is not at the line, when it is not (silence says nothing)."""
    if measured.silent:
        return None
    if asked is not None and asked.lufs - measured.input_i > LU_MISS:
        return (
            f"{MISSED} {measured.words()}, not the line's {asked.lufs:g} LUFS: loudnorm stops "
            f"short of the {asked.true_peak:g} dBTP ceiling on a mix with loud peaks "
            f"(`mnm song export {run_id} --lufs {measured.input_i:.0f}` asks for what it "
            f"reaches; `mnm song master {run_id} --reference FILE` masters it to a track)."
        )
    if ceiling is not None and measured.input_tp > ceiling + TP_SLACK:
        return f"{MISSED} {measured.words()}, over the line's {ceiling:g} dBTP ceiling."
    return None


def _from(
    house: HouseFormat, folder: Path, loudness: Loudness | None, mp3: bool, source: Path
) -> None:
    """`mix.*` at the loudness line from the song's own mix or a render (copied beside it for
    ffmpeg's one mount)."""
    name = song.MIX_FILE
    if source.parent != folder:
        shutil.copyfile(source, folder / SOURCE)
        name = SOURCE
    try:
        _write(house, folder, loudness, mp3, name)
    finally:
        (folder / SOURCE).unlink(missing_ok=True)


def _master(house: HouseFormat, folder: Path, source: Path, ceiling: float, mp3: bool) -> str:
    """`mix.*` from the master as made (the reference set its loudness), turned down only
    where its true peak passes the ceiling (a master can read -0.8 dBTP: Matchering limits
    sample peaks, and a lossy encode of a true-peak over clips); the words that says so."""
    shutil.copyfile(source, folder / SOURCE)
    try:
        name, said = SOURCE, ""
        peak = house.measure(folder, SOURCE, Loudness()).input_tp
        if math.isfinite(peak) and peak > ceiling:
            down = peak - ceiling
            job = Conversion(source=SOURCE, target=MASTER_DOWN, filters=[f"volume=-{down:.2f}dB"])
            name = house.convert(folder, job).name
            said = f", turned down {down:.1f} dB to {ceiling:g} dBTP"
        _write(house, folder, None, mp3, name)
    finally:
        for scratch in (SOURCE, MASTER_DOWN):
            (folder / scratch).unlink(missing_ok=True)
    return said


def _write(
    house: HouseFormat, folder: Path, loudness: Loudness | None, mp3: bool, name: str
) -> None:
    if loudness is None:
        house.convert(folder, Conversion(source=name, target=TARGET))
    else:
        first = house.measure(folder, name, loudness)
        house.normalised(folder, name, TARGET, Codec.WAV24, loudness, first)
    for codec in [Codec.FLAC, *([Codec.MP3] if mp3 else [])]:
        target = f"{song.EXPORT_FOLDER}/mix{codec.suffix}"
        house.convert(folder, Conversion(source=TARGET, target=target, codec=codec))
