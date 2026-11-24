"""The vocals stage: the separator's vocals cleaned before a voice sings them again.

What the user asks of a song's vocals (`mnm song vocals <run> clean=…`, the Song page's
Vocals panel) runs on the vocals stem in a fixed order, each from the one before:

- `debleed`: the stems stage's own separator run again on the vocals, its vocals kept (what
  other instruments leaked into the stem the first pass left);
- `dereverb`: anvuew's room dereverb (`hub/converted.py`, a checkpoint mnm converts for
  audio.cpp), the voice it gives back kept, mono, written to both channels;
- `denoise`, `deess`, `gate`: ffmpeg in the house format, one chain: `afftdn` at the stem's
  own noise floor (the tenth percentile of its slices' loudness, `studio/peaks`; a fixed
  floor took 0.3 dB of noise off where the measured one took 12),
  `deesser`, and `agate` a little above that floor.

The separator's vocals are kept once as `vocals-raw.wav`, and every run of the stage starts
from them, so cleaning twice never cleans a cleaned stem; nothing asked puts them back
(`as_separated`), as new stems do. `vocals.json` beside them records what was done.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel

from makenomistakes.audio_format import Codec, Conversion, HouseFormat
from makenomistakes.engine_answers import MalformedAnswerError
from makenomistakes.hub.converted import converted_by_file, converted_for_purpose, kit_of
from makenomistakes.hub.music import MusicFamily
from makenomistakes.instances import InstanceService
from makenomistakes.music_api import parse_answer
from makenomistakes.music_api import split_request as audio_request
from makenomistakes.pipeline import song, vocals_layers
from makenomistakes.pipeline.song_steps import audio_seconds, post_task
from makenomistakes.pipeline.song_tools import read_structure
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.recipes.model import Recipe
from makenomistakes.runs import RunRecord, RunStore
from makenomistakes.studio import keys, peaks
from makenomistakes.studio.keys import Key
from makenomistakes.studio.tools import StudioTools
from makenomistakes.studio.tools_answers import ToolsError

SETTING = "vocals_clean"  # the run's setting (`SongInputs.vocals_clean`): what was asked
VOCALS = song.stem_file("vocals")
RAW = vocals_layers.RAW  # the separator's vocals, kept for another clean-up or none
RECORD = "vocals.json"  # what was done to them, beside them
WORK = ".vocals-work.wav"
ENGINE_IN = ".vocals-engine.wav"
FLOOR_PERCENTILE = 0.10  # of the stem's slices: its quiet parts, the floor of what plays
FLOOR_RANGE = (-80.0, -20.0)  # what ffmpeg's `afftdn` takes as its noise floor
GATE_ABOVE = 10.0  # dB over the floor where the gate opens
DEESS = "deesser=i=0.4:m=0.5:f=0.5:s=o"


class Clean(StrEnum):
    DEBLEED = "debleed"
    DEREVERB = "dereverb"
    DENOISE = "denoise"
    DEESS = "deess"
    GATE = "gate"
    TUNE = "tune"  # last: the cleaned voice toward the key's notes (the studio tools)


ORDER = tuple(Clean)
ENGINES = (Clean.DEBLEED, Clean.DEREVERB)  # on audio.cpp
FFMPEG = (Clean.DENOISE, Clean.DEESS, Clean.GATE)  # one chain in the house format
WORDS = {
    Clean.DEBLEED: "the other instruments' bleed taken out by the separator again",
    Clean.DEREVERB: "the room's reverb taken off",
    Clean.DENOISE: "the noise under them lowered",
    Clean.DEESS: "the sibilants softened",
    Clean.GATE: "the silences gated",
    Clean.TUNE: "tuned",
}
KEY = "vocals_key"  # the run's setting: the key tuning moves toward (else the structure map's)
STRENGTH = "vocals_strength"  # the run's setting: how far, 0 to 100
DEFAULT_STRENGTH = 80
RETUNE_MS = 50  # how fast a sung note is moved: slow enough that vibrato and slides stay
CHAIN = "vocals-before-tuning.wav"  # the chain's result (the tools take no dot-name)


@dataclass(frozen=True)
class Tuning:
    """What tuning needs: the tools, the key and how far."""

    tools: StudioTools
    key: Key
    strength: int
    note: str  # where the key came from


class VocalsDone(BaseModel):
    clean: Clean
    by: str  # an instance's name, or `ffmpeg`
    note: str = ""


class VocalsTake(BaseModel):
    """What the stage did to the separator's vocals, in order (`vocals.json`)."""

    done: list[VocalsDone]

    def words(self) -> str:
        parts = [f"{WORDS[d.clean]}{f' ({d.note})' if d.note else ''}" for d in self.done]
        return "; ".join(parts) or "as separated"


def done_in(folder: Path) -> list[str]:
    """What the vocals stage did, as words, from `vocals.json` (the export's provenance)."""
    try:
        take = VocalsTake.model_validate_json((folder / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [f"{WORDS[d.clean]}{f' ({d.note})' if d.note else ''}" for d in take.done]


def parse(text: str) -> list[Clean]:
    """`dereverb,denoise` as the stage's order; `none` or nothing is none."""
    names = [n.strip().lower() for n in text.replace(" ", ",").split(",") if n.strip()]
    if names in ([], ["none"], ["off"]):
        return []
    known = {c.value for c in Clean}
    if unknown := [n for n in names if n not in known]:
        raise PipelineError(
            f"Not a clean-up mnm makes: {', '.join(unknown)}. It makes "
            f"{', '.join(c.value for c in ORDER)} (in that order), or `none`."
        )
    return [c for c in ORDER if c.value in names]


def asked(settings: dict[str, str]) -> list[Clean]:
    return parse(settings.get(SETTING, ""))


def settings_with(  # noqa: PLR0913 - the settings, what the user asked, and what checks it
    settings: dict[str, str],
    clean: str,
    key: str | None,
    strength: int | None,
    service: InstanceService,
    tools: StudioTools,
    *,
    store: RunStore | None = None,
    run_id: str = "",
) -> dict[str, str]:
    """The run's settings with this clean-up, checked before anything is made again (the
    CLI, `song new --clean` and the web share it): names mnm makes, the dereverb added, the
    tools installed for tuning, a key it reads, and for a song already made (`store` and
    `run_id`) a key to tune to now, before the voice's work is thrown away."""
    wanted = parse(clean)
    if Clean.DEREVERB in wanted:
        dereverber(service)
    if strength is not None and not 0 <= strength <= 100:
        raise PipelineError(f"--strength {strength}: tuning's strength is 0 to 100.")
    if key is not None:
        _parsed(key)
    changed = {k: v for k, v in settings.items() if k not in (SETTING, KEY, STRENGTH)}
    changed[SETTING] = ",".join(c.value for c in wanted)
    if key is not None:
        changed[KEY] = key
    if strength is not None:
        changed[STRENGTH] = str(strength)
    if Clean.TUNE in wanted:
        _tools_ready(tools)
        if store is not None:  # the key tuning will read: the settings as they are saved
            _key_for(changed, store.folder(store.get(run_id)), run_id)
    return changed


def dereverber(service: InstanceService) -> Recipe:
    """A saved instance of a converted checkpoint whose purpose is dereverb."""
    rows = converted_for_purpose(Clean.DEREVERB.value)
    for recipe in service.all():
        row = converted_by_file(recipe.files[0]) if recipe.files else None
        if row is not None and row in rows:
            return recipe
    raise PipelineError(
        f"Dereverb runs on {rows[0].kit.label}, which is not added: `mnm add {rows[0].repo_id}` "
        "(118 MB; mnm converts it for audio.cpp once)."
    )


def clean(  # noqa: PLR0913 - the run, what was asked and who plays each step, beside the seams
    store: RunStore,
    run: RunRecord,
    wanted: list[Clean],
    http: httpx.Client,
    house: HouseFormat,
    *,
    separator: Recipe | None,
    dereverb: Recipe | None,
    tuning: Tuning | None = None,
) -> VocalsTake:
    """The separator's vocals with `wanted` done to them, written as the vocals stem."""
    folder = store.folder(run)
    raw = _raw(store, run)
    shutil.copyfile(raw, folder / WORK)
    done: list[VocalsDone] = []
    try:
        for step in (c for c in wanted if c in ENGINES):
            player = separator if step is Clean.DEBLEED else dereverb
            if player is None:
                raise PipelineError(f"No instance plays {step.value}.")
            _engine_pass(player, http, house, folder, run)
            done.append(VocalsDone(clean=step, by=player.name))
        filters, notes = _chain([c for c in wanted if c in FFMPEG], house, folder)
        tune = Clean.TUNE in wanted
        job = Conversion(source=WORK, target=CHAIN if tune else VOCALS, filters=filters)
        made = house.convert(folder, job)
        done += [
            VocalsDone(clean=c, by="ffmpeg", note=notes.get(c, "")) for c in wanted if c in FFMPEG
        ]
        if tune:
            if tuning is None:
                raise PipelineError("Tuning was asked with nothing to tune by.")
            done.append(_tuned(tuning, folder))
            made = folder / VOCALS
        store.keep(run, made)
    finally:
        for name in (WORK, CHAIN):
            (folder / name).unlink(missing_ok=True)
    take = VocalsTake(done=done)
    record = folder / RECORD
    record.write_text(take.model_dump_json(indent=2) + "\n", encoding="utf-8")
    store.keep(run, record)
    return take


def tuning_for(tools: StudioTools, settings: dict[str, str], folder: Path, run_id: str) -> Tuning:
    """The key tuning moves toward and how far, the tools installed, or a refusal saying so."""
    _tools_ready(tools)
    key, note = _key_for(settings, folder, run_id)
    return Tuning(tools, key, int(settings.get(STRENGTH) or DEFAULT_STRENGTH), note)


def _key_for(settings: dict[str, str], folder: Path, run_id: str) -> tuple[Key, str]:
    """The user's key, else the structure map's (said when it was heard unsurely)."""
    if settings.get(KEY):
        return _parsed(settings[KEY]), " (the key you gave)"
    structure = read_structure(folder)
    if structure is None or structure.key is None:
        raise PipelineError(
            "Tuning needs the song's key and the structure map names none: give it (the "
            f'Vocals panel\'s Key, or `mnm song vocals {run_id} tune --key "A minor"`).'
        )
    unsure = (
        " (the structure map's key, heard unsurely: --key makes sure)"
        if structure.unsure
        else " (the structure map's key)"
    )
    return _parsed(structure.key), unsure


def _parsed(key: str) -> Key:
    try:
        return keys.parse(key)
    except keys.KeyNameError as exc:
        raise PipelineError(str(exc)) from exc


def _tools_ready(tools: StudioTools) -> None:
    try:
        tools.installed()
    except ToolsError as exc:
        raise PipelineError(f"Tuning runs in the studio tools: {exc}") from exc


def _tuned(tuning: Tuning, folder: Path) -> VocalsDone:
    try:
        answer = tuning.tools.tune(folder, CHAIN, VOCALS, tuning.key, tuning.strength, RETUNE_MS)
    except ToolsError as exc:  # the step's error, worded, never a traceback
        raise StepError(f"Tuning: {exc}") from exc
    note = (
        f"to {tuning.key.words()}{tuning.note}, {tuning.strength}%: the sung notes moved "
        f"{answer.moved_cents:g} cents (median)"
    )
    return VocalsDone(clean=Clean.TUNE, by="the studio tools", note=note)


def as_separated(store: RunStore, run: RunRecord) -> bool:
    """The separator's vocals back in place, the record dropped; every reset of the stems or
    of this stage goes through here (after the voice stage's own, `song_voice.as_sung`). The
    dereverb's credit leaves with the item's (`song_credits.drop`)."""
    vocals_layers.drop(store, run, RECORD)
    return vocals_layers.put_back(store, run, RAW)


def _raw(store: RunStore, run: RunRecord) -> Path:
    kept = vocals_layers.keep(store, run, RAW)
    if kept is None:
        raise PipelineError(vocals_layers.no_stem(run, "clean"))
    return kept


def _engine_pass(
    recipe: Recipe, http: httpx.Client, house: HouseFormat, folder: Path, run: RunRecord
) -> None:
    """The work file through a separator (its rate and channels), its `vocals` kept."""
    kit = kit_of(MusicFamily(recipe.voice_args.family), recipe.files[0])
    job = Conversion(
        source=WORK, target=ENGINE_IN, codec=Codec.WAV16, rate=kit.sample_rate or 44_100,
        channels=kit.channels,
    )  # fmt: skip
    try:
        wav = house.convert(folder, job).read_bytes()
    finally:
        (folder / ENGINE_IN).unlink(missing_ok=True)
    raw = post_task(
        recipe, http, audio_request(recipe, wav).body(), audio_seconds(folder / WORK, run)
    )
    try:
        outputs = dict(parse_answer(raw).files())
    except MalformedAnswerError as exc:
        raise StepError(f"{recipe.name}: {exc}.") from exc
    if "vocals" not in outputs:
        raise StepError(f"{recipe.name} gave {', '.join(outputs) or 'nothing'}, not vocals.")
    (folder / ENGINE_IN).write_bytes(outputs["vocals"])
    try:  # back as the work file in the house format (a mono answer to both channels)
        house.convert(folder, Conversion(source=ENGINE_IN, target=WORK))
    finally:
        (folder / ENGINE_IN).unlink(missing_ok=True)


def _chain(
    wanted: list[Clean], house: HouseFormat, folder: Path
) -> tuple[list[str], dict[Clean, str]]:
    """ffmpeg's filters for what was asked, at the stem's own noise floor."""
    if not wanted:
        return [], {}
    floor = noise_floor(house, folder, WORK)
    filters: list[str] = []
    notes: dict[Clean, str] = {}
    if Clean.DENOISE in wanted:
        filters.append(f"afftdn=nr=20:nf={floor:.0f}")
        notes[Clean.DENOISE] = f"its floor read at {floor:.0f} dB"
    if Clean.DEESS in wanted:
        filters.append(DEESS)
    if Clean.GATE in wanted:
        threshold = 10 ** ((floor + GATE_ABOVE) / 20)
        filters.append(f"agate=threshold={threshold:.6f}:ratio=2:attack=10:release=250:range=0.125")
        notes[Clean.GATE] = f"opening {GATE_ABOVE:g} dB above the floor"
    return filters, notes


def noise_floor(house: HouseFormat, folder: Path, name: str) -> float:
    """The quiet tenth of the stem's slices, in dB, within what `afftdn` takes."""
    try:
        measured = peaks.measure(house, folder, [name]).get(name)
        levels = measured.rms if measured is not None else []
    finally:  # a measure of a working file is no waveform to keep
        (folder / peaks.FOLDER / f"{name}.json").unlink(missing_ok=True)
    # the waveform's levels are linear, 0 to 1 (`peaks._level`); digital silence says nothing
    audible = sorted(20 * math.log10(v) for v in levels if math.isfinite(v) and v > 0)
    if not audible:
        return FLOOR_RANGE[0]
    found = audible[int(len(audible) * FLOOR_PERCENTILE)]
    return min(max(found, FLOOR_RANGE[0]), FLOOR_RANGE[1])
