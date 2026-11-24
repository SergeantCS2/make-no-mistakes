"""The voice stage: the picked take's vocals sung again in a library voice.

The separators give `stem-vocals.wav`; the voice stage keeps it as `vocals-as-sung.wav` and
writes the converted vocals in its place, so the mix, the mixer and the export use them as
any stem. Beside them `voice.json` records what sang them: the voice as it was then (its
files by sha256, whose voice it is), the converter, the shift and the similarity reading, so
an export says what sang even after the library's row changed. A song with no voice keeps
the song model's own voice (the stage is skipped); choosing none later, new stems or a new
pick put the vocals as sung back first (`as_sung`), so a conversion is never converted again.

The converter is the instance whose family is the voice's engine and that sings (a Seed-VC
added for speech never does): the user's pick for the `voice` stage first, then the
registry's package for the tier, then any. SoulX-Singer voices run through ComfyUI, which
this stage does not take yet.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx

from makenomistakes.audio_format import Codec, Conversion, HouseFormat
from makenomistakes.engine_answers import MalformedAnswerError
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.engine_args_music import ConvertOptions
from makenomistakes.instances import InstanceService
from makenomistakes.music_api import ConverterVoice, convert_request, parse_answer, voice_path
from makenomistakes.pipeline import song, vocals_layers
from makenomistakes.pipeline.song_steps import audio_seconds, post_task
from makenomistakes.pipeline.steps import PipelineError, StepError
from makenomistakes.recipes.model import Recipe
from makenomistakes.runs import RunRecord, RunStore
from makenomistakes.studio import stages
from makenomistakes.studio.stage_instances import is_package
from makenomistakes.studio.tools import StudioTools
from makenomistakes.studio.tools_answers import ToolsError, ToolsMissingError
from makenomistakes.studio.voice_library import (
    ENGINE_WORDS,
    TAKE_FILE,
    VoiceEngine,
    VoiceError,
    VoiceLibrary,
    VoiceRow,
    VoiceTake,
)

VOICE = song.VOICE  # the stage, and the run's setting naming the library voice
SHIFT = "voice_shift"  # the run's setting: semitones the voice moves the vocals by
VOCALS = song.stem_file("vocals")
AS_SUNG = vocals_layers.AS_SUNG  # its input (the cleaned vocals, or the separator's), kept
TAKE = TAKE_FILE  # what sang the vocals (`VoiceTake`), beside them
REFERENCE_COPY = "voice-reference"  # the reference beside the stems while the tools read it
CONVERTER_RATE = 44_100  # what the stem is sent at (Seed-VC and RVC resample it inside)
# where `mnm add` finds the engines the registry's `voice` row does not name
PACKAGES = {
    VoiceEngine.RVC: "audio-cpp/audio.cpp-gguf/RVC-GGUF",
    VoiceEngine.VEVO2: "audio-cpp/audio.cpp-gguf/Vevo2-GGUF",
}
SINGING_TASK = {VoiceEngine.SEED_VC: "svc"}  # a converter of these sings only on this task


def voice_of(library: VoiceLibrary, settings: dict[str, str]) -> VoiceRow | None:
    """The library voice a run names, or None (the song model's own voice)."""
    name = settings.get(VOICE, "")
    if not name:
        return None
    try:
        return library.get(name)
    except VoiceError as exc:
        raise PipelineError(str(exc)) from exc


def _family(recipe: Recipe) -> str | None:
    return recipe.args.family if isinstance(recipe.args, AudioCppArgs) else None


def _sings(recipe: Recipe, engine: VoiceEngine) -> bool:
    wanted = SINGING_TASK.get(engine)
    return wanted is None or (isinstance(recipe.args, AudioCppArgs) and recipe.args.task == wanted)


def converter(service: InstanceService, state_dir: Path, voice: VoiceRow, tier: str) -> Recipe:
    """The instance that sings in `voice`: of the voice's engine and singing, the user's
    `voice` pick first, then the registry's package for the tier, then any; a refusal naming
    the `mnm add` (or what to do with one added for speech) when none."""
    if voice.engine is VoiceEngine.SOULX:
        raise PipelineError(
            f"{voice.name} is a SoulX-Singer voice, which runs through a ComfyUI workflow: the "
            "voice stage takes audio.cpp's converters only so far (Seed-VC, RVC, Vevo2)."
        )
    ours = [r for r in service.all() if _family(r) == voice.engine.value]
    singing = [r for r in ours if _sings(r, voice.engine)]
    try:
        resolved = stages.resolve(VOICE, "full" if tier == "full" else "fast", state_dir)
    except stages.StudioError as exc:  # the picks file unreadable: worded, never a 500
        raise PipelineError(str(exc)) from exc
    chosen = (
        next((r for r in singing if r.name == resolved.instance), None)
        or next((r for r in singing for p in resolved.packages if is_package(r, p)), None)
        or (singing[0] if singing else None)
    )
    if chosen is not None:
        return chosen
    if ours:  # added before the voice stage, for speech
        task = ours[0].args.task if isinstance(ours[0].args, AudioCppArgs) else "?"
        raise PipelineError(
            f"{ours[0].name} was added for speech (task {task}) before the studio's voice "
            f"stage: `mnm rm {ours[0].name}`, then `mnm add {_package(voice.engine, resolved)}` "
            "adds it for singing (its files stay downloaded)."
        )
    raise PipelineError(
        f"No {ENGINE_WORDS[voice.engine]} converter is added for the voice {voice.name}: "
        f"`mnm add {_package(voice.engine, resolved)}`."
    )


def _package(engine: VoiceEngine, resolved: stages.Resolved) -> str:
    if engine is VoiceEngine.SEED_VC and resolved.packages:
        return resolved.packages[0].reference  # the registry's row names Seed-VC's
    return PACKAGES[engine]


def settings_with(
    settings: dict[str, str],
    voice: str | None,
    shift: int,
    library: VoiceLibrary,
    service: InstanceService,
    state_dir: Path,
) -> dict[str, str]:
    """A run's settings with this voice and shift, checked: the shift's bounds, the voice in
    the library, its converter added; an empty voice is the song model's own."""
    changed = {**settings, VOICE: voice or "", SHIFT: str(shift)}
    try:
        inputs = song.SongInputs.from_settings(changed)
    except ValueError as exc:
        raise PipelineError(f"The voice's shift: {exc}") from exc
    found = voice_of(library, changed)
    if found is not None:
        converter(service, state_dir, found, inputs.tier)
    return {k: v for k, v in changed.items() if v != ""}


def convert(  # noqa: PLR0913 - the run, the voice and its converter beside the seams
    store: RunStore,
    run: RunRecord,
    voice: VoiceRow,
    recipe: Recipe,
    http: httpx.Client,
    house: HouseFormat,
    *,
    shift: int,
    library: VoiceLibrary,
) -> VoiceTake:
    """The vocals as sung, converted in `voice` and written as the vocals stem; what sang
    them kept as `voice.json`. The run's shift wins; without one the instance's own."""
    try:  # the converter reads these files, and the take credits the row's hashes of them
        library.check_files(voice)
    except VoiceError as exc:
        raise PipelineError(str(exc)) from exc
    folder = store.folder(run)
    sung = _as_sung(store, run)
    wav = _engine_input(house, folder, sung)
    own = recipe.args.options if isinstance(recipe.args, AudioCppArgs) else None
    options = own if isinstance(own, ConvertOptions) else ConvertOptions()
    moved = shift or options.semitone_shift
    try:
        body = convert_request(
            recipe, wav, _paths(voice), semitone_shift=moved,
            retrieval_blend=options.retrieval_blend,
        ).body()  # fmt: skip
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    raw = post_task(recipe, http, body, audio_seconds(sung, run))
    try:
        files = parse_answer(raw).files()
    except MalformedAnswerError as exc:
        raise StepError(f"{recipe.name}: {exc}.") from exc
    if not files:
        raise StepError(f"{recipe.name} answered with no audio.")
    engine_file = folder / ".engine-voice.wav"
    engine_file.write_bytes(files[0][1])
    try:
        made = house.convert(folder, Conversion(source=engine_file.name, target=VOCALS))
    finally:
        engine_file.unlink(missing_ok=True)
    store.keep(run, made)
    run.voices = [voice.name]
    take = VoiceTake(  # the row's hashes are the files' own: `check_files` read them
        voice=voice.name, engine=voice.engine, source=voice.source, files=voice.files,
        converter=recipe.name, shift=moved,
    )  # fmt: skip
    _write_take(store, run, take)
    return take


def words(take: VoiceTake) -> str:
    """The step's line: `the vocals in alto (Seed-VC, -2 semitones); 0.45 alike …`."""
    moved = f", {take.shift:+d} semitones" if take.shift else ""
    line = f"the vocals in {take.voice} ({ENGINE_WORDS[take.engine]}{moved})"
    now, before = take.similarity.get("vocals"), take.similarity.get("as_sung")
    if now is not None and before is not None:
        line += f"; {now:.2f} alike to {take.voice}'s reference (as sung {before:.2f})"
    return line


def _paths(voice: VoiceRow) -> ConverterVoice:
    found = {f.role: voice_path(voice.name, f.name) for f in voice.files}
    return ConverterVoice(
        model=found.get("model"), index=found.get("index"), reference=found.get("reference")
    )


def as_sung(store: RunStore, run: RunRecord) -> bool:
    """The song model's own voice back: the vocals as sung in place, the take forgotten.
    Every reset of the stems or the take goes through here, so the vocals as sung are never
    taken for a conversion's output."""
    run.voices = []
    vocals_layers.drop(store, run, TAKE)
    return vocals_layers.put_back(store, run, AS_SUNG)


def no_tools_note(run_id: str) -> str:
    return (
        "No similarity reading for the voice: the studio tools are not installed. After "
        f"`mnm studio install`, `mnm song redo {run_id} voice` sings it again and reads it."
    )


def similarity(
    tools: StudioTools,
    store: RunStore,
    run: RunRecord,
    voice: VoiceRow,
    library: VoiceLibrary,
    take: VoiceTake,
) -> str:
    """How alike the vocals now sound to the voice's reference, beside the vocals as sung,
    kept on the take; "" or why there is none. A reading never fails the voice: an RVC
    voice has no recording to compare with, and without the tools the run says so once."""
    reference = voice.file("reference")
    if reference is None:
        return ""
    folder = store.folder(run)
    copy = folder / f"{REFERENCE_COPY}{Path(reference.name).suffix}"
    try:
        shutil.copyfile(library.folder(voice.name) / reference.name, copy)
        answer = tools.similarity(folder, copy.name, VOCALS, AS_SUNG)
    except ToolsMissingError:
        if no_tools_note(run.id) not in run.notes:
            run.notes.append(no_tools_note(run.id))
        return ""
    except (ToolsError, OSError) as exc:
        return f"no similarity reading ({exc})"
    finally:
        copy.unlink(missing_ok=True)
    run.notes = [n for n in run.notes if n != no_tools_note(run.id)]
    vocals, sung = answer.similarity.get(VOCALS), answer.similarity.get(AS_SUNG)
    if vocals is None or sung is None:
        return "no similarity reading (the tools named other files)"
    # + 0.0: a reading just under zero reads 0.00, never -0.00
    take.similarity = {"vocals": round(vocals, 2) + 0.0, "as_sung": round(sung, 2) + 0.0}
    take.reader = answer.model
    _write_take(store, run, take)
    return ""


def _write_take(store: RunStore, run: RunRecord, take: VoiceTake) -> None:
    path = store.folder(run) / TAKE
    path.write_text(take.model_dump_json(indent=2) + "\n", encoding="utf-8")
    store.keep(run, path)


def _as_sung(store: RunStore, run: RunRecord) -> Path:
    """The voice stage's input, kept once beside the stem the conversion replaces."""
    kept = vocals_layers.keep(store, run, AS_SUNG)
    if kept is None:
        raise PipelineError(vocals_layers.no_stem(run, "sing in another voice"))
    return kept


def _engine_input(house: HouseFormat, folder: Path, source: Path) -> bytes:
    """The stem as the converters read it: 16-bit mono at 44.1 kHz."""
    target = ".voice-input.wav"
    job = Conversion(
        source=source.name, target=target, codec=Codec.WAV16, rate=CONVERTER_RATE, channels=1
    )
    try:
        return house.convert(folder, job).read_bytes()
    finally:
        (folder / target).unlink(missing_ok=True)
