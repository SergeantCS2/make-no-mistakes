"""What a Song step does: ask the desk's `chat` for lyrics as JSON, make one candidate
(a child run) through the song instance's task route, split the picked candidate into stems
through the separators, every file in the house format (`audio_format`)."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from makenomistakes.audio_format import Codec, Conversion, HouseFormat, HouseFormatError
from makenomistakes.consumers import own_origin, router_origin
from makenomistakes.engine_answers import MalformedAnswerError
from makenomistakes.hub.converted import kit_of
from makenomistakes.hub.music import MusicFamily, MusicKit
from makenomistakes.make import credit, settings_used
from makenomistakes.media_bytes import wav_file_seconds
from makenomistakes.music_api import TASK_PATH, OptionValue, parse_answer, song_request
from makenomistakes.music_api import split_request as audio_request
from makenomistakes.music_speed import music_timeout
from makenomistakes.pipeline import song
from makenomistakes.pipeline.song import Lyrics, SongError, SongInputs
from makenomistakes.pipeline.steps import StepError, chat_json
from makenomistakes.recipes import Recipe
from makenomistakes.routing import UPSTREAM
from makenomistakes.run_parts import InputRecord
from makenomistakes.runs import RunKind, RunRecord, RunStore
from makenomistakes.smoke import TryError, answered
from makenomistakes.studio import style_use
from makenomistakes.studio.stage_instances import Player
from makenomistakes.studio.stages import StudioError
from makenomistakes.studio.style_library import StyleRow

CANDIDATE_FILE = "song.wav"
ENGINE_INPUT = ".engine-input.wav"  # what an engine is sent, made in the run's folder, removed


def ask_lyrics(chat: Recipe, http: httpx.Client, inputs: SongInputs) -> Lyrics:
    """The chat model's lyrics; one more try, told what was wrong, before giving up. Through
    the desk's `chat` role (`/upstream/chat`) when the run names the role, so the desk's
    choice at that moment answers."""
    origin = (
        f"{router_origin('127.0.0.1')}{UPSTREAM}{song.DESK_CHAT}"
        if inputs.chat == song.DESK_CHAT
        else own_origin(chat)
    )
    schema = ("lyrics", song.LYRICS_SCHEMA)
    messages = [{"role": "user", "content": song.lyrics_prompt(inputs)}]
    answer = chat_json(chat, http, messages, schema, origin)
    try:
        return _with_tags(song.parse_lyrics(answer), inputs)
    except SongError as exc:
        messages += [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": f"That is not usable: {exc}. Answer with the JSON only."},
        ]
    answer = chat_json(chat, http, messages, schema, origin)
    try:
        return _with_tags(song.parse_lyrics(answer), inputs)
    except SongError as exc:
        raise StepError(f"{chat.name}'s lyrics are not usable twice over: {exc}.") from exc


def _with_tags(lyrics: Lyrics, inputs: SongInputs) -> Lyrics:
    """The user's own style line wins over the chat model's."""
    return lyrics.model_copy(update={"tags": inputs.tags}) if inputs.tags else lyrics


@dataclass(frozen=True)
class Recording:
    """The recording the user gave, as kept in the run's folder, and how it is used."""

    name: str
    path: Path
    use: str  # complete, vocal2bgm, reference
    sha256: str


def recording_options(use: str) -> dict[str, OptionValue]:
    """ACE-Step's route for a recording (`song.ROUTES`) and what the route needs."""
    options: dict[str, OptionValue] = {"route": song.ROUTES[use]}
    if use == "vocal2bgm":
        options["complete_track_classes"] = json.dumps(song.ACCOMPANIMENT)
    if use == "reference":
        options["audio_cover_strength"] = song.REFERENCE_STRENGTH
    return options


def engine_wav(
    house: HouseFormat, folder: Path, source: Path, rate: int, channels: int = 2
) -> bytes:
    """`source` as an engine reads it (16-bit at its rate and channels), made beside it and
    read back."""
    relative = source.relative_to(folder).as_posix()
    job = Conversion(
        source=relative, target=ENGINE_INPUT, codec=Codec.WAV16, rate=rate, channels=channels
    )
    try:
        return house.convert(folder, job).read_bytes()
    finally:
        (folder / ENGINE_INPUT).unlink(missing_ok=True)


def make_candidate(  # noqa: PLR0913 - the candidate's parts beside the seams
    store: RunStore,
    parent: RunRecord,
    recipe: Recipe,
    http: httpx.Client,
    house: HouseFormat,
    *,
    lyrics: Lyrics,
    inputs: SongInputs,
    recording: Recording | None,
    made: Callable[[RunRecord], None] = lambda _child: None,
    style: StyleRow | None = None,
) -> RunRecord:
    """One candidate: a child run of `parent` holding the song, in the house format. A
    failure is kept as a failed child and raised as a StepError; `made` is told the child
    as soon as it exists (the step names it, so a retry replaces it). An instance with a
    style sings with its trigger word in front of the style line, credited."""
    seed = secrets.randbelow(2**32)
    used = settings_used(recipe) | {"lyrics": lyrics.sung(), "seed": str(seed)}
    used["seconds"] = f"{inputs.seconds:g}"
    tags = style_use.styled(lyrics.tags, style)
    child = store.new(RunKind.INSTANCE, recipe.name, tags, used, parent=parent.id, role="candidate")
    child.models = credit(recipe)
    if style is not None:
        child.models.append(style_use.credit(style))
    made(child)
    request = song_request(recipe, tags, lyrics.sung(), inputs.seconds, seed)
    if recording is not None:
        child.inputs_recorded.append(
            InputRecord(
                role="recording",
                name=recording.name,
                sha256=recording.sha256,
                source=f"run:{parent.id}",
                stage=song.SONG,
            )
        )
    store.save(child)
    try:
        if recording is not None:  # a file ffmpeg cannot read fails the child, never orphans it
            wav = engine_wav(house, store.folder(parent), recording.path, song_rate(recipe))
            request.request.audio_base64 = base64.b64encode(wav).decode("ascii")
            request.request.options.update(recording_options(recording.use))
        path, answer, took = _sung(store, child, recipe, http, house, request.body(), inputs)
    except (StepError, TryError, HouseFormatError, OSError) as exc:
        store.fail(child, str(exc))
        raise StepError(str(exc)) from exc
    except BaseException as exc:
        store.fail(child, str(exc) or f"Interrupted ({type(exc).__name__}).")
        raise
    return store.finish(child, [path], answer, took)


def song_rate(recipe: Recipe) -> int:
    return _kit(recipe).sample_rate or 48_000


def _kit(recipe: Recipe) -> MusicKit:
    """The instance's own kit: a converted checkpoint's, else its family's."""
    return kit_of(MusicFamily(recipe.voice_args.family), recipe.files[0])


def post_task(recipe: Recipe, http: httpx.Client, body: dict[str, object], seconds: float) -> bytes:
    timeout = music_timeout(recipe, seconds)
    try:
        response = http.post(own_origin(recipe) + TASK_PATH, json=body, timeout=timeout)
    except httpx.TimeoutException as exc:  # never "is the router up?" for this
        raise StepError(
            f"{recipe.name} gave no answer within {timeout / 60:.0f} minutes, the most this "
            f"box waits for {seconds:g} s of audio ({exc.__class__.__name__}); it may still be "
            f"working: `mnm logs {recipe.name}` shows it, and resume tries again."
        ) from exc
    except httpx.TransportError as exc:
        raise StepError(
            f"{recipe.name} did not answer ({exc.__class__.__name__}): is the router up "
            "(`mnm service status`)?"
        ) from exc
    try:
        return answered(recipe, response)
    except TryError as exc:  # the router's or the engine's refusal: the step's error, worded
        raise StepError(str(exc)) from exc


def _sung(
    store: RunStore,
    child: RunRecord,
    recipe: Recipe,
    http: httpx.Client,
    house: HouseFormat,
    body: dict[str, object],
    inputs: SongInputs,
) -> tuple[Path, str, float | None]:
    raw = post_task(recipe, http, body, inputs.seconds)
    try:
        answer = parse_answer(raw)
        files = answer.files()
    except MalformedAnswerError as exc:
        raise StepError(f"{recipe.name}: {exc}.") from exc
    folder = store.folder(child)
    engine_file = folder / ".engine.wav"
    engine_file.write_bytes(files[0][1])
    try:
        path = house.convert(folder, Conversion(source=engine_file.name, target=CANDIDATE_FILE))
    finally:
        engine_file.unlink(missing_ok=True)
    took = answer.seconds
    return path, f"{inputs.seconds:g} s song ({took or 0:.0f} s of work)", took


def split(
    store: RunStore,
    run: RunRecord,
    player: Player,
    source: Path,
    http: httpx.Client,
    house: HouseFormat,
) -> list[Path]:
    """The stems the player gives (`Player.stems`: its package's, or every one a pick of
    another family gives), from `source` (the take in use), each in the house format in the
    run's folder."""
    recipe = player.recipe
    folder = store.folder(run)
    wav = engine_wav(house, source.parent, source, song_rate(recipe), _kit(recipe).channels)
    raw = post_task(recipe, http, audio_request(recipe, wav).body(), audio_seconds(source, run))
    try:
        stems = dict(parse_answer(raw).files())
    except MalformedAnswerError as exc:
        raise StepError(f"{recipe.name}: {exc}.") from exc
    try:
        kept = player.stems(list(stems))
    except StudioError as exc:
        raise StepError(str(exc)) from exc
    if not kept:
        raise StepError(f"{recipe.name} gave no stems.")
    made: list[Path] = []
    for name in kept:
        engine_file = folder / f".engine-{name}.wav"
        engine_file.write_bytes(stems[name])
        try:
            job = Conversion(source=engine_file.name, target=song.stem_file(name))
            made.append(house.convert(folder, job))
        finally:
            engine_file.unlink(missing_ok=True)
    return made


def audio_seconds(path: Path, run: RunRecord) -> float:
    """How long the file sent to an engine is, which the wait on its answer follows (an
    Extend makes the take longer than the song's length setting); the song's length where the
    file's header cannot say."""
    return wav_file_seconds(path) or float(run.settings.get("seconds", song.DEFAULT_SECONDS))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
