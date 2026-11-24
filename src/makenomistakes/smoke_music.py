"""`mnm try` and `mnm run` on the studio's types: one request to audio.cpp's task route
through the router, what came back saved where the command runs (a try) or in the run's folder
(a run).

A song instance makes a clip from tags and lyrics (`mnm try`: the card's tags at 10 s; a Run:
the user's tags and lyrics at the instance's own length). A separator splits a mix (`--audio`,
or a made-up one) into its stems; a converter or a scorer takes the clip it is given. The
input goes inline (`music_api`) at the rate the model reads (the house format's
ffmpeg, `audio_format.conform_bytes`, which falls back to `wav_rate` without Docker)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import httpx

from makenomistakes import audio_format
from makenomistakes.engine_answers import MalformedAnswerError
from makenomistakes.harness_media import SONG_LYRICS, SONG_SECONDS, SONG_TAGS
from makenomistakes.hub.converted import kit_of
from makenomistakes.hub.facts import Modality
from makenomistakes.hub.music import MusicFamily
from makenomistakes.media_bytes import wav_seconds
from makenomistakes.music_api import TASK_PATH, TaskAnswer, parse_answer, song_request
from makenomistakes.music_api import split_request as audio_request
from makenomistakes.music_speed import music_timeout, own_length
from makenomistakes.recipes.model import Recipe
from makenomistakes.smoke_replies import TryError, TryResult
from makenomistakes.try_figures import Figure
from makenomistakes.wav_rate import WavError, sample_mix

AnswerCheck = Callable[[Recipe, httpx.Response], bytes]


def music_try(  # noqa: PLR0913, PLR0917 - the request's parts beside the seams
    recipe: Recipe,
    http: httpx.Client,
    root: str,
    out_dir: Path,
    answered: AnswerCheck,
    audio: Path | None = None,
    prompt: str | None = None,
    lyrics: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TryResult:
    """One task, timed; `answered` is smoke's status check (a refusal worded with the log
    rules). A Run gives `prompt` (and `lyrics`); a try sends the card's words at 10 s."""
    family = MusicFamily(recipe.voice_args.family)
    if recipe.modality is Modality.SONG:
        if audio is not None:
            raise TryError(f"{recipe.name} makes a song from words: --audio is for a separator.")
        seconds = SONG_SECONDS if prompt is None else own_length(recipe) or SONG_SECONDS
        words = (prompt or SONG_TAGS).strip()
        request = song_request(
            recipe, words, lyrics if lyrics is not None else (SONG_LYRICS if prompt is None else "")
        )
        if prompt is None:
            request.request.options["duration_sec"] = seconds
    elif recipe.modality is Modality.VOICE_CONVERT:  # A voice from the library
        raise TryError(
            f"{recipe.name} sings a stem in a library voice: `mnm voices add <name> --reference "
            f"voice.wav`, then `mnm song new … --voice <name>` (or `mnm song voice <run> <name>` "
            "on a song made). Its card's curl line sends a clip by hand."
        )
    else:
        if prompt is not None and recipe.modality is not Modality.STEMS:
            raise TryError(f"{recipe.name} works on a clip, not on words: give it --audio.")
        wav = _input(recipe, family, audio)
        seconds = wav_seconds(wav) or SONG_SECONDS
        request = audio_request(recipe, wav)
    started = clock()
    response = http.post(
        root + TASK_PATH,
        json=request.body(),
        timeout=music_timeout(recipe, seconds),
    )
    raw = answered(recipe, response)
    took = clock() - started
    try:
        answer = parse_answer(raw)
        files = answer.files()
    except MalformedAnswerError as exc:
        raise TryError(f"{recipe.name}: {exc}.") from exc
    return _saved(recipe, answer, files, took, out_dir)


def _input(recipe: Recipe, family: MusicFamily, audio: Path | None) -> bytes:
    kit = kit_of(family, recipe.files[0])
    if audio is None:
        if recipe.modality is not Modality.STEMS:
            raise TryError(
                f"{recipe.name} works on a clip: `mnm try {recipe.name} --audio clip.wav`."
            )
        return sample_mix(rate=kit.sample_rate or 44_100)
    try:
        data = audio.read_bytes()
    except OSError as exc:
        raise TryError(f"Cannot read the clip {audio}: {exc.strerror or exc}.") from exc
    try:
        if not kit.sample_rate:
            return data
        return audio_format.default_house().conform_bytes(data, kit.sample_rate, kit.channels)
    except WavError as exc:
        raise TryError(f"{audio.name}: {exc}.") from exc


def _saved(
    recipe: Recipe,
    answer: TaskAnswer,
    files: list[tuple[str, bytes]],
    took: float,
    out_dir: Path,
) -> TryResult:
    paths: list[Path] = []
    for name, data in files:
        out = out_dir / (f"{recipe.name}.wav" if name == "song" else f"{recipe.name}-{name}.wav")
        out.unlink(missing_ok=True)  # never write through a link left under this name
        out.write_bytes(data)
        paths.append(out)
    paths += _midi(recipe, answer, out_dir)
    length = wav_seconds(files[0][1]) or 0.0
    work = answer.seconds or took
    rate = length / work if work > 0 else 0.0
    numbers = {Figure.AUDIO_SECONDS: length, Figure.REALTIME: rate}
    size = sum(p.stat().st_size for p in paths) / 2**20
    if recipe.modality is Modality.STEMS:
        names = ", ".join(name for name, _ in files)
        line = f"{len(files)} stems of {length:.1f} s ({names}; {size:.1f} MiB) in {work:.1f} s"
    else:
        line = f"{length:.1f} s of audio ({size:.1f} MiB, {rate:.2f}x realtime)"
    where = paths[0] if len(paths) == 1 else out_dir
    return TryResult(f"{line}: {where}", took, numbers, saved=paths[0], extra=tuple(paths[1:]))


def _midi(recipe: Recipe, answer: TaskAnswer, out_dir: Path) -> list[Path]:
    """A scorer's MIDI artifacts, written as `.mid` (MuScriptor, SheetSage2)."""
    made: list[Path] = []
    for artifact in answer.artifacts:
        data = artifact.midi()
        if data is None:
            continue
        out = out_dir / f"{recipe.name}-{len(made) + 1}.mid"
        out.unlink(missing_ok=True)
        out.write_bytes(data)
        made.append(out)
    return made
