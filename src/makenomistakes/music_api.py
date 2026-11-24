"""audio.cpp's task route for the studio: `POST /v1/tasks/run`, its request and its
answer typed at the edge.

What the `full-cpu` image answered on 2026-10-31 (from `app/server/api.md` and the engine
itself):

- The body is `{"model": <the config's id>, "request": {...}}`; the request is one CLI
  request: `text` (a song's tags or caption), `lyrics`, `seed`, an input WAV as `audio` (a
  path inside the container) or `audio_base64` (the WAV itself, inline), and `options` (the
  family's request options: ACE-Step's `route`, `duration_sec`, `num_inference_steps` …).
  `artifacts[]` carries embeddings, tokens and MIDI between requests, not the audio. mnm
  sends inputs inline (`audio_base64`, within the server's 2 GiB body limit), so music
  containers see only the models folder, as every engine's does.
- A song comes back as one `audio` (a base64 WAV) with `sample_rate`, `channels` and
  `timing` (`wall_ms`, `audio_duration_ms`, `rtf`); stems as `named_audio_outputs`, one
  `{id, audio, sample_rate, channels}` each (`vocals` and `instrumental` from BS-RoFormer;
  drums, bass, other, vocals, guitar, piano from HTDemucs 6 stems).
- A refusal is `{"error": {"message", "type"}}`: HTTP 500 for an unknown model id or an
  unreadable path, 400 for a body that is not a WAV.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from makenomistakes.engine_answers import ErrorAnswer, MalformedAnswerError, parse, said_error
from makenomistakes.media_bytes import wav_header
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR, Recipe

TASK_ROUTE = "/tasks/run"
TASK_PATH = f"/v1{TASK_ROUTE}"
OptionValue = str | int | float | bool


class TaskInput(BaseModel):
    """The `request` object: what this one task runs on."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    lyrics: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    audio_base64: str | None = None
    # a target voice's recording, as a path inside the container (the generic task route takes
    # no inline reference: "json value is not a string", `error-voice-ref-object.json`)
    voice_ref: str | None = None
    options: dict[str, OptionValue] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    request: TaskInput

    def body(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TaskTiming(_Loose):
    wall_ms: float | None = None
    audio_duration_ms: float | None = None
    rtf: float | None = None


class NamedAudio(_Loose):
    # a stem's name, which names its file (`stem-<id>.wav`, `mnm try`'s `<name>-<id>.wav`):
    # a word, never a path
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    audio: str
    sample_rate: int | None = None
    channels: int | None = None


class Artifact(_Loose):
    """One `artifacts[]` entry: what a task hands back beside its audio (MuScriptor's MIDI as
    a base64 standard MIDI file, `kind: midi`; embeddings and tokens otherwise)."""

    id: str | None = None
    kind: str = ""
    payload: str | None = None

    def midi(self) -> bytes | None:
        """The MIDI file it carries, decoded and checked; None when it carries no MIDI, a
        `MalformedAnswerError` when it says MIDI and is not one."""
        if self.kind != "midi" or not self.payload:
            return None
        try:
            data = base64.b64decode(self.payload.split(",", 1)[-1], validate=True)
        except (binascii.Error, ValueError):
            raise MalformedAnswerError("the engine's MIDI artifact is not base64") from None
        if not data.startswith(b"MThd"):
            raise MalformedAnswerError("the engine's MIDI artifact is not a MIDI file")
        return data


class TaskAnswer(_Loose):
    """A run's result: one `audio`, or named stems, or both absent (a refusal reads as an
    `ErrorAnswer` first)."""

    audio: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    named_audio_outputs: list[NamedAudio] = Field(default_factory=list)
    timing: TaskTiming = TaskTiming()
    artifacts: list[Artifact] = Field(default_factory=list)

    def files(self) -> list[tuple[str, bytes]]:
        """Each WAV the answer carries, by name (`song`, or each stem's id), decoded and
        checked: a payload that is not a WAV is a `MalformedAnswerError`."""
        found = [("song", self.audio)] if self.audio else []
        found += [(stem.id, stem.audio) for stem in self.named_audio_outputs]
        if not found:
            raise MalformedAnswerError("the engine's task answer carried no audio")
        return [(name, _wav(name, payload)) for name, payload in found]

    @property
    def seconds(self) -> float | None:
        return None if self.timing.wall_ms is None else self.timing.wall_ms / 1000


def is_task(path: str) -> bool:
    """A request to the task route (`/v1/tasks/run`, under `/upstream/<name>` too)."""
    return path.split("?", 1)[0].rstrip("/").endswith(TASK_ROUTE)


class _AskedInput(_Loose):
    audio_base64: str | None = None
    options: dict[str, OptionValue] = Field(default_factory=dict)


class _AskedTask(_Loose):
    request: _AskedInput


@dataclass(frozen=True)
class Asked:
    """What a task request asks of the engine, as far as its wait goes (None: not said)."""

    seconds: float | None = None  # the audio asked for (`duration_sec`) or sent inline
    steps: int | None = None  # ACE-Step's `num_inference_steps`


HEADER_CHARS = 88 * 1024  # base64 of a WAV's first 64 KiB: its format and data chunks


def asked(body: bytes) -> Asked:
    """`Asked` of a request's body, read leniently: a body that does not parse, or an option
    that is not a number, says nothing (the engine answers the request itself). An inline
    WAV's length is read from its header and the base64's length, never decoded whole (a
    3-minute mix is about 45 MB of it)."""
    try:
        found = _AskedTask.model_validate_json(body).request
    except ValueError:
        return Asked()
    seconds = _number(found.options.get("duration_sec"))
    if (seconds is None or seconds <= 0) and found.audio_base64:
        seconds = _inline_seconds(found.audio_base64.split(",", 1)[-1])
    steps = _number(found.options.get("num_inference_steps"))
    return Asked(
        seconds=seconds if seconds is not None and seconds > 0 else None,
        steps=int(steps) if steps is not None and steps >= 1 else None,
    )


def _number(value: OptionValue | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _inline_seconds(payload: str) -> float | None:
    try:
        head = base64.b64decode(payload[: HEADER_CHARS - HEADER_CHARS % 4], validate=True)
    except (binascii.Error, ValueError):
        return None
    header = wav_header(head)
    if header is None or not header.byte_rate:
        return None
    whole = len(payload) * 3 // 4 - payload[-2:].count("=")
    held = min(header.data_bytes, whole - header.data_start)
    return held / header.byte_rate if held > 0 else None


def _wav(name: str, payload: str) -> bytes:
    try:
        data = base64.b64decode(payload.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise MalformedAnswerError(f"the engine's {name} audio is not base64") from None
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise MalformedAnswerError(f"the engine's {name} audio is not a WAV file")
    return data


def parse_answer(raw: bytes | str) -> TaskAnswer:
    return parse(TaskAnswer, raw, "task answer")


def refusal(raw: bytes | str) -> str:
    """The engine's own words for a refused task, or the start of what it sent."""
    try:
        said = said_error(parse(ErrorAnswer, raw, "error").error)
    except MalformedAnswerError:
        said = ""
    text = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
    return said or text.strip()[:200] or "no reason given"


def song_request(
    recipe: Recipe,
    tags: str,
    lyrics: str = "",
    seconds: float | None = None,
    seed: int | None = None,
) -> TaskRequest:
    """A song from tags and lyrics: ACE-Step's `text2music` (its default route) at the
    instance's own length unless `seconds` says another."""
    options: dict[str, OptionValue] = {}
    if seconds is not None:
        options["duration_sec"] = seconds
    return TaskRequest(
        model=recipe.served_name,
        request=TaskInput(text=tags, lyrics=lyrics, seed=seed, options=options),
    )


def split_request(recipe: Recipe, wav: bytes) -> TaskRequest:
    """A mix to split, sent inline: the container reads nothing outside the models folder."""
    encoded = base64.b64encode(wav).decode("ascii")
    return TaskRequest(model=recipe.served_name, request=TaskInput(audio_base64=encoded))


# Each reference-based converter's singing route (its `svc` task); RVC's voice is a checkpoint.
SINGING_ROUTES = {"seed_vc": "v1_svc", "vevo2": "style_preserved_svc"}
VOICE_FOLDER = "voice-library"  # under the models folder: where a converter reads a voice


def voice_path(voice: str, file: str) -> str:
    """A voice's file where the converter's container reads it (it mounts the models folder)."""
    return f"{CONTAINER_MODELS_DIR}/{VOICE_FOLDER}/{voice}/{file}"


class ConverterVoice(BaseModel):
    """A voice as a converter reads it: container paths, one kind per engine."""

    model: str | None = None  # RVC's checkpoint
    index: str | None = None  # RVC's retrieval index
    reference: str | None = None  # Seed-VC's and Vevo2's reference recording


def convert_request(
    recipe: Recipe,
    wav: bytes,
    voice: ConverterVoice,
    *,
    semitone_shift: int = 0,
    retrieval_blend: float = 0.0,
    seed: int | None = None,
) -> TaskRequest:
    """A sung stem in a voice: RVC with its checkpoint (its index and blend said whenever it
    has one, so the engine's own default never blends against nothing), Seed-VC and Vevo2 on
    their singing route with the reference. The stem goes inline."""
    encoded = base64.b64encode(wav).decode("ascii")
    family = getattr(recipe.args, "family", "")
    options: dict[str, OptionValue] = {"semitone_shift": semitone_shift}
    if family == "rvc":
        if voice.model is None:
            raise ValueError("an RVC voice is its checkpoint, and this one names none")
        options["voice_model_path"] = voice.model
        blend = retrieval_blend if voice.index is not None else 0.0
        options["retrieval_blend"] = blend
        if voice.index is not None and blend > 0:
            options["retrieval_index_file"] = voice.index
        return TaskRequest(
            model=recipe.served_name, request=TaskInput(audio_base64=encoded, options=options)
        )
    route = SINGING_ROUTES.get(str(family))
    if voice.reference is None or route is None:
        raise ValueError(f"{recipe.name} ({family or 'no family'}) sings no reference voice")
    options["route"] = route
    return TaskRequest(
        model=recipe.served_name,
        request=TaskInput(
            audio_base64=encoded, voice_ref=voice.reference, seed=seed, options=options
        ),
    )
