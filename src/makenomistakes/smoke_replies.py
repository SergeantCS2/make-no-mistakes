"""What `mnm try` and a Run read back: each engine's reply typed at
the edge, worded as one line for the terminal, and its file saved where the command runs."""

from __future__ import annotations

import base64
import binascii
import json
import struct
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.harness_media import image_size
from makenomistakes.hub.facts import Modality
from makenomistakes.media_bytes import PNG_SIGNATURE, wav_seconds
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import error_message
from makenomistakes.try_figures import Figure


class TryError(RuntimeError):
    """The instance did not answer, or answered something the card would not accept."""


class NoAnswerError(TryError):
    """Nothing answered, or the router found no engine behind it (HTTP 502): the
    container may have died during the request, which only Docker can tell."""


class EmptyTranscriptError(TryError):
    """A transcription came back well formed and empty (a silent clip reads as nothing)."""


@dataclass(frozen=True)
class TryResult:
    answer: str  # one line, already worded for the terminal
    seconds: float
    numbers: dict[Figure, float] = dataclass_field(default_factory=dict)  # tok/s, dims, scores
    saved: Path | None = None  # the image or clip written by an image or video instance
    extra: tuple[Path, ...] = ()  # What else it wrote (a separator's other stems)


ANSWER_LINE = 200  # a reply's start on the result line; a Run keeps the whole in a file
REFUSAL_LIMIT = 2000  # an error's message is read whole; a runaway one is still cut
BODY_LIMIT = 300  # a body that is not an error message: its start only


def refusal_text(body: str) -> str:
    """What a non-200 answer says: an error's message whole (the router's refusals end with
    what to do, such as "pinned … keep=auto", which a 300-character cut would lose),
    else the body's start."""
    message = error_message(body)
    return message[:REFUSAL_LIMIT] if message else body[:BODY_LIMIT]


class _JobError(BaseModel):
    code: str = ""
    message: str = ""


class Job(BaseModel):
    """sd-server's async job: queued by `POST /sdcpp/v1/vid_gen` (202), read at `poll_url`."""

    id: str = ""
    status: str = ""
    poll_url: str = ""
    result: _VideoResult | None = None
    error: _JobError | str | None = None


class _VideoResult(BaseModel):
    b64_json: str = ""
    mime_type: str = ""
    fps: int = 0
    frame_count: int = 0


class _Message(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None


class _Choice(BaseModel):
    message: _Message = Field(default_factory=_Message)


class _Timings(BaseModel):  # llama-server's own counters, beside the OpenAI fields
    prompt_n: float | None = None
    prompt_per_second: float | None = None
    predicted_n: float | None = None
    predicted_per_second: float | None = None
    # With a draft model: the tokens it drafted for this reply and those the model kept
    # (the same counts the engine's /metrics sums as `llamacpp:spec_decode_num_*_total`).
    draft_n: float | None = None
    draft_n_accepted: float | None = None


class _Usage(BaseModel):  # the OpenAI counters every server sends (vLLM has no timings)
    completion_tokens: int = 0


class ChatReply(BaseModel):
    choices: list[_Choice] = Field(default_factory=list)
    timings: _Timings | None = None
    usage: _Usage | None = None


class _Vector(BaseModel):
    embedding: list[float] = Field(default_factory=list)


class _EmbeddingReply(BaseModel):
    data: list[_Vector] = Field(default_factory=list)


class _Score(BaseModel):
    index: int
    relevance_score: float


class _RerankReply(BaseModel):
    results: list[_Score] = Field(default_factory=list)


class _Image(BaseModel):
    b64_json: str | None = None


class _ImageReply(BaseModel):
    data: list[_Image] = Field(default_factory=list)


class _Transcription(BaseModel):
    text: str = ""


def parse_reply[M: BaseModel](model: type[M], reply: object, recipe: Recipe) -> M:
    try:
        return model.model_validate(reply)
    except ValidationError as exc:
        raise TryError(f"{recipe.name} answered an unexpected shape: {str(reply)[:200]}") from exc


def summarise(
    recipe: Recipe, reply: object, seconds: float, out_dir: Path, speed: bool
) -> TryResult:
    match recipe.modality:
        case Modality.CHAT | Modality.CODING | Modality.VISION_CHAT:
            return _chat(recipe, parse_reply(ChatReply, reply, recipe), seconds, speed)
        case Modality.EMBEDDING:
            vectors = [v.embedding for v in parse_reply(_EmbeddingReply, reply, recipe).data]
            if not vectors or not all(vectors):
                raise TryError(f"{recipe.name} returned no vectors: {json.dumps(reply)[:200]}")
            dims = len(vectors[0])
            return TryResult(
                f"{len(vectors)} vectors of {dims} dimensions", seconds, {Figure.DIMS: dims}
            )
        case Modality.RERANK:
            scores = {
                r.index: r.relevance_score for r in parse_reply(_RerankReply, reply, recipe).results
            }
            if set(scores) != {0, 1}:
                raise TryError(f"{recipe.name} returned no scores: {json.dumps(reply)[:200]}")
            order = "above" if scores[0] > scores[1] else "NOT above"
            return TryResult(
                f'"fruit" {scores[0]:.2f} {order} "car" {scores[1]:.2f} for "apple"',
                seconds,
                {Figure.FRUIT: scores[0], Figure.CAR: scores[1]},
            )
        case Modality.IMAGE_GEN | Modality.IMAGE_EDIT:
            return _image(recipe, parse_reply(_ImageReply, reply, recipe), seconds, out_dir)
        case Modality.SPEECH_TO_TEXT:
            # whisper puts a line break between segments: one line reads better
            text = " ".join(parse_reply(_Transcription, reply, recipe).text.split())
            if not text:
                raise EmptyTranscriptError(
                    f"{recipe.name} returned no text: {json.dumps(reply)[:200]}"
                )
            return TryResult(f'"{text}"', seconds)
    raise TryError(f"`mnm try` has no request for {recipe.modality.value} models.")


def _chat(recipe: Recipe, reply: ChatReply, seconds: float, speed: bool) -> TryResult:
    message = reply.choices[0].message if reply.choices else _Message()
    content = (message.content or "").strip()
    timings = reply.timings or _Timings()
    numbers = {Figure(k): v for k, v in timings.model_dump().items() if v is not None}
    if speed:
        if timings.predicted_per_second is None:
            return _wall_clock_speed(recipe, reply, seconds)
        if timings.draft_n:
            numbers[Figure.DRAFT_ACCEPTED_PERCENT] = (
                100 * (timings.draft_n_accepted or 0) / (timings.draft_n)
            )
        return TryResult(
            f"decode {timings.predicted_per_second:.1f} tok/s over "
            f"{timings.predicted_n or 0:.0f} tokens, prompt "
            f"{timings.prompt_per_second or 0:.1f} tok/s" + _drafted(recipe, timings),
            seconds,
            numbers,
        )
    if not content:
        why = " (it spent the reply on thinking)" if message.reasoning_content else ""
        raise TryError(f"{recipe.name} answered with empty content{why}.")
    return TryResult(f'"{cut(content)}"', seconds, numbers)


def _drafted(recipe: Recipe, timings: _Timings) -> str:
    """With a draft model: how many of its tokens the model kept."""
    if timings.draft_n is not None:
        kept = timings.draft_n_accepted or 0
        share = f" ({100 * kept / timings.draft_n:.0f}%)" if timings.draft_n else ""
        return f"; the draft kept {kept:.0f} of {timings.draft_n:.0f} drafted tokens{share}"
    if recipe.draft is not None:
        return (
            f"; no draft counts came back (is {recipe.name} running with its draft? `mnm show "
            f"{recipe.name}` names the command)"
        )
    return ""


def answer_reply(recipe: Recipe, reply: ChatReply, seconds: float, out_dir: Path) -> TryResult:
    """A Run's chat answer: the whole text saved as `<name>.md`, its start on the line."""
    message = reply.choices[0].message if reply.choices else _Message()
    content = (message.content or "").strip()
    if not content:
        why = " (it spent the reply on thinking)" if message.reasoning_content else ""
        raise TryError(f"{recipe.name} answered with empty content{why}.")
    out = out_dir / f"{recipe.name}.md"
    out.unlink(missing_ok=True)  # never write through a link left under this name
    out.write_text(content + "\n", encoding="utf-8")
    timings = reply.timings or _Timings()
    numbers = {Figure(k): v for k, v in timings.model_dump().items() if v is not None}
    return TryResult(f'"{cut(" ".join(content.split()))}"', seconds, numbers, saved=out)


def cut(text: str, limit: int = ANSWER_LINE) -> str:
    """A reply's start for one line; a longer one ends in "…", so a cut never reads as where
    the model stopped (such as "… than longer wavelengths (red a")."""
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _wall_clock_speed(recipe: Recipe, reply: ChatReply, seconds: float) -> TryResult:
    """A server without llama.cpp's timings (vLLM): tokens over the whole request, prompt
    included, so the rate reads a little low."""
    tokens = reply.usage.completion_tokens if reply.usage else 0
    if not tokens or seconds <= 0:
        raise TryError(f"{recipe.name} reported neither timings nor a token count.")
    return TryResult(
        f"{tokens / seconds:.1f} tok/s over {tokens} tokens (the whole request, prompt "
        "included: the engine reports no decode timings)",
        seconds,
        {
            Figure.COMPLETION_TOKENS: float(tokens),
            Figure.REQUEST_TOKENS_PER_SECOND: tokens / seconds,
        },
    )


def _image(recipe: Recipe, reply: _ImageReply, seconds: float, out_dir: Path) -> TryResult:
    encoded = reply.data[0].b64_json if reply.data else None
    if not encoded:
        raise TryError(f"{recipe.name} returned no image: {reply.model_dump_json()[:300]}")
    try:
        png = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TryError(f"{recipe.name} returned an image that is not base64.") from exc
    width, height = png_size(png)
    out = out_dir / f"{recipe.name}.png"
    # A new file every time: a symlink or hard link left under this name must not carry
    # the write to a file elsewhere.
    out.unlink(missing_ok=True)
    out.write_bytes(png)
    wanted = image_size(recipe)
    note = (
        "" if (width, height) == wanted else f" (the instance is set to {wanted[0]} x {wanted[1]})"
    )
    return TryResult(
        f"a {width} x {height} image{note}: {out}",
        seconds,
        {Figure.WIDTH: width, Figure.HEIGHT: height},
        saved=out,
    )


def video_reply(recipe: Recipe, job: Job, seconds: float, out_dir: Path) -> TryResult:
    if job.status != "completed" or job.result is None or not job.result.b64_json:
        error = job.error
        why = (
            error.message or error.code
            if isinstance(error, _JobError)
            else error or job.status or "no status"
        )
        raise TryError(f"{recipe.name}'s clip did not complete: {why}")
    try:
        clip = base64.b64decode(job.result.b64_json, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TryError(f"{recipe.name} returned a clip that is not base64.") from exc
    suffix = _VIDEO_SUFFIX.get(job.result.mime_type, ".webm")
    out = out_dir / f"{recipe.name}{suffix}"
    out.unlink(missing_ok=True)  # never write through a link left under this name
    out.write_bytes(clip)
    return TryResult(
        f"a {job.result.frame_count}-frame clip at {job.result.fps} fps "
        f"({len(clip) / 1024:.0f} KiB): {out}",
        seconds,
        {Figure.FRAMES: job.result.frame_count, Figure.FPS: job.result.fps},
        saved=out,
    )


_VIDEO_SUFFIX = {"video/webm": ".webm", "image/webp": ".webp", "video/x-msvideo": ".avi"}


def speech_reply(
    recipe: Recipe, wav: bytes, seconds: float, out_dir: Path, loading: bool = False
) -> TryResult:
    """The WAV saved and worded; `loading`: the request also loaded the model (it was
    parked), so its rate is said with the load (else "0.2x realtime" for a voice that
    reads 16.7x warm)."""
    length = wav_seconds(wav)
    if length is None:
        raise TryError(f"{recipe.name} returned something that is not a WAV file.")
    out = out_dir / f"{recipe.name}.wav"
    out.unlink(missing_ok=True)  # never write through a link left under this name
    out.write_bytes(wav)
    rate = length / seconds if seconds > 0 else 0.0
    return TryResult(
        f"{length:.1f} s of speech ({len(wav) / 1024:.0f} KiB, {rate:.1f}x realtime"
        f"{' counting the model load' if loading else ''}): {out}",
        seconds,
        {Figure.AUDIO_SECONDS: length, Figure.REALTIME: rate},
        saved=out,
    )


def png_size(png: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR chunk; TryError for anything else."""
    if len(png) < 24 or not png.startswith(PNG_SIGNATURE) or png[12:16] != b"IHDR":
        raise TryError("the image the engine returned is not a PNG")
    width, height = struct.unpack(">II", png[16:24])
    return width, height
