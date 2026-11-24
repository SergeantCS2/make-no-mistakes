"""`mnm try` and `mnm vram`: send an instance the request its card shows, read the GPU.

The request is the curl card's own (`harness.sample_body`), so a passing `mnm try` means the
card works as printed. Images are saved where the command runs; nothing is written anywhere
else. The GPU readings during a request are `gpu_peak`'s. The request takes its client and
clocks as arguments so tests never touch a GPU or a socket. What comes back is read in
`smoke_replies`; its errors and result are re-exported here, where callers have always
found them.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.consumers import own_origin
from makenomistakes.diagnose import diagnose
from makenomistakes.harness_media import (
    EDIT_PROMPT,
    IMAGE_PROMPT,
    client_path,
    image_size,
    needs_first_frame,
)
from makenomistakes.harness_samples import NoCardError, sample_body
from makenomistakes.hub.facts import CHAT_LIKE, MUSIC_MODALITIES, Modality
from makenomistakes.media_bytes import grey_png, image_mime
from makenomistakes.recipes import Recipe
from makenomistakes.smoke_music import music_try
from makenomistakes.smoke_replies import (
    ChatReply,
    EmptyTranscriptError,
    Job,
    NoAnswerError,
    TryError,
    TryResult,
    answer_reply,
    parse_reply,
    png_size,
    refusal_text,
    speech_reply,
    summarise,
    video_reply,
)

__all__ = [  # the reply side's names callers import from here
    "EmptyTranscriptError",
    "NoAnswerError",
    "Picture",
    "TryError",
    "TryResult",
    "answered",
    "png_size",
    "try_instance",
    "unanswered",
]

TRY_TIMEOUT_SECONDS = 600.0  # a first image on a cold engine loads its weights first
VIDEO_TIMEOUT_SECONDS = 3600.0  # a clip is queued and polled: minutes on 12 GB, longer streamed
POLL_SECONDS = 2.0
SPEED_TOKENS = 400  # the speed request: long enough for a steady decode rate
ANSWER_TOKENS = 4096  # a Run's chat answer: room for a whole reply, not a two-word check
EDIT_INPUT_SIDE = 256  # the picture `mnm try` sends an editing instance: a flat grey square


@dataclass(frozen=True)
class Picture:
    """`mnm try --image`, `--mask`: a picture to repaint, edit or animate, and where to;
    with a prompt (a Run), also a picture a vision model is asked about."""

    image: bytes
    mask: bytes | None = None


# The types a picture reaches, and those a mask does: anything else would drop them silently
# and answer about the card's own test input.
PICTURE_TYPES = frozenset({Modality.IMAGE_GEN, Modality.IMAGE_EDIT, Modality.VIDEO})
MASK_TYPES = frozenset({Modality.IMAGE_GEN, Modality.IMAGE_EDIT})


def _check_picture(recipe: Recipe, picture: Picture | None, asked: bool) -> None:
    if picture is None:
        return
    if recipe.modality is Modality.VISION_CHAT and asked and picture.mask is None:
        return  # a Run asks a vision model about it; the test's own question has its picture
    if recipe.modality not in PICTURE_TYPES:
        raise TryError(
            f"{recipe.name} is {recipe.modality.a_label} model: --image is for an image model "
            "(repaint), an editor or a video model (image-to-video)."
        )
    if picture.mask is not None and recipe.modality not in MASK_TYPES:
        raise TryError(
            f"{recipe.name} is {recipe.modality.a_label} model: --mask is for an image model "
            "or an editor."
        )


def try_instance(  # noqa: PLR0913, PLR0917 - the swappable parts beside the request
    recipe: Recipe,
    http: httpx.Client,
    out_dir: Path,
    audio: Path | None = None,
    speed: bool = False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    say: str | None = None,
    picture: Picture | None = None,
    root: str | None = None,
    prompt: str | None = None,
    loading: bool = False,
    lyrics: str | None = None,
) -> TryResult:
    """One request, timed, its answer summarised. `speed` asks a chat model for 400 tokens
    and reports llama.cpp's own rates instead of a two-word reply. A video instance's request
    queues a job that is polled every `POLL_SECONDS` until the clip is done. `say` is the
    text a voice reads instead of the card's sentence. `picture` is what an image model
    repaints (inpaints, with its mask), an editor edits, or a video model animates. `root`
    sends it to that origin instead of the router (the engine's own port). `prompt` (a
    Run) replaces the card's question, picture prompt, clip prompt or sentence; a chat answer
    is then saved whole as `<name>.md` beside the other outputs. `loading`: the model was not
    loaded when it was sent, so its time includes the load (a voice's rate says so).
    `lyrics` is what a song instance sings to the prompt's tags."""
    _check_picture(recipe, picture, prompt is not None)
    direct = root is not None  # the engine itself (`mnm engines test`), not the router
    root = root or own_origin(recipe)
    if recipe.modality in MUSIC_MODALITIES:  # audio.cpp's task route (`smoke_music`)
        try:
            return music_try(recipe, http, root, out_dir, answered, audio, prompt, lyrics)
        except httpx.TransportError as exc:
            raise unanswered(recipe, exc, root, direct) from exc
    repaint = recipe.modality is Modality.IMAGE_GEN and picture is not None
    path = root + ("/v1/images/edits" if repaint else client_path(recipe))
    started = clock()
    try:
        response = _send(recipe, http, path, audio, speed, say or prompt, picture, prompt)
        if recipe.modality is Modality.VIDEO:
            queued = parse_reply(Job, _json(recipe, response, httpx.codes.ACCEPTED), recipe)
            job = _await_job(recipe, http, root, queued, sleep)
            return video_reply(recipe, job, clock() - started, out_dir)
    except httpx.TransportError as exc:  # every request goes through the router
        raise unanswered(recipe, exc, root, direct) from exc
    seconds = clock() - started
    if recipe.modality is Modality.TEXT_TO_SPEECH:  # bytes, not JSON: a WAV file
        return speech_reply(recipe, answered(recipe, response), seconds, out_dir, loading)
    reply = _json(recipe, response, httpx.codes.OK)
    if prompt is not None and recipe.modality in CHAT_LIKE:
        return answer_reply(recipe, parse_reply(ChatReply, reply, recipe), seconds, out_dir)
    return summarise(recipe, reply, seconds, out_dir, speed)


def _send(  # noqa: PLR0913, PLR0917 - the card's request, with what a try or a Run adds
    recipe: Recipe,
    http: httpx.Client,
    path: str,
    audio: Path | None,
    speed: bool,
    say: str | None = None,
    picture: Picture | None = None,
    prompt: str | None = None,
) -> httpx.Response:
    """The card's request: JSON for most kinds, a file upload for speech-to-text and edits."""
    if recipe.modality is Modality.SPEECH_TO_TEXT:
        if audio is None:
            raise TryError("Speech-to-text needs a clip: `mnm try <name> --audio clip.wav`.")
        try:
            clip = audio.open("rb")
        except OSError as exc:
            raise TryError(f"Cannot read the clip {audio}: {exc.strerror or exc}.") from exc
        with clip:
            return http.post(
                path,
                data={"model": recipe.served_name, "response_format": "json"},
                files={"file": (audio.name, clip, "audio/wav")},
            )
    edit = recipe.modality is Modality.IMAGE_EDIT
    if edit or (recipe.modality is Modality.IMAGE_GEN and picture is not None):
        image = picture.image if picture else grey_png(EDIT_INPUT_SIDE)
        files = {"image": ("input.png", image, "image/png")}
        if picture is not None and picture.mask is not None:
            files["mask"] = ("mask.png", picture.mask, "image/png")
        words = prompt or (EDIT_PROMPT if edit else IMAGE_PROMPT)
        return http.post(path, data={"model": recipe.served_name, "prompt": words}, files=files)
    body = _body(recipe, speed) if prompt is None else _asked(recipe, prompt, picture)
    if say is not None and recipe.modality is Modality.TEXT_TO_SPEECH:
        body["input"] = say
    if recipe.modality is Modality.VIDEO and (picture is not None or needs_first_frame(recipe)):
        width, height = image_size(recipe)  # the first frame at the clip's own size
        frame = picture.image if picture else grey_png(width, height)
        body["init_image"] = base64.b64encode(frame).decode()
    return http.post(path, json=body)


def _asked(recipe: Recipe, prompt: str, picture: Picture | None) -> dict[str, object]:
    """The card's request with the user's words in it (a Run): the question a chat or
    vision model answers in full, or the prompt a picture or clip is made from."""
    body = _body(recipe, speed=False)
    if recipe.modality in CHAT_LIKE:
        content: object = prompt
        if picture is not None:  # a vision model: the question and the picture
            mime = image_mime(picture.image)
            if mime is None:
                raise TryError("The picture is not a PNG, JPEG, GIF or WebP file.")
            url = f"data:{mime};base64,{base64.b64encode(picture.image).decode()}"
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        body.update(messages=[{"role": "user", "content": content}], max_tokens=ANSWER_TOKENS)
    elif recipe.modality in {Modality.IMAGE_GEN, Modality.VIDEO}:
        body["prompt"] = prompt
    return body


def unanswered(
    recipe: Recipe, exc: httpx.TransportError, root: str, direct: bool = False
) -> NoAnswerError:
    """A request that got no answer, worded: nothing listening (the router, or with `direct`
    the engine's own port) or the instance cut off mid-answer."""
    kind = exc.__class__.__name__
    if not isinstance(exc, httpx.ConnectError):
        return NoAnswerError(f"{recipe.name} stopped answering ({kind}).")
    if direct:
        return NoAnswerError(f"Nothing answers at {root} ({kind}).")
    return NoAnswerError(
        f"The router at {root} does not answer ({kind}): `mnm service status` says whether "
        "its service runs (`mnm service install` sets it up)."
    )


def answered(recipe: Recipe, response: httpx.Response, expected: int = httpx.codes.OK) -> bytes:
    """The body of an answer with the `expected` status; a 502 is `NoAnswerError` (no
    engine behind the router), any other status a `TryError` with what the log rules make
    of it."""
    if response.status_code == httpx.codes.BAD_GATEWAY:
        raise NoAnswerError(
            f"{recipe.name}'s engine does not answer on port {recipe.port} behind the router "
            f"(`mnm logs {recipe.name}`)."
        )
    if response.status_code != expected:
        # an error the log rules explain
        known = diagnose(response.text, recipe.engine, recipe_backend_or_none(recipe))
        why = f"\n{known[0].cause} {known[0].remedy}" if known else ""
        raise TryError(
            f"{recipe.name} answered HTTP {response.status_code}: "
            f"{refusal_text(response.text)}{why}"
        )
    return response.content


def _json(recipe: Recipe, response: httpx.Response, expected: int) -> object:
    answered(recipe, response, expected)
    try:
        reply: object = response.json()
    except ValueError as exc:
        raise TryError(f"{recipe.name} answered with something that is not JSON.") from exc
    return reply


_PENDING = frozenset({"queued", "generating"})


def _await_job(
    recipe: Recipe, http: httpx.Client, root: str, queued: Job, sleep: Callable[[float], None]
) -> Job:
    """Poll the queued job every `POLL_SECONDS` until it leaves the queue."""
    if not queued.poll_url:
        raise TryError(f"{recipe.name} queued no job: {queued.model_dump_json()[:200]}")
    url = root + queued.poll_url  # a poll's reply carries the status, not the URL again
    waited = 0.0
    while True:
        job = parse_reply(Job, _json(recipe, http.get(url), httpx.codes.OK), recipe)
        if job.status not in _PENDING:
            return job
        if waited >= VIDEO_TIMEOUT_SECONDS:
            raise TryError(
                f"{recipe.name} is still {job.status} after {waited:.0f} s (job {job.id})."
            )
        sleep(POLL_SECONDS)
        waited += POLL_SECONDS


def _body(recipe: Recipe, speed: bool) -> dict[str, object]:
    try:
        body: dict[str, object] = {"model": recipe.served_name, **sample_body(recipe)}
    except NoCardError as exc:
        raise TryError(f"{recipe.name}: {exc}.") from exc
    if speed and recipe.modality in CHAT_LIKE:
        body.update(
            messages=[{"role": "user", "content": "Write a long story about a lighthouse."}],
            max_tokens=SPEED_TOKENS,
            ignore_eos=True,  # llama-server and vLLM: always the full count, so rates compare
        )
    return body
