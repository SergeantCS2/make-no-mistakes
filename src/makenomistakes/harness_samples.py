"""One example request per type of model (`_SAMPLES`): what the curl, SDK, Open WebUI and
Continue cards send, and what `mnm try` sends (`sample_body`)."""

from __future__ import annotations

from dataclasses import dataclass, field

from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.harness_media import (
    EDIT_PROMPT,
    IMAGE_PROMPT,
    SONG_LYRICS,
    SONG_SECONDS,
    SONG_TAGS,
    SPEECH_TEXT,
    VIDEO_PATH,
    VIDEO_PROMPT,
    image_size,
)
from makenomistakes.hub.facts import Modality
from makenomistakes.recipes.model import Recipe


@dataclass(frozen=True)
class _Sample:
    """One example request per modality: feeds the curl, SDK and Continue cards."""

    sdk: str  # Python lines after `client = OpenAI(...)`
    continue_role: str | None  # Continue's `roles:` entry; None when it has no such role
    open_webui: str  # where to paste in Open WebUI
    body: dict[str, object] = field(default_factory=dict)  # a JSON request
    form: dict[str, str] = field(default_factory=dict)  # or a multipart upload (curl -F)


# An 8x8 red PNG: small enough to paste, and it proves the projector loaded (a text-only
# answer to it is a missing --mmproj, not a chat-template problem).
_RED_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR42mO4IyeHFTEMLQkAid1GAQVcagkAAAAASUVORK5CYII="
)
_CHAT_ROLES = "chat, edit, apply"
_NO_STUDIO = "Open WebUI has no music setting; `mnm run <name>` and the web Run tab use it."
_NO_THINKING = {"enable_thinking": False}
_SDK_NO_THINKING = 'extra_body={"chat_template_kwargs": {"enable_thinking": False}}'

_CHAT_WEBUI = (
    "Admin → Settings → Connections → OpenAI API: URL above. {key} Model `{model}` appears in "
    "the picker."
)
_CHAT_SAMPLE = _Sample(
    body={
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 16,
        "chat_template_kwargs": _NO_THINKING,
    },
    sdk=(
        "reply = client.chat.completions.create(model=MODEL, "
        f'messages=[{{"role": "user", "content": "Say OK"}}], {_SDK_NO_THINKING})\n'
        "print(reply.choices[0].message.content)\n"
    ),
    continue_role=_CHAT_ROLES,
    open_webui=_CHAT_WEBUI,
)
_SAMPLES: dict[Modality, _Sample] = {
    Modality.CHAT: _CHAT_SAMPLE,
    Modality.CODING: _CHAT_SAMPLE,
    Modality.VISION_CHAT: _Sample(  # the chat API with an image part
        body={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this image? One word."},
                        {"type": "image_url", "image_url": {"url": _RED_PNG}},
                    ],
                }
            ],
            "max_tokens": 16,
            "chat_template_kwargs": _NO_THINKING,
        },
        sdk=(
            f'IMAGE = "{_RED_PNG}"\n'
            'reply = client.chat.completions.create(model=MODEL, messages=[{"role": "user", '
            '"content": [{"type": "text", "text": "What colour is this image?"}, '
            f'{{"type": "image_url", "image_url": {{"url": IMAGE}}}}]}}], {_SDK_NO_THINKING})\n'
            "print(reply.choices[0].message.content)\n"
        ),
        continue_role=_CHAT_ROLES,
        open_webui=_CHAT_WEBUI,
    ),
    Modality.EMBEDDING: _Sample(
        body={"input": ["a red apple", "the stock market"]},
        sdk=(
            'vectors = client.embeddings.create(model=MODEL, input=["a red apple"])\n'
            "print(vectors.data[0].embedding[:4])\n"
        ),
        continue_role="embed",
        open_webui="Admin → Settings → Documents: Embedding Model Engine OpenAI, URL above, "
        "model `{model}`. {key}",
    ),
    Modality.RERANK: _Sample(
        body={"query": "apple", "documents": ["fruit", "car"]},
        sdk=(
            'print(client.post("/rerank", cast_to=dict, body={"model": MODEL, '
            '"query": "apple", "documents": ["fruit", "car"]}))\n'
        ),
        continue_role="rerank",
        open_webui="Open WebUI has no OpenAI rerank setting; the curl card checks the endpoint.",
    ),
    Modality.IMAGE_GEN: _Sample(
        body={"prompt": "a lighthouse at dusk", "size": "1024x1024"},
        sdk=(
            'image = client.images.generate(model=MODEL, prompt="a lighthouse at dusk", '
            'size="1024x1024")\n'
            "print(image.data[0].b64_json[:40])\n"
        ),
        continue_role=None,
        open_webui="Admin → Settings → Images: engine OpenAI, URL above, model `{model}`. {key}",
    ),
    Modality.IMAGE_EDIT: _Sample(
        form={"image": "@input.png", "prompt": f"'{EDIT_PROMPT}'"},
        sdk=(
            'with open("input.png", "rb") as picture:\n'
            f'    image = client.images.edit(model=MODEL, image=picture, prompt="{EDIT_PROMPT}")\n'
            "print(image.data[0].b64_json[:40])\n"
        ),
        continue_role=None,
        open_webui="Open WebUI's image settings generate only; its Images page has no edit "
        "step. Use the engine page or the curl card.",
    ),
    Modality.VIDEO: _Sample(
        body={"prompt": VIDEO_PROMPT, "output_format": "webm"},
        sdk=(  # not an OpenAI route: the SDK's raw client reaches it from the server root
            "import base64, time\n"
            f'job = client.post("{VIDEO_PATH}", cast_to=dict, body={{"prompt": "{VIDEO_PROMPT}", '
            '"output_format": "webm"})\n'
            'while (job := client.get(job["poll_url"], cast_to=dict))["status"] in '
            '("queued", "generating"):\n'
            "    time.sleep(2)\n"
            'open("clip.webm", "wb").write(base64.b64decode(job["result"]["b64_json"]))\n'
        ),
        continue_role=None,
        open_webui="Open WebUI has no video setting; use the engine page or the curl card.",
    ),
    Modality.TEXT_TO_SPEECH: _Sample(
        body={"input": SPEECH_TEXT, "voice": "default", "response_format": "wav"},
        sdk=(
            'speech = client.audio.speech.create(model=MODEL, voice="default", '
            f'input="{SPEECH_TEXT}", response_format="wav")\n'
            'speech.write_to_file("speech.wav")\n'
        ),
        continue_role=None,
        open_webui="Admin → Settings → Audio: text-to-speech engine OpenAI, URL above, model "
        "`{model}`, voice one of the model's. {key}",
    ),
    # The studio's types answer on audio.cpp's task route, which no chat client calls
    Modality.SONG: _Sample(
        body={
            "request": {
                "text": SONG_TAGS,
                "lyrics": SONG_LYRICS,
                "options": {"duration_sec": SONG_SECONDS},
            }
        },
        sdk=(  # not an OpenAI route: the raw client posts it
            "import base64\n"
            'song = client.post("/tasks/run", cast_to=dict, body={"model": MODEL, "request": '
            f'{{"text": "{SONG_TAGS}", "options": {{"duration_sec": {SONG_SECONDS:g}}}}}}})\n'
            'open("song.wav", "wb").write(base64.b64decode(song["audio"]))\n'
        ),
        continue_role=None,
        open_webui=_NO_STUDIO,
    ),
    **{
        modality: _Sample(
            body={"request": {"audio_base64": "<the WAV, base64>"}},
            sdk=(
                "import base64\n"
                'wav = base64.b64encode(open("mix.wav", "rb").read()).decode()\n'
                'answer = client.post("/tasks/run", cast_to=dict, body={"model": MODEL, '
                '"request": {"audio_base64": wav}})\n'
                'for stem in answer.get("named_audio_outputs") or [{"id": "out", **answer}]:\n'
                '    open(stem["id"] + ".wav", "wb").write(base64.b64decode(stem["audio"]))\n'
            ),
            continue_role=None,
            open_webui=_NO_STUDIO,
        )
        for modality in (Modality.STEMS, Modality.VOICE_CONVERT, Modality.MUSIC_SCORE)
    },
    Modality.SPEECH_TO_TEXT: _Sample(
        form={"file": "@clip.wav", "response_format": "json"},
        sdk=(
            'with open("clip.wav", "rb") as audio:\n'
            "    text = client.audio.transcriptions.create(model=MODEL, file=audio)\n"
            "print(text.text)\n"
        ),
        continue_role=None,
        open_webui="Admin → Settings → Audio: speech-to-text engine OpenAI, URL above, model "
        "`{model}`. {key}",
    ),
}


class NoCardError(LookupError):
    """A kind of model this card module has no sample request for."""


def sample_body(recipe: Recipe) -> dict[str, object]:
    """The JSON a card sends (without `model`), at the instance's own size for images: the
    same request `mnm try` makes. Empty for speech-to-text, which uploads a file instead."""
    body = dict(sample_for(recipe).body)
    if recipe.modality is Modality.IMAGE_GEN:
        width, height = image_size(recipe)
        body.update(prompt=IMAGE_PROMPT, size=f"{width}x{height}")
    if isinstance(recipe.args, AudioCppArgs) and not recipe.args.music:  # its default voice
        body["voice"] = recipe.args.voice
    return body


# Said beside every chat card that turns thinking off (a raw curl without it can spend 32
# tokens on `reasoning_content` and answer nothing).
THINKING_OFF = (
    "`enable_thinking: false` keeps a thinking model (Qwen3) from spending this short reply on "
    "its reasoning; leave it out to get the reasoning too, in `reasoning_content`, with a "
    "larger `max_tokens`."
)


def thinking_off(recipe: Recipe) -> bool:
    """The card's request turns thinking off (chat, coding, vision)."""
    return "chat_template_kwargs" in sample_for(recipe).body


def sample_for(recipe: Recipe) -> _Sample:
    """Every servable modality has a sample: a new one raises `NoCardError`
    (a card saying so in `consumer_urls`, a `TryError` in `mnm try`), never chat cards."""
    try:
        return _SAMPLES[recipe.modality]
    except KeyError:
        raise NoCardError(
            f"no card for a {recipe.modality.value} model yet (`mnm url {recipe.name}` shows "
            "the Odysseus card)"
        ) from None
