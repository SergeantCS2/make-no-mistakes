"""Copy-ready URLs and instructions per consumer. Pure data; no API coupling."""

from __future__ import annotations

import json

from makenomistakes.consumers import (
    CONSUMER_LABELS,
    ODYSSEUS_HOST,
    Consumer,
    HarnessUrl,
    browser_note,
    browser_origin,
    context_tokens,
    curl_key,
    docker_base_url,
    instance_origin,
    key_set,
    local_base_url,
    ollama_origin,
    output_tokens,
    shared_base_url,
    with_key_hint,
)
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.hardware import platform as platforms
from makenomistakes.harness_agents import AGENT_CLIENTS, AGENT_MODALITIES, agent_cards
from makenomistakes.harness_apps import APPS, app_cards
from makenomistakes.harness_media import (
    client_path,
    endpoint,
    image_curl_card,
    image_size,
    image_urls,
    repaint_curl_card,
    stitch_card,
    video_curl_card,
    video_urls,
)
from makenomistakes.harness_music import NOT_A_CHAT_MODEL, music_curl_card, music_urls
from makenomistakes.harness_samples import (
    THINKING_OFF,
    NoCardError,
    sample_body,
    sample_for,
    thinking_off,
)
from makenomistakes.hub.facts import CHAT_LIKE, MUSIC_MODALITIES, Modality
from makenomistakes.recipes.audiocpp import voice_names
from makenomistakes.recipes.llamacpp import VISION_ID_RE, vision_name
from makenomistakes.recipes.model import Recipe

__all__ = ["NoCardError", "consumer_urls", "harness_urls", "sample_body"]
# Odysseus takes a full URL for embeddings (its .env) and a base URL for everything else;
# rerank has nowhere to go in Odysseus, so its card shows the base URL like the others.
_ODYSSEUS_FULL_URL = frozenset({Modality.EMBEDDING})


def harness_urls(recipe: Recipe) -> list[HarnessUrl]:
    """Odysseus's cards (and two for other apps), the router's key named when it asks."""
    return with_key_hint(_harness_urls(recipe))


def _harness_urls(recipe: Recipe) -> list[HarnessUrl]:
    base_odysseus = docker_base_url(recipe)
    base_local = local_base_url(recipe)
    if recipe.modality in (Modality.IMAGE_GEN, Modality.IMAGE_EDIT):
        return image_urls(recipe, base_odysseus, base_local)
    if recipe.modality is Modality.VIDEO:
        return video_urls(recipe, base_local)
    if recipe.modality in CHAT_LIKE:
        return _shared_urls(recipe, base_local)
    if recipe.modality in MUSIC_MODALITIES:  # The studio's task route
        return music_urls(recipe, base_local)
    return [
        HarnessUrl(
            label="Odysseus (running in Docker)",
            url=_odysseus_url(recipe, base_odysseus),
            note=_odysseus_note(recipe),
            snippet=_odysseus_snippet(recipe, base_odysseus),
        ),
        HarnessUrl(
            label=platforms.current().apps_label,
            url=base_local,
            note=f"OpenAI-compatible base URL; model id `{recipe.served_name}`. Clients POST to "
            f"{client_path(recipe)} under it.",
        ),
        _browser_check(recipe, base_local),
    ]


def _shared_urls(recipe: Recipe, base_local: str) -> list[HarnessUrl]:
    """A chat, coding or vision instance: the router's one URL for every such model, the
    instance name as the model; this instance's own URL for a client that wants it."""
    return [
        HarnessUrl(
            label="Odysseus (running in Docker)",
            url=shared_base_url(ODYSSEUS_HOST, chat_only=True),
            note=_odysseus_note(recipe)
            + " One endpoint serves every chat model mnm runs: new ones appear in its list, and "
            "a model that is not loaded loads at its first message (its thinking panel shows "
            "the load).",
        ),
        HarnessUrl(
            label=platforms.current().apps_label,
            url=shared_base_url(),
            note=f"OpenAI-compatible base URL for every instance; model id `{recipe.name}`. "
            f"Clients POST to {client_path(recipe)} under it. `/chat/v1` lists the chat models "
            "only.",
        ),
        HarnessUrl(
            label="This instance alone",
            url=base_local,
            note=f"For a client that wants a URL of its own; model id `{recipe.served_name}` "
            f"(or `{recipe.name}`).",
        ),
        _browser_check(recipe, base_local),
        *_ollama_cards(),
    ]


def _ollama_cards() -> list[HarnessUrl]:
    """With `MNM_OLLAMA_PORT` set: what Odysseus gets from the router's Ollama port."""
    origin = ollama_origin(ODYSSEUS_HOST)
    if origin is None:
        return []
    return [
        HarnessUrl(
            label="Odysseus Cookbook (Ollama's port)",
            url=origin,
            note="Nothing to paste: the Cookbook's model scan asks this port's /api/tags "
            "(routes/cookbook_helpers.py:559 in 0.20.19), so mnm's chat and embedding models "
            "also show in its model list, as Ollama models. Chat with them through the "
            "endpoint above: the Cookbook's own Serve would start an Ollama of its own. Other "
            "Ollama clients (Home Assistant, n8n, Enchanted) take this URL: `mnm url <name> "
            "--for home-assistant`.",
        )
    ]


def _browser_check(recipe: Recipe, base_local: str) -> HarnessUrl:
    if recipe.modality is Modality.SPEECH_TO_TEXT:  # whisper-server has no /v1/models (404)
        return HarnessUrl(
            label="Check in a browser",
            url=f"{browser_origin(recipe)}/",
            note="whisper-server's own page: upload a clip to test the instance directly."
            + browser_note(),
            opens_in_browser=True,
        )
    return HarnessUrl(
        label="Check in a browser",
        url=f"{browser_origin(recipe)}/v1/models",
        note="Plain GET that lists the served model id. The base URL itself answers 404 in "
        "a browser by design." + browser_note(),
        opens_in_browser=True,
    )


def _odysseus_url(recipe: Recipe, base: str) -> str:
    """Odysseus takes a full URL for its embeddings setting and a base URL for everything else."""
    return base + endpoint(recipe) if recipe.modality in _ODYSSEUS_FULL_URL else base


def _odysseus_snippet(recipe: Recipe, base: str) -> str | None:
    """The .env lines Odysseus reads for embeddings; everything else is pasted in its UI."""
    if recipe.modality is not Modality.EMBEDDING:
        return None
    return f"EMBEDDING_URL={_odysseus_url(recipe, base)}\nEMBEDDING_MODEL={recipe.served_name}\n"


_NOT_A_CHAT_MODEL = (
    "Do not add this URL under Added Models: it is not a chat model, and a chat with it "
    "fails (HTTP 502)."
)


# Odysseus: the .env route names no key; the admin `POST /api/embeddings/endpoint` takes one.
_EMBEDDING_KEY = (
    " This router asks for MNM_API_KEY, and the .env lines carry none, so Odysseus would fall "
    "back to its built-in model: give it the endpoint with its key through Odysseus's admin "
    "embeddings endpoint (url, model, api_key) instead (not checked on the box yet)."
)


def _odysseus_note(recipe: Recipe) -> str:  # noqa: PLR0911 - one sentence per type
    match recipe.modality:
        case Modality.CHAT | Modality.CODING | Modality.VISION_CHAT | Modality.UNKNOWN:
            # An Image endpoint at a chat URL makes Odysseus route every message of that chat to
            # its image tools (routes/chat_routes.py `_is_image_generation_session`).
            images = (
                " Attach an image in a chat with it."
                if recipe.modality is Modality.VISION_CHAT
                else ""
            )
            if recipe.modality is Modality.VISION_CHAT and not VISION_ID_RE.search(recipe.name):
                images = (  # Odysseus decides image upload by the id's name
                    f" Odysseus offers image upload only for ids with `vl` or `vision`, and "
                    f"`{recipe.name}` has neither: `mnm rm {recipe.name}` (its downloads stay), "
                    f"then add it again with `--name {vision_name(recipe.name)}`."
                )
            return (
                "Settings → Add Models → Add Local Models, type LLM (not Image: Odysseus would "
                f"send every message to its image tools), paste the URL. Model id `{recipe.name}`."
                + images
            )
        case Modality.EMBEDDING:
            # Odysseus v1.0.3 has no embeddings setting in its UI (admin.js says so): it embeds
            # with its built-in FastEmbed model unless EMBEDDING_URL is set in its .env.
            return (
                "Odysseus has no embeddings setting in its UI; it reads EMBEDDING_URL and "
                "EMBEDDING_MODEL from its .env. With this instance running, add the two lines "
                "below to ~/odysseus/.env, then run `docker compose up -d` in ~/odysseus (if "
                "the URL does not answer then, Odysseus uses its built-in model until its next "
                "restart). Stored memories and documents are re-embedded with this model. "
                + _NOT_A_CHAT_MODEL
                + (_EMBEDDING_KEY if key_set() else "")
            )
        case Modality.RERANK:
            return (
                "Odysseus has no reranker setting; the curl card checks the endpoint. "
                + _NOT_A_CHAT_MODEL
            )
        case Modality.SPEECH_TO_TEXT:
            # Odysseus 0.20.19: no speech-to-text settings in its UI (static/js/settings/
            # speech.js:176); src/settings.py:62-65 has the keys, routes/auth_routes.py:760
            # takes them from an admin; static/js/voiceRecorder.js:10 sends recordings then.
            return (
                "Odysseus (this build) has no speech-to-text settings page. Add this URL under "
                "Settings → Add Models → Add Local Models, type LLM (Odysseus finds one model "
                f"there, `{recipe.name}`: the router lists it, loaded or not); then an admin "
                "sets it through Odysseus's API: POST "
                '/api/auth/settings with {"stt_enabled": true, "stt_provider": '
                f'"endpoint:<that endpoint\'s id>", "stt_model": "{recipe.served_name}"}}. The '
                "chat's recorder then sends what you say here."
            )
        case Modality.TEXT_TO_SPEECH:
            # Odysseus 0.20.19 hides read-aloud: the TTS settings card (static/index.html:
            # 1697-1700) and the chat's TTS Mode (index.html:1155-1156); src/settings.py:57-61
            # has the keys, routes/auth_routes.py:760 takes them from an admin.
            return (
                "Odysseus (this build) hides read-aloud and its text-to-speech settings, so "
                "nothing in its UI plays speech. The only route is its admin API: add this URL "
                "under Settings → Add Models → Add Local Models, type LLM (Odysseus finds one "
                "model there), then POST "
                '/api/auth/settings with {"tts_provider": "endpoint:<that endpoint\'s id>", '
                f'"tts_model": "{recipe.served_name}", "tts_voice": "<a voice>"}}.'
                f"{_voices_line(recipe)}"
            )
        case Modality.IMAGE_GEN | Modality.IMAGE_EDIT | Modality.VIDEO:  # their own card sets
            return "Settings → Add Models → Add Local Models, type Image, paste the URL."
        case Modality.SONG | Modality.STEMS | Modality.VOICE_CONVERT | Modality.MUSIC_SCORE:
            return NOT_A_CHAT_MODEL


def _voices_line(recipe: Recipe) -> str:
    if not isinstance(recipe.args, AudioCppArgs):
        return ""
    return (
        f" The voice must be one of this model's: {', '.join(voice_names(recipe))} (another "
        "answers with an error)."
    )


def _api_key_note() -> str:
    if key_set():
        return "API key: your MNM_API_KEY (the router asks every request for it)."
    return "API key: anything, e.g. `none` (no key is set, so nothing checks it)."


def consumer_urls(recipe: Recipe, consumer: Consumer) -> list[HarnessUrl]:
    """The cards for one consumer. Odysseus keeps the original three-card layout."""
    if consumer is Consumer.ODYSSEUS:
        return harness_urls(recipe)
    if consumer in AGENT_CLIENTS:
        return with_key_hint(agent_cards(recipe, consumer))
    if consumer in APPS:
        return with_key_hint(app_cards(recipe, consumer))
    try:
        return with_key_hint(_client_urls(recipe, consumer))
    except NoCardError as exc:  # a kind of model with no sample yet: a card saying so, no crash
        return [
            HarnessUrl(label=CONSUMER_LABELS[consumer], url=local_base_url(recipe), note=str(exc))
        ]


def _client_urls(recipe: Recipe, consumer: Consumer) -> list[HarnessUrl]:
    """Open WebUI, Continue, the OpenAI SDK and curl."""
    base = local_base_url(recipe)
    if consumer is Consumer.OPEN_WEBUI:
        note = sample_for(recipe).open_webui.format(model=recipe.served_name, key=_api_key_note())
        return [
            HarnessUrl(
                label="Open WebUI in Docker",  # reaches WSL through the bridge host, like Odysseus
                url=docker_base_url(recipe),
                note=note,
            ),
            HarnessUrl(
                label=f"Open WebUI {platforms.current().here} (not in Docker)",
                url=base,
                note="Same settings with this URL.",
            ),
        ]
    if consumer is Consumer.CONTINUE:
        return [_continue_card(recipe, base)]
    if consumer is Consumer.OPENAI_SDK:
        sdk = sample_for(recipe).sdk
        if recipe.modality is Modality.IMAGE_GEN:  # a request's size beats the instance's
            width, height = image_size(recipe)
            sdk = sdk.replace('size="1024x1024"', f'size="{width}x{height}"')
        if isinstance(recipe.args, AudioCppArgs):
            sdk = sdk.replace('voice="default"', f'voice="{recipe.args.voice}"')
        # The video route hangs off the server root, not `/v1`: the client is built there.
        sdk_base = instance_origin(recipe) if recipe.modality is Modality.VIDEO else base
        key = 'os.environ["MNM_API_KEY"]' if key_set() else '"none"'
        snippet = (
            ("import os\n" if key_set() else "")
            + f'from openai import OpenAI\n\nMODEL = "{recipe.served_name}"\n'
            f'client = OpenAI(base_url="{sdk_base}", api_key={key})\n' + sdk
        )
        why = f" {THINKING_OFF}" if thinking_off(recipe) else ""
        return [
            HarnessUrl(
                label="Python", url=sdk_base, note=f"pip install openai.{why}", snippet=snippet
            )
        ]
    media = _media_curl_cards(recipe, base)
    if media is not None:
        return media
    sample = sample_for(recipe)
    if sample.form:
        fields = {**sample.form, "model": recipe.served_name}
        payload = " ".join(f"-F {key}={value}" for key, value in fields.items())
    else:
        body = {"model": recipe.served_name, **sample_body(recipe)}
        payload = f"-H 'content-type: application/json' -d '{json.dumps(body)}'"
    speech = recipe.modality is Modality.TEXT_TO_SPEECH
    output = f" -o {recipe.name}.wav" if speech else ""
    # --fail: an engine error is not saved as the clip; -S still prints curl's error line
    flags = f"-s{'Sf' if speech else ''}{curl_key()}"
    curl = f"curl {flags} {base}{endpoint(recipe)} {payload}{output}\n"
    note = f"From {platforms.current().shell_label}." + (
        f" Writes {recipe.name}.wav in the current directory (nothing when the engine answers "
        f"an error: `mnm logs {recipe.name}` says why); `mnm try {recipe.name}` does the "
        "same and says how long the clip is."
        if output
        else f" {THINKING_OFF}"
        if thinking_off(recipe)
        else ""
    )
    return [HarnessUrl(label="Terminal", url=base, note=note, snippet=curl)]


def _media_curl_cards(recipe: Recipe, base: str) -> list[HarnessUrl] | None:
    """Pictures and clips have their own one-paste lines (`harness_media`)."""
    match recipe.modality:
        case Modality.IMAGE_GEN:  # every image model also repaints and inpaints
            return [image_curl_card(recipe, base), repaint_curl_card(recipe, base)]
        case Modality.IMAGE_EDIT:
            return [image_curl_card(recipe, base)]
        case Modality.VIDEO:
            return [video_curl_card(recipe), stitch_card()]
        case Modality.SONG | Modality.STEMS | Modality.VOICE_CONVERT | Modality.MUSIC_SCORE:
            return [music_curl_card(recipe, base)]
    return None


def _continue_card(recipe: Recipe, base: str) -> HarnessUrl:
    role = sample_for(recipe).continue_role
    if role is None:
        return HarnessUrl(
            label="Continue", url=base, note=f"Continue has no {recipe.modality.value} role."
        )
    model = recipe.served_name
    snippet = (
        "models:\n"
        f"  - name: {model}\n"
        "    provider: openai\n"
        f"    model: {model}\n"
        f"    apiBase: {base}\n"
        + (  # Continue's own secrets: `MNM_API_KEY=…` in ~/.continue/.env
            "    apiKey: ${{ secrets.MNM_API_KEY }}\n" if key_set() else "    apiKey: none\n"
        )
        + f"    roles: [{role}]\n"
    )
    if recipe.modality in AGENT_MODALITIES:
        # Agent mode needs tools, and Continue detects them only for vendor model names
        # (`gpt-*`, `o*`, `gemma*` …); its default window is 32,768 whatever the engine runs.
        snippet += (
            "    capabilities: [tool_use]\n"
            "    defaultCompletionOptions:\n"
            f"      contextLength: {context_tokens(recipe)}\n"
            f"      maxTokens: {output_tokens(recipe)}\n"
        )
    return HarnessUrl(
        label=f"Continue config.yaml ({platforms.current().editor_label})",
        url=base,
        note="Paste into ~/.continue/config.yaml; when the file already has a `models:` "
        "list, add only the `- name:` entry to it"
        + platforms.current().localhost_note
        + (
            " The key comes from Continue's own secrets: add `MNM_API_KEY=<your key>` to "
            "~/.continue/.env."
            if key_set()
            else ""
        ),
        snippet=snippet,
    )
