"""The studio's cards: a music instance answers on audio.cpp's task route, which no
chat client calls, so its cards say where mnm uses it (`mnm try`, `mnm run`, the web's Run
tab) and give the one curl line that reaches it through the router."""

from __future__ import annotations

import json

from makenomistakes.consumers import HarnessUrl, curl_key
from makenomistakes.hardware import platform as platforms
from makenomistakes.harness_samples import sample_body
from makenomistakes.hub.facts import Modality
from makenomistakes.music_api import SINGING_ROUTES, TASK_PATH, voice_path
from makenomistakes.recipes.model import Recipe

NOT_A_CHAT_MODEL = (
    "A music model answers audio.cpp's task route, not a chat one: chat clients (Odysseus, "
    "Open WebUI) never list it, and the router leaves it out of `/v1/models`."
)
_DECODE = (
    'python3 -c "import base64, json, sys; a = json.load(open(sys.argv[1])); '
    "[open(s.get('id', 'song') + '.wav', 'wb').write(base64.b64decode(s['audio'])) "
    "for s in a.get('named_audio_outputs') or [a]]\" {answer}"
)


def music_urls(recipe: Recipe, base_local: str) -> list[HarnessUrl]:
    """What the Odysseus card set shows for a music instance."""
    return [
        HarnessUrl(
            label="mnm's studio",
            url=base_local,
            note=f"{NOT_A_CHAT_MODEL} `mnm try {recipe.name}` checks it; "
            f"`mnm run {recipe.name}` keeps what it makes as a run (the web's Run tab too).",
        ),
        music_curl_card(recipe, base_local),
    ]


def music_curl_card(recipe: Recipe, base: str) -> HarnessUrl:
    """One paste: the request (a song's tags, or the clip inline as base64), the answer into
    a file, its WAVs written beside it."""
    answer = f"{recipe.name}-answer.json"
    url = f"{base.removesuffix('/v1')}{TASK_PATH}"
    decode = _DECODE.format(answer=answer)
    if recipe.modality is Modality.SONG:
        body = json.dumps({"model": recipe.served_name, **sample_body(recipe)})
        send = f"curl -sSf{curl_key()} {url} -H 'content-type: application/json' -d '{body}'"
        note = "Writes song.wav from the answer"
    else:
        extra, note = _clip_parts(recipe)
        send = (
            f'printf \'{{"model": "{recipe.served_name}", "request": {{"audio_base64": "%s"'
            f'{extra}}}}}\' "$(base64 -w0 clip.wav)" > request.json\n'
            f"curl -sSf{curl_key()} {url} -H 'content-type: application/json' -d @request.json"
        )
    snippet = f"{send} -o {answer}\n{decode}\n"
    return HarnessUrl(
        label="Terminal",
        url=base,
        note=f"From {platforms.current().shell_label}. {note} (a WAV the engine sends as base64 "
        f"in JSON). `mnm try {recipe.name}` does the same and times it.",
        snippet=snippet,
    )


def _clip_parts(recipe: Recipe) -> tuple[str, str]:
    """What a clip's request adds beside the audio, and the card's note: a converter names a
    library voice (its files where the container reads them), a separator nothing."""
    if recipe.modality is not Modality.VOICE_CONVERT:
        return "", "Sends clip.wav from the current folder; writes each stem or output as a WAV"
    family = getattr(recipe.args, "family", "")
    if family == "rvc":
        model = voice_path("<voice>", "model.pth")
        part = f', "options": {{"voice_model_path": "{model}"}}'
    else:
        route = SINGING_ROUTES.get(str(family), "")
        reference = voice_path("<voice>", "reference.wav")
        part = f', "voice_ref": "{reference}", "options": {{"route": "{route}"}}'
    note = (
        "Sends clip.wav (a sung stem) from the current folder in a library voice: put its name "
        "for <voice> (`mnm voices ls`); writes the converted WAV"
    )
    return part, note
