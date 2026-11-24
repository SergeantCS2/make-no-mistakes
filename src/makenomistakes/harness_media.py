"""The picture and clip cards: what an image, edit or video instance's URLs look like,
how a client reaches the engine's routes, and the one-paste curl lines that save the result.

Shared by `harness.py` (the consumer cards) and `smoke.py` (`mnm try` sends the same request).
"""

from __future__ import annotations

import json

from makenomistakes.consumers import (
    ODYSSEUS_HOST,
    HarnessUrl,
    browser_note,
    browser_origin,
    curl_key,
    instance_origin,
)
from makenomistakes.engine_args import SdCppArgs
from makenomistakes.hardware import platform as platforms
from makenomistakes.hub.facts import Modality
from makenomistakes.hub.kit_types import ImageFamily
from makenomistakes.recipes.model import Recipe
from makenomistakes.recipes.whispercpp import WHISPERCPP_IMAGE

__all__ = [
    "EDIT_PROMPT",
    "IMAGE_PROMPT",
    "SPEECH_TEXT",
    "VIDEO_PATH",
    "VIDEO_PROMPT",
    "client_path",
    "endpoint",
    "image_curl_card",
    "image_size",
    "image_urls",
    "needs_first_frame",
    "repaint_curl_card",
    "stitch_card",
    "video_curl_card",
    "video_urls",
]

_ENDPOINT: dict[Modality, str] = {
    Modality.EMBEDDING: "/embeddings",
    Modality.RERANK: "/rerank",
    Modality.IMAGE_GEN: "/images/generations",
    Modality.IMAGE_EDIT: "/images/edits",  # multipart: an image (or several), a mask, a prompt
    Modality.SPEECH_TO_TEXT: "/audio/transcriptions",
    Modality.TEXT_TO_SPEECH: "/audio/speech",  # JSON in, `audio/wav` out
    # audio.cpp's task route, not an OpenAI one (`music_api`): a WAV comes back base64
    Modality.SONG: "/tasks/run",
    Modality.STEMS: "/tasks/run",
    Modality.VOICE_CONVERT: "/tasks/run",
    Modality.MUSIC_SCORE: "/tasks/run",
}
# sd-server's own video route is not under `/v1`: a job is queued (HTTP 202 with a
# `poll_url`) and polled until it completes; the clip comes back base64 in `result.b64_json`.
VIDEO_PATH = "/sdcpp/v1/vid_gen"

IMAGE_PROMPT = "a lighthouse at dusk, oil painting"  # every image card and `mnm try`
VIDEO_PROMPT = "a lighthouse at dusk, waves rolling in, the beam turning slowly"
EDIT_PROMPT = "make it night, with the lighthouse lamp lit"
SPEECH_TEXT = "The lighthouse keeper lit the lamp at dusk."
SONG_TAGS = "warm synth pop, female vocal, bright drums"  # the song cards and `mnm try`
SONG_LYRICS = "[verse]\nThe lighthouse keeper lit the lamp at dusk"
SONG_SECONDS = 10.0  # `mnm try`'s clip: long enough to hear, short enough on a first load
# A thinking model (Qwen3) spends a 16-token check on `reasoning_content` and answers with an
# empty `content`: the check turns thinking off. Templates without it ignore
# the key; Odysseus sends the same one.


def endpoint(recipe: Recipe) -> str:
    """The path under the `/v1` base URL a client calls for this kind of model (a video
    instance's route is `VIDEO_PATH` from the root instead)."""
    return _ENDPOINT.get(recipe.modality, "/chat/completions")  # chat and vision chat


def client_path(recipe: Recipe) -> str:
    """The request path from the server root: `/v1` plus `endpoint`, or the video route."""
    if recipe.modality is Modality.VIDEO:
        return VIDEO_PATH
    return "/v1" + endpoint(recipe)


def image_size(recipe: Recipe) -> tuple[int, int]:
    """The size an image instance runs at: what its cards ask for, and with the router's
    rewrite what every image gets unless `client-size` is on."""
    args = recipe.args
    return (args.width, args.height) if isinstance(args, SdCppArgs) else (1024, 1024)


def image_urls(recipe: Recipe, base_odysseus: str, base_local: str) -> list[HarnessUrl]:
    width, height = image_size(recipe)
    client_size = isinstance(recipe.args, SdCppArgs) and recipe.args.client_size
    # sd-server reads native settings embedded in the prompt and they override the request's
    # size (verified on the CPU build, f89d9b1): Odysseus passes a direct message through.
    size = json.dumps({"width": width, "height": height}, separators=(",", ":"))
    sizing = (
        "Odysseus picks the size (client-size is on): 512x512 when you message the image "
        "model directly (hard-coded), 1024x1024 through a chat model's image tool. To make a "
        f"message use this instance's {width}x{height}, end it with the line below."
        if client_size
        else f"Every image is this instance's {width}x{height}: mnm's router drops the size "
        "Odysseus asks for (its caption still shows it: 512x512 for a direct message). For "
        "another size in one message, end it with a line like the one below; "
        f"`mnm set {recipe.name} client-size=on` lets clients choose."
    )
    edit_size = (
        "the request's `size` (client-size is on)"
        if client_size
        else "the picture's own size (the router drops the request's `size`)"
    )
    if recipe.modality is Modality.IMAGE_EDIT:
        editing = (
            " Odysseus's image tools also edit: a picture plus an instruction in a chat goes to "
            f"`/v1/images/edits`, which this engine answers at {edit_size}, with the first image "
            "as the reference."
        )
    else:  # an image model repaints what it is sent; it does not follow an instruction
        editing = (
            " A picture Odysseus sends it (a chat upload, an inpaint with a mask) is repainted "
            f"at {edit_size} and the instance's strength (`mnm set {recipe.name} strength=…`), "
            "not edited by instruction: that is an editor's job (Kontext)."
        )
    return [
        HarnessUrl(
            label="Odysseus (running in Docker)",
            url=base_odysseus,
            note="Settings → Add Models → Add Local Models, type Image, paste the URL. The server "
            f"lists one model, `{recipe.served_name}`. Every request uses this instance's "
            "defaults: negative prompt, steps, CFG, sampler, scheduler and always-on LoRAs "
            f"(`mnm set {recipe.name} …`, `mnm lora …`); Odysseus's Quality setting does not "
            f"reach this engine. {sizing}{editing}",
            snippet=f"<sd_cpp_extra_args>{size}</sd_cpp_extra_args>\n",
        ),
        HarnessUrl(
            label=platforms.current().apps_label,
            url=base_local,
            note=f"OpenAI-compatible base URL; clients POST to {client_path(recipe)} under it.",
        ),
        HarnessUrl(
            label="Full control in a browser",
            url=f"{browser_origin(recipe)}/",
            note="The engine's own page, starting from this instance's defaults: prompt and "
            "negative prompt, size, steps, CFG, sampler, scheduler, seed, batch, CLIP skip, "
            "and this instance's LoRAs with a strength each." + browser_note(),
            opens_in_browser=True,
        ),
    ]


def video_urls(recipe: Recipe, base_local: str) -> list[HarnessUrl]:
    """Odysseus has only `llm` and `image` endpoint types, and the
    engine's video route is its own, not OpenAI's: the cards name what does talk to it."""
    args = recipe.args
    clip = (
        f"{args.frames} frames at {args.fps} fps, {args.width}x{args.height}"
        if isinstance(args, SdCppArgs) and args.frames
        else "the instance's clip settings"
    )
    return [
        HarnessUrl(
            label="Odysseus (running in Docker)",
            url=instance_origin(recipe, ODYSSEUS_HOST),
            note="Odysseus has no video endpoint type (only LLM and Image), so this instance "
            "cannot be added there; its image tools would send `/v1/images/generations`, which "
            "a video model refuses (HTTP 400). Make clips from the engine page, the curl card "
            f"or any client that POSTs to {VIDEO_PATH} (`mnm url {recipe.name} --for curl`). "
            "In a workflow, a chat model writes the shots and an image model draws the "
            "keyframes this instance animates (docs/GUIDE.md §9).",
        ),
        HarnessUrl(
            label=platforms.current().apps_label,
            url=instance_origin(recipe),
            note=f"Clients POST JSON to {VIDEO_PATH} under it (`prompt`, optional `init_image` "
            f"as base64 for image-to-video, `output_format` webm) and get a job to poll at its "
            f"`poll_url`. Every clip is {clip} unless the request says otherwise "
            f"(`mnm set {recipe.name} frames=49`).",
        ),
        HarnessUrl(
            label="Full control in a browser",
            url=f"{browser_origin(recipe)}/",
            note="The engine's own page in video mode: prompt, a first (and last) frame, size, "
            "frames, fps, steps, CFG, sampler, seed, and this instance's LoRAs." + browser_note(),
            opens_in_browser=True,
        ),
    ]


# Writes the PNG, or says what came back instead: an error reply (curl saves the body of a
# 4xx/5xx too, since the card does not pass --fail) or a page that is not JSON at all.
_SAVE_PNG = (
    "import base64,json,sys; raw=open(sys.argv[1]).read(); "
    'd=json.loads(raw) if raw.lstrip().startswith("{") else {"reply": raw[:300]}; '
    'img=(d.get("data") or [{}])[0].get("b64_json"); '
    'sys.exit(("no image in the reply: " + json.dumps(d))[:400]) if not img else None; '
    'open(sys.argv[2],"wb").write(base64.b64decode(img)); print("wrote", sys.argv[2])'
)


def repaint_curl_card(recipe: Recipe, base: str) -> HarnessUrl:
    """Every image model repaints a picture through `/v1/images/edits` (img2img at the
    instance's strength), and inpaints with a mask: white is redrawn, black kept."""
    reply = f"{recipe.name}.reply.json"
    snippet = (
        f"time curl -sS{curl_key()} {base}/images/edits -F image=@input.png -F mask=@mask.png "
        f"-F prompt='{IMAGE_PROMPT}' -o {reply} "
        f"&& python3 -c '{_SAVE_PNG}' {reply} {recipe.name}.png\n"
    )
    return HarnessUrl(
        label="Terminal: repaint or inpaint",
        url=base,
        note="Repaints input.png from the current directory; with mask.png (same size, white "
        "where to redraw, black to keep) only the white part changes. Drop `-F mask=@mask.png` "
        f"to repaint the whole picture. How much changes is `mnm set {recipe.name} strength=0.5` "
        f"(the engine's 0.75 when unset). `mnm try {recipe.name} --image input.png --mask "
        "mask.png` does the same in one step.",
        snippet=snippet,
    )


def image_curl_card(recipe: Recipe, base: str) -> HarnessUrl:
    """One paste: generate at the instance's own size (or edit `input.png`), save
    `<name>.png`, time it."""
    width, height = image_size(recipe)
    size = f"{width}x{height}"
    reply = f"{recipe.name}.reply.json"  # beside the PNG: never outside the folder it runs in
    if recipe.modality is Modality.IMAGE_EDIT:
        payload = f"-F image=@input.png -F prompt='{EDIT_PROMPT}'"
        what = "edits input.png from the current directory and writes"
    else:
        body = json.dumps({"prompt": IMAGE_PROMPT, "size": size})
        payload = f"-H 'content-type: application/json' -d '{body}'"
        what = "writes"
    snippet = (
        f"time curl -sS{curl_key()} {base}{endpoint(recipe)} {payload} -o {reply} "
        f"&& python3 -c '{_SAVE_PNG}' {reply} {recipe.name}.png\n"
    )
    return HarnessUrl(
        label="Terminal",
        url=base,
        note=f"From {platforms.current().shell_label}; {what} {recipe.name}.png (and the raw "
        f"reply, {recipe.name}.reply.json) in the current directory. "
        f"`mnm try {recipe.name}` does the same in one step.",
        snippet=snippet,
    )


# Polls the job the video route queued and writes the clip, or says what came back.
_SAVE_VIDEO = (
    "import base64,json,os,sys,time,urllib.request\n"
    'key=os.environ.get("MNM_API_KEY")\n'
    'auth={"Authorization": "Bearer " + key} if key else {}\n'
    "job=json.load(open(sys.argv[1]))\n"
    'if "poll_url" not in job: sys.exit(("no job in the reply: " + json.dumps(job))[:400])\n'
    'url=sys.argv[2]+job["poll_url"]\n'
    'while job.get("status") in ("queued", "generating"):\n'
    "    time.sleep(2); request=urllib.request.Request(url, headers=auth)\n"
    "    job=json.load(urllib.request.urlopen(request))\n"
    'result=job.get("result") or {}\n'
    'if not result.get("b64_json"): '
    'sys.exit(("the job failed: " + json.dumps(job.get("error")))[:400])\n'
    'open(sys.argv[3],"wb").write(base64.b64decode(result["b64_json"]))\n'
    'print("wrote", sys.argv[3], result.get("frame_count"), "frames at", result.get("fps"), "fps")'
)


# The request body with input.png as the first frame: base64 is too long for a shell line.
_I2V_BODY = (
    "import base64,json,sys; print(json.dumps({'prompt': sys.argv[1], 'output_format': 'webm', "
    "'init_image': base64.b64encode(open('input.png','rb').read()).decode()}))"
)


def needs_first_frame(recipe: Recipe) -> bool:
    """Image-to-video builds take a picture on every request (Wan 2.1 I2V, Wan 2.2 A14B I2V)."""
    family = recipe.image_family
    named_i2v = "i2v" in f"{recipe.repo_id} {' '.join(recipe.files)}".lower()
    return family is ImageFamily.WAN_I2V or (family is ImageFamily.WAN22_A14B and named_i2v)


def video_curl_card(recipe: Recipe) -> HarnessUrl:
    """One paste: queue a clip at the instance's own settings, poll it, save `<name>.webm`
    (from input.png as the first frame, for an image-to-video build)."""
    root = instance_origin(recipe)
    reply = f"{recipe.name}.job.json"
    if needs_first_frame(recipe):
        send = (
            f"python3 -c \"{_I2V_BODY}\" '{VIDEO_PROMPT}' | "
            f"curl -sS{curl_key()} {root}{VIDEO_PATH} -H 'content-type: application/json' -d @-"
        )
    else:
        body = json.dumps({"prompt": VIDEO_PROMPT, "output_format": "webm"})
        send = (
            f"curl -sS{curl_key()} {root}{VIDEO_PATH} -H 'content-type: application/json' "
            f"-d '{body}'"
        )
    snippet = (
        f"time ({send} -o {reply} && python3 -c '{_SAVE_VIDEO}' {reply} {root} "
        f"{recipe.name}.webm)\n"
    )
    return HarnessUrl(
        label="Terminal",
        url=root,
        note=f"From {platforms.current().shell_label}; queues the clip, polls the job and writes "
        f"{recipe.name}.webm in the current directory (minutes on 12 GB: the plan's s/video). "
        f"`mnm try {recipe.name}` does the same in one step; `mnm try {recipe.name} --image "
        "input.png` animates a picture (image-to-video).",
        snippet=snippet,
    )


# Clips plus a voice track into one file, with the ffmpeg the whisper.cpp image carries
# (nothing to install on the box; the engine images are mnm's already).
STITCH = (
    "printf \"file '%s'\\n\" shot*.webm > shots.txt && docker run --rm "
    '--user "$(id -u):$(id -g)" -v "$PWD":/w -w /w --entrypoint ffmpeg '
    f"{WHISPERCPP_IMAGE} -f concat -safe 0 -i shots.txt -i voice.wav "
    "-c:v copy -c:a libopus -shortest film.webm\n"
)


def stitch_card() -> HarnessUrl:
    return HarnessUrl(
        label="Terminal: stitch clips and a voice",
        url="",
        note="Joins the shot*.webm clips in the current directory in name order (shot01, "
        "shot02 … sort right; shot10 comes before shot2) and lays voice.wav under them, into "
        'film.webm: `mnm try <voice> --say "…"` writes <voice>.wav, so copy it to voice.wav. '
        "It runs ffmpeg from the whisper.cpp engine image, so nothing is installed.",
        snippet=STITCH,
    )
