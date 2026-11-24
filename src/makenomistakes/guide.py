"""When a reference cannot be served, say what it is and what to use instead.

Every branch produces a `Guidance` with a one-paragraph explanation and, where the Hub can
tell us, concrete alternatives (most-downloaded GGUF conversions of the same base model,
or GGUF repos matching an Ollama-style name).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from makenomistakes.hub import HubClient, HubError, ModelFacts, SearchHit
from makenomistakes.hub.facts import Format, Modality
from makenomistakes.hub.kits import FAMILY_LABELS, KITS, ImageFamily, ImageKit, detect_family
from makenomistakes.hub.model_info import GGUF_ARCHITECTURES
from makenomistakes.hub.music import music_family
from makenomistakes.hub.pipelines import parse_pipeline_index, pipeline_family
from makenomistakes.hub.refs import ollama_link
from makenomistakes.hub.voices import AUDIOCPP_REPO, kit_for

_MODALITY_LABEL = {
    Modality.IMAGE_GEN: "an image generation",
    Modality.IMAGE_EDIT: "an image editing",
    Modality.VIDEO: "a video",
    Modality.SPEECH_TO_TEXT: "a speech-to-text",
    Modality.TEXT_TO_SPEECH: "a text-to-speech",
}
_RUNNABLE = ", ".join(k.label for k in KITS.values() if not k.min_vram_bytes and not k.is_video)
_RUNNABLE_VIDEO = ", ".join(k.label for k in KITS.values() if k.is_video)
_ENGINE_PHASE = {
    Modality.IMAGE_GEN: f"Image generation runs these families: {_RUNNABLE}. Each runs from "
    "GGUF or a single safetensors file (a checkpoint, or a diffusion model such as ComfyUI's "
    "`split_files/diffusion_models/`); SD 1.x and SDXL also from diffusers folders.",
    Modality.TEXT_TO_SPEECH: "Text-to-speech runs on audio.cpp's server (`/v1/audio/speech`) "
    "from its GGUF packages: paste a folder of `audio-cpp/audio.cpp-gguf` "
    "(`PocketTTS-GGUF/english`, `Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF`; docs/GUIDE.md §10). "
    "The studio's songs and stems come from the same repo (`ACE-Step1.5-GGUF/turbo`, "
    "`BS-RoFormer-ep368-GGUF`, `HTDemucs-6stems-GGUF`; docs/GUIDE.md §28). Other voice models "
    "and other formats are not run here.",
    Modality.IMAGE_EDIT: "Image editing (an image and an instruction in, an image out) runs "
    "through `/v1/images/edits` on FLUX.1 Kontext and on FLUX.2 klein, which edits natively; "
    "FLUX.1 Fill and Qwen-Image-Edit (20B) have no kit here (docs/GUIDE.md §8).",
    Modality.VIDEO: f"Video generation runs these families: {_RUNNABLE_VIDEO}, each from GGUF "
    "or a single diffusion-model safetensors file, through `/sdcpp/v1/vid_gen` "
    "(docs/GUIDE.md §9). Speech-to-video (Wan 2.2 S2V) and audio-video (MiniMax-H3) models "
    "need an audio track in and have no kit.",
}
WHISPER_CPP_REPO = "ggerganov/whisper.cpp"
# A text encoder's GGUF tagged with the image model it serves (`…/Qwen3-4b-Z-Image-Turbo-…`).
_TEXT_ENCODER_RE = re.compile(r"qwen\d|umt5|t5[-_]|clip[-_]|gemma|text[-_]?encoder", re.I)


class Alternative(BaseModel):
    repo_id: str
    downloads: int
    note: str


class Guidance(BaseModel):
    what_it_is: str
    what_to_do: str
    alternatives: list[Alternative] = Field(default_factory=list)


def guide_voice_image(facts: ModelFacts) -> Guidance | None:
    """A voice package the engine's official image cannot speak with (Kokoro), or None."""
    kit = kit_for(facts.voice_family) if facts.voice_family is not None else None
    if kit is None or not kit.missing:
        return None
    pocket = f"{AUDIOCPP_REPO}/PocketTTS-GGUF/english"
    return Guidance(
        what_it_is=f"{facts.display_name} is {kit.label}: {kit.missing}.",
        what_to_do=f"Use PocketTTS (`mnm add {pocket} --name pocket`, English voices) or "
        "Qwen3-TTS: both run from the official image. Kokoro can return once an audio.cpp "
        "build ships eSpeak-ng (`mnm engines update audiocpp`).",
        alternatives=[Alternative(repo_id=pocket, downloads=0, note="PocketTTS, English")],
    )


def guide_unservable(facts: ModelFacts, hub: HubClient) -> Guidance:
    """Called when a resolved repo has no GGUF we can run."""
    guidance = _guide_unservable(facts, hub)
    return guidance.model_copy(update={"alternatives": _not_itself(facts, guidance.alternatives)})


def _guide_unservable(facts: ModelFacts, hub: HubClient) -> Guidance:
    if facts.pasted_path and facts.file_count == 0:  # before any kind: nothing was read
        return Guidance(
            what_it_is=f"'{facts.pasted_path}' is not a file or folder in {facts.repo_id}.",
            what_to_do="A quant goes after a colon (`org/name:Q4_K_M`); a file or folder "
            "path is spelled as the repo's tree shows it.",
            alternatives=[],
        )
    if facts.is_lora:
        base = facts.base_model or "the base model it was trained on"
        return Guidance(
            what_it_is=f"{facts.display_name} is a LoRA: a small add-on trained for one style, "
            f"character or concept, applied on top of a checkpoint of {base}. It does not run "
            "on its own.",
            what_to_do="Add an image instance of a matching base (an SDXL LoRA needs an SDXL "
            "checkpoint such as Illustrious or Pony), then `mnm lora add <instance> "
            f"{facts.repo_id}`. Its weight is set per image or as an always-on default.",
        )
    if facts.format is Format.ADAPTER:
        base = facts.base_model or "its base model"
        return Guidance(
            what_it_is=f"{facts.display_name} is a LoRA/PEFT adapter: a small set of weights "
            f"that modifies {base}, not a runnable model on its own.",
            what_to_do="Run the base model instead (a GGUF of it is listed below if the Hub "
            "has one). mnm does not run an adapter on top of its base model yet.",
            alternatives=_gguf_conversions(hub, facts.base_model) if facts.base_model else [],
        )
    by_modality = _guide_by_modality(facts, hub)
    if by_modality is not None:
        return by_modality
    return _guide_by_format(facts, hub)


_VIDEO_TAGS = frozenset({"text-to-video", "image-to-video", "video-to-video"})


def _guide_by_format(facts: ModelFacts, hub: HubClient) -> Guidance:
    """Nothing an engine here runs: say what the files are and where to look."""
    if facts.pipeline_tag in _VIDEO_TAGS or facts.modality is Modality.VIDEO:
        return _video_guidance(facts, hub)
    if facts.format is Format.GGUF:
        return Guidance(
            what_it_is=f"{facts.display_name} has GGUF files, but the Hub metadata does not say "
            "what kind of model it is (no pipeline tag, no architecture).",
            what_to_do="Open the repo page and check the model card. If it is a chat, vision, "
            "embedding or rerank model, a better-labelled GGUF repo of the same family is "
            "listed below; otherwise it is not something this tool can run yet.",
            alternatives=_gguf_search(hub, facts.family_name),
        )
    if facts.format is Format.SAFETENSORS:
        return Guidance(
            what_it_is=f"{facts.display_name} ships as {facts.format.value} weights "
            f"({facts.safetensors_bytes / 2**30:.1f} GiB), the training/transformers format.",
            what_to_do="llama.cpp needs the GGUF format. Pick a GGUF conversion of the same "
            "model (most downloaded first). Without a GGUF, vLLM runs chat, coding and vision "
            "models whose configuration it knows (docs/GUIDE.md §7.1); this one is not among "
            "them (an embedding or audio-in model, a brand-new architecture, or custom code).",
            alternatives=_gguf_conversions(hub, facts.repo_id),
        )
    found = ", ".join(facts.extensions) or "no files"
    return Guidance(
        what_it_is=f"{facts.display_name} contains {facts.file_count} files ({found}) and no "
        "model weights in a format this tool knows (GGUF or safetensors).",
        what_to_do="Check the repo page on huggingface.co: it may be a dataset, a space, a "
        "tokenizer-only repo, or weights in an unsupported format. Paste a GGUF repo instead.",
        alternatives=_gguf_search(hub, facts.family_name),
    )


def _video_guidance(facts: ModelFacts, hub: HubClient) -> Guidance:
    """A video model no kit here runs: an audio-in family, or one sd.cpp does not load."""
    family = facts.image_family
    what = (
        f"{facts.display_name} is a speech-to-video model: it animates a portrait to a WAV "
        "track through an audio encoder, which no kit here loads."
        if family is ImageFamily.WAN22_S2V
        else f"{facts.display_name} is an audio-video model (a soundtrack in and out), which "
        "no kit here loads."
        if family is ImageFamily.MINIMAX_H3
        else f"{facts.display_name} is a video model (the Hub tags it "
        f"`{facts.pipeline_tag or 'video'}`), but not one of the families this tool has a kit "
        "for."
    )
    return Guidance(
        what_it_is=what,
        what_to_do=_ENGINE_PHASE[Modality.VIDEO] + " Paste one of the repos below.",
        alternatives=[
            a for a in _gguf_search(hub, "Wan2.2-TI2V-5B-GGUF") if a.repo_id != facts.repo_id
        ],
    )


def _guide_by_modality(facts: ModelFacts, hub: HubClient) -> Guidance | None:  # noqa: PLR0911
    """Kinds of model an engine here could run, in a form it cannot (yet)."""
    pipeline = _guide_pipeline(facts, hub) if facts.format is Format.DIFFUSERS else None
    if pipeline is not None:
        return pipeline
    if facts.modality is Modality.VIDEO:
        return _video_guidance(facts, hub)
    if facts.modality is Modality.IMAGE_GEN and facts.format is Format.GGUF:
        known = _RUNNABLE
        return Guidance(
            what_it_is=f"{facts.display_name} is an image generation model in GGUF form, "
            + (
                f"a {KITS[facts.image_family].label} build that needs more VRAM than this "
                "machine has."
                if facts.image_family in KITS
                else "but not one of the families this tool has a kit for."
            ),
            what_to_do=f"Families that run here today: {known}. Paste one of the repos below "
            "(the runner fetches the VAE and text encoder it needs).",
            alternatives=_gguf_search(hub, _first_runnable_kit_query()),
        )
    if facts.modality is Modality.SPEECH_TO_TEXT:
        return Guidance(
            what_it_is=f"{facts.display_name} is a speech-to-text model in "
            f"{facts.format.value} form; whisper.cpp, the engine here, runs "
            "`ggml-<size>.bin` files.",
            what_to_do=f"Paste `{WHISPER_CPP_REPO}` (every official size, quantised too) or one "
            "of the GGML conversions below; `mnm add` picks a size that fits.",
            alternatives=[
                Alternative(repo_id=WHISPER_CPP_REPO, downloads=0, note="whisper.cpp's own models"),
                *(
                    a
                    for a in _ggml_conversions(hub, facts.display_name)
                    if a.repo_id != facts.repo_id
                ),
            ],
        )
    if facts.modality in _ENGINE_PHASE:
        if (builds := _music_builds(facts)) is not None:
            return builds
        gguf_note = (
            " It is in GGUF form, but not one this tool's engines load for that kind of model."
            if facts.format is Format.GGUF and facts.modality is not Modality.TEXT_TO_SPEECH
            else ""
        )
        later = (
            ""
            if facts.modality is Modality.TEXT_TO_SPEECH  # its text names the packages
            else " Paste a repo of one of those."
        )
        return Guidance(
            what_it_is=f"{facts.display_name} is {_MODALITY_LABEL[facts.modality]} "
            f"model ({facts.format.value} files).{gguf_note}",
            what_to_do=_ENGINE_PHASE[facts.modality] + later,
        )
    return None


def _music_builds(facts: ModelFacts) -> Guidance | None:
    """A studio package's parent folder: `ACE-Step1.5-GGUF` holds `base/` and
    `turbo/`, so the paste names no one build."""
    pasted = (facts.pasted_path or "").strip("/")
    if not pasted or music_family(pasted) is None:
        return None
    folders = sorted({v.files[0].rsplit("/", 1)[0] for v in facts.gguf_variants})
    inside = [f for f in folders if f.startswith(f"{pasted}/")]
    if not inside:
        return None
    named = ", ".join(f"`{facts.repo_id}/{f}`" for f in inside)
    return Guidance(
        what_it_is=f"{pasted} holds {len(inside)} builds of one model, each its own package.",
        what_to_do=f"Paste one of them: {named} (docs/GUIDE.md §28).",
    )


def _guide_pipeline(facts: ModelFacts, hub: HubClient) -> Guidance | None:
    """A diffusers folder of a transformer model: the family from its `model_index.json`,
    and the repos whose files run here. None when the index cannot be read."""
    if facts.pipeline_index is None:
        return None
    try:
        index = parse_pipeline_index(
            hub.json_file(facts.repo_id, facts.pipeline_index, facts.revision)
        )
    except (HubError, ValueError):
        return None
    family = pipeline_family(index, facts.repo_id)
    label = FAMILY_LABELS[family] if family is not ImageFamily.UNKNOWN else index.label
    code = [*index.custom_components, *facts.pipeline_code]
    what = (
        f"{facts.display_name} is a diffusers pipeline ({index.class_name}): {label}, with "
        "its transformer, text encoder and VAE in separate folders."
    )
    if code:
        what += f" It ships Python ({', '.join(code)}), which this tool never runs."
    kit = KITS.get(family)
    if kit is not None and kit.min_vram_bytes:
        return Guidance(
            what_it_is=what,
            what_to_do=f"{kit.label} needs about {kit.min_vram_bytes // 2**30} GB of VRAM in "
            f"any form ({kit.note}), more than this machine has.",
        )
    if kit is None or not kit.runs_from:
        why = (
            f"There is no {label} kit here yet."
            if family is not ImageFamily.UNKNOWN
            else f"{label} changes a picture you give it. {_ENGINE_PHASE[Modality.IMAGE_EDIT]}"
            if index.edits
            else f"{label} is not a family this tool runs."
        )
        return Guidance(what_it_is=what, what_to_do=f"{why} {_ENGINE_PHASE[Modality.IMAGE_GEN]}")
    return Guidance(
        what_it_is=what,
        what_to_do="stable-diffusion.cpp loads only Stable Diffusion UNets from diffusers "
        f"folders; it runs {label} from single files (a diffusers transformer is often fp32 "
        "shards: 24.6 GB for Z-Image, against 3.5 GB as GGUF). Paste one of the repos below; "
        "the runner adds the text encoder and VAE.",
        alternatives=_pipeline_routes(facts, kit, family, hub),
    )


def _pipeline_routes(
    facts: ModelFacts, kit: ImageKit, family: ImageFamily, hub: HubClient
) -> list[Alternative]:
    """The base model when the repo names one, the kit's own repos, then GGUF conversions of
    this repo that are the same family; each once, at most five."""
    routes: list[Alternative] = []
    if facts.base_model and facts.base_model != facts.repo_id:
        routes.append(Alternative(repo_id=facts.base_model, downloads=0, note="its base model"))
    routes += [Alternative(repo_id=r, downloads=0, note=f"{kit.label}") for r in kit.runs_from]
    routes += [
        a
        for a in _gguf_conversions(hub, facts.repo_id)
        if detect_family(a.repo_id) is family and not _TEXT_ENCODER_RE.search(a.repo_id)
    ]
    unique: dict[str, Alternative] = {}
    for route in routes:
        unique.setdefault(route.repo_id, route)
    return list(unique.values())[:5]


def _first_runnable_kit_query() -> str:
    """The lightest family the kit table can run, as a Hub search phrase."""
    kit = next(k for k in KITS.values() if not k.min_vram_bytes)
    return kit.label.split(" (")[0].lower() + " gguf"


def guide_ollama_tag(tag: str, hub: HubClient) -> Guidance:
    """`qwen2.5:14b` -> the GGUF repos on the Hub that match the name and size."""
    name, _, size = tag.partition(":")
    query = f"{name} {size}".strip().replace("-instruct", "")
    return Guidance(
        what_it_is=f"'{tag}' is an Ollama library tag, not a Hugging Face repo.",
        what_to_do=f"Paste its link ({ollama_link(tag)}) to run Ollama's own build, or "
        "pick one of the matching GGUF repos below (same model family, Instruct variants are "
        "the chat ones).",
        alternatives=_gguf_search(hub, query),
    )


def gguf_builds(facts: ModelFacts, hub: HubClient) -> Guidance:
    """A model too large for the engine that would serve it as it is: its GGUF builds."""
    return Guidance(
        what_it_is=f"{facts.repo_id} is {facts.modality.a_label} model in safetensors.",
        what_to_do="Plan a GGUF build of it with `mnm fit <repo>`: the plan says what it needs."
        + _layout_note(facts),
        alternatives=_not_itself(facts, _gguf_conversions(hub, facts.repo_id)),
    )


def _layout_note(facts: ModelFacts) -> str:
    """A model whose GGUFs come from several converters, of which llama.cpp reads one."""
    if GGUF_ARCHITECTURES.get(facts.architecture or "") != "deepseek41":
        return ""
    return (
        " Its GGUFs come in three layouts and official llama.cpp reads only the mainline "
        "converter's (unsloth's `DeepSeek-V4.1-Flash-1017-GGUF`); `mnm fit` refuses the others."
    )


def _gguf_conversions(hub: HubClient, base_repo: str | None) -> list[Alternative]:
    if not base_repo:
        return []
    hits = _search(hub, filters=(f"base_model:quantized:{base_repo}", "gguf"))
    if not hits:
        hits = _search(hub, search=base_repo.split("/")[-1], filters=("gguf",))
    # Fine-tunes are tagged as quantized from the same base; keep the plain conversions
    # (the base model's name in the repo name) when there are any.
    base_name = base_repo.split("/")[-1].lower()
    plain = [h for h in hits if base_name in h.id.split("/")[-1].lower()]
    return [_alternative(h, "GGUF conversion") for h in (plain or hits)]


def _ggml_conversions(hub: HubClient, name: str) -> list[Alternative]:
    """`whisper-large-v3` -> repos named like `…whisper-large-v3…-ggml` (not ONNX re-exports)."""
    hits = _search(hub, search=f"{name}-ggml")
    return [
        _alternative(h, "GGML conversion")
        for h in hits
        if h.id != WHISPER_CPP_REPO and "onnx" not in h.id.lower()
    ]


def _gguf_search(hub: HubClient, query: str) -> list[Alternative]:
    return [_alternative(h, "GGUF repo") for h in _search(hub, search=query, filters=("gguf",))]


def _search(
    hub: HubClient, *, search: str | None = None, filters: tuple[str, ...] = ()
) -> list[SearchHit]:
    try:
        return hub.search(search=search, filters=filters, limit=5)
    except HubError:
        return []


def _not_itself(facts: ModelFacts, alternatives: list[Alternative]) -> list[Alternative]:
    """The repo just refused is never its own alternative."""
    return [a for a in alternatives if a.repo_id != facts.repo_id]


def _alternative(hit: SearchHit, note: str) -> Alternative:
    return Alternative(repo_id=hit.id, downloads=hit.downloads, note=note)
