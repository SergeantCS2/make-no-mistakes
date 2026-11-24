"""Derive everything the rest of the system needs to know about a model from Hub data.

Classification order: Hub metadata (pipeline_tag, library_name, config, gguf) first,
file layout second (mmproj, model_index.json), repo-name heuristics last.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field

from makenomistakes.hub.client import HubModel, TreeEntry
from makenomistakes.hub.converted import converted_kit, converted_variant
from makenomistakes.hub.gguf_files import (
    basename,
    extension,
    folder_of,
    gguf_model,
    gguf_variants,
    is_projector,
    is_whisper_bin,
    other_draft_heads,
    projector_rank,
    unused_mtp_heads,
    whisper_variants,
    with_companions,
    with_mtp_heads,
    with_sibling_shards,
    within,
)
from makenomistakes.hub.image_files import (
    CHECKPOINT_MIN_BYTES,
    ImageWeights,
    image_weights,
    pick_label,
)
from makenomistakes.hub.kits import VIDEO_FAMILIES, ImageFamily, detect_family
from makenomistakes.hub.modality import (
    CHAT_LIKE,
    EMBEDDING_MODALITIES,
    LLAMACPP_MODALITIES,
    MUSIC_MODALITIES,
    RETYPABLE,
    SDCPP_MODALITIES,
    SERVABLE_MODALITIES,
    TYPE_LABELS,
    TYPE_ORDER,
    VLLM_MODALITIES,
    Modality,
)
from makenomistakes.hub.model_info import (
    TEXT_ONLY_ARCHITECTURES,
    card_base_model,
    card_negative_prompt,
    model_architecture,
    param_billions,
    vision_architecture,
)
from makenomistakes.hub.music import MusicFamily, music_family, one_package
from makenomistakes.hub.music import kit_for as music_kit
from makenomistakes.hub.pipelines import PIPELINE_INDEX_FILES
from makenomistakes.hub.refs import ModelRef
from makenomistakes.hub.variants import DraftHead, GgufVariant
from makenomistakes.hub.video_files import pair_noise_halves
from makenomistakes.hub.vllm_files import pick_folder
from makenomistakes.hub.voices import VoiceFamily, voice_family, with_voice_files

__all__ = [  # the model types from `hub/modality.py`, which every importer reads here
    "CHAT_LIKE",
    "EMBEDDING_MODALITIES",
    "LLAMACPP_MODALITIES",
    "MUSIC_MODALITIES",
    "RETYPABLE",
    "SDCPP_MODALITIES",
    "SERVABLE_MODALITIES",
    "TYPE_LABELS",
    "TYPE_ORDER",
    "VLLM_MODALITIES",
    "Modality",
]


class Engine(StrEnum):
    """Which engine serves a model. Facts decide it; the recipe records it."""

    LLAMACPP = "llamacpp"
    SDCPP = "sdcpp"
    WHISPERCPP = "whispercpp"
    AUDIOCPP = "audiocpp"  # text-to-speech
    VLLM = "vllm"  # chat models that exist only as safetensors


class Format(StrEnum):
    GGUF = "gguf"
    GGML = "ggml"  # whisper.cpp's `ggml-<size>[-<quant>].bin` files
    SAFETENSORS = "safetensors"
    DIFFUSERS = "diffusers"
    ADAPTER = "adapter"  # LoRA/PEFT weights that need a base model
    UNKNOWN = "unknown"


class ModelFacts(BaseModel):
    repo_id: str
    revision: str
    display_name: str
    via: str = ""  # another source's link behind it (LM Studio, the Civitai Archive)
    modality: Modality
    format: Format
    architecture: str | None = None
    params_b: float | None = None
    context_length: int | None = None
    gated: bool = False
    license: str | None = None
    pipeline_tag: str | None = None  # the Hub's own word for the kind, kept for the guide
    base_model: str | None = None
    has_chat_template: bool = False
    supports_tools: bool = False
    reasoning_hint: bool = False
    gguf_variants: list[GgufVariant] = Field(default_factory=list)
    projector_files: list[str] = Field(default_factory=list)  # preferred first
    projector_bytes: int = 0  # size of the preferred projector (the one a recipe uses)
    # MTP heads no variant drafts with (a head without a quant label): listed, never run.
    unused_mtp_heads: list[str] = Field(default_factory=list)
    # Draft heads of the kinds mnm does not run (DSpark, DFlash, EAGLE-3): listed only.
    draft_heads: list[DraftHead] = Field(default_factory=list)
    safetensors_bytes: int = 0
    total_bytes: int = 0
    file_count: int = 0
    extensions: list[str] = Field(default_factory=list)  # what the repo holds, for guidance
    image_family: ImageFamily = ImageFamily.UNKNOWN  # image models only
    image_weights: list[ImageWeights] = Field(default_factory=list)  # oldest version first
    voice_family: VoiceFamily | None = None  # text-to-speech packages audio.cpp runs
    voice_package: str | None = None  # that package's folder in the repo (the paste)
    music_family: MusicFamily | None = None  # the studio's packages on audio.cpp
    music_package: str | None = None  # that package's folder in the repo (the paste)
    pasted_path: str | None = None  # the file or folder the reference named, if any
    negative_prompt: str = ""  # the model card's own example negative prompt, if any
    is_lora: bool = False  # small safetensors tagged or named as a LoRA
    # An image repo holding several versions (checkpoint files, diffusers folders, or GGUF
    # models): every label, and the one these facts were narrowed to (`narrow_to_version`).
    versions: list[str] = Field(default_factory=list)
    version: str | None = None
    # A diffusers repo's pipeline index at the top of the selection (`model_index.json` or
    # `modular_model_index.json`) and the Python files it ships: read for guidance,
    # never run.
    pipeline_index: str | None = None
    pipeline_code: list[str] = Field(default_factory=list)
    # A transformers model folder vLLM can load (`hub/vllm_files`): weights first, then the
    # small files beside them; empty when the repo has none.
    vllm_files: list[str] = Field(default_factory=list)
    vllm_bytes: int = 0

    def narrowed_to(self, weights: ImageWeights, labels: list[str] | None = None) -> ModelFacts:
        """These facts narrowed to one version's weights: `versions` lists every label when
        there is a choice, `version` names the pick. The one place that writes the three
        fields (`narrow_to_version`, the plan's fallback walk)."""
        labels = labels if labels is not None else [w.label for w in self.image_weights]
        several = len(labels) > 1
        return self.model_copy(
            update={
                "image_weights": [weights],
                "versions": labels if several else [],
                "version": weights.label if several else None,
            }
        )

    @property
    def default_image_weights(self) -> ImageWeights | None:
        """The newest version; without version numbers, the smallest file (an fp16 or
        EMA-only build of the same weights)."""
        if not self.image_weights:
            return None
        return max(self.image_weights, key=lambda w: (w.version, -w.size_bytes))

    @property
    def engine(self) -> Engine | None:
        """The engine that can serve these files, or None (then guidance applies)."""
        if self.format is Format.GGML:
            return Engine.WHISPERCPP if self.modality is Modality.SPEECH_TO_TEXT else None
        if self.modality is Modality.TEXT_TO_SPEECH:
            # audio.cpp runs the families its kit table names, from GGUF packages
            return Engine.AUDIOCPP if self.voice_family is not None and self.gguf_variants else None
        if self.modality in MUSIC_MODALITIES:  # the studio's packages, audio.cpp too
            return Engine.AUDIOCPP if self.music_family is not None and self.gguf_variants else None
        if self.modality in SDCPP_MODALITIES:
            if self.image_weights and not self.gguf_variants:
                return Engine.SDCPP  # the family comes from the checkpoint's header at plan time
            # GGUF of a known family, whatever the tags say: a GGUF repo tagged `diffusers`
            # is servable too.
            known = bool(self.gguf_variants) and self.image_family is not ImageFamily.UNKNOWN
            return Engine.SDCPP if known else None
        return self._language_engine()

    def _language_engine(self) -> Engine | None:
        """A GGUF runs on llama.cpp; a repo with only safetensors weights on vLLM."""
        if self.format is Format.GGUF and self.modality in LLAMACPP_MODALITIES:
            return Engine.LLAMACPP
        if self.format is Format.SAFETENSORS and self.modality in VLLM_MODALITIES:
            return Engine.VLLM if self.vllm_files else None
        return None

    @property
    def servable(self) -> bool:
        return self.engine is not None

    @property
    def family_name(self) -> str:
        """Display name without the publisher's -GGUF suffix: `Qwen3-8B`, `gemma-3-4b-it`."""
        name = self.display_name
        return name[:-5] if name.lower().endswith("-gguf") else name

    def variant(self, quant: str) -> GgufVariant | None:
        wanted = quant.upper()
        return next((v for v in self.gguf_variants if v.quant.upper() == wanted), None)


_PIPELINE_TO_MODALITY = {
    "text-generation": Modality.CHAT,
    "image-text-to-text": Modality.VISION_CHAT,
    "image-to-text": Modality.VISION_CHAT,
    "text-to-image": Modality.IMAGE_GEN,
    "image-to-image": Modality.IMAGE_GEN,
    "text-to-video": Modality.VIDEO,
    "image-to-video": Modality.VIDEO,
    "automatic-speech-recognition": Modality.SPEECH_TO_TEXT,
    "text-to-speech": Modality.TEXT_TO_SPEECH,
    "feature-extraction": Modality.EMBEDDING,
    "sentence-similarity": Modality.EMBEDDING,
    "text-ranking": Modality.RERANK,
}
# A chat model trained or tuned for code: coding agents are its consumers.
_CODING_RE = re.compile(
    r"(?<!\bde)(?<!\ben)coder|codex|devstral|codestral|codegemma|codellama|codeqwen|"
    r"(?<![a-z])code(?![a-z])",  # OpenCoder, WizardCoder, Magicoder; not "decoder"/"encoder"
    re.I,
)
# Image models that take an image in: edit-trained (Kontext, Fill), inpainting, Qwen-Image-Edit.
_IMAGE_EDIT_RE = re.compile(
    r"kontext|(?<![a-z])fill(?![a-z])|inpaint|qwen[-_]?image[-_]?edit", re.I
)
_IMAGE_TOKEN_RE = re.compile(r"flux|qwen[-_]?image|sdxl|stable[-_]?diffusion|\bsd\d", re.I)
_NAME_HINTS: tuple[tuple[re.Pattern[str], Modality], ...] = (
    (re.compile(r"whisper|parakeet|\basr\b", re.I), Modality.SPEECH_TO_TEXT),
    (re.compile(r"\btts\b|-tts|kokoro|audio\.cpp", re.I), Modality.TEXT_TO_SPEECH),
    (re.compile(r"rerank", re.I), Modality.RERANK),
    (re.compile(r"embed|\bbge\b|\be5\b|minilm", re.I), Modality.EMBEDDING),
    (re.compile(r"stable-diffusion|\bsd\d", re.I), Modality.IMAGE_GEN),
    (re.compile(r"-vl-|-vl\b|vision|\bvlm\b", re.I), Modality.VISION_CHAT),
)


_LLAMA_HINTED = frozenset({Modality.RERANK, Modality.EMBEDDING, Modality.VISION_CHAT})


def _natural(label: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key that orders `preview2` before `preview10`."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", label)
        if part
    )


def narrow_to_version(
    facts: ModelFacts, wanted: str | None, quant: str | None = None
) -> ModelFacts:
    """An image repo narrowed to one version: `wanted` (`--version`, matched by `pick_label`)
    or the default. Checkpoint files and diffusers folders default to the newest version
    (`ModelFacts.default_image_weights`); a GGUF repo of several models to the last in
    natural order (`anima-preview3-base` after `anima-preview2`) that holds `quant` (a
    reference's `:Q3_K_L` tag) when one is given. ValueError when `wanted` matches nothing,
    or the repo has no versions to pick from."""
    if facts.image_weights and not facts.gguf_variants:
        labels = [w.label for w in facts.image_weights]
        default = facts.default_image_weights
        chosen = pick_label(labels, wanted) if wanted else default.label if default else labels[0]
        return facts.narrowed_to(next(w for w in facts.image_weights if w.label == chosen), labels)
    models = sorted({gguf_model(v) for v in facts.gguf_variants}, key=_natural)
    if facts.modality not in SDCPP_MODALITIES or len(models) < 2:
        if wanted:
            what = "quants: pick one with --quant" if facts.gguf_variants else "one model"
            raise ValueError(f"{facts.repo_id} has no versions to pick from (it holds {what})")
        return facts
    holding = [
        m
        for m in models
        if quant is None
        or any(gguf_model(v) == m and v.quant.upper() == quant.upper() for v in facts.gguf_variants)
    ]
    chosen = pick_label(models, wanted) if wanted else (holding or models)[-1]
    return facts.model_copy(
        update={
            "gguf_variants": [v for v in facts.gguf_variants if gguf_model(v) == chosen],
            "versions": models,
            "version": chosen,
        }
    )


def build_facts(info: HubModel, tree: list[TreeEntry], ref: ModelRef) -> ModelFacts:
    files = [e for e in tree if e.is_file and within(e.path, ref.path)]
    files = with_sibling_shards(files, tree, ref.path)
    files = with_companions(files, tree, ref.path)
    variants = with_mtp_heads(pair_noise_halves(gguf_variants(files)), files) or whisper_variants(
        files
    )
    modality_hint = _PIPELINE_TO_MODALITY.get(info.pipeline_tag or "")
    # A voice package is one folder of the repo, named by the paste: the family is read from
    # that folder's name only, and every variant must sit in it (the repo root or a parent
    # folder holds a hundred packages; the first file's family would pick one at random).
    tts = modality_hint is Modality.TEXT_TO_SPEECH or "text-to-speech" in info.tags
    package = ref.path.strip("/") if tts and ref.path else None
    one_folder = bool(variants) and all(folder_of(v.files[0]) == package for v in variants)
    # The studio's packages share the repo: a music folder is read as music first.
    music = music_family(package) if package and one_package(variants, package) else None
    if (converted := converted_kit(info.id, ref.path)) is not None:  # mnm converts it
        variants = [converted_variant(converted, {e.path: e.size for e in files})]
        music = converted.kit.family
    voices = voice_family(package) if package is not None and one_folder and not music else None
    if voices is not None:
        variants = with_voice_files(variants, {e.path: e.size for e in files}, package or "")
    text_only = model_architecture(info) in TEXT_ONLY_ARCHITECTURES
    projector_entries = sorted(
        (e for e in files if is_projector(e.path) and not text_only),
        key=projector_rank,
    )
    projectors = [e.path for e in projector_entries]
    safetensors_bytes = sum(e.size for e in files if e.path.endswith(".safetensors"))
    fmt = _format(info, files, variants, safetensors_bytes)
    modality = _modality(info, fmt, projectors)
    if text_only and modality is Modality.VISION_CHAT:
        modality = Modality.CHAT
    if music is not None:
        modality = music_kit(music).modality
    template = info.chat_template or ""
    diffusion = modality in SDCPP_MODALITIES
    weights = image_weights(files, ref.path) if diffusion else []
    served, folders = (
        pick_folder(files, ref.path) if fmt is Format.SAFETENSORS and not diffusion else ([], [])
    )
    sizes = {e.path: e.size for e in files}

    return ModelFacts(
        pipeline_tag=info.pipeline_tag,
        repo_id=info.id,
        revision=ref.revision,
        # A voice package is one folder of a repo of many: named by the folder it was pasted
        # as (`PocketTTS-GGUF/english`), so the instance's default name says which.
        display_name=(
            ref.path if (voices or music) is not None and ref.path else info.id.split("/")[-1]
        ),
        modality=modality,
        format=fmt,
        architecture=model_architecture(info),
        params_b=param_billions(info),
        context_length=info.gguf.context_length,
        gated=info.is_gated,
        license=info.card_data.license,
        base_model=card_base_model(info),
        has_chat_template=bool(template),
        supports_tools="tools" in template,
        reasoning_hint=any(k in template for k in ("<think>", "enable_thinking", "reasoning")),
        gguf_variants=variants,
        projector_files=projectors,
        projector_bytes=projector_entries[0].size if projector_entries else 0,
        unused_mtp_heads=unused_mtp_heads(variants, files),
        draft_heads=other_draft_heads(files),
        safetensors_bytes=safetensors_bytes,
        total_bytes=sum(e.size for e in files),
        file_count=len(files),
        extensions=sorted({extension(e.path) for e in files} - {""}),
        image_family=(
            detect_family(info.id, tuple(v.files[0] for v in variants))
            if diffusion
            else ImageFamily.UNKNOWN
        ),
        image_weights=weights,
        voice_family=voices,
        voice_package=package if voices is not None else None,
        music_family=music,
        music_package=package if music is not None else None,
        pasted_path=ref.path or None,
        negative_prompt=card_negative_prompt(info),
        is_lora=diffusion and _looks_like_lora(info, files, weights, variants),
        pipeline_index=_pipeline_index(files, ref.path) if fmt is Format.DIFFUSERS else None,
        pipeline_code=sorted(e.path for e in files if e.path.endswith(".py"))[:5],
        vllm_files=served,
        vllm_bytes=sum(sizes[f] for f in served),
        **({"versions": folders, "version": folders[0]} if folders else {}),
    )


def _pipeline_index(files: list[TreeEntry], path: str | None) -> str | None:
    paths = {e.path for e in files}
    prefix = f"{path.strip('/')}/" if path else ""
    return next((prefix + n for n in PIPELINE_INDEX_FILES if prefix + n in paths), None)


def _looks_like_lora(
    info: HubModel,
    files: list[TreeEntry],
    weights: list[ImageWeights],
    variants: list[GgufVariant],
) -> bool:
    small = [e for e in files if e.path.endswith(".safetensors") and e.size < CHECKPOINT_MIN_BYTES]
    named = "lora" in info.id.lower() or any("lora" in tag.lower() for tag in info.tags)
    return bool(small) and named and not weights and not variants


def _format(
    info: HubModel, files: list[TreeEntry], variants: list[GgufVariant], safetensors_bytes: int
) -> Format:
    names = {basename(e.path) for e in files}
    if "adapter_config.json" in names or "peft" in info.library_name_lower:
        return Format.ADAPTER
    if "model_index.json" in names or info.library_name == "diffusers":
        return Format.DIFFUSERS
    if variants:
        return Format.GGML if is_whisper_bin(variants[0].files[0]) else Format.GGUF
    if safetensors_bytes:
        return Format.SAFETENSORS
    return Format.UNKNOWN


def _modality(info: HubModel, fmt: Format, projectors: list[str]) -> Modality:
    modality = _modality_by_metadata(info, fmt, projectors)
    if modality is Modality.CHAT and _CODING_RE.search(info.id):
        return Modality.CODING  # the same engine; the cards and the agent profile differ
    if modality is Modality.IMAGE_GEN and detect_family(info.id) in VIDEO_FAMILIES:
        return Modality.VIDEO  # a Wan repo tagged text-to-image or `diffusers`
    if modality is Modality.IMAGE_GEN and _IMAGE_EDIT_RE.search(info.id):
        return Modality.IMAGE_EDIT  # needs an image in: not the generate route
    return modality


def _modality_by_metadata(info: HubModel, fmt: Format, projectors: list[str]) -> Modality:
    if fmt is Format.DIFFUSERS:
        return Modality.IMAGE_GEN
    by_tag = _PIPELINE_TO_MODALITY.get(info.pipeline_tag or "")
    if by_tag is Modality.CHAT and (projectors or vision_architecture(info)):
        return Modality.VISION_CHAT
    if by_tag is not None:
        return by_tag
    if projectors:
        return Modality.VISION_CHAT
    # An image GGUF carries no architecture, so a Hub-reported one makes this a llama.cpp
    # model whatever the name says (`…-Z-Image-Turbo-…` is Z-Image's text encoder): only the
    # llama.cpp types' hints apply then (an untagged embedding or reranker GGUF is not chat).
    by_name = _llama_name_hint(info.id) if info.gguf.architecture else _modality_from_name(info.id)
    if by_name is not None:
        return by_name
    return Modality.CHAT if model_architecture(info) else Modality.UNKNOWN


def _modality_from_name(repo_id: str) -> Modality | None:
    edit_named = _IMAGE_EDIT_RE.search(repo_id) and _IMAGE_TOKEN_RE.search(repo_id)
    family = detect_family(repo_id)
    if family in VIDEO_FAMILIES:
        return Modality.VIDEO
    if family is not ImageFamily.UNKNOWN or edit_named:
        return Modality.IMAGE_GEN  # `_modality` tells an edit-trained one apart
    return next((m for pattern, m in _NAME_HINTS if pattern.search(repo_id)), None)


def _llama_name_hint(repo_id: str) -> Modality | None:
    """A GGUF with an architecture runs on llama.cpp: an embedding, rerank or vision name
    still says which kind of instance (Qwen3-Embedding, bge-reranker), an image one does not."""
    return next(
        (m for pattern, m in _NAME_HINTS if m in _LLAMA_HINTED and pattern.search(repo_id)),
        None,
    )
