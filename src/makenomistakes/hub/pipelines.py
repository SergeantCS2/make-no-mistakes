"""A diffusers repo's pipeline index: the pipeline class, and each folder's class.

`model_index.json` (`modular_model_index.json` for a modular pipeline) names the pipeline
class (`ZImagePipeline`, `AnimaModularPipeline` …) and, per component folder, the library and
class that load it. The runner reads it to name the family and the files that run here
instead. It never runs a repo's Python: a `pipeline.py`, a custom component module
(`["modeling_llm_adapter", "AnimaLLMAdapter"]`) or modular blocks are only named.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from makenomistakes.hub.kits import ImageFamily, detect_family

PIPELINE_INDEX_FILES = ("model_index.json", "modular_model_index.json")
_LIBRARIES = frozenset({"diffusers", "transformers"})

# Text-to-image pipeline classes of the families the runner knows (exact names: FluxFill,
# FluxKontext and the img2img/inpaint classes are other tasks). SD 2.x shares
# StableDiffusionPipeline with SD 1.x; the UNet header tells them apart.
PIPELINE_FAMILIES: dict[str, ImageFamily] = {
    "StableDiffusionPipeline": ImageFamily.SD1,
    "StableDiffusionXLPipeline": ImageFamily.SDXL,
    "StableDiffusion3Pipeline": ImageFamily.SD3,
    "ZImagePipeline": ImageFamily.Z_IMAGE,
    "FluxPipeline": ImageFamily.FLUX1,
    "Flux2KleinPipeline": ImageFamily.FLUX2_KLEIN,  # 9B: by name, as for GGUF repos
    "Flux2Pipeline": ImageFamily.FLUX2,
    "QwenImagePipeline": ImageFamily.QWEN_IMAGE,
    "ChromaPipeline": ImageFamily.CHROMA,
    "QwenImage21Pipeline": ImageFamily.QWEN_IMAGE_2_1,
    "ErnieImagePipeline": ImageFamily.ERNIE_IMAGE,
    "AnimaPipeline": ImageFamily.ANIMA,
    "AnimaTextToImagePipeline": ImageFamily.ANIMA,
    "AnimaModularPipeline": ImageFamily.ANIMA,
    "WanPipeline": ImageFamily.WAN,
    "WanImageToVideoPipeline": ImageFamily.WAN_I2V,
    "LingBotVideoPipeline": ImageFamily.LINGBOT_VIDEO,
    "HunyuanVideo15Pipeline": ImageFamily.HUNYUAN_VIDEO,
    "LTX2Pipeline": ImageFamily.LTX2,
}
_EDIT_WORDS = ("Edit", "Kontext", "Img2Img", "Inpaint", "Fill")
# Readable names for pipelines of models the runner has no family for.
PIPELINE_LABELS: dict[str, str] = {
    "PixArtSigmaPipeline": "PixArt-Sigma",
    "PixArtAlphaPipeline": "PixArt-Alpha",
    "SanaPipeline": "Sana",
    "Lumina2Pipeline": "Lumina Image 2.0",
    "HiDreamImagePipeline": "HiDream-I1",
    "HunyuanDiTPipeline": "Hunyuan-DiT",
    "CogView4Pipeline": "CogView4",
    "KolorsPipeline": "Kolors",
    "AuraFlowPipeline": "AuraFlow",
    "FluxKontextPipeline": "FLUX.1 Kontext (image editing)",
    "FluxFillPipeline": "FLUX.1 Fill (inpainting)",
    "QwenImageEditPipeline": "Qwen-Image-Edit (image editing)",
    "QwenImageEditPlusPipeline": "Qwen-Image-Edit Plus (image editing)",
    "StableDiffusionXLImg2ImgPipeline": "an SDXL image-to-image stage (the refiner)",
}


class PipelineComponent(BaseModel):
    library: str | None = None  # "diffusers", "transformers", or a module in the repo
    class_name: str | None = None


class PipelineIndex(BaseModel):
    class_name: str
    diffusers_version: str | None = None
    blocks_class_name: str | None = None  # a modular pipeline's blocks
    components: dict[str, PipelineComponent] = Field(default_factory=dict)

    @property
    def custom_components(self) -> list[str]:
        """Components loaded by Python shipped in the repo, not by diffusers or transformers."""
        return sorted(
            f"{name} ({c.library}.{c.class_name})"
            for name, c in self.components.items()
            if c.library is not None and c.library not in _LIBRARIES
        )

    @property
    def edits(self) -> bool:
        """An image editing pipeline (Qwen-Image-Edit, its Plus, FLUX.1 Kontext): it takes a
        picture to change, which a text-to-image instance never gets."""
        return any(word in self.class_name for word in _EDIT_WORDS)

    @property
    def label(self) -> str:
        """`Sana`, or the class without its suffix for a pipeline nobody named (an editing
        one says so, rather than "QwenImageEditPlus is not a family")."""
        known = PIPELINE_LABELS.get(self.class_name)
        if known:
            return known
        bare = self.class_name.removesuffix("Pipeline")
        return f"{bare} (image editing)" if self.edits else bare


def parse_pipeline_index(data: object) -> PipelineIndex:
    """The typed index; ValueError when the payload is not one (no `_class_name`)."""
    if not isinstance(data, dict) or not isinstance(data.get("_class_name"), str):
        raise ValueError("not a diffusers pipeline index (no _class_name)")
    components: dict[str, PipelineComponent] = {}
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, list) or len(value) < 2:
            continue  # settings such as `is_distilled`, `force_zeros_for_empty_prompt`
        library, class_name = value[0], value[1]
        components[key] = PipelineComponent(
            library=library if isinstance(library, str) else None,
            class_name=class_name if isinstance(class_name, str) else None,
        )
    version, blocks = data.get("_diffusers_version"), data.get("_blocks_class_name")
    return PipelineIndex(
        class_name=data["_class_name"],
        diffusers_version=version if isinstance(version, str) else None,
        blocks_class_name=blocks if isinstance(blocks, str) else None,
        components=components,
    )


def pipeline_family(index: PipelineIndex, repo_id: str) -> ImageFamily:
    family = PIPELINE_FAMILIES.get(index.class_name, ImageFamily.UNKNOWN)
    if family is ImageFamily.FLUX2_KLEIN and detect_family(repo_id) is ImageFamily.FLUX2_KLEIN_9B:
        return ImageFamily.FLUX2_KLEIN_9B  # its own kit: Qwen3-8B, not the 4B's Qwen3-4B
    return family
