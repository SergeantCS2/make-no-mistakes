"""Image plans: GGUF kits, safetensors checkpoints and diffusers folders, single DiT files.

Every branch reads only what it needs from the Hub (a tree, a companion's size, a
safetensors header through two HTTP ranges) and returns a `Plan`, or a `PlanError` with
guidance: what the weights are and what runs here instead.
"""

from __future__ import annotations

from makenomistakes.fit import Budget
from makenomistakes.fit.image import (
    plan_checkpoint_profiles,
    plan_dit_profiles,
    plan_image_profiles,
)
from makenomistakes.fit.image_memory import checkpoint_sizes, kit_sizes
from makenomistakes.guide import Guidance, guide_unservable
from makenomistakes.hub import HubClient, HubError, ModelFacts
from makenomistakes.hub.image_files import ImageWeights
from makenomistakes.hub.kits import (
    FAMILY_LABELS,
    KITS,
    UPSCALER_PATH,
    UPSCALER_REPO,
    ImageKit,
    kit_for,
)
from makenomistakes.plan_types import (
    CheckpointChoice,
    NoRoomError,
    Plan,
    PlanError,
    SourceReadError,
)
from makenomistakes.safetensors.reader import (
    SafetensorsError,
    SafetensorsHeader,
    read_safetensors_header,
)
from makenomistakes.safetensors.sd import (
    DitParts,
    checkpoint_parts,
    family_of,
    image_parts,
    lora_info,
)

MAX_VERSIONS_TRIED = 4  # header reads before a repo of partial weights is refused


class PartialWeightsError(PlanError):
    """These weights lack a part; another version of the same repo may be whole."""


def image_plan(facts: ModelFacts, hub: HubClient, budget: Budget) -> Plan:
    """GGUF diffusion models: the kit's companion sizes come from the Hub, no header is read."""
    kit = kit_for(facts.image_family)
    if kit is None:
        raise PlanError(f"{facts.repo_id}: unknown image family.", guide_unservable(facts, hub))
    _check_vram_floor(kit, facts, hub, budget)
    sizes = companion_sizes(kit, hub)
    profiles = plan_image_profiles(facts, kit, kit_sizes(kit.companions, sizes), budget)
    if not profiles:
        raise PlanError(f"{facts.repo_id} has no GGUF variants with a recognisable quant.")
    return Plan(facts, None, budget, profiles, kit, sizes, upscaler_bytes=upscaler_bytes(kit, hub))


def checkpoint_plan(facts: ModelFacts, hub: HubClient, budget: Budget, picked: bool) -> Plan:
    """Safetensors image weights: a checkpoint, a diffusers folder or a single DiT file.

    Without a pick (`--version`, or a pasted file link) the newest version is tried first,
    then the next ones when it lacks a part: a repo like SDXL-Lightning keeps bare `_unet`
    files beside the whole checkpoints, and the smallest file is not always a whole model."""
    ordered = sorted(facts.image_weights, key=lambda w: (w.version, -w.size_bytes), reverse=True)
    if not ordered:
        raise PlanError(f"{facts.repo_id} has no image weights.")
    candidates = ordered[:1] if picked or len(ordered) == 1 else ordered[:MAX_VERSIONS_TRIED]
    labels = facts.versions or [w.label for w in facts.image_weights]  # `--version` narrowed
    first: PartialWeightsError | None = None
    for weights in candidates:
        try:
            return _weights_plan(facts.narrowed_to(weights, labels), weights, hub, budget)
        except PartialWeightsError as exc:
            first = first or exc
    assert first is not None  # every candidate raised
    raise first


def _weights_plan(facts: ModelFacts, weights: ImageWeights, hub: HubClient, budget: Budget) -> Plan:
    """One set of weights. Its header names the family and, for a UNet model, how much of
    it is UNet, text encoders and VAE; the kit for that family adds what it lacks (SDXL:
    the fp16-fix VAE, since SDXL's own VAE overflows in F16)."""
    headers: dict[str, SafetensorsHeader] = {}

    def header_of(path: str) -> SafetensorsHeader:  # each file's header is read once
        if path not in headers:
            source = hub.file_source(facts.repo_id, facts.revision, path)
            headers[path] = read_safetensors_header(source)
        return headers[path]

    try:
        first = header_of(weights.files[0])
        if (lora := lora_info(first)) is not None:  # a 1 GiB+ LoRA is still a LoRA
            raise PlanError(
                f"{weights.label} is a LoRA for {lora.label}, not a model that runs on its own.",
                guide_unservable(facts.model_copy(update={"is_lora": True}), hub),
            )
        family = family_of(first)  # the checkpoint, the UNet or the DiT
        kit = kit_for(family)
        if kit is not None and kit.takes_dit_files and weights.layout == "file":
            return _dit_plan(facts, weights, header_of(weights.files[0]), kit, hub, budget)
        if kit is None or kit.checkpoint_text_encoders is None:
            raise PlanError(
                f"{weights.label} is {FAMILY_LABELS[family]}. This tool plans SD 1.x and SDXL "
                "checkpoints (Illustrious, Pony and other SDXL fine-tunes included) and single "
                f"diffusion-model files of {_dit_labels()}; other families have no kit here "
                "yet.",
                image_guidance(facts, hub),
            )
        sizes = companion_sizes(kit, hub)
        if kit.brings_vae and weights.layout == "diffusers":  # the kit's VAE wins
            weights = weights.without([f for f in weights.files if f.split("/")[-2] == "vae"])
        parts = image_parts(weights.layout, weights.files, header_of)
    except (HubError, SafetensorsError) as exc:
        raise SourceReadError(_unreadable(facts, exc), image_guidance(facts, hub)) from exc
    missing = parts.missing(kit.checkpoint_text_encoders, vae_supplied=kit.brings_vae)
    if missing:
        raise PartialWeightsError(
            f"{weights.label} is not a whole {parts.label} model: it lacks "
            f"{' and '.join(missing)} (a UNet, ControlNet or adapter on its own). Paste the "
            "repo of the full checkpoint or diffusers folder"
            + (
                f", or pick another version ({', '.join(facts.versions)})."
                if facts.versions
                else "."
            ),
            image_guidance(facts, hub),
        )
    planned = facts.model_copy(update={"image_family": parts.family})
    split = checkpoint_sizes(parts, kit_sizes(kit.companions, sizes))
    profiles = plan_checkpoint_profiles(planned, kit, weights, split, sum(sizes.values()), budget)
    return Plan(
        planned,
        None,
        budget,
        profiles,
        kit,
        sizes,
        checkpoint=CheckpointChoice(weights, parts),
        upscaler_bytes=upscaler_bytes(kit, hub),
    )


def _dit_plan(
    facts: ModelFacts,
    weights: ImageWeights,
    header: SafetensorsHeader,
    kit: ImageKit,
    hub: HubClient,
    budget: Budget,
) -> Plan:
    """A transformer's diffusion model in one file (ComfyUI's split files, a fine-tune): the
    kit brings the text encoder and the VAE, as for the family's GGUF builds."""
    parts = checkpoint_parts(header)
    if parts.dtype == "U8":  # NVFP4/MXFP packs its weights as bytes: no sd.cpp code path
        raise PartialWeightsError(
            f"{weights.label} stores its weights packed as bytes (NVFP4 or MXFP), a format "
            "stable-diffusion.cpp cannot read"
            + (f"; pick another version ({', '.join(facts.versions)})." if facts.versions else ".")
        )
    if not isinstance(parts, DitParts):  # a family that is not a DiT is planned as a checkpoint
        raise PlanError(f"{weights.label} is not a transformer's diffusion model.")
    if parts.bundled:
        raise PlanError(
            f"{weights.label} is an all-in-one {parts.label} file: it carries its text encoders "
            f"or VAE as well. This engine runs {parts.label} from the diffusion model alone, "
            "with the kit's text encoder and VAE: pick the repo's diffusion-model file, or a "
            "GGUF build.",
            image_guidance(facts, hub),
        )
    _check_vram_floor(kit, facts, hub, budget)
    sizes = companion_sizes(kit, hub)
    planned = facts.model_copy(update={"image_family": parts.family})
    profiles = plan_dit_profiles(
        planned,
        kit,
        weights,
        parts,
        kit_sizes(kit.companions, sizes),
        sum(sizes.values()),
        budget,
    )
    return Plan(
        planned,
        None,
        budget,
        profiles,
        kit,
        sizes,
        checkpoint=CheckpointChoice(weights, parts),
        upscaler_bytes=upscaler_bytes(kit, hub),
    )


def _check_vram_floor(kit: ImageKit, facts: ModelFacts, hub: HubClient, budget: Budget) -> None:
    if kit.min_vram_bytes and budget.vram_free_bytes < kit.min_vram_bytes:
        raise NoRoomError(
            f"{facts.repo_id} needs about {kit.min_vram_bytes / 2**30:.0f} GiB of VRAM "
            f"({kit.note}); this machine has {budget.vram_free_bytes / 2**30:.1f} GiB free.",
            guide_unservable(facts, hub),
        )


def _dit_labels() -> str:
    names = [FAMILY_LABELS[k.family] for k in KITS.values() if k.takes_dit_files]
    return ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)


def companion_sizes(kit: ImageKit, hub: HubClient) -> dict[str, int]:
    sizes: dict[str, int] = {}
    try:
        for companion in kit.companions:
            sizes[companion.key] = _hub_size(
                hub, companion.repo_id, companion.path, companion.revision
            )
    except HubError as exc:
        raise SourceReadError(f"Could not read a companion repo: {exc}") from exc
    return sizes


def upscaler_bytes(kit: ImageKit, hub: HubClient) -> int:
    """The ESRGAN upscaler's size for a picture kit, for the download total; 0 for a video
    kit (never offered) or when the Hub cannot say (the download trusts the file, and an
    optional 18 MB asset never blocks a plan). One extra tree listing per picture plan,
    `--no-upscaler` included: the plan does not know yet whether the user wants it."""
    if kit.is_video:
        return 0
    try:
        return _hub_size(hub, UPSCALER_REPO, UPSCALER_PATH, "main")
    except (HubError, PlanError):
        return 0


def _hub_size(hub: HubClient, repo_id: str, path: str, revision: str) -> int:
    entries = {e.path: e.size for e in hub.tree(repo_id, revision)}
    if path not in entries:
        raise PlanError(
            f"{repo_id} no longer has {path}; the kit table in hub/kits.py needs updating."
        )
    return entries[path]


def image_guidance(facts: ModelFacts, hub: HubClient) -> Guidance:
    """Guidance for image weights that cannot be planned: what runs here instead."""
    return guide_unservable(facts.model_copy(update={"image_weights": []}), hub)


def _unreadable(facts: ModelFacts, exc: Exception) -> str:
    return unreadable_message(facts, exc, "safetensors header")


def unreadable_message(facts: ModelFacts, exc: Exception, what: str) -> str:
    """A header read that failed: the gated remedy when the repo is gated (a 401 on a gated
    GGUF said only "HTTP 401" before), else the plain reason."""
    if facts.gated:
        return (
            f"{facts.repo_id} is gated: accept its terms on its Hugging Face page and store a "
            f"token (`mnm setup`, HF_TOKEN), then try again ({exc})."
        )
    return f"Could not read the {what}: {exc}"
