"""llama.cpp recipes: build one from a profile, render it to `docker run`."""

from __future__ import annotations

import re
from pathlib import Path

from makenomistakes.engine_args import DRAFT_KV, LlamaCppArgs, LoadMode
from makenomistakes.fit.cpu_rates import CpuSide
from makenomistakes.fit.estimate import ngl_for
from makenomistakes.fit.profiles import Profile
from makenomistakes.gguf.tensors import ENGRAM_TENSOR
from makenomistakes.hub.facts import Engine, Modality, ModelFacts
from makenomistakes.prompt_cache import PlannedCache
from makenomistakes.recipes.cards import device_args, split_argv
from makenomistakes.recipes.llamacpp_cache import cache_volume, slot_dir
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR, Recipe, container_name, publish

LLAMACPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
_MODE_FLAGS: dict[Modality, list[str]] = {
    Modality.CHAT: [],
    Modality.CODING: [],
    Modality.VISION_CHAT: [],
    Modality.EMBEDDING: ["--embedding"],
    Modality.RERANK: ["--rerank"],
}  # every servable modality; anything else is a bug upstream (see ModelFacts.servable)


def llamacpp_recipe(  # noqa: PLR0913 - the plan's extras are keyword-only
    facts: ModelFacts,
    profile: Profile,
    name: str,
    port: int,
    image: str = LLAMACPP_IMAGE,
    *,
    cpu: CpuSide | None = None,
    load_mode: LoadMode | None = None,
    cache: PlannedCache | None = None,
) -> Recipe:
    """`cpu`: a CPU plan (no card): every block in RAM (`-ngl 0`), the physical cores
    as threads and the NUMA mode. `load_mode`: the plan's own (`plan.pool_load`: `none` on a
    unified pool); an Engram table's lazy reads keep mmap whatever it says. `cache`:
    the plan's prompt cache (`prompt_cache.planned`); None leaves it llama.cpp's own."""
    if not facts.servable:
        raise ValueError(f"{facts.repo_id} ({facts.modality.value}) cannot run on llama-server")
    args = LlamaCppArgs(
        ctx=profile.ctx,
        gpu_layers=0 if cpu is not None else ngl_for(profile.fit),
        threads=cpu.threads if cpu is not None else None,
        numa=cpu.numa if cpu is not None else None,
        # experts in RAM; a placement says which itself
        cpu_moe_layers=(profile.fit.cpu_moe_layers or None) if profile.placement is None else None,
        kv_type=profile.kv_type,
        batch=profile.ubatch,  # the profile sets both for embedding models (whole input)
        ubatch=profile.ubatch,
        image_min_tokens=QWEN_VL_IMAGE_TOKENS
        if _QWEN_VL_RE.match(facts.architecture or "")
        else None,
        extra=_MODE_FLAGS[facts.modality],
        tensor_split=profile.tensor_split,  # several cards only
        main_gpu=profile.main_gpu,
        placement=profile.placement,
        # An Engram table stays on disk, read row by row (`--lazy-mode on`)
        lazy_tensors=(ENGRAM_TENSOR,) if profile.variant.engram is not None else (),
        load_mode=load_mode if profile.variant.engram is None else None,
        cache_ram_mib=cache.cache_ram_mib if cache is not None else None,
        cache_disk_gib=cache.cache_disk_gib if cache is not None else None,
    )
    return Recipe(
        name=name,
        display_name=facts.display_name,
        repo_id=facts.repo_id,
        revision=facts.revision,
        modality=facts.modality,
        engine=Engine.LLAMACPP,
        image=image,
        port=port,
        served_name=served_name(facts, profile.quant),
        files=list(profile.variant.files),
        architecture=facts.architecture,
        projector=facts.projector_files[0] if facts.projector_files else None,
        mtp=profile.variant.mtp,
        engram=profile.variant.engram.file if profile.variant.engram is not None else None,
        draft_heads=list(facts.draft_heads),
        args=args,
        profile=profile.name,
        quant=profile.quant,
        # a CPU plan holds nothing on a card: its size is the RAM it takes (the floor's)
        vram_estimate_bytes=0 if cpu is not None else profile.fit.gpu_bytes,
        fit_level=profile.fit.level,
    )


VISION_ID_RE = re.compile(r"vl|vision", re.IGNORECASE)


def text_only_name(slug: str) -> str:
    """A name no client reads as vision (a vision repo added as chat or coding runs
    text-only): every `VISION_ID_RE` match taken out, as Odysseus matches them anywhere in
    an id (`qwen3-vl-8b` → `qwen3-8b`, `internvl3-8b` → `intern3-8b`), empty words dropped."""
    words = (VISION_ID_RE.sub("", word) for word in slug.split("-"))
    return "-".join(word for word in words if word)


def vision_name(name: str) -> str:
    """An instance name with `-vision` added, within the 64 characters a name may have: the
    name is the id Odysseus matches, and it offers image upload by that id."""
    return f"{name[:57].rstrip('-')}-vision".lstrip("-")


# llama.cpp's Qwen-VL projectors warn below 1024 image tokens ("require at minimum 1024 image
# tokens to function correctly on grounding tasks", tools/mtmd/clip.cpp); the flag silences
# it by meeting it. Architectures: qwen2vl, qwen25vl, qwen3vl, qwen3vlmoe.
_QWEN_VL_RE = re.compile(r"qwen\d*vl", re.IGNORECASE)
QWEN_VL_IMAGE_TOKENS = 1024


def served_name(facts: ModelFacts, quant: str) -> str:
    """Model id clients see. Keeps the family name so Odysseus's name heuristics work.

    Odysseus enables image upload only when the id matches `vl`/`vlm`/`vision`, so a vision
    family without such a token (gemma-3-4b-it) gets `-vision` added.
    """
    name = facts.family_name
    if facts.modality is Modality.VISION_CHAT and not VISION_ID_RE.search(name):
        name = f"{name}-vision"
    return f"{name}-{quant}".lower()


def llamacpp_run_argv(recipe: Recipe, models_dir: Path) -> list[str]:
    if not recipe.downloaded:
        raise ValueError(f"{recipe.name} is not fully downloaded")
    argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name(recipe.name),
        *device_args(recipe),
        *recipe.llama_args.cpuset_argv(),  # `numa=node:N`, the node's CPUs and memory
        "--publish",
        publish(recipe),
        "--volume",
        f"{models_dir}:{CONTAINER_MODELS_DIR}:ro",
        *cache_volume(recipe, models_dir),  # Its saved slots (`keeps_slots`)
        "--restart",
        "no",
        "--label",
        f"mnm.instance={recipe.name}",
        # The image exports LLAMA_ARG_HOST=0.0.0.0 and LLAMA_ARG_PORT=8080 (the server stage
        # of upstream's `.devops/cuda.Dockerfile`, read at d854d98): a flag beside either
        # makes the server warn "environment variable is set, but will be overwritten" at
        # every start. Set the variables, not the flags.
        "--env",
        "LLAMA_ARG_HOST=0.0.0.0",
        "--env",
        f"LLAMA_ARG_PORT={recipe.port}",
        recipe.image,
        "--alias",
        recipe.served_name,
        "--model",
        recipe.container_path(recipe.files[0]),
        # `/metrics` for the router's: the server keeps these counters either way; the
        # flag only opens the endpoint (on 127.0.0.1, like the engine itself).
        "--metrics",
    ]
    if recipe.projector:
        argv += ["--mmproj", recipe.container_path(recipe.projector)]
    split = split_argv(recipe)  # several cards only
    engine = recipe.llama_args.to_cli(slot_dir(recipe))  # With its folder's mount
    return argv + engine + split + draft_argv(recipe, models_dir)


def draft_argv(recipe: Recipe, models_dir: Path) -> list[str]:
    """The draft model's flags, as this engine's build takes them: `--spec-type
    draft-simple` with the file (`-md`, under the models folder the engine already mounts
    read-only), `--spec-draft-n-max` (3 by default, llama.cpp's own) and `--spec-draft-p-min`
    (0.75: a token the draft is unsure of is not drafted), every layer of it on the GPU
    (`-ngld all`) and its KV cache at f16 (`-ctkd/-ctvd`: a small model's cache is small, and
    a quantised one only lowers what the draft gets right). Checked against llama-server
    c7fc509, which had renamed `--draft-max` to `--spec-draft-n-max` and refuses the old
    name. An MTP head (`draft=mtp`) is `--spec-type draft-mtp` with the same flags: the
    head is a model file of its own (`-md`), which llama.cpp runs in an MTP context."""
    draft = recipe.draft
    if draft is None:
        return []
    if not (models_dir / draft.path).is_file():
        raise ValueError(
            f"{recipe.name}'s draft model ({draft.file}, {draft.source}) is no longer on disk: "
            f"`mnm set {recipe.name} draft=` runs it without one."
        )
    args = recipe.llama_args
    kv = DRAFT_KV.value
    return [
        "--spec-type",
        draft.spec_type,
        "-md",
        f"{CONTAINER_MODELS_DIR}/{draft.path}",
        "--spec-draft-n-max",
        str(args.draft_max),
        "--spec-draft-p-min",
        f"{args.draft_p_min:g}",
        "-ngld",
        "all",
        "-ctkd",
        kv,
        "-ctvd",
        kv,
    ]
