"""The Recipe: the single source of truth for one runnable instance.

Everything the UI edits lives here as typed fields. The `docker run` command is derived
from it, never edited as a string.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from makenomistakes.engine_args import AudioCppArgs, LlamaCppArgs, SdCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.engine_args_whisper import WhisperArgs
from makenomistakes.fit.estimate import FitLevel
from makenomistakes.fit.profiles import ProfileName
from makenomistakes.hub.civitai import civitai_page, is_civitai
from makenomistakes.hub.elsewhere import elsewhere
from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.hub.kits import KITS, ImageFamily, Role
from makenomistakes.hub.variants import DraftHead
from makenomistakes.recipes.migrations import MIGRATIONS
from makenomistakes.recipes.quality import QualityTuned
from makenomistakes.recipes.tuned import Tuned
from makenomistakes.safetensors.dit import DIT_FAMILIES

CONTAINER_MODELS_DIR = "/models"
HF_CACHE_SUBDIR = "hf"  # downloads land in <models>/hf, the Hugging Face cache layout


def companion_key(repo_id: str, path: str) -> str:
    """How a file from another repository is keyed in `Recipe.local_paths`."""
    return f"{repo_id}:{path}"


INSTANCE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


class RepoFile(BaseModel):
    """A companion file from another repository (VAE, text encoder), keyed for local_paths."""

    role: Role
    repo_id: str
    path: str
    revision: str = "main"
    size_bytes: int = 0

    @property
    def key(self) -> str:
        return companion_key(self.repo_id, self.path)


class LoraFile(BaseModel):
    """A LoRA the instance offers: the engine page lists it by `name`, and an always-on
    one is applied to every request (Odysseus's included) at `weight`."""

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")  # the link's stem
    repo_id: str
    path: str
    revision: str = "main"
    size_bytes: int = 0
    family: ImageFamily = ImageFamily.UNKNOWN  # read from its header when added
    weight: float = Field(default=1.0, ge=-4.0, le=4.0)
    always_on: bool = False

    @property
    def key(self) -> str:
        return companion_key(self.repo_id, self.path)


class UpscalerFile(BaseModel):
    """The ESRGAN upscaler an image instance offers per request (`/sdcpp/v1/upscale`, the
    engine page's upscale and hires steps): a folder-served asset like a LoRA, not a part
    the engine loads at start."""

    repo_id: str
    path: str
    revision: str = "main"
    size_bytes: int = 0

    @property
    def key(self) -> str:
        return companion_key(self.repo_id, self.path)

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


class DraftModel(BaseModel):
    """Another saved llama.cpp instance's model, drafting tokens for this one (speculative
    decoding): its file as it was when `mnm set <name> draft=<instance>` checked it.
    Kept here whole, so the draft runs (and is priced) whatever becomes of that instance.
    Or (`kind` mtp) the instance's own MTP head, `mnm set <name> draft=mtp`: a head
    trained beside the model, run as `--spec-type draft-mtp`."""

    # `dspark`: DeepSeek-V4.1's DSpark head, `--spec-type draft-dspark`; not run yet
    # (added once a box measures a gain), so no recipe carries one.
    kind: Literal["model", "mtp", "dspark"] = "model"
    instance: str  # the instance it came from, as the user named it (itself for an MTP head)
    repo_id: str
    file: str  # its repo file (the first shard when it has several)
    path: str  # where it is, relative to the models folder (`/models/<path>` in the engine)
    quant: str
    size_bytes: int = Field(ge=0)  # every shard

    @property
    def is_mtp(self) -> bool:
        return self.kind == "mtp"

    @property
    def setting(self) -> str:
        """Its `draft=` value: the instance it came from, or `mtp`."""
        return self.instance if self.kind == "model" else self.kind

    @property
    def spec_type(self) -> str:
        """llama-server's `--spec-type` for it."""
        return {"mtp": "draft-mtp", "dspark": "draft-dspark"}.get(self.kind, "draft-simple")

    @property
    def source(self) -> str:
        """Where it comes from, in words (the parts list, a refusal)."""
        if self.kind == "model":
            return f"from {self.instance}"
        return "its MTP head" if self.is_mtp else "its DSpark head"


class Eviction(BaseModel):
    """The disk policy deleted the instance's own files: when, and what they
    held. The recipe stays; its next start (`mnm start`, the router's load) downloads them
    again and clears this."""

    at: datetime
    freed_bytes: int = Field(ge=0)
    last_used: datetime | None = None  # when it was last used before it went


# Bump SCHEMA_VERSION and add a migration (`recipes.migrations`) whenever a saved field is
# added without a default, renamed, or changes meaning.
SCHEMA_VERSION = 27
IDLE_SECONDS = 600  # the router's idle unload: ten minutes without a request


class NewerRecipeError(ValueError):
    """A recipe file written by a newer makenomistakes than this one."""

    def __init__(self, version: int) -> None:
        super().__init__(f"schema {version}; this version reads up to {SCHEMA_VERSION}")
        self.version = version


# The user's own fields of a recipe that no plan sets (a `gpus=` pin, the idle limit, `keep`,
# an instance switched off): a re-plan and a quality tune's save keep them as they are.
OWNER_FIELDS = ("gpus", "idle", "keep", "off")


def load_recipe_json(text: str) -> Recipe:
    """A saved recipe file, brought up to `SCHEMA_VERSION` one numbered step at a time.

    Migrations run here, at the file boundary, never on a `Recipe(...)` built in code. A
    file without `schema_version` is version 1. Raises NewerRecipeError for a file from a
    newer runner and ValueError (pydantic's ValidationError included) for anything else
    this version cannot read.
    """
    return load_recipe_data(json.loads(text))


def load_recipe_data(data: object) -> Recipe:
    """`load_recipe_json` of a recipe already parsed (a queued quality plan's)."""
    if not isinstance(data, dict):
        raise ValueError("a recipe file holds one JSON object")
    version = data.get("schema_version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"schema_version must be a positive integer, not {version!r}")
    if version > SCHEMA_VERSION:
        raise NewerRecipeError(version)
    for step in range(version, SCHEMA_VERSION):
        data = MIGRATIONS[step](data)
    return Recipe.model_validate({**data, "schema_version": SCHEMA_VERSION})


class Recipe(BaseModel):
    name: str
    display_name: str
    repo_id: str
    revision: str
    modality: Modality
    engine: Engine
    image: str
    port: int = Field(ge=1024, le=65535)
    served_name: str
    files: list[str]
    # llama.cpp's architecture name (`deepseek2`, `qwen35moe` …): `split=tensor` is
    # refused for those llama.cpp does not split by tensor; None in an older recipe
    architecture: str | None = None
    projector: str | None = None
    companions: list[RepoFile] = Field(default_factory=list)  # image kits
    # Image weights: one file, or a diffusers folder the engine reads through a link view.
    layout: Literal["file", "diffusers"] = "file"
    image_family: ImageFamily = ImageFamily.UNKNOWN  # image recipes; LoRAs must match it
    loras: list[LoraFile] = Field(default_factory=list)  # sd.cpp instances
    # Wan 2.2 A14B: the high-noise diffusion model (a repo file) that runs before `files[0]`.
    high_noise: str | None = None
    # llama.cpp: the repo's multi-token-prediction head for this quant (a repo file),
    # downloaded with the model; it runs only as the draft `mnm set <name> draft=mtp` names.
    mtp: str | None = None
    # The shard holding an Engram table (`per_layer_token_embd.weight`, DeepSeek-V4.1):
    # downloaded, kept on disk and read row by row (`LlamaCppArgs.lazy_tensors`), never in RAM
    engram: str | None = None
    # The repo's other draft heads (DSpark, DFlash, EAGLE-3), listed as unused by `mnm
    # show`, never downloaded
    draft_heads: list[DraftHead] = Field(default_factory=list)
    upscaler: UpscalerFile | None = None  # sd.cpp picture instances (`mnm add --upscaler`)
    local_paths: dict[str, str] = Field(default_factory=dict)  # repo file or RepoFile.key
    args: LlamaCppArgs | SdCppArgs | WhisperArgs | AudioCppArgs | VllmArgs = Field(
        discriminator="engine"
    )
    profile: ProfileName
    quant: str
    schema_version: int = SCHEMA_VERSION  # the shape of this file; see MIGRATIONS
    vram_estimate_bytes: int | None = (
        None  # what the plan puts on the GPU; None: older file or changed by `mnm set`
    )
    # How the plan fitted it (comfortable, tight, offload); None: a file from before it was
    # kept. The start-time `-ngl` upgrade reads it: a planned offload at exactly the block
    # count is the planner's own split, not a full fit an older mnm saved at the block count.
    fit_level: FitLevel | None = None
    # The swap workflow this instance takes turns in: its start may park the group's
    # others, and the router loads it at the next request.
    swap_group: str | None = None
    # The user stopped it (`mnm stop`, `workflow stop`): with no container it stays off, and
    # the router neither lists nor loads it. Without it, a stopped instance is parked: the next
    # request loads it. A running container is loaded whatever this says.
    off: bool = False
    # The router unloads it this many seconds after its last reply finished, with nothing in
    # flight (0: never); it loads again at the next request.
    idle: int = Field(default=IDLE_SECONDS, ge=0)
    # `always` pins it: never unloaded for idleness or to make room, and a load that needs its
    # room is refused with the reason.
    keep: Literal["auto", "always"] = "auto"
    # llama.cpp chat, coding and vision instances: a smaller model of the same vocabulary
    # drafts tokens this one checks (`--spec-type draft-simple`); None runs it alone.
    draft: DraftModel | None = None
    # The cards (nvidia-smi's indices) it is pinned to, `mnm set <name> gpus=0,1`;
    # None runs it where `recipes.cards.recipe_cards` says (`MNM_GPU`'s).
    gpus: tuple[int, ...] | None = None
    # The disk policy evicted its files (`local_paths` cleared); None: never, or they
    # were downloaded again since
    evicted: Eviction | None = None
    # What the tuner chose and measured, and on what (`Tuned.key`); None:
    # never tuned, or the user's `mnm set` of a setting it chose dropped it.
    tuned: Tuned | None = None
    # The quality tune's choice (schema 27); None: never quality-tuned, or
    # planned since at another quant. Its speed tune leaves the KV cache as it is.
    quality: QualityTuned | None = None
    # The box the plan was made on (`MachineFingerprint.id`, `cards_id`), set by `mnm
    # add`, `replan`, `upgrade` and the tuner's save; None: planned before mnm kept them
    planned_on: str | None = None
    planned_cards: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return validate_instance_name(value)

    @field_validator("created_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        """A hand-edited file without a zone would make `mnm ls` fail to sort the list."""
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value

    @model_validator(mode="after")
    def _one_engine(self) -> Recipe:
        """`engine` and `args.engine` name the same engine, or the recipe fails deep later
        (a TypeError at `docker run`, a bare ValueError from `mnm set`)."""
        if str(self.args.engine) != self.engine.value:
            raise ValueError(f"engine is {self.engine.value} but the args are {self.args.engine}'s")
        return self

    @property
    def drops_references(self) -> bool:
        """The router keeps an edit's picture from this instance's text encoder (it would
        crash on it): an `image` instance of a family whose kit encoder cannot see
        (`ImageKit.references_need_vision`). An editor keeps its references."""
        kit = KITS.get(self.image_family)
        return (
            self.modality is Modality.IMAGE_GEN and kit is not None and kit.references_need_vision
        )

    @property
    def drafts_with_mtp(self) -> bool:
        """It drafts with its own MTP head (`draft=mtp`)."""
        return self.draft is not None and self.draft.is_mtp

    @property
    def pinned(self) -> bool:
        """Never unloaded by the router, for idleness or to make room (`keep=always`)."""
        return self.keep == "always"

    def with_args(self, **updates: object) -> Recipe:
        """A copy with some engine args changed, validated, and the plan's estimate forgotten
        unless every changed setting is one its args class names as leaving the GPU as
        planned (`MEMORY_NEUTRAL`: without it `mnm set moe load=none` would forget the
        estimate, and the next re-plan would read "unknown → 8.4").

        `mnm set` and auto-fix both go through here, so a recipe can never carry settings
        its args class would refuse, nor an estimate made for other settings.
        """
        unknown = set(updates) - set(type(self.args).model_fields)
        if unknown:  # model_validate would drop them silently
            raise ValueError(f"{type(self.args).__name__} has no {', '.join(sorted(unknown))}")
        validated = type(self.args).model_validate({**self.args.model_dump(), **updates})
        moved = {k for k in updates if getattr(validated, k) != getattr(self.args, k)}
        neutral = moved <= type(self.args).MEMORY_NEUTRAL
        estimate = self.vram_estimate_bytes if neutral else None
        return self.model_copy(update={"args": validated, "vram_estimate_bytes": estimate})

    @property
    def source_url(self) -> str:
        """The page the files came from, at the pinned revision (a Civitai version's page for
        a `civitai/<model id>` repo)."""
        if is_civitai(self.repo_id):
            return civitai_page(self.repo_id, self.revision)
        if (where := elsewhere(self.repo_id)) is not None:
            return where.page(self.repo_id, self.revision)
        return f"https://huggingface.co/{self.repo_id}/tree/{self.revision}"

    @property
    def settings_summary(self) -> str:
        """One cell for tables: the settings that matter most for this engine."""
        return self.args.summary

    @property
    def safetensors_weights(self) -> bool:
        """Image weights as a safetensors checkpoint or diffusers folder: their headers hold
        the UNet, text encoder and VAE split (a GGUF image file is sized as a whole)."""
        return self.engine is Engine.SDCPP and self.files[0].endswith(".safetensors")

    @property
    def converts_dit(self) -> bool:
        """A single DiT file whose diffusion model sd.cpp can convert while loading
        (`--tensor-type-rules`): a safetensors file of a transformer family. A GGUF is
        quantized already; a UNet checkpoint's split is not priced for a conversion."""
        return (
            self.safetensors_weights and self.layout == "file" and self.image_family in DIT_FAMILIES
        )

    @property
    def own_files(self) -> list[str]:
        """The repo files of its own repository it runs: the model (every shard), then the
        projector, the high-noise half and the MTP head when it has them."""
        extra = (self.projector, self.high_noise, self.mtp)
        return [*self.files, *(f for f in extra if f)]

    @property
    def wanted_keys(self) -> list[str]:
        """Every file the instance needs, as local_paths keys."""
        own = self.own_files
        extras = [self.upscaler.key] if self.upscaler else []
        return own + [c.key for c in self.companions] + [lora.key for lora in self.loras] + extras

    @property
    def downloaded(self) -> bool:
        return set(self.wanted_keys) <= set(self.local_paths)

    @property
    def llama_args(self) -> LlamaCppArgs:
        if not isinstance(self.args, LlamaCppArgs):
            raise TypeError(f"{self.name} is a {self.engine.value} recipe, not llama.cpp")
        return self.args

    @property
    def sd_args(self) -> SdCppArgs:
        if not isinstance(self.args, SdCppArgs):
            raise TypeError(f"{self.name} is a {self.engine.value} recipe, not sd.cpp")
        return self.args

    @property
    def whisper_args(self) -> WhisperArgs:
        if not isinstance(self.args, WhisperArgs):
            raise TypeError(f"{self.name} is a {self.engine.value} recipe, not whisper.cpp")
        return self.args

    @property
    def voice_args(self) -> AudioCppArgs:
        if not isinstance(self.args, AudioCppArgs):
            raise TypeError(f"{self.name} is a {self.engine.value} recipe, not audio.cpp")
        return self.args

    @property
    def vllm_args(self) -> VllmArgs:
        if not isinstance(self.args, VllmArgs):
            raise TypeError(f"{self.name} is a {self.engine.value} recipe, not vLLM")
        return self.args

    def container_path(self, repo_file: str) -> str:
        """Where a downloaded repo file appears inside the engine container."""
        return f"{CONTAINER_MODELS_DIR}/{self.local_paths[repo_file]}"


def validate_instance_name(value: str) -> str:
    if not INSTANCE_NAME_RE.match(value):
        raise ValueError(
            f"'{value}' is not a valid instance name: lowercase letters, digits and dashes, "
            "max 64 characters"
        )
    return value


# `mnm engines update`'s copies (`engine_selftest`) run as `selftest-<engine>`: an instance
# under such a name would be stopped and its links removed by the self-test.
SELFTEST_PREFIX = "selftest-"
# The tuner's trials run a copy as `tune-<name>`, apart from the instance's own container.
TUNE_PREFIX = "tune-"


def container_name(recipe_name: str) -> str:
    return f"mnm-{recipe_name}"


def publish(recipe: Recipe) -> str:
    """`docker run --publish`: on 127.0.0.1 only. Clients reach every instance through the
    router, which forwards here; nothing else on the network sees the engine."""
    return f"127.0.0.1:{recipe.port}:{recipe.port}"
