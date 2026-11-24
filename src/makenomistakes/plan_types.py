"""What a plan is: the facts, budget and profiles a reference resolved to, and the errors.

`plan.py` builds one (`make_plan`); the CLI and the web UI choose a profile and turn it into
a recipe through it. It is a module of its own so the image planners (`plan_images.py`) can
build plans without importing the orchestration that calls them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn

from makenomistakes.busy_card import refusal_note
from makenomistakes.engine_args import MIN_CTX
from makenomistakes.engine_args_vllm import VLLM_MIN_CTX
from makenomistakes.fit import Budget, FitLevel, KvType, Profile, ProfileName, plan_custom
from makenomistakes.fit.audio import plan_audio_custom
from makenomistakes.fit.budget import pool_load
from makenomistakes.fit.image import dit_precisions, plan_dit_custom, plan_image_custom
from makenomistakes.fit.image_memory import kit_sizes
from makenomistakes.fit.music import plan_music_custom
from makenomistakes.fit.plan_floor import RamFloorError
from makenomistakes.fit.profiles import plannable
from makenomistakes.fit.quant_ladder import below
from makenomistakes.fit.vllm import plan_vllm_custom
from makenomistakes.fit.voice import plan_voice_custom
from makenomistakes.gguf import GgufHeader
from makenomistakes.guide import Guidance
from makenomistakes.hub import ModelFacts
from makenomistakes.hub.facts import Engine, Modality
from makenomistakes.hub.hf_config import HfConfig
from makenomistakes.hub.image_files import ImageWeights
from makenomistakes.hub.kits import UPSCALER_PATH, UPSCALER_REPO, ImageKit
from makenomistakes.prompt_cache import planned as planned_cache
from makenomistakes.recipes import (
    Recipe,
    UpscalerFile,
    audiocpp_recipe,
    llamacpp_recipe,
    sdcpp_recipe,
    spec,
    whispercpp_recipe,
)
from makenomistakes.recipes.sdcpp import GGUF_KIT_LOAD, CheckpointLoad
from makenomistakes.recipes.vllm import vllm_recipe
from makenomistakes.safetensors.sd import AnyParts, DitParts

CustomPlanner = Callable[[ModelFacts, Budget, str], Profile]

ONE_SET = (
    "{repo} holds one set of weights ({quant}); for another quant, paste the repo that holds "
    "it (an AWQ or FP8 build of the same model)."
)


class PlanError(RuntimeError):
    """A user-facing reason the plan could not be made, with guidance when the Hub can help."""

    def __init__(self, message: str, guidance: Guidance | None = None) -> None:
        super().__init__(message)
        self.guidance = guidance


class NoRoomError(PlanError):
    """The model does not fit this machine at any quant it was planned at: the box's size,
    never a refusal of the model (a RAM floor, an image kit's VRAM floor, vLLM's weights over
    the cards). `mnm what` reads it as *does not fit*."""


class SourceReadError(PlanError):
    """The source or the model's file could not be read (a source that did not answer or
    refused access, an unreadable header): nothing is known of whether it would run.
    `mnm what` reads it as *cannot read*."""


class NoSuchQuantError(PlanError):
    """The quant asked for is not among the repo's (such as a saved instance's quant the repo
    no longer lists, which `mnm why` answers with `mnm replan <name> --any-quant`)."""


class TypeRefusedError(PlanError):
    """The model is of a type it cannot be added as (`--type`): `modality` is its own.
    `mnm what --type coder` leaves such a hit out as another type rather than refusing it."""

    def __init__(self, message: str, modality: Modality) -> None:
        super().__init__(message)
        self.modality = modality


@dataclass(frozen=True)
class CheckpointChoice:
    """The checkpoint version or diffusers folder a plan runs, and what its header holds."""

    weights: ImageWeights
    parts: AnyParts


def base_profile(name: ProfileName) -> ProfileName:
    """The named profile a plan sizes `name` from: a custom profile (an instance added with
    `--quant`, `--ctx` or `--for-agents`) starts from balanced, overriding what it set."""
    return name if name is not ProfileName.CUSTOM else ProfileName.BALANCED


@dataclass(frozen=True)
class Plan:
    facts: ModelFacts
    header: GgufHeader | None  # None for image and speech-to-text models: no header read
    budget: Budget
    profiles: dict[ProfileName, Profile]
    kit: ImageKit | None = None
    kit_sizes: dict[str, int] = field(default_factory=dict)  # companion key -> bytes
    checkpoint: CheckpointChoice | None = None  # safetensors image weights: what runs
    tag_quant: str | None = None  # `hf.co/org/repo:Q8_0`: the quant unless one is passed
    upscaler_bytes: int = 0  # the ESRGAN upscaler's size, read with the kit's companions
    hf_config: HfConfig | None = None  # vLLM plans: the model's config.json

    @property
    def speed_note(self) -> str | None:
        """What this card measured of the family's picture time, for the plan."""
        if self.kit is None:
            return None
        return self.budget.speed_notes.get(self.kit.family)

    def upscaler_file(self) -> UpscalerFile:
        return UpscalerFile(
            repo_id=UPSCALER_REPO, path=UPSCALER_PATH, size_bytes=self.upscaler_bytes
        )

    @property
    def image_summary(self) -> str:
        """`SDXL checkpoint waiIllustriousSDXL_v170.safetensors (F16: UNet 4.8 GiB, text
        encoders 1.5 GiB); kit: …`, one line for `mnm fit` and the add page."""
        if self.kit is None:
            return ""
        kit = ", ".join(c.summary for c in self.kit.companions)
        if self.checkpoint is None:
            kind = "video" if self.kit.is_video else "image"
            return f"{self.kit.label} {kind} model" + (f"; kit: {kit}" if kit else "")
        weights, parts = self.checkpoint.weights, self.checkpoint.parts
        line = parts.describe(weights.files[0].rsplit("/", 1)[-1], weights.layout)
        return line + (f"; kit: {kit}" if kit else "")

    @property
    def quant_choices(self) -> list[tuple[str, int | None]]:
        """What `--quant` may name, with the download size when there is one: a GGUF repo's
        variants, or a DiT file's precisions (as stored, or Q8_0 while loading), whether or
        not the profiles chose them (`choose` prices the one asked for)."""
        parts = self.checkpoint.parts if self.checkpoint is not None else None
        if self.checkpoint is not None and isinstance(parts, DitParts):
            precisions = dit_precisions(self.checkpoint.weights, parts)
            return [(q, None) for q in precisions]
        return [(v.quant, v.size_bytes) for v in self.facts.gguf_variants]

    def smaller_quants(self, name: ProfileName = ProfileName.BALANCED) -> list[str]:
        """The quants below a profile's that this plan runs, the most faithful first
        (`quant_ladder.below` over the variants the planner itself plans: known bits, an
        Engram table that loads); where `mnm what` steps a coder down when its agent context
        does not fit at the profile's quant."""
        variants = [v for v in self.facts.gguf_variants if plannable(v)]
        return [v.quant for v in below(variants, self.profiles[name].variant)]

    def check_quant(self, quant: str) -> None:
        """PlanError unless this plan can run `quant`: the one check a tag (`mnm fit`), the
        flag (`mnm add`) and the web share, so they never disagree."""
        wanted = quant.upper()
        if self.facts.engine is Engine.VLLM:  # one build: another tag is refused, never ignored
            held = next(iter(self.profiles.values())).quant
            if wanted != held.upper():
                raise PlanError(ONE_SET.format(repo=self.facts.repo_id, quant=held))
            return
        if self.checkpoint is not None and not isinstance(self.checkpoint.parts, DitParts):
            if wanted not in {p.quant.upper() for p in self.profiles.values()}:
                raise PlanError(
                    f"{quant}: a checkpoint runs at F16 and has no quants to pick; to plan "
                    "another version, pass `--version` (or paste that file's link)."
                )
            return
        choices = [q for q, _ in self.quant_choices]
        if wanted in {c.upper() for c in choices}:
            return
        if self.checkpoint is not None:
            raise PlanError(
                f"{quant} is not a choice for this file (choices: {', '.join(choices)}: as "
                "stored, or quantized while loading)."
            )
        available = ", ".join(choices) or "none"
        raise NoSuchQuantError(
            f"{self.facts.repo_id} has no '{quant}' variant (available: {available})."
        )

    @property
    def other_versions(self) -> list[str]:
        """The repo's other versions (checkpoint files, diffusers folders, GGUF models): pass
        one to `--version` to plan it."""
        return [label for label in self.facts.versions if label != self.facts.version]

    def choose(
        self,
        name: ProfileName,
        quant: str | None = None,
        ctx: int | None = None,
        kv: KvType | None = None,
    ) -> Profile:
        """A named profile, or a custom one when quant or context are overridden. Never NO_FIT."""
        quant = quant or self.tag_quant
        base = self.profiles[base_profile(name)]
        engine = self.facts.engine
        if engine is Engine.VLLM:
            return self._choose_vllm(base, quant, ctx, kv)
        if engine is not None and (row := spec(engine)).knobs is not None:
            if ctx is not None or kv is not None:
                raise PlanError(
                    f"Context and KV cache do not apply to {row.label} models; {row.knobs}."
                )
            if engine is Engine.SDCPP:
                return self._choose_image(base, quant)
            if engine is Engine.AUDIOCPP:
                music = self.facts.music_family is not None
                return self._choose_audio(
                    base, quant, plan_music_custom if music else plan_voice_custom
                )
            return self._choose_audio(base, quant, plan_audio_custom)
        if ctx is not None:
            self._check_ctx(ctx)
        chosen = base
        if self.header is not None and (quant is not None or ctx is not None or kv is not None):
            try:
                chosen = plan_custom(
                    self.facts,
                    self.header,
                    self.budget,
                    quant or base.quant,
                    ctx or base.ctx,
                    kv or base.kv_type,
                    floor=True,
                )
            except RamFloorError as exc:
                raise NoRoomError(str(exc)) from exc
            except ValueError as exc:
                raise PlanError(str(exc)) from exc
        if chosen.fit.level is FitLevel.NO_FIT:
            self._refuse(
                f"{chosen.quant} at {chosen.ctx:,} context needs "
                f"{chosen.estimate.total_bytes / 2**30:.1f} GiB and does not fit this machine "
                "even with offloading.",
                "Pick a smaller quant or context",
            )
        return chosen

    def _refuse(self, needs: str, smaller: str) -> NoReturn:
        """A pick that does not fit, with what to do: a lighter pick only when one of the
        profiles fits, and VRAM held elsewhere when that is why (with another program holding
        8 GiB, every quant of wan5b would be refused with "pick a smaller quant")."""
        fits = any(p.fit.level is not FitLevel.NO_FIT for p in self.profiles.values())
        busy = refusal_note(self.budget)
        if fits:
            hint = f"{smaller} ({busy})" if busy else smaller
        elif busy:
            hint = f"None of its profiles fits right now: {busy}"
        else:
            box = "this box's RAM" if self.budget.no_card else "this card"
            hint = f"None of its profiles fits {box}; pick a smaller model"
        raise NoRoomError(f"{needs} {hint}.")

    def _choose_vllm(
        self, base: Profile, quant: str | None, ctx: int | None, kv: KvType | None
    ) -> Profile:
        """A vLLM repo is one set of weights: the quant is the repo's, the context and the KV
        precision are planned (any quantised KV type means FP8 on vLLM)."""
        if quant is not None and quant.upper() != base.quant.upper():
            raise PlanError(ONE_SET.format(repo=self.facts.repo_id, quant=base.quant))
        chosen = base
        if (ctx is not None or kv is not None) and self.hf_config is not None:
            native = self.hf_config.text.max_position_embeddings
            if ctx is not None and (ctx < VLLM_MIN_CTX or (native and ctx > native)):
                raise PlanError(
                    f"Context {ctx:,} is outside {VLLM_MIN_CTX:,} to {native or 0:,} tokens."
                )
            base_args = base.vllm
            kv_name = (
                ("auto" if kv is KvType.F16 else "fp8")
                if kv is not None
                else (base_args.kv_cache_dtype if base_args else "auto")
            )
            chosen = plan_vllm_custom(
                self.facts,
                self.hf_config,
                self.budget,
                ctx or base.ctx,
                kv_name,
                base_args.tensor_parallel if base_args else None,
            )
        if chosen.fit.level is FitLevel.NO_FIT:
            self._refuse(
                f"{chosen.quant} at {chosen.ctx:,} context needs "
                f"{chosen.fit.gpu_bytes / 2**30:.1f} GiB of the card and only "
                f"{self.budget.card_free_bytes / 2**30:.1f} GiB is free.",  # the table's
                "Pick a smaller context, `--kv q8_0` (FP8), or an AWQ build of the model",
            )
        return chosen

    def _check_ctx(self, ctx: int) -> None:
        """An explicit context must be one the engine args accept and the model was trained
        for; said as a PlanError, never as pydantic's ValidationError from inside `recipe`."""
        if ctx < MIN_CTX:
            raise PlanError(f"Context {ctx:,} is below the minimum of {MIN_CTX:,} tokens.")
        native = self.header.context_length if self.header is not None else None
        if native and ctx > native:
            raise PlanError(
                f"Context {ctx:,} is above the {native:,} tokens this model was trained for."
            )

    def recipe(self, profile: Profile, name: str, port: int, upscaler: bool = False) -> Recipe:
        """The recipe for a chosen profile: the plan owns the engine decision. `upscaler`
        gives a picture instance the ESRGAN upscaler (never a video one: clips are not
        upscaled); its size is what the plan read from the Hub (`upscaler_bytes`). Its image
        is the engine's for the box's backend, when the budget read the box."""
        built = self._recipe(profile, name, port, upscaler)
        image = self.image
        return built if image is None else built.model_copy(update={"image": image})

    @property
    def image(self) -> str | None:
        """The engine's tag for this box's backend and arch; None for a budget that did not
        read the box (`--vram`): the recipe keeps the image every box ran before.
        A box the engine has no image for is refused while planning (`plan.make_plan`)."""
        budget = self.budget
        if budget.backend is None or self.facts.engine is None:
            return None
        found = spec(self.facts.engine).image_for(budget.backend, budget.arch or "x86_64")
        return found if isinstance(found, str) else None

    def _recipe(self, profile: Profile, name: str, port: int, upscaler: bool) -> Recipe:
        match self.facts.engine:
            case Engine.SDCPP if self.kit is not None:
                return sdcpp_recipe(
                    self.facts,
                    self.kit,
                    self.kit_sizes,
                    profile,
                    name,
                    port,
                    CheckpointLoad(
                        layout=self.checkpoint.weights.layout,
                        weight_type=self.checkpoint.parts.load_type,
                    )
                    if self.checkpoint is not None
                    else GGUF_KIT_LOAD,
                    self.upscaler_file() if upscaler and not self.kit.is_video else None,
                )
            case Engine.WHISPERCPP:
                return whispercpp_recipe(self.facts, profile, name, port)
            case Engine.AUDIOCPP:
                return audiocpp_recipe(self.facts, profile, name, port, cpu=self.budget.cpu)
            case Engine.LLAMACPP:
                return llamacpp_recipe(
                    self.facts,
                    profile,
                    name,
                    port,
                    cpu=self.budget.cpu,
                    load_mode=pool_load(self.budget),
                    cache=planned_cache(self.budget, profile, self.facts.modality),
                )
            case Engine.VLLM:
                return vllm_recipe(self.facts, profile, name, port)
        raise PlanError(f"{self.facts.repo_id}: no recipe for this plan.")

    def _dit_quant(self, kit: ImageKit, checkpoint: CheckpointChoice, quant: str) -> Profile:
        """A DiT file at the precision asked for: as stored or converted while loading."""
        parts = checkpoint.parts
        if not isinstance(parts, DitParts):  # `check_quant` let no other quant through
            raise PlanError(f"{quant}: a checkpoint runs at F16 and has no quants to pick.")
        try:
            return plan_dit_custom(
                self.facts,
                kit,
                checkpoint.weights,
                parts,
                self.kit_sizes,
                self.budget,
                quant,
            )
        except ValueError as exc:
            raise PlanError(str(exc)) from exc

    def _choose_audio(self, base: Profile, size: str | None, custom: CustomPlanner) -> Profile:
        """A whisper size or a voice package's quant: one label picks one file."""
        chosen = base
        if size is not None and size.lower() != base.quant.lower():
            try:
                chosen = custom(self.facts, self.budget, size)
            except ValueError as exc:
                raise PlanError(str(exc)) from exc
        if chosen.fit.level is FitLevel.NO_FIT:
            # a box without a card plans in RAM: "free VRAM" would name what it lacks
            where = "of RAM for models" if self.budget.no_card else "of free VRAM"
            self._refuse(
                f"{chosen.quant} needs {chosen.estimate.total_bytes / 2**30:.1f} GiB and does "
                f"not fit the {self.budget.card_free_bytes / 2**30:.1f} GiB {where}.",
                "Pick a smaller size or stop an instance",
            )
        return chosen

    def _choose_image(self, base: Profile, quant: str | None) -> Profile:
        if self.kit is None:
            raise PlanError(f"{self.facts.repo_id} is not an image plan.")
        chosen = base
        if quant is not None and self.checkpoint is not None:
            self.check_quant(quant)
            if quant.upper() != base.quant.upper():  # a DiT file: as stored, or Q8_0
                chosen = self._dit_quant(self.kit, self.checkpoint, quant)
        if quant is not None and quant.upper() != base.quant.upper() and self.checkpoint is None:
            try:
                chosen = plan_image_custom(
                    self.facts,
                    self.kit,
                    kit_sizes(self.kit.companions, self.kit_sizes),
                    self.budget,
                    quant,
                )
            except ValueError as exc:
                raise PlanError(str(exc)) from exc
        if chosen.fit.level is FitLevel.NO_FIT:
            smaller = (
                "Stop another instance, or pick a smaller model"
                if self.checkpoint is not None
                else "Pick a smaller quant"
            )
            self._refuse(
                f"{chosen.quant} needs {chosen.estimate.total_bytes / 2**30:.1f} GiB with its "
                "text encoder and VAE and does not fit this machine, even streaming weights "
                "from RAM.",
                smaller,
            )
        return chosen
