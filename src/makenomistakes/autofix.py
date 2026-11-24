"""What to change when a start fails for a reason the runner can fix (out of memory).

Pure: takes the recipe, the diagnoses and, when the service could gather them, the
resources (GGUF header, file sizes, free VRAM and RAM). With resources the next recipe is
*planned* with the same fit model `mnm add` used, so one retry lands on settings that fit
the card as it is now (the hard rule: every recipe is planned against this machine). Without
them (image models carry no readable header) a short ladder of safer settings is walked.
`InstanceService.start` applies the fix, saves it, and retries a bounded number of times,
announcing every change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.diagnose import Cause, Diagnosis
from makenomistakes.engine_args import KvType, LlamaCppArgs, SdCppArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.estimate import KV_BYTES_PER_VALUE, FitLevel, ngl_for
from makenomistakes.fit.profiles import CTX_LADDER, EMBEDDING_CTX, Planner, ProfileName
from makenomistakes.gguf.reader import GgufHeader
from makenomistakes.hub.facts import EMBEDDING_MODALITIES
from makenomistakes.hub.variants import GgufVariant
from makenomistakes.recipes.model import Recipe
from makenomistakes.recipes.registry import spec_for

MAX_ATTEMPTS = 3  # the first start plus two retries
MIN_CTX = 4096
SMALLER_SIDE = 768  # image fallback resolution
GIB = 1024**3


@dataclass(frozen=True)
class Resources:
    """What the fit model needs, gathered by the service from disk and nvidia-smi."""

    header: GgufHeader
    weights_bytes: int
    projector_bytes: int
    budget: Budget | None  # the card right after the dead container was removed; None: unread


@dataclass(frozen=True)
class Fix:
    recipe: Recipe
    what: str  # "Out of memory: retrying with context 8,192 (was 16,384)."


@dataclass(frozen=True)
class _Target:
    """llama.cpp settings a retry may move to: never more of anything than what died."""

    ctx: int
    kv: KvType
    layers: int
    cpu_moe: int = 0  # layers whose experts move to RAM; never fewer than before

    def shrinks(self, args: LlamaCppArgs, on_gpu: int) -> bool:
        smaller_kv = KV_BYTES_PER_VALUE[self.kv] < KV_BYTES_PER_VALUE[args.kv_type]
        more_in_ram = self.cpu_moe > (args.cpu_moe_layers or 0)
        return self.ctx < args.ctx or smaller_kv or self.layers < on_gpu or more_in_ram


def can_fix(recipe: Recipe, diagnoses: list[Diagnosis]) -> bool:
    """Could a retry with other settings help? Decided before anything is stopped or probed,
    so a failure auto-fix cannot touch keeps its container and log for `mnm logs`. Whisper
    has nothing to shrink (one small model): its diagnosis says what to stop instead."""
    oom = any(d.cause_code is Cause.OUT_OF_MEMORY for d in diagnoses)
    return oom and spec_for(recipe).autofix_ladder


def next_fix(
    recipe: Recipe, diagnoses: list[Diagnosis], resources: Resources | None = None
) -> Fix | None:
    """The next recipe to try for this failure, or None when nothing safe is left.

    Every fix uses less GPU memory than the start that died: context, KV precision and GPU
    layers only ever go down. With resources the target is planned by the fit model; when
    the plan shrinks nothing, the engine has disproved the estimate and one blind step down
    follows instead.
    """
    args = recipe.args
    if not can_fix(recipe, diagnoses):
        return None
    if isinstance(args, SdCppArgs):
        return _less_sd_memory(recipe, args)
    if isinstance(args, LlamaCppArgs):
        return _llama_fix(recipe, args, resources)
    return None  # unreachable: can_fix admits only the two engines with a ladder


def shrinker(
    diagnoses: list[Diagnosis], resources: Resources | None, applied: list[Fix]
) -> Callable[[Recipe], Recipe]:
    """A store update: the next fix for whatever recipe is on disk now (edits meanwhile win)."""

    def shrink(current: Recipe) -> Recipe:
        fix = next_fix(current, diagnoses, resources)
        if fix is None:
            return current
        applied.append(fix)
        return fix.recipe

    return shrink


def _llama_fix(recipe: Recipe, args: LlamaCppArgs, resources: Resources | None) -> Fix | None:
    embedding = recipe.modality in EMBEDDING_MODALITIES  # one signal for both paths
    blocks = resources.header.block_count if resources else None
    on_gpu = min(args.gpu_layers, blocks or args.gpu_layers)  # 99 means "all" to llama.cpp
    budget = resources.budget if resources else None
    if resources is None or budget is None:  # no plan without a header and a GPU reading
        return _less_llama_memory(recipe, args, embedding, on_gpu)
    target = _plan(recipe, args, resources, budget, embedding, on_gpu)
    if target is None:
        return None  # nothing fits this card now, not even offloaded to RAM
    if not target.shrinks(args, on_gpu):
        return _less_llama_memory(recipe, args, embedding, on_gpu)
    update: dict[str, object] = {
        "ctx": target.ctx,
        "kv_type": target.kv,
        "gpu_layers": target.layers,
        "cpu_moe_layers": target.cpu_moe or None,
    }
    if embedding:
        update.update(args.whole_input_batches(target.ctx))
    layers = blocks or on_gpu
    experts = f", experts of {target.cpu_moe} layers in RAM" if target.cpu_moe else ""
    return Fix(
        recipe.with_args(**update),
        f"Out of memory: retrying with context {target.ctx:,}, kv {target.kv.value}, "
        f"{min(target.layers, layers)}/{layers} layers on the GPU{experts} (planned against "
        f"{budget.vram_free_bytes / GIB:.1f} GiB free VRAM; was context "
        f"{args.ctx:,}, kv {args.kv_type.value}, {on_gpu} layers).",
    )


def _plan(
    recipe: Recipe,
    args: LlamaCppArgs,
    res: Resources,
    budget: Budget,
    embedding: bool,
    on_gpu: int,
) -> _Target | None:
    """The planner's own choice (best fit level, largest context) at or below the current
    context and the native one, clamped so no setting rises above what just died."""
    kv = args.kv_type if embedding or args.kv_type is not KvType.F16 else KvType.Q8_0
    cap = min(args.ctx, res.header.context_length or args.ctx)
    # Embeddings use their own contexts (8192 / 2048 / 1024): the chat ladder stopped at 4096
    # while 2048 and 1024 still fit.
    ladder = sorted(set(EMBEDDING_CTX.values()), reverse=True) if embedding else CTX_LADDER
    ctxs = [c for c in ladder if c <= cap] or [cap]
    variant = GgufVariant(
        quant=recipe.quant, files=list(recipe.files), size_bytes=res.weights_bytes
    )
    planned = Planner(res.header, budget, res.projector_bytes).place(
        ProfileName.CUSTOM, variant, ctxs, kv, "", whole_input=embedding, ubatch=args.ubatch
    )
    if planned.fit.level is FitLevel.NO_FIT:
        return None
    # Everything on the GPU is `-ngl blocks + 1` (the output layer); never more than died.
    layers = min(ngl_for(planned.fit), args.gpu_layers)
    # Experts in RAM were priced with every block on the GPU; with blocks in RAM already the
    # two together are a placement no fit model priced, so the experts stay as they were.
    whole = layers == ngl_for(planned.fit)
    cpu_moe = max(planned.fit.cpu_moe_layers if whole else 0, args.cpu_moe_layers or 0)
    return _Target(ctx=planned.ctx, kv=kv, layers=layers, cpu_moe=cpu_moe)


def _less_llama_memory(
    recipe: Recipe, args: LlamaCppArgs, embedding: bool, on_gpu: int
) -> Fix | None:
    """One safe step down: 8-bit KV (chat), then half the context, then fewer GPU layers.
    `on_gpu` is the layer count that actually sat on the GPU (99 means "all" to llama.cpp)."""
    if args.kv_type is KvType.F16 and not embedding:
        return Fix(
            recipe.with_args(kv_type=KvType.Q8_0),
            "Out of memory: retrying with an 8-bit KV cache (kv=q8_0; quality is unchanged in "
            "practice).",
        )
    floor = min(EMBEDDING_CTX.values()) if embedding else MIN_CTX
    if args.ctx > floor:
        ctx = max(floor, args.ctx // 2)
        update: dict[str, object] = {"ctx": ctx}
        if embedding:  # a chat micro-batch is left alone
            update.update(args.whole_input_batches(ctx))
        return Fix(
            recipe.with_args(**update),
            f"Out of memory: retrying with context {ctx:,} (was {args.ctx:,}).",
        )
    if on_gpu > 1:
        layers = max(1, on_gpu * 3 // 4)
        return Fix(
            recipe.with_args(gpu_layers=layers),
            f"Out of memory: retrying with {layers} layers on the GPU (was {on_gpu}); "
            "the rest run from RAM, slower.",
        )
    return None


@dataclass(frozen=True)
class _Rung:
    """One step of the sd.cpp ladder: the setting it changes, and when it applies."""

    field: str
    value: object
    what: str
    applies: Callable[[Recipe], bool] = lambda _recipe: True

    def next_for(self, recipe: Recipe, args: SdCppArgs) -> Fix | None:
        if not self.applies(recipe) or getattr(args, self.field) == self.value:
            return None
        return Fix(recipe.with_args(**{self.field: self.value}), f"Out of memory: {self.what}.")


# The cheapest concession first: a whole 1024x1024 decode needs 6.5 GiB of its own, a tiled
# one 0.4 GiB (fit/image.py); an encoder on the CPU costs seconds per new prompt; Q8_0 while
# loading halves a DiT for a slight quality loss; streaming is 3x slower per image.
_SD_LADDER: tuple[_Rung, ...] = (
    _Rung("vae_tiling", True, "retrying with VAE tiling (decodes in tiles, same output)"),
    _Rung("text_encoder_on_cpu", True, "retrying with the text encoder in RAM (--backend te=cpu)"),
    _Rung("vae_on_cpu", True, "retrying with the VAE in RAM"),
    _Rung(
        "diffusion_type",
        "q8_0",
        "retrying with the diffusion model converted to Q8_0 while loading (half its memory, "
        "slight quality loss)",
        applies=lambda recipe: recipe.converts_dit,
    ),
    _Rung("offload_to_cpu", True, "retrying with weights streamed from RAM (slower per image)"),
)


def _less_sd_memory(recipe: Recipe, args: SdCppArgs) -> Fix | None:
    for rung in _SD_LADDER:
        if (fix := rung.next_for(recipe, args)) is not None:
            return fix
    if max(args.width, args.height) > SMALLER_SIDE:
        update = {"width": min(args.width, SMALLER_SIDE), "height": min(args.height, SMALLER_SIDE)}
        return Fix(
            recipe.with_args(**update),
            f"Out of memory: retrying at {update['width']}x{update['height']} "
            f"(was {args.width}x{args.height}).",
        )
    return None
