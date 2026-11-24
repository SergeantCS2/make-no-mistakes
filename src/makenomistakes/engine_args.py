"""Engine arguments, named after what they control rather than their letters.

Kept apart from the recipe so the fit layer can describe a profile with the same typed
object the recipe will run with (no parallel settings class to copy field by field).
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from makenomistakes.card_placement import Placement
from makenomistakes.engine_args_music import (
    OPTIONS_KIND,
    SPEECH_FAMILIES,
    MusicFamilyName,
    MusicOptions,
    MusicTask,
)
from makenomistakes.hub.kits import Sampler, Scheduler

MIN_CTX = 512  # llama.cpp's smallest useful context; the planner refuses less
# A draft model's tokens per step: llama.cpp's own default (`--spec-draft-n-max`, 3).
# qwen3-8b with Qwen3-0.6B at 16 on an RTX 4070: 75.6 tok/s alone fell to 27.1, 8% of 2,710
# drafted tokens kept, the rest checked for nothing at Qwen3's sampling.
DRAFT_MAX = 3
# The least probability the draft must give a token before it drafts it
# (`--spec-draft-p-min`; llama.cpp's own is 0, drafting even its wild guesses).
DRAFT_P_MIN = 0.75


class KvType(StrEnum):
    """KV cache element type (`-ctk`/`-ctv`); `fit.estimate` knows the bytes per value."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


DRAFT_KV = KvType.F16  # a draft model's KV cache (`-ctkd/-ctvd`), whatever its target's
# Where an instance's saved slots sit in its container (`<models>/prompt-cache/<name>`)
SLOT_CACHE_DIR = "/prompt-cache"
# The prompt cache's fields (`prompt_cache`): RAM and disk beside the engine, so they key
# no reading and leave the planned estimate (`measurements.settings_key`, `MEMORY_NEUTRAL`)
PROMPT_CACHE_FIELDS = ("cache_ram_mib", "slot_cache", "cache_disk_gib")


LoadMode = Literal["none", "mmap", "dio"]  # unset is llama.cpp's own `auto`
NUMA_PATTERN = r"^(distribute|isolate|off|node:\d{1,3})$"
CPU_LIST_PATTERN = r"^\d+(-\d+)?(,\d+(-\d+)?)*$"  # the kernel's list format: `0-7,32-39`
NUMA_FLAGS = ("distribute", "isolate")  # what llama.cpp's `--numa` takes of them


class LlamaCppArgs(BaseModel):
    """Flags for llama-server, named after what they control rather than their letters."""

    engine: Literal["llamacpp"] = "llamacpp"
    # Settings that leave what it holds on the GPU as planned: changing only these keeps the
    # saved estimate (`Recipe.with_args`; an RTX 4070 read 9.80 GiB with `load=none`
    # against 9.85 without).
    MEMORY_NEUTRAL: ClassVar[frozenset[str]] = frozenset(
        {
            "load_mode",
            "threads",
            "numa",
            "numa_cpus",
            "reasoning_format",
            "jinja",
            "sleep_idle_seconds",
            "draft_max",
            "draft_p_min",
            *PROMPT_CACHE_FIELDS,  # RAM and disk beside the card, never on it
        }
    )
    ctx: int = Field(ge=MIN_CTX)
    # the model's layer count means "everything on the GPU"; 0 every block in RAM, the CPU
    # backend's only plan (no card)
    gpu_layers: int = Field(ge=0)
    kv_type: KvType = KvType.F16
    flash_attn: Literal["on", "off", "auto"] = "auto"
    parallel: int = Field(default=1, ge=1, le=16)
    threads: int | None = Field(default=None, ge=1)
    # The CPU backend on a box of several NUMA nodes: `--numa distribute` (1.96 → 4.69 tok/s
    # on a 4-node Rome); None on one node and on a card. The user's `numa=` (`numa_setting`):
    # `distribute` and `isolate` are llama.cpp's own `--numa`; `node:N` binds the container to
    # node N's CPUs and memory (Docker's `--cpuset-cpus numa_cpus --cpuset-mems N`: the image
    # has no `numactl`); `off` renders nothing and keeps a re-plan from choosing one
    numa: str | None = Field(default=None, pattern=NUMA_PATTERN)
    numa_cpus: str | None = Field(default=None, pattern=CPU_LIST_PATTERN)
    batch: int | None = Field(default=None, ge=1)
    ubatch: int | None = Field(default=None, ge=1)
    cpu_moe_layers: int | None = Field(default=None, ge=0)
    # `--moe-cache-mib`: this many MiB of the card cache the hottest of the experts kept in
    # RAM (`cpu_moe_layers`); None or 0 is off, llama.cpp's own default.
    moe_cache_mib: int | None = Field(default=None, ge=0, le=1024 * 1024)
    reasoning_format: Literal["auto", "none", "deepseek"] = "auto"
    jinja: bool = True
    sleep_idle_seconds: int = Field(default=0, ge=0)  # 0 keeps the model loaded forever
    # Vision: the fewest tokens an image becomes. Qwen-VL asks for 1024 (llama.cpp warns
    # below it: grounding degrades); None keeps the projector's own minimum.
    image_min_tokens: int | None = Field(default=None, ge=1)
    pooling: Literal["none", "mean", "cls", "last", "rank"] | None = None  # embeddings only
    # How the weights are read (`--load-mode`); None is the engine's own (`auto`: mmap).
    # mlock is left out: Docker's default locked-memory limit refuses a model's worth.
    load_mode: LoadMode | None = None
    # Tensors read from disk row by row as needed (an Engram table): `--load-mode mmap
    # --lazy-mode on -ot <name>=CPU`; their bytes count toward disk only, never RAM or VRAM.
    lazy_tensors: tuple[str, ...] = ()
    # With a draft model (`Recipe.draft`): the most tokens it drafts per step
    # (`--spec-draft-n-max`); the engine drafts fewer where the draft grows unsure.
    draft_max: int = Field(default=DRAFT_MAX, ge=1, le=64)
    draft_p_min: float = Field(default=DRAFT_P_MIN, ge=0.0, le=1.0)  # `--spec-draft-p-min`
    # Several cards (`recipes.cards.split_argv` renders them; nothing on one card):
    # each card's share of the `-ngl` layers in nvidia-smi's order (`-ts`, the MiB each had
    # free when planned), `-sm` (None: layer, llama.cpp's default) and the card, by its place
    # among the instance's cards, for intermediate results (`-mg`).
    tensor_split: tuple[int, ...] | None = None
    split_mode: Literal["layer", "tensor"] | None = None
    main_gpu: int | None = Field(default=None, ge=0, le=15)
    # Several cards (`card_placement`, rendered by `recipes.cards.split_argv` with its
    # `--device`, `-ts`, `-mg` and one `-ot`): a MoE's blocks and experts per card, the rest of
    # the experts in RAM; and the user's choice of where experts go (`experts=`, None: auto)
    placement: Placement | None = None
    experts: Literal["auto", "ram", "cards"] | None = None
    # The prompt cache (`prompt_cache`): `--cache-ram`, the most MiB of RAM llama.cpp keeps
    # earlier prompts' states in (0 off; None renders nothing: llama.cpp's own 8 GiB, an older
    # recipe until `mnm replan`); `slot_cache`, the slots saved to disk when the
    # instance unloads and restored at its next start (None: auto, on for a coding instance,
    # `llamacpp_cache.keeps_slots`; rendered with its folder's mount there, which knows the
    # type); `cache_disk_gib`, the most those files take
    cache_ram_mib: int | None = Field(default=None, ge=0, le=1024 * 1024)
    slot_cache: bool | None = None
    cache_disk_gib: float | None = Field(default=None, gt=0, le=4096)
    extra: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        """One table cell: the settings that matter most."""
        return f"ctx {self.ctx:,}, KV {self.kv_type.value}"

    @property
    def experts_in_ram(self) -> bool:
        """Some routed experts stay in RAM: `--n-cpu-moe` on one card, a placement's on
        several. The expert cache (`moe_cache_mib`) caches only those."""
        return bool(self.cpu_moe_layers) or bool(self.placement and self.placement.in_ram)

    @staticmethod
    def whole_input_micro_batch(ctx: int, whole_input: bool, ubatch: int | None) -> int | None:
        """The planner's micro-batch: the whole input for embedding and rerank (one pass per
        input), capped by a micro-batch the recipe already runs; None keeps the engine default.
        The recipe builder then sets batch = ubatch to it (the one rule for both)."""
        limits = [n for n in (ctx if whole_input else None, ubatch) if n is not None]
        return min(limits) if limits else None

    def whole_input_batches(self, ctx: int) -> dict[str, object]:
        """For embedding and rerank the whole input is one micro-batch, so a batch tied to the
        context (equal to it) follows a new context up or down; one the user set lower stays,
        capped at the new context; one left unset keeps the engine default. Auto-fix (which
        only lowers) and `mnm set ctx=` both apply this one rule."""
        current = {"ubatch": self.ubatch, "batch": self.batch}
        return {
            key: self.whole_input_micro_batch(ctx, True, None if value == self.ctx else value)
            for key, value in current.items()
            if value is not None
        }

    def to_cli(self, slot_dir: str | None = None) -> list[str]:
        """The flags; `slot_dir` the saved slots' folder in the container, given only with its
        mount (`llamacpp_cache.slot_dir`: whether it keeps them depends on the type)."""
        argv = [
            # llama-server fits unset arguments to the card by itself (`--fit on`, its default
            # in current builds): mnm plans `-ngl`, `--n-cpu-moe` and the context, so the
            # engine's own fit is off, or an engine update could re-plan an instance.
            "--fit",
            "off",
            "-c",
            str(self.ctx),
            "-ngl",
            str(self.gpu_layers),
            "-fa",
            self.flash_attn,
            "-ctk",
            self.kv_type.value,
            "-ctv",
            self.kv_type.value,
            "-np",
            str(self.parallel),
            "--reasoning-format",
            self.reasoning_format,
            "--jinja" if self.jinja else "--no-jinja",
        ]
        for flag, value in self.optional_flags(slot_dir):
            if value is not None:
                argv += [flag, str(value)]
        if self.lazy_tensors:  # a placement's `-ot` carries them first (one list)
            argv += ["--lazy-mode", "on"]
            if self.placement is None:
                argv += ["-ot", ",".join(f"{t}=CPU" for t in self.lazy_tensors)]
        return argv + list(self.extra)

    def optional_flags(self, slot_dir: str | None = None) -> list[tuple[str, object | None]]:
        """The flags `to_cli` renders when set, in its order; the quality tune's
        `llama-perplexity` line takes the ones every llama.cpp tool takes
        (`quality.perplexity.LEFT_OUT` the rest)."""
        return [
            ("-t", self.threads),
            ("--numa", self.numa if self.numa in NUMA_FLAGS else None),
            ("-b", self.batch),
            ("-ub", self.ubatch),
            ("--n-cpu-moe", self.cpu_moe_layers),
            # only beside experts in RAM (a re-plan or auto-fix that keeps none drops it); on
            # several cards llama.cpp splits the one value like `-ts`
            ("--moe-cache-mib", (self.moe_cache_mib or None) if self.experts_in_ram else None),
            ("--sleep-idle-seconds", self.sleep_idle_seconds or None),
            ("--pooling", self.pooling),
            ("--image-min-tokens", self.image_min_tokens),
            ("--load-mode", "mmap" if self.lazy_tensors else self.load_mode),
            ("--cache-ram", self.cache_ram_mib),
            ("--slot-save-path", slot_dir),
        ]

    @property
    def numa_node(self) -> int | None:
        """The node `numa=node:N` binds to; None otherwise."""
        if self.numa is None or not self.numa.startswith("node:"):
            return None
        return int(self.numa.removeprefix("node:"))

    def cpuset_argv(self) -> list[str]:
        """Docker's binding for `numa=node:N`: the node's CPUs and its memory only."""
        node = self.numa_node
        if node is None or self.numa_cpus is None:
            return []
        return ["--cpuset-cpus", self.numa_cpus, "--cpuset-mems", str(node)]

    @model_validator(mode="after")
    def _node_needs_cpus(self) -> LlamaCppArgs:
        """`node:N` renders the node's CPUs: a binding without them (or CPUs without one)
        would start unbound in silence."""
        if (self.numa_node is None) != (self.numa_cpus is None):
            raise ValueError(
                "numa=node:N needs the node's CPUs (numa_cpus) and they go only with it: set it "
                "with `mnm set <name> numa=node:N`, which reads them from the box"
            )
        return self

    @model_validator(mode="after")
    def _lazy_needs_mmap(self) -> LlamaCppArgs:
        """A tensor read from disk as needed is read through the memory map: `load=none`
        (read whole into RAM) or `dio` would load a 111 GB Engram table into RAM."""
        if self.lazy_tensors and self.load_mode not in (None, "mmap"):
            raise ValueError(
                f"load={self.load_mode} cannot run this model: its Engram table "
                f"({', '.join(self.lazy_tensors)}) stays on disk and is read row by row through "
                "the memory map (`--load-mode mmap --lazy-mode on`); leave load empty or mmap"
            )
        return self


# sd.cpp prefixes the diffusion model's tensors with this when it loads `--diffusion-model`;
# `--tensor-type-rules` matches it as a regex (verified on a CPU build: only the DiT changed).
DIFFUSION_TENSORS = r"^model\.diffusion_model\."


PACED_FAMILIES = frozenset({"kokoro_tts"})  # voices that take a speaking rate


class AudioCppArgs(BaseModel):
    """audio.cpp's server: what its JSON config says about the one model it serves.
    For text-to-speech `voice` is a voice id of the family (the config declares it as the
    default preset); `speed` and `seed` are the request defaults a client may override. A
    music instance names a studio family, its task and its typed `options`."""

    engine: Literal["audiocpp"] = "audiocpp"
    # Settings that leave what it holds on the GPU as planned: changing only these keeps the
    # saved estimate (`Recipe.with_args`; request defaults).
    MEMORY_NEUTRAL: ClassVar[frozenset[str]] = frozenset({"voice", "speed", "seed", "threads"})
    family: Literal["kokoro_tts", "pocket_tts", "qwen3_tts"] | MusicFamilyName
    voice: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)  # only where `paced` says so
    seed: int | None = Field(default=None, ge=0)
    threads: int | None = Field(default=None, ge=1)
    language: str | None = None  # PocketTTS: the package folder's language (a load option)
    # Qwen3-TTS VoiceDesign describes the voice instead (`vdes`); the studio's tasks
    task: Literal["tts", "vdes"] | MusicTask = "tts"
    options: MusicOptions | None = Field(default=None, discriminator="kind")

    @property
    def paced(self) -> bool:
        """Whether the family takes a speaking rate at all. PocketTTS and Qwen3-TTS answer
        500 "does not support runtime style controls" to any `speed`, 1.0 included (CPU
        build, 2026-10-15), so the router drops a client's and the config sets none."""
        return self.family in PACED_FAMILIES

    @property
    def drops_speed(self) -> bool:
        """The router drops a client's `speed` for this voice: it refuses one."""
        return not self.paced and not self.music

    @property
    def music(self) -> bool:
        """A studio instance: a song, stems, a conversion or a score, not a voice."""
        return self.family not in SPEECH_FAMILIES

    @model_validator(mode="after")
    def _speed_where_it_applies(self) -> AudioCppArgs:
        if self.speed != 1.0 and not self.paced:
            raise ValueError(f"{self.family} reads at its own pace: `speed` applies to Kokoro only")
        speech_task = self.task in {"tts", "vdes"}
        if self.music == speech_task:
            raise ValueError(f"{self.family} does not run the task `{self.task}`")
        if self.options is not None and self.options.kind != OPTIONS_KIND.get(self.family):
            raise ValueError(f"{self.family} takes no {self.options.kind} options")
        return self

    @property
    def summary(self) -> str:
        if self.music:
            seed = f", seed {self.seed}" if self.seed is not None else ""
            what = self.options.summary if self.options is not None else self.task
            return f"{what}{seed}"
        seed = f", seed {self.seed}" if self.seed is not None else ""
        speed = f", speed {self.speed:g}" if self.paced else ""
        return f"voice {self.voice}{speed}{seed}"


class SdCppArgs(BaseModel):
    """How an image instance runs: sd-server's startup defaults, and how its port is served.
    A request's `size` overrides width/height per call unless the router drops it
    (`client_size` off, the default); `client_size` is the router's setting, never a flag."""

    engine: Literal["sdcpp"] = "sdcpp"
    # Settings that leave what it holds on the GPU as planned: changing only these keeps the
    # saved estimate (`Recipe.with_args`; per-step and prompt defaults: size, frames and
    # placement do move it).
    MEMORY_NEUTRAL: ClassVar[frozenset[str]] = frozenset(
        {
            "steps",
            "cfg_scale",
            "sampler",
            "scheduler",
            "flow_shift",
            "seed",
            "clip_skip",
            "negative_prompt",
            "guidance",
            "strength",
            "high_noise_steps",
            "threads",
        }
    )
    width: int = Field(default=1024, ge=64, le=4096, multiple_of=16)
    height: int = Field(default=1024, ge=64, le=4096, multiple_of=16)
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    sampler: Sampler = "euler"
    scheduler: Scheduler | None = None  # None: the engine's choice for the model family
    flow_shift: float | None = Field(default=None, ge=0.0)
    seed: int | None = Field(default=None, ge=0)  # None: a new random seed per image
    clip_skip: int | None = Field(default=None, ge=1, le=12)
    negative_prompt: str = ""  # default for every request that sends none
    guidance: float | None = Field(default=None, ge=0.0, le=30.0)  # distilled guidance (FLUX)
    # How much a picture sent to `/v1/images/edits` is repainted (`--strength`; the engine's
    # 0.75 when unset): 0.3 keeps its layout, 1.0 redraws it. A mask limits it to the white.
    strength: float | None = Field(default=None, gt=0.0, le=1.0)
    # A video instance: the clip's length and rate (`--video-frames`, `--fps`); None for images.
    # (frames - 1) should divide by the family's temporal compression (33, 49, 81 for Wan).
    frames: int | None = Field(default=None, ge=1, le=257)
    fps: int | None = Field(default=None, ge=1, le=60)
    # Wan 2.2 A14B: steps the high-noise model takes before the low-noise one's `steps`.
    high_noise_steps: int | None = Field(default=None, ge=1, le=150)
    flash_attn: bool = True  # in the diffusion model only: less memory, faster on CUDA
    offload_to_cpu: bool = False  # weights live in RAM and stream to the GPU per stage
    vae_tiling: bool = False
    text_encoder_on_cpu: bool = False
    vae_on_cpu: bool = False
    threads: int | None = Field(default=None, ge=1)
    main_flag: Literal["--diffusion-model", "--model"] = "--diffusion-model"
    weight_type: Literal["f16"] | None = None  # `--type f16`: a F32/BF16 checkpoint loads as F16
    # A DiT file converted while loading, the diffusion model only: `--type` would also
    # re-quantize a GGUF text encoder (sd.cpp converts any tensor it is told to).
    diffusion_type: Literal["bf16", "q8_0"] | None = None
    # Off: the router drops a client's `size`, so every image is width x height
    # (Odysseus asks 512x512 for a direct message). On: the client's size wins.
    client_size: bool = False
    extra: list[str] = Field(default_factory=list)

    @property
    def keeps_size(self) -> bool:
        """The router passes a client's `size` on (`client-size=on`)."""
        return self.client_size

    @property
    def summary(self) -> str:
        if self.frames is not None:
            return f"{self.width}x{self.height}, {self.frames} frames, {self.steps} steps"
        return f"{self.width}x{self.height}, {self.steps} steps"

    def _video_cli(self) -> list[str]:
        """A video instance's clip defaults; nothing for an image instance."""
        argv: list[str] = []
        if self.frames is not None:
            argv += ["--video-frames", str(self.frames)]
        if self.fps is not None:
            argv += ["--fps", str(self.fps)]
        if self.high_noise_steps is not None:
            # The high-noise stage shares the sampler and cfg; its default (-1) would be
            # "auto" through `--moe-boundary` instead of a step count.
            argv += ["--high-noise-steps", str(self.high_noise_steps)]
            argv += ["--high-noise-cfg-scale", f"{self.cfg_scale:g}"]
            argv += ["--high-noise-sampling-method", self.sampler]
        return argv

    def _sampling_cli(self) -> list[str]:
        """The request defaults past size and steps, each only when set."""
        argv: list[str] = []
        if self.flow_shift is not None:
            argv += ["--flow-shift", f"{self.flow_shift:g}"]
        if self.scheduler is not None:
            argv += ["--scheduler", self.scheduler]
        # sd-server's own default seed is a fixed 42; a negative seed means "random per image".
        argv += ["-s", str(self.seed if self.seed is not None else -1)]
        if self.clip_skip is not None:
            argv += ["--clip-skip", str(self.clip_skip)]
        if self.negative_prompt:
            argv += ["-n", self.negative_prompt]
        if self.guidance is not None:
            argv += ["--guidance", f"{self.guidance:g}"]
        if self.strength is not None:
            argv += ["--strength", f"{self.strength:g}"]
        return argv

    def to_cli(self) -> list[str]:
        argv = [
            "-W",
            str(self.width),
            "-H",
            str(self.height),
            "--steps",
            str(self.steps),
            "--cfg-scale",
            f"{self.cfg_scale:g}",
            "--sampling-method",
            self.sampler,
        ]
        argv += self._sampling_cli()
        argv += self._video_cli()
        if self.threads is not None:
            argv += ["-t", str(self.threads)]
        if self.weight_type is not None:
            argv += ["--type", self.weight_type]
        if self.diffusion_type is not None:
            argv += ["--tensor-type-rules", f"{DIFFUSION_TENSORS}={self.diffusion_type}"]
        if self.weight_type is not None or self.diffusion_type is not None:
            # sd.cpp's `auto` LoRA mode reads the types as stored, not the ones it converts
            # to, so it merges a LoRA "immediately" into weights it is quantizing: 0 of 840
            # tensors applied, then GGML_ASSERT (reproduced on a CPU build of f89d9b1). At
            # runtime the same LoRA applies 840 of 840.
            argv += ["--lora-apply-mode", "at_runtime"]
        if self.flash_attn:
            # `--diffusion-fa` is deprecated for this, the same modes (sd.cpp `common.cpp`);
            # the build warned at every start.
            argv += ["--flash-attn", "off,diffusion=on"]
        for flag, on in (
            ("--offload-to-cpu", self.offload_to_cpu),
            ("--vae-tiling", self.vae_tiling),
        ):
            if on:
                argv.append(flag)
        # `--clip-on-cpu`/`--vae-on-cpu` are deprecated upstream in favour of module backends.
        on_cpu = [m for m, on in (("te", self.text_encoder_on_cpu), ("vae", self.vae_on_cpu)) if on]
        if on_cpu:
            argv += ["--backend", ",".join(["all=cuda0", *(f"{m}=cpu" for m in on_cpu)])]
        return argv + list(self.extra)
