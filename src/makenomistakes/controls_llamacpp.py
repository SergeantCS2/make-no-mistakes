"""llama.cpp's settings: chat, coding, vision, embedding and rerank instances."""

from __future__ import annotations

from makenomistakes.controls_catalog import CHAT_MODALITIES, Control, Kind, check_controls
from makenomistakes.engine_args import DRAFT_MAX, DRAFT_P_MIN, LlamaCppArgs
from makenomistakes.hub.facts import EMBEDDING_MODALITIES, Modality
from makenomistakes.recipes.model import Recipe

LLM_CONTROLS: tuple[Control, ...] = (
    Control(
        key="ctx",
        field="ctx",
        label="Context length",
        help="Tokens the model can see at once. Doubling it roughly doubles KV cache memory.",
        kind=Kind.INT,
        recommended="16384 for chat, 32768 for coding, as high as fits for long documents",
    ),
    Control(
        key="gpu-layers",
        field="gpu_layers",
        label="Layers on GPU",
        help=(
            "The model's layer count plus one (llama.cpp counts the output layer too) keeps "
            "everything on the GPU; lower it only when VRAM is short (slower)."
        ),
        kind=Kind.INT,
        recommended="all layers (the profile sets this from the fit estimate)",
    ),
    Control(
        key="kv",
        modalities=CHAT_MODALITIES,
        field="kv_type",
        label="KV cache precision",
        help=(
            "f16 is exact; q8_0 halves KV memory with no visible loss; q4_0 quarters it "
            "with slight loss."
        ),
        kind=Kind.CHOICE,
        choices=("f16", "q8_0", "q4_0"),
        recommended="q8_0 above 16k context, f16 otherwise",
    ),
    Control(
        key="flash-attn",
        modalities=CHAT_MODALITIES,
        field="flash_attn",
        label="Flash attention",
        help="Faster attention with less memory. auto lets llama.cpp decide per model.",
        kind=Kind.CHOICE,
        choices=("auto", "on", "off"),
        recommended="auto",
    ),
    Control(
        key="parallel",
        field="parallel",
        label="Parallel requests",
        help="Slots served at once. Each slot gets ctx / parallel tokens, so raise ctx too.",
        kind=Kind.INT,
        recommended="1 for one person; 2 when a pipeline runs several agents at once",
    ),
    Control(
        key="threads",
        field="threads",
        label="CPU threads",
        help="Only matters for layers left on the CPU. Leave empty to let llama.cpp pick.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty, or the number of physical cores when offloading",
    ),
    Control(
        key="numa",
        field="numa",
        label="NUMA (several memory nodes)",
        help=(
            "On a box of several NUMA nodes (an EPYC in NPS4, two sockets): distribute spreads "
            "the threads over every node; isolate keeps them on the node llama.cpp starts on; "
            "node:N binds the container to node N's CPUs and memory; auto picks node:N when the "
            "model's RAM part fits one node, else distribute; off sets none. Empty on one node."
        ),
        kind=Kind.TEXT,
        nullable=True,
        recommended="auto on a box of several nodes (`mnm doctor` says how many); empty on one",
    ),
    Control(
        key="batch",
        field="batch",
        label="Batch size",
        help="Prompt-processing batch. Larger is faster for long prompts and uses more memory.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty (2048); embedding instances get the context from the profile",
    ),
    Control(
        key="ubatch",
        field="ubatch",
        label="Micro-batch size",
        help="Physical batch per step. Lower it if prompt processing runs out of memory.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty (512); 256 when VRAM is tight; embedding instances get the context",
    ),
    Control(
        key="cpu-moe-layers",
        modalities=CHAT_MODALITIES,
        field="cpu_moe_layers",
        label="MoE expert layers on CPU",
        help="Mixture-of-experts models only: keep this many layers' experts in RAM.",
        kind=Kind.INT,
        nullable=True,
        recommended="empty for dense models",
    ),
    Control(
        key="moe-cache",
        modalities=CHAT_MODALITIES,
        field="moe_cache_mib",
        label="MoE expert cache on GPU (MiB)",
        help=(
            "With experts kept in RAM (cpu-moe-layers): this many MiB of the card hold the "
            "experts used most, so fewer are read over PCIe per token. Empty or 0 is off."
        ),
        kind=Kind.INT,
        nullable=True,
        recommended="empty; about a tenth of the RAM-held experts' size to try (an experiment)",
    ),
    Control(
        key="reasoning",
        modalities=CHAT_MODALITIES,
        field="reasoning_format",
        label="Thinking output",
        help="auto separates <think> blocks into reasoning_content; none leaves them inline.",
        kind=Kind.CHOICE,
        choices=("auto", "none", "deepseek"),
        recommended="auto",
    ),
    Control(
        key="jinja",
        modalities=CHAT_MODALITIES,
        field="jinja",
        label="Model chat template",
        help=(
            "Use the template shipped in the GGUF (needed for tool calling). Turn off only "
            "for broken templates."
        ),
        kind=Kind.BOOL,
        recommended="on",
    ),
    Control(
        key="image-min-tokens",
        modalities=frozenset({Modality.VISION_CHAT}),
        field="image_min_tokens",
        label="Image tokens (minimum)",
        help="The fewest tokens an image becomes; more tokens see finer detail, cost more.",
        kind=Kind.INT,
        nullable=True,
        recommended="1024 for Qwen-VL (set by its profile); empty for other vision models",
    ),
    Control(
        key="pooling",
        modalities=EMBEDDING_MODALITIES,
        field="pooling",
        label="Embedding pooling",
        help=(
            "How token vectors become one embedding. Empty uses the model's own setting; "
            "`last` for Qwen3-Embedding, `mean` for nomic/bge if results look wrong."
        ),
        kind=Kind.CHOICE,
        choices=("none", "mean", "cls", "last", "rank"),
        nullable=True,
        recommended="empty (from the model)",
    ),
    Control(
        key="load",
        field="load_mode",
        label="Model loading",
        help=(
            "How the weights are read: mmap maps the file (auto, the default, does unless a "
            "device cannot); none reads it into memory, which llama.cpp suggests when MoE "
            "experts stay in RAM; dio reads past the page cache where the disk allows."
        ),
        kind=Kind.CHOICE,
        choices=("none", "mmap", "dio"),  # empty is llama.cpp's own auto (one "auto" listed)
        nullable=True,
        recommended="empty (auto); try none for a MoE with experts in RAM and compare tok/s",
    ),
    Control(
        key="cache",
        modalities=CHAT_MODALITIES,
        field="slot_cache",
        label="Keep the prompt across unloads",
        help=(
            "Saves the slots' prompts to disk when the model unloads (idle, making room, a "
            "stop) and puts the newest back at its next load, so a long prompt is not read "
            "again. The files hold the conversation; `off` deletes them."
        ),
        kind=Kind.BOOL,
        nullable=True,
        recommended="on for coding (an agent's long prompt), off otherwise; the plan's default",
    ),
    Control(
        key="cache-ram",
        modalities=CHAT_MODALITIES,
        field="cache_ram_mib",
        label="Prompt cache in RAM (MiB)",
        help=(
            "The most RAM llama.cpp keeps earlier prompts' states in, so a conversation a side "
            "request pushed out comes back without being read again. 0 is off; empty is "
            "llama.cpp's own 8 GiB, which nothing plans around."
        ),
        kind=Kind.INT,
        nullable=True,
        recommended="the plan's (a quarter of the RAM the model leaves, at most 8192)",
    ),
    Control(
        key="cache-disk",
        modalities=CHAT_MODALITIES,
        field="cache_disk_gib",
        label="Saved prompts on disk (GiB)",
        help=(
            "With cache=on: the most its saved slots take under the models folder; the oldest "
            "go first. A slot larger than this is not saved."
        ),
        kind=Kind.FLOAT,
        nullable=True,
        recommended="the plan's (one full-context prompt, at least 2)",
    ),
    Control(
        key="idle-sleep",
        field="sleep_idle_seconds",
        label="Unload after idle (seconds)",
        help=(
            "Frees VRAM after this long without requests; the port stays up and the model "
            "reloads on the next request. 0 disables."
        ),
        kind=Kind.INT,
        recommended="0 for daily drivers; 600 for models you use occasionally",
    ),
)

# A draft model: another instance's file, kept on the recipe (`draft.with_draft`).
DRAFT_CONTROL = Control(
    key="draft",
    modalities=CHAT_MODALITIES,
    field="draft",
    label="Draft model (speculative decoding)",
    help=(
        "Another saved llama.cpp instance of the same vocabulary (its family's smallest "
        "size) drafts tokens this model checks in one pass: the same reply, often faster. "
        "`mtp` drafts with the model's own MTP head instead, when its repo ships one. "
        "Its weights and KV cache take VRAM beside this model's. Empty runs it alone."
    ),
    kind=Kind.TEXT,
    nullable=True,
    recommended="empty; mtp where the repo has a head, else the family's 0.5-1.7B model",
    on_recipe=True,
)


DRAFT_MAX_CONTROL = Control(
    key="draft-max",
    modalities=CHAT_MODALITIES,
    field="draft_max",
    label="Draft tokens per step",
    help=(
        "With a draft model: the most tokens it drafts before the model checks them. "
        "More gains on predictable text (code, lists) and loses where the draft is wrong."
    ),
    kind=Kind.INT,
    recommended=f"{DRAFT_MAX} (llama.cpp's own); compare with `mnm try <name> --speed`",
)

DRAFT_P_MIN_CONTROL = Control(
    key="draft-p-min",
    modalities=CHAT_MODALITIES,
    field="draft_p_min",
    label="Draft only what it is sure of",
    help=(
        "With a draft model: the least probability the draft gives a token before it drafts "
        "it (0 to 1). Lower drafts more and wastes more checks when the model disagrees."
    ),
    kind=Kind.FLOAT,
    recommended=f"{DRAFT_P_MIN}; 0 is llama.cpp's own",
)


check_controls([(c, LlamaCppArgs) for c in (*LLM_CONTROLS, DRAFT_MAX_CONTROL, DRAFT_P_MIN_CONTROL)])
check_controls([(DRAFT_CONTROL, Recipe)])
