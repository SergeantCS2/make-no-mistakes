"""What a known engine warning means, and whether it is one a user can safely ignore.

`diagnose.review_log` attaches the first matching note for the log's engine to each warning
or error line (a note naming one engine's settings never lands on another's line); a
harmless one (the open-port security note, tokenizer quirks llama.cpp corrects itself, the
pooling an embedding server asks for) is counted in one summary line after a start, while
the rest print in full.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from makenomistakes.hub.facts import Engine


class Note(NamedTuple):
    pattern: re.Pattern[str]
    text: str
    harmless: bool = False
    engines: frozenset[Engine] = frozenset(Engine)  # whose lines it explains
    # The backends (`backend.Backend` values) whose images print it; empty: every one
    backends: frozenset[str] = frozenset()


VLLM = frozenset({Engine.VLLM})
_NO_GPU = re.compile(
    r"no usable GPU found|compiled without GPU support|docs/build.md|auto-fit: no GPU devices",
    re.IGNORECASE,
)

NOTES: tuple[Note, ...] = (
    # vLLM's own warnings (as Qwen3-8B-AWQ on vLLM 0.32.0 printed them).
    Note(
        re.compile(r"Attempted to load weight .* with dtype \S+ into parameter with dtype"),
        "The checkpoint stores these weights (norms, AWQ scales) in another precision than "
        "the one vLLM runs the model in (float16 here), so it converts them while loading. "
        "Expected for AWQ checkpoints; harmless.",
        harmless=True,
        engines=VLLM,
    ),
    Note(
        re.compile(r"Pinned memory is not available on this platform"),
        "vLLM does not use pinned host memory under WSL and keeps those buffers in GPU "
        "memory instead (its own fallback). Expected on this box; harmless.",
        harmless=True,
        engines=VLLM,
    ),
    Note(
        re.compile(r"Default vLLM sampling parameters have been overridden by the model"),
        "The model's generation_config.json sets its sampling defaults (temperature, top_k, "
        "top_p); a request that sends its own values overrides them. Harmless.",
        harmless=True,
        engines=VLLM,
    ),
    Note(
        re.compile(r"llama_kv_cache: layer +\d+: sharing with layer +\d+"),
        "A layer that reuses another layer's KV cache instead of keeping its own: Gemma 4's "
        "last layers do, and its MTP head (`draft=mtp`) reads the model's cache this way. "
        "Expected; harmless.",
        harmless=True,
        engines=frozenset({Engine.LLAMACPP}),
    ),
    Note(
        re.compile(r"model is not multilingual", re.IGNORECASE),
        "An English-only (`.en`) whisper model: it ignores the language setting and always "
        "transcribes English. Harmless; pick a multilingual size for other languages.",
        harmless=True,
    ),
    Note(  # ggml-vulkan on Mesa's software device (whisper.cpp prints it at a start)
        re.compile(r"Device type is CPU|\bllvmpipe\b", re.IGNORECASE),
        "Vulkan runs on Mesa's software device (lavapipe), not a card: slow. A card's Vulkan "
        "driver is missing, or this box has none (the CPU backend is faster there).",
        backends=frozenset({"vulkan"}),
    ),
    Note(  # The CPU image says it at every start, and it is the plan
        _NO_GPU,
        "Expected on the CPU backend: the CPU image has no GPU code, and every layer runs in "
        "RAM as planned.",
        harmless=True,
        backends=frozenset({"cpu"}),
    ),
    Note(
        _NO_GPU,
        "The engine cannot see a GPU, so everything runs on the CPU. With the CUDA image that "
        "means Docker lost GPU access: `mnm doctor --deep`, then `mnm setup`.",
    ),
    Note(
        re.compile(r"align generation request", re.IGNORECASE),
        "The requested size was rounded up to what the model's latent grid needs; the image "
        "comes back slightly larger than asked.",
    ),
    Note(
        re.compile(
            r"n_batch .* n_ubatch|setting n_batch = n_ubatch|increase the physical batch",
            re.IGNORECASE,
        ),
        "The engine's batch sizes disagree; `mnm set <name> batch=<ctx> ubatch=<ctx>` for "
        "embedding models.",
    ),
    Note(
        re.compile(r"flash.?attn|flash attention", re.IGNORECASE),
        "Flash attention is unavailable for this model, so a quantised KV cache (kv=q8_0) "
        "may be refused; `mnm set <name> kv=f16` if loading fails.",
        engines=frozenset({Engine.LLAMACPP}),  # sd.cpp's `--diffusion-fa` notice is not this
    ),
    Note(
        re.compile(r"tensor overrides to CPU are used with mmap", re.IGNORECASE),
        "llama.cpp's hint for experts kept in RAM (`--n-cpu-moe`): loading without mmap may "
        "be faster, which mnm has not measured. mnm keeps the engine's default loading, "
        "the mode its speed estimate is fitted to (28.6 tok/s on an RTX 4070). Harmless.",
        harmless=True,
        engines=frozenset({Engine.LLAMACPP}),
    ),
    Note(
        re.compile(r"n_ctx_per_seq \(\d+\) < n_ctx_train|full capacity of the model will not"),
        "The context is below the model's native one. Fine for chat; raise it with "
        "`mnm set <name> ctx=<n>` when you need longer inputs (more VRAM).",
        harmless=True,
    ),
    Note(
        # A recipe saved above the model's training context: llama-server runs at the
        # training context instead and says so.
        re.compile(r"exceeds the training context of the model \(\d+\) - capping"),
        "llama-server capped the context to the model's training context (the second number) "
        "and runs at that, not at the saved ctx. `mnm set <name> ctx=<that number>` saves "
        "what runs; mnm refuses a ctx above it.",
        harmless=True,
        engines=frozenset({Engine.LLAMACPP}),
    ),
    Note(
        # A Qwen3.5/3.6 GGUF carries its MTP head's blk.N.* and nextn.* tensors inside
        # the main file; llama.cpp skips what the model code does not load
        # (src/llama-model-loader.cpp: "model has unused tensor %s ... -- ignoring").
        re.compile(r"model has unused tensor .* -- ignoring"),
        "A tensor in the file that this model's code does not load, skipped (for a Qwen3.5 or "
        "3.6 GGUF, the multi-token-prediction head's `blk.N.*` and `nextn.*` tensors until "
        "`mnm set <name> draft=mtp`). Harmless.",
        harmless=True,
        engines=frozenset({Engine.LLAMACPP}),
    ),
    Note(
        # llama-server c7fc509 prints this WRN at every start of a template that keeps
        # reasoning between turns (tools/server/server-context.cpp); nothing failed.
        re.compile(r"chat template (?:supports|does NOT support) preserving reasoning"),
        "Informational: the chat template keeps the model's reasoning in the history between "
        "turns (on by default in this build; a few more tokens a turn). Harmless.",
        harmless=True,
        engines=frozenset({Engine.LLAMACPP}),
    ),
    Note(
        re.compile(r"n_ctx_per_seq \(\d+\) > n_ctx_train|possible training context overflow"),
        "The context is above what the model was trained on; answers degrade past that "
        "point. `mnm set <name> ctx=<native>` keeps it honest.",
    ),
    Note(
        # llama.cpp's image exports LLAMA_ARG_HOST and LLAMA_ARG_PORT=8080; a `--port` flag
        # beside the second (an older mnm passed one) makes every container print this.
        re.compile(r"environment variable is set, but will be overwritten", re.IGNORECASE),
        "Harmless: the engine image sets LLAMA_ARG_HOST and LLAMA_ARG_PORT (8080) as its "
        "defaults, and the flag on the command line wins. mnm passes both as variables; a "
        "container an older mnm made still "
        "prints this: `mnm stop <name>` then `mnm start <name>` recreates it, and nothing "
        "downloads.",
        harmless=True,
        engines=frozenset({Engine.LLAMACPP}),
    ),
    Note(
        re.compile(r"no API key is set and CORS allows all origins", re.IGNORECASE),
        "Expected: the engine listens on 127.0.0.1 only, behind the router, which asks for "
        "the key (MNM_API_KEY) when one is set.",
        harmless=True,
    ),
    Note(
        re.compile(r"control-looking token|special_eos_id|special_eog_ids", re.IGNORECASE),
        "A quirk in the GGUF's tokenizer metadata that llama.cpp corrects while loading; harmless.",
        harmless=True,
    ),
    Note(
        re.compile(r"not marked as outputs", re.IGNORECASE),
        "Expected for embedding and rerank models: every input token becomes an output so "
        "the server can pool them (which is also why their batch buffer is large).",
        harmless=True,
    ),
    Note(
        re.compile(r"speculative decoding not supported", re.IGNORECASE),
        "Embedding and rerank contexts cannot use speculative decoding, and nothing asked "
        "for it; harmless.",
        harmless=True,
    ),
    Note(
        re.compile(r"default pooling_type is \[-1\], but \[\d\] was specified", re.IGNORECASE),
        "Expected with --rerank: the server asks for rank pooling, which the GGUF does not "
        "declare.",
        harmless=True,
    ),
    Note(
        re.compile(r"image tokens to function correctly|--image-min-tokens|issues/16842"),
        "Qwen-VL's own advice: fewer than 1024 image tokens hurts grounding (boxes and "
        "positions). `mnm set <name> image-min-tokens=1024`, then restart; new Qwen-VL "
        "recipes set it already.",
    ),
    Note(
        re.compile(r"deprecated|renamed|no longer", re.IGNORECASE),
        "A flag this tool passes is being retired by the engine; it still works. Report it so "
        "the recipe renderer can be updated.",
    ),
    Note(
        re.compile(r"vocab missing newline token|key not found", re.IGNORECASE),
        "Optional metadata missing from the GGUF; harmless unless chat or tools misbehave.",
        harmless=True,
    ),
    Note(
        re.compile(r"missing.*(?:token|template)", re.IGNORECASE),
        "Metadata missing from the GGUF. If chat or tool calls misbehave, this is where to "
        "look (`mnm set <name> jinja=off` falls back to a generic template).",
    ),
    Note(
        # llama-server's own failure lines (tools/server/server-context.cpp, common/chat.cpp at
        # c7fc509): "chat template parsing error", "please consider disabling jinja via
        # --no-jinja", "failed to initialize chat template", "failed to parse tool use chat
        # template (ignoring it)", "failed to apply chat template". Any other line naming the
        # template is informational.
        re.compile(
            r"chat template parsing error|consider disabling jinja|"
            r"failed to (?:initialize|parse tool use|apply) chat template|"
            r"this custom template is not supported",
            re.IGNORECASE,
        ),
        "The model's chat template could not be used as-is; `mnm set <name> jinja=off` falls "
        "back to a generic one (tool calling stops working).",
        engines=frozenset({Engine.LLAMACPP}),
    ),
)


def note_for(text: str, engine: Engine | None = None, backend: str | None = None) -> Note | None:
    """The first note for this line; without an engine, any engine's; a note scoped to some
    backends only when `backend` is one of them."""
    return next(
        (
            note
            for note in NOTES
            if (engine is None or engine in note.engines)
            and (not note.backends or backend in note.backends)
            and note.pattern.search(text)
        ),
        None,
    )
