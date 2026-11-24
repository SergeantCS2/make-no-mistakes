"""The settings of a vLLM instance, offered instead of llama.cpp's for chat, coding and
vision models that run on vLLM. Same keys where they mean the same (`ctx`, `kv`, `parallel`),
so `mnm set <name> ctx=16384` reads the same on either engine. The share of the card is not
a setting: every start sizes it for these (`start_fixes.vllm_share_upgrade`)."""

from __future__ import annotations

from makenomistakes.controls_catalog import CHAT_MODALITIES, Control, Kind, check_controls
from makenomistakes.engine_args_vllm import VllmArgs

VLLM_CONTROLS: tuple[Control, ...] = (
    Control(
        key="ctx",
        field="max_model_len",
        label="Context length",
        help="Tokens one request can hold (`--max-model-len`). vLLM refuses to start when its "
        "KV cache cannot hold this many; the next start sizes the GPU share for it.",
        kind=Kind.INT,
        recommended="16384 for chat, 32768 for coding, as high as fits for long documents",
        modalities=CHAT_MODALITIES,
    ),
    Control(
        key="kv",
        field="kv_cache_dtype",
        label="KV cache precision",
        help="auto keeps the model's own precision; fp8 halves the KV cache for a slight loss.",
        kind=Kind.CHOICE,
        choices=("auto", "fp8"),
        recommended="auto; fp8 when a long context does not fit",
        modalities=CHAT_MODALITIES,
    ),
    Control(
        key="parallel",
        field="max_num_seqs",
        label="Requests at once",
        help="Sequences vLLM batches together; more raises the sampler's memory a little.",
        kind=Kind.INT,
        recommended="4 for one person; 16 when several agents share it",
        modalities=CHAT_MODALITIES,
    ),
    Control(
        key="eager",
        field="enforce_eager",
        label="Eager mode",
        help="Skips CUDA graph capture: about 0.5 GiB less and a faster start, slower replies.",
        kind=Kind.BOOL,
        recommended="off; on when the card is just short",
        modalities=CHAT_MODALITIES,
    ),
)

check_controls([(c, VllmArgs) for c in VLLM_CONTROLS])
