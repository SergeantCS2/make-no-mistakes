"""vLLM's OpenAI server flags: the engine for chat models that exist only as safetensors.

vLLM takes a share of the whole card (`--gpu-memory-utilization`), refuses to start when less
than that share is free, and fills what its weights, activations and CUDA graphs leave with
KV cache, which must hold `--max-model-len` tokens. The plan sets the share from what the
instance needs, so other instances keep the rest of the card.
"""

from __future__ import annotations

import json
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

VLLM_MIN_CTX = 512
# Tool-call and reasoning parsers vLLM ships, by the model's `model_type` (config.json).
# vLLM's parser names by config.json `model_type`; the families Odysseus's Cookbook maps by
# name are here by model type. A Qwen3 coder speaks its own
# XML-style calls (`qwen3_coder`, picked by the repo's name in `fit.vllm.vllm_args`).
TOOL_PARSERS: dict[str, str] = {
    "qwen2": "hermes",
    "qwen3": "hermes",
    "qwen3_moe": "hermes",
    "qwen2_5_vl": "hermes",
    "qwen3_vl": "hermes",
    "llama": "llama3_json",
    "llama4": "llama4_json",
    "mistral": "mistral",
    "granite": "granite",
    "deepseek_v3": "deepseek_v3",
    "glm4_moe": "glm45",
}
CODER_TOOL_PARSER = "qwen3_coder"
REASONING_PARSERS: dict[str, str] = {"qwen3": "qwen3", "qwen3_moe": "qwen3", "glm4_moe": "glm45"}


class VllmArgs(BaseModel):
    """Flags for `vllm serve`."""

    engine: Literal["vllm"] = "vllm"
    # Settings that leave what it holds on the GPU as planned: changing only these keeps the
    # saved estimate (`Recipe.with_args`; parsers of the reply).
    MEMORY_NEUTRAL: ClassVar[frozenset[str]] = frozenset({"tool_parser", "reasoning_parser"})
    max_model_len: int = Field(ge=VLLM_MIN_CTX)
    gpu_memory_utilization: float = Field(ge=0.1, le=0.95)
    kv_cache_dtype: Literal["auto", "fp8"] = "auto"
    max_num_seqs: int = Field(default=4, ge=1, le=256)  # one person: few at once, less memory
    enforce_eager: bool = False  # no CUDA graphs: about 0.5 GiB less, slower decode
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    images_per_prompt: int | None = Field(default=None, ge=0)  # vision: caps the profile run
    # `--tensor-parallel-size` across this many equal cards (vLLM asserts every worker
    # reports the same KV blocks); None on one card
    tensor_parallel: int | None = Field(default=None, ge=2, le=16)
    extra: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        kv = "" if self.kv_cache_dtype == "auto" else f", KV {self.kv_cache_dtype}"
        eager = ", eager" if self.enforce_eager else ""
        # The share is left out: every start sizes it again for these settings, so a saved
        # one goes stale after `mnm set`; `mnm show` prints the exact flag.
        return f"ctx {self.max_model_len:,}{kv}{eager}"

    @property
    def plan_summary(self) -> str:
        """A plan's summary: its share was sized just now, so it is shown."""
        return f"{self.summary}, {self.gpu_memory_utilization:.0%} of the GPU"

    def to_cli(self) -> list[str]:
        argv = [
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            f"{self.gpu_memory_utilization:.2f}",
            "--max-num-seqs",
            str(self.max_num_seqs),
        ]
        if self.kv_cache_dtype != "auto":
            argv += ["--kv-cache-dtype", self.kv_cache_dtype]
        if self.enforce_eager:
            argv.append("--enforce-eager")
        if self.tool_parser:
            argv += ["--enable-auto-tool-choice", "--tool-call-parser", self.tool_parser]
        if self.reasoning_parser:
            argv += ["--reasoning-parser", self.reasoning_parser]
        if self.tensor_parallel:
            argv += ["--tensor-parallel-size", str(self.tensor_parallel)]
        if self.images_per_prompt is not None:
            argv += ["--limit-mm-per-prompt", json.dumps({"image": self.images_per_prompt})]
        return argv + list(self.extra)
