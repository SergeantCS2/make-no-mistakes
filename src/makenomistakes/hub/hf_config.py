"""A transformers `config.json`, typed: what the vLLM fit model needs.

Vision-language configs keep the language model's sizes under `text_config`; the fields
below read from there when the top level lacks them. A config with `auto_map` needs the
repo's own Python code (`--trust-remote-code`), which this tool never runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    quant_method: str = ""
    bits: int | None = None
    fmt: str | None = None  # FP8 configs: "e4m3"


class TextConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    vocab_size: int | None = None


class HfConfig(TextConfig):
    """The parts of `config.json` the plan reads; everything else is ignored."""

    architectures: list[str] = Field(default_factory=list)
    model_type: str = ""
    torch_dtype: str | None = None
    dtype: str | None = None  # what newer transformers writes instead of torch_dtype
    quantization_config: QuantizationConfig | None = None
    auto_map: dict[str, object] | None = None
    text_config: TextConfig | None = None

    @property
    def text(self) -> TextConfig:
        """The language model's sizes: the top level, filled in from `text_config`."""
        inner = self.text_config or TextConfig()
        merged = {
            name: getattr(self, name) if getattr(self, name) is not None else getattr(inner, name)
            for name in TextConfig.model_fields
        }
        return TextConfig(**merged)

    @property
    def needs_remote_code(self) -> bool:
        return bool(self.auto_map)

    @property
    def quant_label(self) -> str:
        """`AWQ-INT4`, `GPTQ-INT8`, `FP8`, or the stored dtype (`BF16`)."""
        quant = self.quantization_config
        if quant is not None and quant.quant_method:
            method = quant.quant_method.upper().replace("_", "-")
            if method in {"AWQ", "GPTQ"} and quant.bits:
                return f"{method}-INT{quant.bits}"
            return method
        return (
            (self.torch_dtype or self.dtype or "bf16")
            .upper()
            .replace("BFLOAT16", "BF16")
            .replace("FLOAT16", "F16")
        )

    def kv_bytes_per_token(self, kv_bytes: int) -> int:
        """K and V for every layer: 2 x layers x KV heads x head size x bytes per value."""
        text = self.text
        if not (text.num_hidden_layers and text.num_attention_heads and text.hidden_size):
            raise ValueError("config.json has no layer or head counts")
        heads_kv = text.num_key_value_heads or text.num_attention_heads
        head_dim = text.head_dim or text.hidden_size // text.num_attention_heads
        return 2 * text.num_hidden_layers * heads_kv * head_dim * kv_bytes
