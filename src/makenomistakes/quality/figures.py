"""What a quality run measured, typed: a `Kind.QUALITY` reading's figures and
a quality trial's. A leaf module: `measurements` and the tuner's trials read it without the
quality tune's machinery.

Every score is one quant's against the tune's **reference** (Q8_0, or the most faithful quant
the box runs) on one **text** (`quality.texts`): KL divergence and how often the most likely
token stayed the same are llama.cpp's own words for how far a quant's answers drift from the
reference's; perplexity is beside them, comparable only among one model's quants."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QualityFigures(BaseModel):
    """One quant's scores; the reference's own carry no KLD (it is the yardstick)."""

    model_config = ConfigDict(frozen=True)

    repo_id: str
    quant: str
    file: str  # the GGUF's name in the repo (its first shard)
    reference: str  # the quant every score is against
    text: str  # `quality.texts.TextName`: prose or code
    text_sha256: str  # the text's first 12 hex: another text never compares
    chunks: int = Field(ge=1)  # 512-token chunks read (each scores its second half)
    ppl: float = Field(gt=0)  # perplexity over the chunks, as `llama-perplexity` printed it
    ppl_err: float = Field(ge=0)
    kld: float | None = Field(default=None, ge=-1e-3)  # mean KL divergence from the reference
    kld_err: float | None = Field(default=None, ge=0)
    kld_p99: float | None = Field(default=None, ge=-1e-3)  # its 99th percentile
    same_top: float | None = Field(default=None, ge=0, le=100)  # % of tokens, the same top one
    same_top_err: float | None = Field(default=None, ge=0)
    rms_dp: float | None = Field(default=None, ge=0)  # % RMS change of the right token's chance
    ppl_ratio: float | None = Field(default=None, gt=0)  # PPL(quant) / PPL(reference)
    # Its decode at its serving plan, the median of the tune's speed trial (on the engine's
    # own image, a start of a copy as the tuner's trials are); None: not timed. Kept here, never
    # as the reading's `speed`, which the speed learners read as a request's.
    decode: float | None = Field(default=None, gt=0)
    decode_noisy: bool = False  # its three readings spread past the tuner's 10%

    @property
    def is_reference(self) -> bool:
        return self.quant == self.reference

    @property
    def drift(self) -> float:
        """The KLD the choice compares (the reference's is 0)."""
        return 0.0 if self.is_reference or self.kld is None else max(self.kld, 0.0)

    @property
    def drift_err(self) -> float:
        return 0.0 if self.is_reference or self.kld_err is None else self.kld_err

    @property
    def words(self) -> str:
        """`KLD 0.0723 ± 0.0015 against Q8_0, the same top token 85.5%, PPL 21.69` (the
        reference: `the reference, PPL 20.85`)."""
        ppl = f"PPL {self.ppl:.2f}"
        if self.is_reference or self.kld is None:
            return f"the reference, {ppl}"
        same = f", the same top token {self.same_top:.1f}%" if self.same_top is not None else ""
        err = f" ± {self.kld_err:.4f}" if self.kld_err is not None else ""
        return f"KLD {self.drift:.4f}{err} against {self.reference}{same}, {ppl}"

    @property
    def cells(self) -> tuple[str, str, str]:
        """The tables' KLD, same top token and decode cells (`mnm calibration`, the web)."""
        kld = (
            "-"
            if self.is_reference or self.kld is None
            else f"{self.drift:.4f} ± {self.drift_err:.4f}"
        )
        same = "-" if self.same_top is None else f"{self.same_top:.1f}%"
        spread = " (spread)" if self.decode_noisy else ""
        decode = f"{self.decode:.1f} tok/s{spread}" if self.decode else "-"
        return kld, same, decode
