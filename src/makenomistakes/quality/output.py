"""`llama-perplexity`'s own words, read. Read at b12359 (b1b770d9, 2026-11-13):

- a base run (`--kl-divergence-base FILE` with `-f TEXT`) logs `Final estimate: PPL = x +/- y`
  on stderr behind a timestamp, glued to the ETA line before it (no newline between them);
- a KL-divergence run (`--kl-divergence --kl-divergence-base FILE`) prints its statistics on
  stdout: `Mean PPL(Q)`, `Mean PPL(Q)/PPL(base)`, `Mean    KLD`, `99.0%   KLD`, `RMS Δp` and
  `Same top p`, each `value ± error` (percentages with `%`).

Both streams come back together from `docker logs`, so each figure is searched for anywhere;
a run that printed none of its figures is a `QualityOutputError` with the last lines it said."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

NUMBER = r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
PM = r"\s*(?:±|\+/-)\s*"


class QualityOutputError(ValueError):
    """`llama-perplexity` ended without the figures a run prints."""


@dataclass(frozen=True)
class BaseFigures:
    ppl: float
    ppl_err: float


@dataclass(frozen=True)
class KldFigures:
    ppl: float
    ppl_err: float
    ppl_ratio: float
    kld: float
    kld_err: float
    kld_p99: float | None
    same_top: float
    same_top_err: float
    rms_dp: float | None


def _last(pattern: str, text: str) -> tuple[float, ...] | None:
    found = re.findall(pattern, text)
    if not found:
        return None
    last = found[-1]
    values = tuple(float(v) for v in (last if isinstance(last, tuple) else (last,)))
    return values if all(math.isfinite(v) for v in values) else None


def _tail(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return " / ".join(lines[-3:])[-400:] if lines else "nothing"


def base_figures(text: str) -> BaseFigures:
    found = _last(rf"Final estimate: PPL = {NUMBER}{PM}{NUMBER}", text)
    if found is None:
        raise QualityOutputError(
            f"llama-perplexity printed no final estimate; its last lines: {_tail(text)}"
        )
    return BaseFigures(ppl=found[0], ppl_err=found[1])


def kld_figures(text: str) -> KldFigures:
    ppl = _last(rf"Mean PPL\(Q\)\s*:\s*{NUMBER}{PM}{NUMBER}", text)
    ratio = _last(rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*{NUMBER}{PM}{NUMBER}", text)
    kld = _last(rf"Mean\s+KLD:\s*{NUMBER}{PM}{NUMBER}", text)
    same = _last(rf"Same top p:\s*{NUMBER}{PM}{NUMBER}\s*%", text)
    if ppl is None or ratio is None or kld is None or same is None:
        raise QualityOutputError(
            f"llama-perplexity printed no KL-divergence statistics; its last lines: {_tail(text)}"
        )
    p99 = _last(rf"99\.0%\s+KLD:\s*{NUMBER}", text)
    rms = _last(rf"RMS Δp\s*:\s*{NUMBER}{PM}{NUMBER}\s*%", text)
    return KldFigures(
        ppl=ppl[0],
        ppl_err=ppl[1],
        ppl_ratio=ratio[0],
        kld=kld[0],
        kld_err=kld[1],
        kld_p99=p99[0] if p99 is not None else None,
        same_top=same[0],
        same_top_err=same[1],
        rms_dp=rms[0] if rms is not None else None,
    )
