"""The quality tune: each quant of an instance's repo that runs on this box,
scored by llama.cpp's own `llama-perplexity` against a near-lossless reference, beside the
speed the tuner measures; the most faithful one over the user's speed floor is kept."""

# `<models>/quality/<name>/`: a tune's text and the reference's log-probabilities while it runs
# (1.9 GB for Qwen3's vocabulary at 24 chunks); `mnm disk` counts it, `uninstall.sh` clears it.
WORK_FOLDER = "quality"

# The decode floor's bounds (tok/s), one pair for `--floor`, the web's field, the settings
# (`MNM_FLOOR_CHAT`, `MNM_FLOOR_AGENT`) and the plan.
FLOOR_MIN = 0.5
FLOOR_MAX = 10_000.0


class QualityPlanError(ValueError):
    """A quality tune that cannot be planned, approved or queued here, in words."""
