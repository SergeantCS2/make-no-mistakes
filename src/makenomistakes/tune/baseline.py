"""What the device row says a tune's seed should decode at: the planner's figure for
the recipe's own settings on this box's card, from its row in the device table with this
box's learned figures over it, so the first trial (mnm's plan, measured) is compared with the
row that priced it. The tune's two figures then feed the learned efficiency like any reading
(`device_learning.tuned_speeds`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.device_row import DeviceRow
from makenomistakes.fit.devices import device_for
from makenomistakes.fit.estimate import Fit, FitLevel, estimate_tokens_per_second
from makenomistakes.machine_facts import box_priced
from makenomistakes.model_files import read_model_files
from makenomistakes.recipes import Recipe

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService


@dataclass(frozen=True)
class Baseline:
    tokens_per_second: float
    row: DeviceRow

    def words(self, measured: float) -> str:
        """`the RTX 4070's row priced it at 75.1 tok/s (measured)`, beside what was read."""
        name = self.row.name or "the card"
        gap = measured / self.tokens_per_second - 1 if self.tokens_per_second else 0.0
        return (
            f"mnm's plan measured {measured:.1f} tok/s; the {name}'s row priced it at "
            f"{self.tokens_per_second:.1f} ({self.row.source_word}, {gap:+.0%})"
        )


def baseline(service: InstanceService, recipe: Recipe) -> Baseline | None:
    """None when the files, the card or the settings say nothing (not llama.cpp, no header)."""
    args = recipe.args
    if not isinstance(args, LlamaCppArgs):
        return None
    files = read_model_files(recipe, service.models_dir)
    gpu = service.memory().gpu
    if files is None or files.header is None or gpu is None:
        return None
    header = files.header.for_quant(recipe.quant)
    blocks = header.block_count or 0
    row = device_for(gpu)
    budget = Budget(vram_free_bytes=0, ram_free_bytes=0, gpu_bandwidth=row.bandwidth, device=row)
    priced = box_priced(budget, service.settings.expanded_state_dir())
    whole = args.gpu_layers > blocks and not args.cpu_moe_layers
    fit = Fit(
        level=FitLevel.COMFORTABLE if whole else FitLevel.OFFLOAD,
        gpu_layers=min(args.gpu_layers, blocks),
        headroom_bytes=0,
        gpu_bytes=0,
        cpu_moe_layers=args.cpu_moe_layers or 0,
    )
    speed = estimate_tokens_per_second(
        header,
        files.weights_bytes,
        fit,
        priced.device.bandwidth,
        priced.expert_rate,
        device=priced.device,
        layer_rate=priced.layer_rate,
    )
    return Baseline(speed, priced.device)
