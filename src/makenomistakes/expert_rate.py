"""The rate RAM-held experts decode at, learned from this box's MoE readings: each
`tok/s` reading of a llama.cpp instance that kept experts in RAM (`--n-cpu-moe`) turned back
through the speed model (`fit.estimate.decode_parts`): what the token's time leaves after
the card's share and the launches is the experts' bytes over the rate. The median of the
readings, from the first one, as `CPU_EXPERT_RATE` was fitted to one (28.6 tok/s on an RTX
4070). Warm and cold alike: decode speed does not load weights."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.fit.cpu_rates import EXPERT_MODEL
from makenomistakes.fit.devices import device_for
from makenomistakes.fit.estimate import Fit, FitLevel, decode_parts
from makenomistakes.gguf.reader import GgufHeader
from makenomistakes.hub.facts import Engine
from makenomistakes.machine_facts import MachineFacts
from makenomistakes.measurements import Reading, SpeedUnit, card_names
from makenomistakes.model_files import read_model_files

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService


class Learned(BaseModel):
    gbps: float
    readings: int


def implied_rate(header: GgufHeader, weights_bytes: int, reading: Reading) -> float | None:
    """The expert rate one reading says, bytes a second (`header` read for the instance's
    quant, `GgufHeader.for_quant`); None when it says nothing (no experts in RAM, no speed,
    or a speed the card's share alone cannot reach)."""
    args = reading.args
    if not isinstance(args, LlamaCppArgs) or not args.cpu_moe_layers:
        return None
    if reading.speed_unit is not SpeedUnit.TOKENS or not reading.speed:
        return None
    blocks = header.block_count or 0
    fit = Fit(
        level=FitLevel.OFFLOAD,
        gpu_layers=min(args.gpu_layers, blocks),
        headroom_bytes=0,
        gpu_bytes=0,
        cpu_moe_layers=args.cpu_moe_layers,
    )
    names = card_names(reading.gpu)
    device = device_for(names[0] if names else "")
    fixed, from_ram = decode_parts(header, weights_bytes, fit, device.bandwidth, device=device)
    left = 1 / reading.speed - fixed
    return from_ram / left if from_ram and left > 0 else None


def learn(found: list[float]) -> Learned | None:
    rates = [r for r in found if r > 0]
    if not rates:
        return None
    return Learned(gbps=round(statistics.median(rates) / 1e9, 2), readings=len(rates))


def saved(learned: Learned) -> Callable[[MachineFacts], MachineFacts]:
    now = datetime.now(UTC)

    def change(facts: MachineFacts) -> MachineFacts:
        return facts.model_copy(
            update={
                "expert_gbps": learned.gbps,
                "expert_readings": learned.readings,
                "expert_learned_at": now,
                "expert_model": EXPERT_MODEL,  # Learned under today's expert model
            }
        )

    return change


def learn_from(service: InstanceService, readings: dict[str, list[Reading]]) -> Learned | None:
    """The rate every MoE instance's readings imply together (their files read once each)."""
    found: list[float] = []
    for recipe in service.all():
        mine = readings.get(recipe.name) or []
        if recipe.engine is not Engine.LLAMACPP or not any(_offloads(r) for r in mine):
            continue
        files = read_model_files(recipe, service.models_dir)
        if files is None or files.header is None:
            continue
        header = files.header.for_quant(recipe.quant)
        found += [
            rate for r in mine if (rate := implied_rate(header, files.weights_bytes, r)) is not None
        ]
    return learn(found)


def _offloads(reading: Reading) -> bool:
    return isinstance(reading.args, LlamaCppArgs) and bool(reading.args.cpu_moe_layers)
