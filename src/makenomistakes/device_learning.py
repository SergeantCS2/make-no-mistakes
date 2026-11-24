"""This box's decode efficiency and launch term on its card (its readings only), the way
the RTX 4070's were fitted: a decode reading with every block on the card is the
streamed bytes over the card's bandwidth x efficiency plus a launch cost per block, so readings
of two models of different size give both figures (least squares over every reading), and
readings of one size give the efficiency with the row's launch term.

What counts: llama.cpp `tok/s` readings of this box (`readings_filter`) on one card, whose
args put every block on it (`-ngl` above the block count, no expert or block in RAM, no
placement or split), without a draft (a draft's speed is the draft's), of a file that is not a
codebook i-quant (they keep their own efficiency), and the measured trials of each instance's
tune on this box whose candidate kept the plan's layers (the tuner's readings). Each
instance's settings and build count once, at their median speed, weighted by their readings.
`mnm calibration` learns them and saves them in `machine.json` (`machine_facts`)."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.fit.device_row import DeviceRow
from makenomistakes.fit.devices import device_for
from makenomistakes.fit.estimate import Fit, FitLevel, decode_parts
from makenomistakes.gguf.file_types import GRID_QUANT_IDS
from makenomistakes.gguf.reader import GgufHeader
from makenomistakes.hub.facts import Engine
from makenomistakes.machine_facts import MachineFacts
from makenomistakes.measurements import Reading, SpeedUnit, card_names, settings_key
from makenomistakes.model_files import read_model_files
from makenomistakes.readings_filter import Here
from makenomistakes.recipes import Recipe

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

# A size apart by this much is a second size: the launch term can be told from the bytes.
SECOND_SIZE = 1.5
_PLAUSIBLE_EFFICIENCY = (0.2, 1.2)
_PLAUSIBLE_LAUNCH = (0.0, 1e-3)
_LAYER_FIELDS = frozenset({"gpu_layers", "cpu_moe_layers", "placement"})


@dataclass(frozen=True)
class Point:
    """One instance's settings at one speed: what a token streams over the card's bandwidth
    (`seconds_at_peak`), its blocks, and its time a token."""

    seconds_at_peak: float
    blocks: int
    seconds: float
    readings: int


@dataclass(frozen=True)
class DecodeLearned:
    card: str  # nvidia-smi's name of the card
    efficiency: float
    launch_seconds: float | None  # None: one size only, the row's launch term kept
    readings: int
    row: DeviceRow  # the row it was learned over (its efficiency, for the words)


def whole_on_card(header: GgufHeader, args: LlamaCppArgs) -> bool:
    blocks = header.block_count or 0
    return (
        bool(blocks)
        and args.gpu_layers > blocks
        and not args.cpu_moe_layers
        and args.placement is None
        and not args.tensor_split
    )


def point_of(
    header: GgufHeader, weights: int, speeds: list[float], device: DeviceRow
) -> Point | None:
    """The point a set of speeds of one settings make (None: nothing usable)."""
    blocks = header.block_count or 0
    usable = [s for s in speeds if s > 0]
    if not usable or not blocks:
        return None
    bare = device.model_copy(update={"efficiency": 1.0, "launch_seconds": 0.0})
    whole = Fit(level=FitLevel.COMFORTABLE, gpu_layers=blocks, headroom_bytes=0, gpu_bytes=0)
    at_peak, _ram = decode_parts(header, weights, whole, device.bandwidth, device=bare)
    return Point(at_peak, blocks, 1 / statistics.median(usable), len(usable))


def fit_points(points: list[Point], device: DeviceRow) -> tuple[float, float | None] | None:
    """(efficiency, launch term or None) from the points: both by least squares when two
    sizes differ by `SECOND_SIZE` and the result is plausible, else the efficiency alone."""
    if not points:
        return None
    sizes = [p.seconds_at_peak for p in points]
    if max(sizes) >= SECOND_SIZE * min(sizes):
        both = _least_squares(points)
        if both is not None:
            return both
    each = [
        p.seconds_at_peak / left
        for p in points
        for _ in range(p.readings)
        if (left := p.seconds - p.blocks * device.launch_seconds) > 0
    ]
    if not each:
        return None
    efficiency = statistics.median(each)
    low, high = _PLAUSIBLE_EFFICIENCY
    return (efficiency, None) if low <= efficiency <= high else None


def _least_squares(points: list[Point]) -> tuple[float, float] | None:
    """t = x / efficiency + blocks x launch, weighted by each point's readings."""
    sxx = sum(p.readings * p.seconds_at_peak**2 for p in points)
    sll = sum(p.readings * p.blocks**2 for p in points)
    sxl = sum(p.readings * p.seconds_at_peak * p.blocks for p in points)
    sxt = sum(p.readings * p.seconds_at_peak * p.seconds for p in points)
    slt = sum(p.readings * p.blocks * p.seconds for p in points)
    det = sxx * sll - sxl**2
    if det <= 0:
        return None
    inverse = (sxt * sll - sxl * slt) / det
    launch = (sxx * slt - sxl * sxt) / det
    if inverse <= 0:
        return None
    efficiency = 1 / inverse
    if not (_PLAUSIBLE_EFFICIENCY[0] <= efficiency <= _PLAUSIBLE_EFFICIENCY[1]):
        return None
    if not (_PLAUSIBLE_LAUNCH[0] < launch <= _PLAUSIBLE_LAUNCH[1]):
        return None
    return efficiency, launch


def _speeds(
    recipe: Recipe, header: GgufHeader, readings: Iterable[Reading]
) -> dict[str, list[float]]:
    """This instance's usable speeds by (card, settings and build)."""
    alone = settings_key(recipe.model_copy(update={"draft": None}))
    found: dict[str, list[float]] = {}
    for r in readings:
        names = card_names(r.gpu)
        args = r.args
        if r.speed_unit is not SpeedUnit.TOKENS or not r.speed or len(names) != 1:
            continue
        if not isinstance(args, LlamaCppArgs) or not whole_on_card(header, args):
            continue
        if r.settings != alone:  # the settings without a draft: a draft's speed is the draft's
            continue
        found.setdefault(f"{names[0]}\n{r.settings}\n{r.build}", []).append(r.speed)
    return found


def tuned_speeds(recipe: Recipe, here: Here) -> list[float]:
    """A tune's two decode figures (mnm's plan as its first trial measured it, and the
    settings it chose) when it ran on this box and kept the plan's layers and draft."""
    tuned = recipe.tuned
    if tuned is None or not here.owns(tuned.key.machine):
        return []
    if tuned.draft is not None or _LAYER_FIELDS & set(tuned.changes):
        return []
    return [tuned.seed.decode, tuned.score.decode]


def learn_decode(
    service: InstanceService,
    readings: dict[str, list[Reading]],
    here: Here | None = None,
    card: str | None = None,
) -> DecodeLearned | None:
    """This box's decode figures on its card from every instance's usable readings, and the
    tunes run on `here` (taken on `card`, nvidia-smi's name now); None without one."""
    by_card: dict[str, list[Point]] = {}
    for recipe in service.all():
        if recipe.engine is not Engine.LLAMACPP or not isinstance(recipe.args, LlamaCppArgs):
            continue
        files = read_model_files(recipe, service.models_dir)
        if files is None or files.header is None:
            continue
        header = files.header.for_quant(recipe.quant)
        if header.file_type in GRID_QUANT_IDS:
            continue
        groups = [
            (key.split("\n", 1)[0], speeds)
            for key, speeds in _speeds(recipe, header, readings.get(recipe.name) or []).items()
        ]
        tried = tuned_speeds(recipe, here) if here is not None else []
        if tried and card and whole_on_card(header, recipe.args) and recipe.draft is None:
            groups.append((card, tried))
        for name, speeds in groups:
            point = point_of(header, files.weights_bytes, speeds, device_for(name))
            if point is not None:
                by_card.setdefault(name, []).append(point)
    if not by_card:
        return None
    chosen, points = max(by_card.items(), key=lambda item: sum(p.readings for p in item[1]))
    row = device_for(chosen)
    found = fit_points(points, row)
    if found is None:
        return None
    efficiency, launch = found
    count = sum(p.readings for p in points)
    return DecodeLearned(chosen, round(efficiency, 3), launch, count, row)


def saved(learned: DecodeLearned) -> Callable[[MachineFacts], MachineFacts]:
    now = datetime.now(UTC)
    return lambda facts: facts.model_copy(
        update={
            "decode_efficiency": learned.efficiency,
            "launch_seconds": learned.launch_seconds,
            "decode_readings": learned.readings,
            "decode_learned_at": now,
            "decode_card": learned.card,
        }
    )


def learned_words(learned: DecodeLearned) -> str:
    """`decode efficiency on this RTX 3090: 0.84 from 6 readings (the row's 0.91)`."""
    name = learned.row.name or learned.card
    said = (
        f"decode efficiency on this {name}: {learned.efficiency:.2f} from {learned.readings} "
        f"reading{'s' * (learned.readings != 1)} (the row's {learned.row.efficiency:.2f})"
    )
    if learned.launch_seconds is not None:
        said += (
            f", launch term {learned.launch_seconds * 1e3:.3f} ms a block (the row's "
            f"{learned.row.launch_seconds * 1e3:.3f})"
        )
    return said
