"""The doctor's lines about the box beyond the cards: the CPU (cores, NUMA nodes, CCDs),
the RAM's DIMMs and channels, and with `--deep` the measured bandwidth against what each saved
mixture of experts reads from RAM per token ("coder36 reads about 2.3 GB a token from RAM: at
39 GB/s that is 17 tok/s at best"). The last is a ceiling, not a plan: the RAM-held experts'
real rate (the planner's, learned by `mnm calibration`) sits well under the copy's bandwidth.
"""

from __future__ import annotations

from collections.abc import Iterable

from makenomistakes.doctor_types import Check, Status
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.fit.estimate import head_bytes
from makenomistakes.gguf.reader import GgufHeader
from makenomistakes.hardware.box_facts import BoxFacts
from makenomistakes.instances import InstanceService
from makenomistakes.model_files import read_model_files
from makenomistakes.recipes import Recipe
from makenomistakes.tune.cpus import CpuLayout

GB = 10**9
SLOT_WORDS = "`sudo dmidecode -t memory` lists them"


def box_checks(box: BoxFacts | None) -> list[Check]:
    """The CPU and RAM slots lines; none when the box was not read (the web's
    light poll)."""
    if box is None:
        return []
    return [check for check in (_cpu(box), _slots(box)) if check is not None]


def _cpu(box: BoxFacts) -> Check | None:
    cpu = box.cpu
    if cpu is None or not cpu.cores:
        return None
    threads = CpuLayout(cpu.usable_cores, cpu.ccds).choices()
    detail = cpu.words()
    if threads:
        detail += f"; `mnm tune` tries {', '.join(str(n) for n in threads)} threads"
    if cpu.numa_nodes > 1:
        detail += (
            "; several NUMA nodes: experts kept in RAM read fastest from the node their "
            "threads run on (`numa=` per instance, the NUMA lines below)"
        )
    return Check(name="CPU", status=Status.OK, detail=detail)


def _slots(box: BoxFacts) -> Check:
    name = "RAM slots"
    if box.wsl:
        return Check(
            name=name,
            status=Status.OK,
            detail="not readable under WSL (Task Manager → Performance → Memory shows the "
            "slots in use)",
        )
    slots = box.slots
    if slots is None:
        return Check(name=name, status=Status.OK, detail=f"unknown without sudo ({SLOT_WORDS})")
    one_channel = slots.channels == 1 or len(slots.dimms) == 1
    if one_channel:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{slots.words()}: one channel reads about half what two do, and experts "
            "kept in RAM decode at about half the speed",
            remedy="A second DIMM of the same kind in the other channel's slot (the board's "
            "manual names the pair: A2 and B2 on most four-slot boards).",
        )
    detail = slots.words()
    if len(slots.dimms) >= 4 and any(d.kind == "DDR5" for d in slots.dimms):
        detail += "; four DDR5 DIMMs run slower than two on a desktop board"
    return Check(name=name, status=Status.OK, detail=detail)


def ram_bytes_per_token(header: GgufHeader, weights_bytes: int, in_ram: Iterable[int]) -> float:
    """The active experts a decoded token reads from RAM, for the blocks whose experts stay
    there: each block's experts at the variant's bits times its active share, scaled as the
    speed model scales them (`fit.estimate._uneven_reads`)."""
    streamed = weights_bytes - head_bytes(header, weights_bytes)
    sizes = header.expert_layer_bytes()
    shares = header.expert_shares()
    total = sum(sizes)
    scale = min(1.0, streamed / total) if total else 1.0
    return sum(sizes[b] * scale * shares[b] for b in set(in_ram) if b < len(sizes))


def in_ram_blocks(args: LlamaCppArgs, blocks: int) -> list[int]:
    """The blocks whose routed experts stay in RAM: a placement's (several cards), else the
    first `--n-cpu-moe` blocks (one card)."""
    if args.placement is not None:
        return list(args.placement.in_ram)
    return list(range(min(args.cpu_moe_layers or 0, blocks)))


def expert_need_check(service: InstanceService, gbps: float | None) -> Check:
    """`mnm doctor --deep`: each saved model with experts in RAM, what it reads a token there
    and the speed the measured bandwidth allows at best."""
    name = "Experts in RAM"
    found = []
    for recipe in service.all():
        line = _need(service, recipe, gbps)
        if line is not None:
            found.append(line)
    if not found:
        return Check(name=name, status=Status.OK, detail="no saved model keeps experts in RAM")
    if gbps is None:
        found.append("the bandwidth was not measured, so no ceiling is said")
    return Check(name=name, status=Status.OK, detail="; ".join(found))


def _need(service: InstanceService, recipe: Recipe, gbps: float | None) -> str | None:
    args = recipe.args
    if not isinstance(args, LlamaCppArgs) or not args.experts_in_ram:
        return None
    files = read_model_files(recipe, service.models_dir)
    if files is None or files.header is None:
        return f"{recipe.name} keeps experts in RAM (not downloaded: its files say how much)"
    header = files.header.for_quant(recipe.quant)
    blocks = in_ram_blocks(args, header.block_count or 0)
    need = ram_bytes_per_token(header, files.weights_bytes, blocks)
    if not need:
        return None
    line = f"{recipe.name} reads about {need / GB:.1f} GB a token from RAM ({len(blocks)} layers)"
    if gbps:
        line += f": at {gbps:.0f} GB/s that is {gbps * GB / need:.0f} tok/s at best"
    return line
