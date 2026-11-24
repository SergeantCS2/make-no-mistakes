"""`mnm set <name> numa=auto|off|distribute|isolate|node:N`: where a llama.cpp
instance's RAM-held weights and threads sit on a box of several NUMA nodes (an EPYC in NPS4, two
sockets). `distribute` and `isolate` are llama.cpp's own `--numa`; `node:N` binds the container
to node N's CPUs and memory with Docker's `--cpuset-cpus`/`--cpuset-mems` (the official image
has no `numactl`, and mnm never runs it) and sets the threads to that node's physical cores;
`off` renders nothing and keeps a re-plan from choosing one; empty clears it.

`auto` decides now from the box, never at a start: one node needs nothing; the model's RAM part
fits one node's free memory: `node:<the node with most free>`; else `distribute`. A binding
the node's memory cannot hold is refused with the arithmetic (two sockets: one node's RAM is
half the box's). Nothing here drops the page cache `distribute` wants dropped (that needs
root); the doctor says the command."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.hardware.numa_nodes import NumaNode, node
from makenomistakes.recipes import Recipe

GIB = 2**30
CHOICES = ("auto", "off", "distribute", "isolate", "node:N")
CLEAR = ("", "none", "default")


class NumaSettingError(ValueError):
    """numa= cannot be set: why, worded for `mnm set` and the web form."""


@dataclass(frozen=True)
class NumaFacts:
    """What `node:N` and `auto` read of the box, gathered when asked."""

    nodes: list[NumaNode]
    ram_part_bytes: int | None  # what the instance keeps in RAM; None: not downloaded
    margin_bytes: int  # what the system keeps, spread over the nodes


Facts = Callable[[Recipe], NumaFacts]


def with_numa(recipe: Recipe, raw: str, facts: Facts | None) -> Recipe:
    """The recipe with `numa=raw`; the same recipe when it is already that (the web form posts
    every field)."""
    args = recipe.args
    if not isinstance(args, LlamaCppArgs):
        raise NumaSettingError("numa= applies to llama.cpp instances")
    text = raw.strip().lower()
    if text == (args.numa or "") or (text in CLEAR and args.numa is None):
        return recipe
    if text in CLEAR:
        return _set(recipe, None)
    if text in ("off", "distribute", "isolate"):
        return _set(recipe, text)
    if text != "auto" and not text.startswith("node:"):
        raise NumaSettingError(f"numa must be one of {', '.join(CHOICES)}, not '{raw}'")
    if facts is None:
        raise NumaSettingError("numa=auto and numa=node:N are set through `mnm set` or its page")
    found = facts(recipe)
    if not found.nodes:
        raise NumaSettingError(
            "numa=auto and numa=node:N read this box's NUMA nodes from "
            "/sys/devices/system/node, and none were readable here"
        )
    if text == "auto":
        return _auto(recipe, found)
    number = text.removeprefix("node:")
    if not number.isdigit():
        raise NumaSettingError(f"numa=node:N names a node by its number (node:0), not '{raw}'")
    return _bound(recipe, int(number), found)


def _set(recipe: Recipe, value: str | None) -> Recipe:
    """A plain value; leaving a node binding gives the threads back to llama.cpp's own."""
    update: dict[str, object] = {"numa": value, "numa_cpus": None}
    if recipe.llama_args.numa_node is not None:
        update["threads"] = None
    return recipe.with_args(**update)


def _auto(recipe: Recipe, found: NumaFacts) -> Recipe:
    if len(found.nodes) < 2:
        return _set(recipe, None)  # one node: nothing to choose
    roomiest = max(found.nodes, key=lambda n: n.free_bytes or 0)
    part = found.ram_part_bytes
    if part is not None and part <= _room(roomiest, found, free=True):
        return _bound(recipe, roomiest.index, found)
    return _set(recipe, "distribute")


def _room(where: NumaNode, found: NumaFacts, free: bool = False) -> int:
    """What a node holds for the model: its memory (free, for `auto`) less its share of the
    system's margin."""
    held = where.free_bytes if free else where.total_bytes
    return (held or 0) - found.margin_bytes // max(len(found.nodes), 1)


def _bound(recipe: Recipe, index: int, found: NumaFacts) -> Recipe:
    where = node(found.nodes, index)
    if where is None:
        numbers = ", ".join(str(n.index) for n in found.nodes)
        raise NumaSettingError(f"this box has NUMA nodes {numbers}; node {index} is not one")
    part = found.ram_part_bytes
    if part is not None and where.total_bytes and part > _room(where, found):
        share = found.margin_bytes / len(found.nodes) / GIB
        raise NumaSettingError(
            f"numa=node:{index} holds the container's memory to node {index} "
            f"({where.total_bytes / GIB:.0f} GiB, less {share:.0f} GiB of the system's "
            f"share): {recipe.name} keeps about {part / GIB:.0f} GiB in "
            "RAM, so it would not load there; numa=distribute spreads it over every node"
        )
    return recipe.with_args(numa=f"node:{index}", numa_cpus=where.cpus, threads=where.cores)


def words(args: LlamaCppArgs) -> str | None:
    """`numa node:1 (CPUs 8-15,40-47)` for `mnm show`; None when unset."""
    if args.numa is None:
        return None
    if args.numa_node is not None:
        return f"node:{args.numa_node} (CPUs {args.numa_cpus}, its memory only)"
    return args.numa
