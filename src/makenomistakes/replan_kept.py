"""What a re-plan keeps of the saved recipe beside the plan's own settings (moved out of
`replan`): the processor-sized settings (`threads`, the load mode), the prompt cache's caps an
older recipe lacks, and the user's NUMA binding."""

from __future__ import annotations

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.hardware.numa_nodes import node, read_nodes
from makenomistakes.hub.facts import Engine
from makenomistakes.prompt_cache import PlannedCache
from makenomistakes.ram_room import runs_on_processor
from makenomistakes.recipes.model import Recipe

# The settings a plan sizes to the processor, whatever the engine (`kept_processor`): a CPU
# plan's `threads`, the physical cores (llama.cpp's, and audio.cpp's, whose own default is
# one core), and llama.cpp's load mode (`none` in RAM or on a pool).
PROCESSOR_SIZED = ("threads", "load_mode")


def processor_only(old: Recipe, new: Recipe) -> Recipe:
    """`old` with only the new plan's processor-sized settings (`kept_processor`) and the
    prompt cache it never had (`unset_cache`): a withheld re-plan moves nowhere, so only those
    the user left unset."""
    moved = {
        key: value
        for key, value in (kept_processor(old, new) | unset_cache(old, new)).items()
        if getattr(old.args, key) != value
    }
    if not moved:
        return old
    return old.model_copy(update={"args": old.args.model_copy(update=moved)})


def kept_processor(old: Recipe, fresh: Recipe) -> dict[str, object]:
    """Onto or off the processor (its image: `ram_room.runs_on_processor`) every engine's
    `PROCESSOR_SIZED` settings are the plan's; else the plan's only where the user set none
    (`mnm set threads=` and `load=` stand), so a CPU plan's cores and its `none` in RAM reach
    an instance saved before them (else an audio.cpp re-planned onto the processor keeps
    `threads` unset and runs on one core)."""
    moved = runs_on_processor(old) != runs_on_processor(fresh)
    fields = type(old.args).model_fields
    return {
        key: getattr(fresh.args, key)
        for key in PROCESSOR_SIZED
        if key in fields and (moved or getattr(old.args, key) is None)
    }


def unset_cache(old: Recipe, fresh: Recipe) -> dict[str, object]:
    """A withheld re-plan's prompt cache where the recipe has none yet (saved by an older
    mnm): the caps (a re-plan that goes through sizes them again with the rest, `PLANNED`)."""
    if old.engine is not Engine.LLAMACPP:
        return {}
    return {
        key: getattr(fresh.args, key)
        for key in PlannedCache._fields
        if getattr(old.args, key) is None
    }


def kept_numa(old: LlamaCppArgs, fresh: LlamaCppArgs) -> dict[str, object]:
    """The user's `numa=` survives a re-plan, unless it binds to a node whose CPUs
    are no longer this box's (a new board, `mnm machine adopt`): then the plan's."""
    if old.numa is None:
        return {"numa": fresh.numa, "numa_cpus": None}
    node = old.numa_node
    if node is not None and node_cpus(node) != old.numa_cpus:
        return {"numa": fresh.numa, "numa_cpus": None, "threads": fresh.threads}
    return {"numa": old.numa, "numa_cpus": old.numa_cpus}


def node_cpus(index: int) -> str | None:
    found = node(read_nodes(), index)
    return found.cpus if found is not None else None
