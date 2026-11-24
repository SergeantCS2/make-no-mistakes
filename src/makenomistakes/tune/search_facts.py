"""What the tuner's search reads before its first trial: the header's blocks and experts,
the room the plan leaves, the CPU, the draft, the box's other builds and nodes, and the fit
model's word on a candidate."""

from __future__ import annotations

from typing import Any

from makenomistakes.backend import Backend
from makenomistakes.backend_now import box_now
from makenomistakes.backend_setting import CHOICES as BACKEND_CHOICES
from makenomistakes.backend_setting import current as backend_current
from makenomistakes.backend_setting import offered as backend_offered
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.engine_images import IMAGES
from makenomistakes.fit.estimate import FitLevel
from makenomistakes.hardware.numa_nodes import read_nodes
from makenomistakes.hub.facts import Engine
from makenomistakes.instances import InstanceService
from makenomistakes.model_files import read_model_files
from makenomistakes.ram_room import runs_on_processor
from makenomistakes.recipes import Recipe
from makenomistakes.tune.cpus import CpuLayout, read_layout
from makenomistakes.tune.search import SearchFacts
from makenomistakes.tune.trial import CandidateError, candidate_recipe
from makenomistakes.tune.trial_types import Candidate


def search_facts(
    service: InstanceService, recipe: Recipe, cpus: CpuLayout | None = None
) -> SearchFacts:
    """What the stages read: the header's blocks and experts, the VRAM the plan leaves, the
    CPU, the draft, and the fit model's word on a candidate (a step up in memory)."""
    files = read_model_files(recipe, service.models_dir)
    header = files.header if files is not None else None
    snapshot = service.memory()
    here = runs_on_processor(recipe)  # The RAM it sits in, else its cards
    cards = ([snapshot.ram] if snapshot.ram is not None else []) if here else snapshot.cards
    total = sum(c.vram_total_bytes for c in cards)
    used = sum(c.vram_used_bytes for c in cards)
    planned = service.vram_estimate(recipe).bytes
    left = total - planned if cards and planned is not None else None
    held = sum(service.running_vram(in_ram=here).values())

    def fits(candidate: Candidate) -> bool:
        if not cards:
            return False
        try:
            estimate = service.vram_estimate(candidate_recipe(service, recipe, candidate))
        except CandidateError:
            return False
        if estimate.bytes is None:
            return False
        return estimate.bytes <= total - max(used - held, 0)

    return SearchFacts(
        args=recipe.llama_args,
        blocks=header.block_count if header is not None else None,
        moe=bool(header is not None and header.is_moe),
        vram_left_bytes=left,
        cpus=cpus or read_layout(),
        draft=recipe.draft is not None,
        mtp=recipe.mtp is not None and recipe.mtp in recipe.local_paths,
        fits=fits,
        kv_held=recipe.quality is not None,
        **backend_facts(service, recipe),
        **numa_search_facts(recipe),
    )


def numa_search_facts(recipe: Recipe) -> dict[str, Any]:
    """The box's NUMA nodes for the numa stage and the threads per node; none on one
    node."""
    nodes = read_nodes()
    if len(nodes) < 2 or not isinstance(recipe.args, LlamaCppArgs):
        return {}
    args = recipe.args
    in_ram = args.gpu_layers == 0 or args.experts_in_ram or (recipe.fit_level == FitLevel.OFFLOAD)
    roomiest = max(nodes, key=lambda n: n.free_bytes or 0)
    return {
        "nodes": tuple((n.index, n.cores) for n in nodes),
        "roomiest": roomiest.index,
        "in_ram": in_ram,
    }


def backend_facts(service: InstanceService, recipe: Recipe) -> dict[str, Any]:
    """The instance's backend and the other llama.cpp builds its AMD card runs whose image
    is pulled here: the backend stage's two facts; none elsewhere."""
    if recipe.engine is not Engine.LLAMACPP or not backend_offered():
        return {}
    now = backend_current(recipe)
    box = box_now()
    others = []
    for choice in BACKEND_CHOICES:
        if choice == now or box is None:
            continue
        tag = IMAGES[Engine.LLAMACPP].image_for(Backend(choice), box.arch)
        if isinstance(tag, str) and service.build_of(tag) is not None:  # pulled here
            others.append(choice)
    return {"backend": now, "backends": tuple(others)}
