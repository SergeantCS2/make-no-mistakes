"""The vLLM branch of `make_plan`: read the model's `config.json`, refuse what needs the
repo's own code, plan the three profiles."""

from __future__ import annotations

from pydantic import ValidationError

from makenomistakes.fit import Budget
from makenomistakes.fit.estimate import GIB
from makenomistakes.fit.ram_floor import GB
from makenomistakes.fit.vllm import plan_vllm_profiles
from makenomistakes.guide import gguf_builds
from makenomistakes.hub import HubClient, HubError, ModelFacts
from makenomistakes.hub.hf_config import HfConfig
from makenomistakes.hub.vllm_files import config_path
from makenomistakes.plan_types import NoRoomError, Plan, PlanError, SourceReadError


def read_config(facts: ModelFacts, hub: HubClient) -> HfConfig:
    path = config_path(facts.vllm_files)
    try:
        return HfConfig.model_validate(hub.json_file(facts.repo_id, path, facts.revision))
    except (HubError, ValidationError) as exc:
        raise SourceReadError(f"{facts.repo_id}: its {path} could not be read ({exc}).") from exc


def vllm_plan(facts: ModelFacts, hub: HubClient, budget: Budget) -> Plan:
    _refuse_larger_than_cards(facts, hub, budget)
    config = read_config(facts, hub)
    if config.needs_remote_code:
        raise PlanError(
            f"{facts.repo_id} needs its own Python code to load (`auto_map` in config.json, "
            "vLLM's `--trust-remote-code`), which this tool never runs. Look for a build of "
            "the model that transformers supports natively, or a GGUF of it."
        )
    card, tp = one_card(budget, config.text.num_attention_heads or 0)
    try:
        profiles = plan_vllm_profiles(facts, config, card, tp)
    except ValueError as exc:  # a config without layer or head counts
        raise PlanError(f"{facts.repo_id}: {exc}") from exc
    return Plan(facts, None, card, profiles, hf_config=config)


def one_card(budget: Budget, heads: int) -> tuple[Budget, int | None]:
    """The card vLLM plans each worker for: the budget itself on one card; on equal
    cards the one with the least free, with tensor parallel across all of them (vLLM asserts
    that every worker reports the same KV blocks, `kv_cache_utils.py`, so unequal cards
    cannot, and the attention heads must divide by their number); otherwise the first card
    alone (pipeline parallel, for uneven splits, is later)."""
    cards = list(budget.cards)
    if len(cards) < 2:
        return budget, None
    equal = len({(c.name, c.total_bytes) for c in cards}) == 1
    tp = len(cards) if equal and heads and heads % len(cards) == 0 else None
    first = min(cards, key=lambda c: c.free_bytes) if tp else cards[0]
    alone = budget.on_card(first)
    each = {"gpu_bandwidth": first.bandwidth * tp} if tp else {}  # each card reads its shard
    return alone.model_copy(update=each), tp


def _refuse_larger_than_cards(facts: ModelFacts, hub: HubClient, budget: Budget) -> None:
    """vLLM keeps every weight on the cards (it has no offload to RAM): weights larger than
    all of the cards' VRAM are refused with the sizes, before config.json is read, and the
    model's GGUF builds offered, which llama.cpp runs from RAM (DeepSeek-V4.1-Flash's
    510 GB of safetensors on a 12 GiB card)."""
    weights = facts.vllm_bytes or facts.safetensors_bytes
    cards = budget.vram_total_bytes or budget.vram_free_bytes
    if weights <= cards:
        return
    raise NoRoomError(
        f"{facts.repo_id} holds {weights / GB:.0f} GB of weights and vLLM keeps every one of "
        f"them on the cards (no offload to RAM); this machine has {cards / GIB:.0f} GiB of "
        "VRAM. A GGUF build of it runs on llama.cpp, from RAM and the cards together.",
        gguf_builds(facts, hub),
    )


def vllm_folder(facts: ModelFacts, version: str) -> str:
    """The model folder `--version` names: the whole folder, else the one it is part of. A
    repo whose root is the model lists no folders: the name is taken as the folder."""
    if not facts.versions:
        return version.strip("/")
    folders = facts.versions
    exact = [f for f in folders if f == version.strip("/")]
    matches = exact or [f for f in folders if version.lower() in f.lower()]
    if len(matches) != 1:
        raise PlanError(
            f"{facts.repo_id}: '{version}' names {len(matches)} of its model folders "
            f"({', '.join(folders)}); pass one of them."
        )
    return matches[0]
