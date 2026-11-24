"""How much prompt cache a llama.cpp instance gets, planned against this box, and the
words that say it.

llama-server keeps earlier prompts' states in RAM (`--cache-ram`, llama.cpp PR 16391): a
conversation a side request pushed out of the slot comes back without being read again. Its default
is a cap of 8 GiB nobody planned: on a 24 GB WSL box that is beside a MoE's experts, which sit in
RAM as mapped pages a growing cache pushes out to disk. So every plan says the cap, sized from what
the RAM keeps after the model's own part (`planned`), and `mnm show` names it.

Beside it the slots are saved to disk when the instance unloads and restored at its next load
(`slot_files`, `cache=on`, coding instances by default): on the CPU build a 4,311-token prefix
read again in 21.3 s came back in 0.19 s from its 495 MB file (2026-11-02, llama.cpp
a91eab67d). Neither touches what the instance holds on the card."""

from __future__ import annotations

from typing import NamedTuple

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.fit.budget import Budget
from makenomistakes.fit.profiles import Profile
from makenomistakes.hub.facts import EMBEDDING_MODALITIES, Modality

MIB = 1024 * 1024
GIB = 1024 * MIB
RAM_CAP_MIB = 8192  # llama.cpp's own default, the most a plan gives
PROMPTS = 4  # the states worth keeping: an agent's conversation, a subagent's, side requests'
RAM_SHARE = 4  # a plan takes at most this share of what the RAM keeps after the model
RAM_STEP_MIB = 256  # caps in whole steps, so a re-plan does not move them by a few MiB
DISK_FLOOR_GIB = 2.0  # the least a saved-slots folder may take
DISK_STEP_GIB = 0.5
CHECKPOINT_SHARE = 1.25  # a saved slot carries its context checkpoints and draft state too


def prompt_bytes(profile: Profile) -> int:
    """One full prompt's state as llama.cpp keeps it: its KV cache and (a hybrid model) its
    recurrent state, with the context checkpoints and draft state that travel with it."""
    return int((profile.estimate.kv_bytes + profile.estimate.state_bytes) * CHECKPOINT_SHARE)


def _steps(size: int, up: bool) -> int:
    """`size` bytes in whole `RAM_STEP_MIB` steps, as MiB (up or down)."""
    step = RAM_STEP_MIB * MIB
    count = -(-size // step) if up else size // step
    return count * RAM_STEP_MIB


def ram_cap_mib(budget: Budget, profile: Profile, modality: Modality) -> int:
    """`--cache-ram`: room for `PROMPTS` prompts at the plan's context (rounded up to a step),
    at most a quarter of the RAM the box keeps after its system margin and the model's own RAM
    part (rounded down) and at most llama.cpp's 8 GiB; 0 (off) for embedding and rerank, which
    decode nothing worth keeping, and where not even one prompt fits."""
    if modality in EMBEDDING_MODALITIES:
        return 0
    one = prompt_bytes(profile)
    ram = budget.ram_total_bytes if budget.ram_total_bytes is not None else budget.ram_free_bytes
    in_ram = budget.no_card or budget.pool  # the whole model sits in the RAM (or the pool)
    own = profile.estimate.total_bytes if in_ram else profile.ram_bytes
    spare = max(0, ram - budget.margin - own) // RAM_SHARE
    cap = min(_steps(one * PROMPTS, up=True), _steps(spare, up=False), RAM_CAP_MIB)
    return cap if one > 0 and cap * MIB >= one else 0


def disk_cap_gib(profile: Profile) -> float:
    """The most an instance's saved slots take: one full prompt, at least `DISK_FLOOR_GIB`, in
    half-GiB steps."""
    want = prompt_bytes(profile) / GIB
    steps = -(-want // DISK_STEP_GIB)  # ceil
    return max(DISK_FLOOR_GIB, steps * DISK_STEP_GIB)


class PlannedCache(NamedTuple):
    """The plan's prompt cache, as the `LlamaCppArgs` fields it sizes (and `mnm replan` sizes
    again: `replan.PLANNED`). Whether the slots are kept across unloads is no size: `cache`
    left unset is auto, on for a coding instance (`llamacpp_cache.keeps_slots`)."""

    cache_ram_mib: int
    cache_disk_gib: float


def planned(budget: Budget, profile: Profile, modality: Modality) -> PlannedCache:
    """The RAM cap and the disk cap at the plan's context."""
    return PlannedCache(ram_cap_mib(budget, profile, modality), disk_cap_gib(profile))


def words(args: LlamaCppArgs, keeps: bool) -> str:
    """`mnm show`'s line: what the cache may hold, where (`keeps`: its slots kept across
    unloads, `llamacpp_cache.keeps_slots`)."""
    if args.cache_ram_mib is None:
        ram = "llama.cpp's own (up to 8 GiB of RAM; `mnm replan` sizes it to this box)"
    elif args.cache_ram_mib == 0:
        ram = "off in RAM"
    else:
        ram = f"up to {_gib(args.cache_ram_mib / 1024)} GiB of RAM"
    auto = " (auto)" if args.slot_cache is None else ""
    if not keeps:
        return f"{ram}; not kept across unloads{auto} (`cache=on` keeps it)"
    cap = args.cache_disk_gib or DISK_FLOOR_GIB
    return f"{ram}; kept on disk across unloads{auto}, up to {_gib(cap)} GiB"


def _gib(value: float) -> str:
    """1.25 → "1.25", 2.0 → "2", 0.5 → "0.5"."""
    return f"{value:.2f}".rstrip("0").rstrip(".")
