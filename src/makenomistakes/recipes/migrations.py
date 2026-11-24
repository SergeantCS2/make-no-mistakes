"""Saved recipe files outlive the code that wrote them: each schema version and the step
that brings a file at version n to n + 1 (`load_recipe_json` applies them)."""

from __future__ import annotations

from collections.abc import Callable

# Each step takes the raw dict at version n to version n + 1 (`recipes.model.SCHEMA_VERSION`).
Migration = Callable[[dict[str, object]], dict[str, object]]


def _v1_to_v2(data: dict[str, object]) -> dict[str, object]:
    """Added the `args.engine` discriminator. Every earlier recipe ran llama.cpp and
    carries its engine at the top level (always required), so the tag is copied
    from there; nothing is invented."""
    args = data.get("args")
    if "engine" in data and isinstance(args, dict) and "engine" not in args:
        return {**data, "args": {**args, "engine": data["engine"]}}
    return data


def _v2_to_v3(data: dict[str, object]) -> dict[str, object]:
    """Added `args.diffusion_type` (sd.cpp, default none).
    Nothing changes in an older file; the bump is so an older runner refuses a newer file
    instead of dropping the field when it rewrites it."""
    return data


def _v3_to_v4(data: dict[str, object]) -> dict[str, object]:
    """Added `high_noise` (Wan 2.2 A14B's first stage) and `upscaler`, both with defaults
    an older file means (none). The bump makes an older runner refuse a file that names them
    instead of loading it and starting a half of the pair alone."""
    return data


def _v4_to_v5(data: dict[str, object]) -> dict[str, object]:
    """Added the `audiocpp` engine (its own args discriminator): a no-op for every
    earlier file, so an older runner refuses a voice recipe instead of failing on its args."""
    return data


def _v5_to_v6(data: dict[str, object]) -> dict[str, object]:
    """Added `args.strength` (sd.cpp, default unset): a no-op for every earlier file, so
    an older runner refuses a file that sets it instead of dropping it on rewrite."""
    return data


def _v6_to_v7(data: dict[str, object]) -> dict[str, object]:
    """Added the `vllm` engine (its own args discriminator): a no-op for every earlier
    file, so an older runner refuses a vLLM recipe instead of failing on its args."""
    return data


def _v7_to_v8(data: dict[str, object]) -> dict[str, object]:
    """Added `swap_group` (models that take turns on the GPU): a no-op, so an older runner
    refuses a swapping instance instead of running it without its router."""
    return data


def _v8_to_v9(data: dict[str, object]) -> dict[str, object]:
    """Added `off` (the user stopped it: it stays off until started) and the router. An
    older file cannot say whether the user stopped the instance or a swap parked
    it, and calling it parked would load models the user had stopped: every migrated file
    is `off`. One running at the update is loaded whatever `off` says, and the router's own
    park or idle unload clears it; `mnm start` and `workflow start` clear it too."""
    return {**data, "off": True}


def _v9_to_v10(data: dict[str, object]) -> dict[str, object]:
    """Added `idle` (unload after this many seconds without a request; default 600) and
    `keep` (`always` pins it). Both default to what an older file meant (the router's idle
    unload and no pin), so nothing changes; the bump is so an older runner refuses a file
    that pins an instance instead of loading it and unloading the pinned model."""
    return data


def _v10_to_v11(data: dict[str, object]) -> dict[str, object]:
    """Added `args.load_mode` (llama.cpp's `--load-mode`, default unset: the
    engine's own): a no-op, so an older runner refuses a file that sets it instead of
    dropping it on rewrite."""
    return data


def _v11_to_v12(data: dict[str, object]) -> dict[str, object]:
    """Added `draft` (a draft model for speculative decoding, default none) and
    `args.draft_max`: a no-op, so an older runner refuses a file with a draft instead of
    dropping it on rewrite and starting the model without it."""
    return data


def _v12_to_v13(data: dict[str, object]) -> dict[str, object]:
    """Added `args.draft_p_min` (`--spec-draft-p-min`) and
    moved `draft_max`'s default to llama.cpp's 3: a no-op (a saved 16 stays until a draft is
    attached again), so an older runner refuses a file with a p-min instead of dropping it."""
    return data


def _v13_to_v14(data: dict[str, object]) -> dict[str, object]:
    """Added `mtp` (the repo's MTP head, downloaded with the model), `draft.kind` (an
    MTP head as the draft, `draft=mtp`) and `args.moe_cache_mib` (`--moe-cache-mib`), each
    defaulting to what an older file meant: a no-op, so an older runner refuses a file that
    drafts with an MTP head instead of starting it with the head as a plain draft model."""
    return data


def _v14_to_v15(data: dict[str, object]) -> dict[str, object]:
    """Added `gpus` (the cards an instance is pinned to, default none: `MNM_GPU`'s),
    `args.tensor_split`, `args.split_mode` and `args.main_gpu` (llama.cpp across cards) and
    `args.tensor_parallel` (vLLM), each defaulting to what an older file meant (one card): a
    no-op, so an older runner refuses a file planned across cards instead of starting it on
    one card with a plan made for two."""
    return data


def _v15_to_v16(data: dict[str, object]) -> dict[str, object]:
    """Added `evicted` (the disk policy deleted the instance's files; its next start
    downloads them again), default none: a no-op, so an older runner refuses an evicted
    recipe instead of reading it as merely not downloaded and forgetting why."""
    return data


def _v16_to_v17(data: dict[str, object]) -> dict[str, object]:
    """Added `engram` (the shard whose Engram table stays on disk), `draft_heads` (the
    repo's draft heads mnm does not run), `draft.kind` `dspark` and `args.lazy_tensors`
    (`--lazy-mode on -ot <tensor>=CPU`), each defaulting to what an older file meant: a no-op,
    so an older runner refuses a file whose Engram table it would load into RAM."""
    return data


def _v17_to_v18(data: dict[str, object]) -> dict[str, object]:
    """Added `tuned` (the tuner's chosen settings, score and key, default none: never
    tuned): a no-op, so an older runner refuses a tuned file instead of starting it and
    dropping the block a later `mnm ls` would have read."""
    return data


def _v18_to_v19(data: dict[str, object]) -> dict[str, object]:
    """Added `architecture` and, for llama.cpp across several cards, `args.placement`
    (each card's blocks and experts, `--device`/`-ts`/`-ot`) and `args.experts`, each
    defaulting to what an older file meant (none, auto): a no-op, so an older runner refuses
    a file placed per card instead of starting it with every expert on its block's card."""
    return data


def _v19_to_v20(data: dict[str, object]) -> dict[str, object]:
    """Added `planned_on` and `planned_cards` (the box a plan was made on), None in an
    older file ("planned before mnm kept fingerprints"): a no-op, so an older runner refuses
    a file that names its box instead of dropping the field at its next save."""
    return data


def _v20_to_v21(data: dict[str, object]) -> dict[str, object]:
    """Added `stable_id` to each card of a placement (`args.placement.cards`: the card's
    uuid, so a card that moved slot keeps its part), None in an older file ("planned before
    mnm kept card ids", rendered by its indices as before): a no-op, so an older runner
    refuses a file that names its cards instead of dropping the ids at its next save."""
    return data


def _v21_to_v22(data: dict[str, object]) -> dict[str, object]:
    """Added `args.numa` (the CPU backend's `--numa distribute`), None in an older file,
    and let `args.gpu_layers` be 0 (every block in RAM): a no-op, so an older runner refuses a
    CPU plan instead of dropping its NUMA mode or failing on `-ngl 0` at its next read."""
    return data


def _v22_to_v23(data: dict[str, object]) -> dict[str, object]:
    """Widened `args.numa` to the user's setting (`isolate`, `off`, `node:N`) and added
    `args.numa_cpus` (the node's CPUs a `node:N` binds to), None in an older file: a no-op, so
    an older runner refuses a recipe bound to one node instead of starting it unbound."""
    return data


def _v23_to_v24(data: dict[str, object]) -> dict[str, object]:
    """Widened `args.family` and `args.task` of audio.cpp recipes to the studio's music
    families and tasks, made `args.voice` optional and added `args.options` (the per-family
    options), None in an older file: a no-op, so an older runner refuses a music recipe
    instead of starting it as a voice."""
    return data


def _v24_to_v25(data: dict[str, object]) -> dict[str, object]:
    """An ACE-Step instance's `options` may name a style from the library (`style`,
    `style_scale`: the LoRA audio.cpp merges at load), absent in an older file: a no-op, so an
    older runner refuses a recipe with a style instead of starting it without one."""
    return data


def _v25_to_v26(data: dict[str, object]) -> dict[str, object]:
    """A llama.cpp instance's prompt cache (`cache_ram_mib`, `slot_cache`,
    `cache_disk_gib`), absent in an older file: a no-op (absent is llama.cpp's own cache until
    `mnm replan` sizes it), so an older runner refuses a recipe that mounts a cache folder
    instead of starting it without one."""
    return data


def _v26_to_v27(data: dict[str, object]) -> dict[str, object]:
    """The quality tune's choice (`quality`), absent in an older file: a no-op (absent is
    never quality-tuned), so an older runner refuses a recipe whose KV cache a quality tune
    holds instead of letting its speed tune move it."""
    return data


MIGRATIONS: dict[int, Migration] = {
    1: _v1_to_v2,
    2: _v2_to_v3,
    3: _v3_to_v4,
    4: _v4_to_v5,
    5: _v5_to_v6,
    6: _v6_to_v7,
    7: _v7_to_v8,
    8: _v8_to_v9,
    9: _v9_to_v10,
    10: _v10_to_v11,
    11: _v11_to_v12,
    12: _v12_to_v13,
    13: _v13_to_v14,
    14: _v14_to_v15,
    15: _v15_to_v16,
    16: _v16_to_v17,
    17: _v17_to_v18,
    18: _v18_to_v19,
    19: _v19_to_v20,
    20: _v20_to_v21,
    21: _v21_to_v22,
    22: _v22_to_v23,
    23: _v23_to_v24,
    24: _v24_to_v25,
    25: _v25_to_v26,
    26: _v26_to_v27,
}
