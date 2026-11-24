"""What a large GGUF's tensor infos add to a plan: the shard of each split
variant that holds an Engram table, which stays on disk, and each block's own routed experts
when the keys say they differ (Nemotron Bolt: `expert_used_count` an array, experts on 40 of
its 88 blocks).

The header comes from one file (`make_plan` reads the smallest variant's first file, where
unsloth keeps every key and no tensor). The tensor infos are small reads more: the largest
shard of each variant of a model with Engram layers (where unsloth puts the table: a few
hundred bytes each), and every file of the smallest variant when experts differ by block
(about 8 MB into Bolt's single file, past its tokenizer). Nothing else reads them.
"""

from __future__ import annotations

from makenomistakes.gguf import GgufError, GgufHeader
from makenomistakes.gguf.reader import ByteSource, GgufLayout, read_header
from makenomistakes.gguf.tensors import (
    ENGRAM_TENSOR,
    TensorInfo,
    model_tensors,
    needs_tensors,
    read_tensors,
)
from makenomistakes.hub import HubClient, HubError, ModelFacts
from makenomistakes.hub.variants import EngramShard, GgufVariant
from makenomistakes.plan_images import unreadable_message
from makenomistakes.plan_types import PlanError, SourceReadError

# The sentence a quant whose Engram table is MXFP8 is refused with (fit and add alike).
MXFP8_REFUSED = (
    "llama.cpp does not read MXFP8, the type of this quant's Engram table (a build from "
    "2026-10-29 on loads the file's type 43 as plain FP8 and would read the table wrong): "
    "unsloth re-uploads these quants"
)


def model_header(hub: HubClient, revision: str, facts: ModelFacts) -> tuple[ModelFacts, GgufHeader]:
    """The header the plan sizes every variant from: the smallest variant's first file. A
    vision tower a repo names like a model (`DeepSeek-V4.1-Flash-Vision.gguf`: arch `clip` or
    `*-vision`) is no variant: it is dropped and the next smallest read."""
    variants = sorted(facts.gguf_variants, key=lambda v: v.size_bytes)
    for variant in variants:
        try:
            header = read_header(hub.file_source(facts.repo_id, revision, variant.files[0]))
        except (HubError, GgufError) as exc:
            raise SourceReadError(unreadable_message(facts, exc, "GGUF header")) from exc
        if not _vision_tower(header):
            kept = [v for v in facts.gguf_variants if v.size_bytes >= variant.size_bytes]
            return facts.model_copy(update={"gguf_variants": kept}), header
    raise PlanError(f"{facts.repo_id} holds vision towers only (no language model GGUF).")


def _vision_tower(header: GgufHeader) -> bool:
    return header.architecture == "clip" or header.architecture.endswith("-vision")


def with_large_facts(
    hub: HubClient, revision: str, facts: ModelFacts, header: GgufHeader
) -> tuple[ModelFacts, GgufHeader]:
    """The facts with each variant's Engram shard, and the header with each block's experts,
    when the model has them; as they came otherwise (no read made). A GGUF another
    converter wrote (`GgufLayout`) is refused: official llama.cpp reads the mainline one."""
    if header.layout is not GgufLayout.MAINLINE:
        raise PlanError(
            f"{facts.repo_id} was written by {header.layout.value}'s converter, whose "
            f"{header.architecture} layout no official llama.cpp image reads (the merged "
            "support takes the mainline converter's: `ple.layers` and the Engram table as "
            "`per_layer_token_embd.weight`, as unsloth's GGUFs of it are). mnm runs official "
            "images only: pick a GGUF made by llama.cpp's own converter."
        )
    if header.engram_layers:
        variants = [_with_engram(hub, revision, facts, v) for v in facts.gguf_variants]
        facts = facts.model_copy(update={"gguf_variants": variants})
    if needs_tensors(header):
        smallest = min(facts.gguf_variants, key=lambda v: v.size_bytes)
        found: list[TensorInfo] = []
        for file in smallest.files:
            found += _tensors(hub, revision, facts, file)
        header = header.with_tensors(model_tensors(found, header.block_count or 0))
    return facts, header


def _with_engram(
    hub: HubClient, revision: str, facts: ModelFacts, variant: GgufVariant
) -> GgufVariant:
    """The variant with its Engram shard, read from its largest file's tensor infos (the
    table is a model's largest single tensor: 111 GB at IQ4_NL)."""
    largest = variant.largest_file()
    table = next(
        (t for t in _tensors(hub, revision, facts, largest) if t.name == ENGRAM_TENSOR), None
    )
    if table is None:
        return variant
    shard = dict(zip(variant.files, variant.file_bytes, strict=False)).get(largest, 0)
    size = table.nbytes or shard  # a type ggml lacks (43): the whole shard
    return variant.model_copy(
        update={"engram": EngramShard(file=largest, size_bytes=size, tensor_type=table.type)}
    )


def _tensors(hub: HubClient, revision: str, facts: ModelFacts, file: str) -> list[TensorInfo]:
    try:
        source: ByteSource = hub.file_source(facts.repo_id, revision, file)
        return list(read_tensors(source).tensors)
    except (HubError, GgufError) as exc:
        raise SourceReadError(unreadable_message(facts, exc, "GGUF tensor list")) from exc


def refused_variants(facts: ModelFacts) -> list[GgufVariant]:
    """The variants mnm refuses: an Engram table of type MXFP8, which llama.cpp loaded none of
    until 68d20b884 and reads as plain FP8 since."""
    return [v for v in facts.gguf_variants if v.engram is not None and v.engram.unreadable]


def refusal(facts: ModelFacts, quant: str) -> str | None:
    """Why `quant` is refused, naming the quants that load; None when it is not."""
    refused = {v.quant.upper() for v in refused_variants(facts)}
    if quant.upper() not in refused:
        return None
    loads = [v.quant for v in facts.gguf_variants if v.quant.upper() not in refused]
    return f"{quant}: {MXFP8_REFUSED}; {_listed(loads)} load."


def _listed(names: list[str]) -> str:
    if len(names) < 2:
        return names[0] if names else "no other quant"
    return f"{', '.join(names[:-1])} and {names[-1]}"
