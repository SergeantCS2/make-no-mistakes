"""What a re-planned recipe's missing files weigh (`mnm machine adopt` asks once, with the
sizes, before it downloads anything). Read from the facts the plan was
made from, never from the network again: the quant's files (each shard where the Hub's tree
listed it), its MTP head, the projector, the image weights' files, the companions. A file
the facts do not size is left out and counted as "size unknown"."""

from __future__ import annotations

from makenomistakes.hub.facts import ModelFacts
from makenomistakes.recipes.model import Recipe


def file_sizes(facts: ModelFacts, recipe: Recipe) -> dict[str, int]:
    """Path (a companion's too) -> bytes, for each of the recipe's files the facts size."""
    found: dict[str, int] = {c.path: c.size_bytes for c in recipe.companions if c.size_bytes}
    variant = facts.variant(recipe.quant)
    if variant is not None:
        if len(variant.file_bytes) == len(variant.files):
            found.update(zip(variant.files, variant.file_bytes, strict=True))
        elif len(variant.files) == 1 and variant.high_noise is None:
            found[variant.files[0]] = variant.size_bytes
        if variant.mtp and variant.mtp_bytes:
            found[variant.mtp] = variant.mtp_bytes
    for weights in facts.image_weights:
        found.update({path: size for path, size in weights.file_bytes.items() if size})
        if len(weights.files) == 1 and weights.high_noise is None and weights.size_bytes:
            found.setdefault(weights.files[0], weights.size_bytes)
    if facts.projector_files and facts.projector_bytes:
        found[facts.projector_files[0]] = facts.projector_bytes
    return found
