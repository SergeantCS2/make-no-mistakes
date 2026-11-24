"""`mnm what`: where the hits come from, before any is graded (`what_grade`).

- **Curated first**: the starter menu's models for the use (`starter.CATALOG`,
  each item naming its use: what has run on a 12 GB card, each with its one line), the
  words matching its reference, its instance name, the starter's title or its line.
- **Then each source's own order**, never a ranking of mnm's, by the use's row
  (`model_uses.USES`): the Hub's search (most downloaded first; GGUF repos for the llama.cpp
  uses, the pipeline's tag for pictures and clips, the hits of another use left out by their
  own tag), Civitai's for pictures (checkpoints, most downloaded, safe-for-work), and Ollama's
  library page for the chat-like uses (`ollama.com` has no search API; `OllamaClient.search`
  reads the page's links, by the use's category: tools for a coder, embedding for memory).

A source that does not answer is a note, never why the others are not shown. Nothing here
downloads a model file or reads a header: the grade does, over ranged reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from makenomistakes.hub.civitai import CivitaiError
from makenomistakes.hub.civitai import SourcesHub as Sources
from makenomistakes.hub.client import HubError
from makenomistakes.hub.ollama import OLLAMA_DOMAIN, OllamaError
from makenomistakes.model_uses import Use, UseRow, row_for
from makenomistakes.starter import CATALOG

OTHER_SOURCES = 3  # Civitai's and Ollama's rows at most (or `--limit` when smaller)


class Source(StrEnum):
    CURATED = "curated"
    HUB = "Hugging Face"
    CIVITAI = "Civitai"
    OLLAMA = "Ollama"


@dataclass(frozen=True)
class Hit:
    """One model to grade: the reference `mnm add` takes, where it was found, and the
    source's download count (None where the source has none)."""

    reference: str
    source: Source
    use: Use | None
    downloads: int | None = None
    why: str | None = None  # a curated row's line
    quant: str | None = None  # a curated row's quant (`mnm add --quant`)
    name: str | None = None  # a curated row's instance name
    label: str | None = None  # a curated row's starter title

    @property
    def key(self) -> str:
        return self.reference.lower().rstrip("/")

    @property
    def shown(self) -> str:
        """The reference as a row shows it: a link without its scheme."""
        return self.reference.removeprefix("https://")

    @property
    def curated(self) -> bool:
        return self.source is Source.CURATED


@dataclass
class Found:
    hits: list[Hit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # a source that did not answer, said


def curated(use: Use | None) -> list[Hit]:
    """The starter menu's rows for `use` (every use's when None), in the menu's order."""
    return [
        Hit(
            item.reference,
            Source.CURATED,
            item.use,
            why=starter.what,
            quant=item.quant,
            name=item.name,
            label=starter.title,
        )
        for starter in CATALOG
        for item in starter.items
        if item.use is not None and (use is None or item.use is use)
    ]


def _matches(hit: Hit, words: list[str]) -> bool:
    text = " ".join(t for t in (hit.reference, hit.name, hit.label, hit.why) if t).lower()
    return all(w in text for w in words)


def find(sources: Sources, words: str, use: Use | None, limit: int) -> Found:
    """The curated rows the words match (all of them with no words), then each source's hits
    in its own order; one row per reference."""
    found = Found()
    terms = words.lower().split()
    found.hits += [h for h in curated(use) if not terms or _matches(h, terms)]
    row = row_for(use)
    if terms and row.searched:
        if row.hub_filters:
            found.hits += _hub(sources, words, use, row, limit, found.notes)
        others = min(limit, OTHER_SOURCES)
        if row.civitai:
            found.hits += _civitai(sources, words, use, others, found.notes)
        if row.ollama:
            found.hits += _ollama(sources, words, use, row, others, found.notes)
    seen: set[str] = set()
    unique = []
    for hit in found.hits:
        if hit.key not in seen:
            seen.add(hit.key)
            unique.append(hit)
    found.hits = unique
    return found


def _hub(
    sources: Sources, words: str, use: Use | None, row: UseRow, limit: int, notes: list[str]
) -> list[Hit]:
    try:  # three times the rows: the hits of another use are left out after
        hits = sources.search(search=words, filters=row.hub_filters, limit=limit * 3)
    except HubError as exc:
        notes.append(f"Hugging Face's search did not answer: {exc}")
        return []
    kept = [
        Hit(h.id, Source.HUB, use, h.downloads)
        for h in hits
        if row.pipelines is None or h.pipeline_tag in row.pipelines
    ]
    return kept[:limit]


def _civitai(
    sources: Sources, words: str, use: Use | None, limit: int, notes: list[str]
) -> list[Hit]:
    try:
        models = sources.civitai.search(
            words, limit, types=("Checkpoint",), sfw=True, sort="Most Downloaded"
        )
    except CivitaiError as exc:
        notes.append(f"Civitai's search did not answer: {exc}")
        return []
    return [
        Hit(f"https://civitai.com/models/{m.id}", Source.CIVITAI, use, m.stats.download_count)
        for m in models
        if not m.nsfw
    ]


def _ollama(
    sources: Sources, words: str, use: Use | None, row: UseRow, limit: int, notes: list[str]
) -> list[Hit]:
    try:
        names = sources.ollama.search(words, limit, category=row.ollama_category)
    except OllamaError as exc:
        notes.append(str(exc))
        return []
    return [Hit(f"https://{OLLAMA_DOMAIN}/library/{n}", Source.OLLAMA, use) for n in names]
