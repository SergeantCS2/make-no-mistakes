"""What the user wants a model for (`mnm what --type`, `--for`), one row per use: the
planner's types that serve it, the type `mnm add` asks for, and where each source is searched
for it. A leaf module: the starter menu names each item's use with it, `what_sources` searches
by its rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from makenomistakes.hub.modality import Modality


class Use(StrEnum):
    """The uses, by the words `mnm add --type` and the starter menu use."""

    CHAT = "chat"
    CODER = "coder"
    VISION = "vision"
    PICTURES = "pictures"
    VIDEO = "video"
    MEMORY = "memory"
    VOICE = "voice"
    EAR = "ear"

    @property
    def row(self) -> UseRow:
        return USES[self]


@dataclass(frozen=True)
class UseRow:
    """A use's searches and the types it takes. Empty `types` is any use's row (no `--type`)."""

    types: frozenset[Modality]  # the planner's types that serve it
    plans_as: Modality | None = None  # what `mnm add` asks for (a coder: `--for-agents`)
    hub_filters: tuple[str, ...] = ()  # the Hub's search filters; empty: not searched there
    pipelines: frozenset[str | None] | None = None  # a hit's own tag (None: a repo sets none)
    ollama: bool = False  # Ollama's library searched
    ollama_category: str | None = None  # its search page's `c=` (embedding, tools)
    civitai: bool = False  # Civitai's checkpoints searched

    @property
    def searched(self) -> bool:
        """Any source is searched for it; voice and ear are the curated rows only: their
        engines read packages and size files no search names."""
        return bool(self.hub_filters) or self.ollama or self.civitai

    @property
    def for_agents(self) -> bool:
        """Planned as `mnm add --for-agents` plans: type coding, the agent's context."""
        return self.plans_as is Modality.CODING

    def serves(self, modality: Modality) -> bool:
        return not self.types or modality in self.types


_TEXT = frozenset({None, "text-generation"})  # many GGUF repos set no pipeline tag
USES: dict[Use, UseRow] = {
    Use.CHAT: UseRow(  # a coding repo chats; the retype only names it for agents
        types=frozenset({Modality.CHAT, Modality.CODING}),
        hub_filters=("gguf",),
        pipelines=_TEXT,
        ollama=True,
    ),
    Use.CODER: UseRow(
        types=frozenset({Modality.CODING}),
        plans_as=Modality.CODING,
        hub_filters=("gguf",),
        # a vision repo codes too, added text-only (Qwen3.6's repos are image-text-to-text)
        pipelines=_TEXT | {"image-text-to-text"},
        ollama=True,
        ollama_category="tools",  # an agent calls tools: a model without them is no coder
    ),
    Use.VISION: UseRow(  # not Ollama's: mnm refuses its projector layers (`hub/ollama.py`)
        types=frozenset({Modality.VISION_CHAT}),
        hub_filters=("gguf", "image-text-to-text"),
        pipelines=frozenset({"image-text-to-text"}),
    ),
    Use.PICTURES: UseRow(  # an editor draws too
        types=frozenset({Modality.IMAGE_GEN, Modality.IMAGE_EDIT}),
        hub_filters=("text-to-image",),
        pipelines=frozenset({"text-to-image"}),
        civitai=True,
    ),
    Use.VIDEO: UseRow(
        types=frozenset({Modality.VIDEO}),
        hub_filters=("text-to-video",),
        pipelines=frozenset({"text-to-video", "image-to-video"}),
    ),
    Use.MEMORY: UseRow(
        types=frozenset({Modality.EMBEDDING}),
        hub_filters=("gguf",),
        pipelines=frozenset({"feature-extraction", "sentence-similarity"}),
        ollama=True,
        ollama_category="embedding",
    ),
    Use.VOICE: UseRow(types=frozenset({Modality.TEXT_TO_SPEECH})),
    Use.EAR: UseRow(types=frozenset({Modality.SPEECH_TO_TEXT})),
}
ANY_USE = UseRow(types=frozenset(), hub_filters=("gguf",), ollama=True)  # no `--type`
# the words `--type` takes too: `mnm add --type`'s
ALIASES = {"coding": Use.CODER, "image": Use.PICTURES, "embedding": Use.MEMORY}


def row_for(use: Use | None) -> UseRow:
    return ANY_USE if use is None else use.row


def parse_use(text: str | None) -> Use | None:
    """`--type`/`--for`: a use's word, or `mnm add --type`'s (`coding`, `image`)."""
    if text is None:
        return None
    word = text.strip().lower()
    if word in ALIASES:
        return ALIASES[word]
    try:
        return Use(word)
    except ValueError:
        uses = ", ".join(u.value for u in Use)
        raise ValueError(f"`{text}` is not a use mnm grades for: {uses}.") from None
