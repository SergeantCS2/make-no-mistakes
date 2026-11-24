"""Hugging Face Hub: reference grammar, API client, and model facts."""

from makenomistakes.hub.client import (
    GatedOrPrivateError,
    HubClient,
    HubError,
    NotFoundError,
    SearchHit,
)
from makenomistakes.hub.facts import Format, Modality, ModelFacts, build_facts
from makenomistakes.hub.refs import (
    LocalPath,
    ModelRef,
    OllamaTagError,
    UnsupportedReferenceError,
    parse_reference,
)

__all__ = [
    "Format",
    "GatedOrPrivateError",
    "HubClient",
    "HubError",
    "LocalPath",
    "Modality",
    "ModelFacts",
    "ModelRef",
    "NotFoundError",
    "OllamaTagError",
    "SearchHit",
    "UnsupportedReferenceError",
    "build_facts",
    "parse_reference",
]
