"""Links from the other sources, turned into what the planner reads: an LM Studio
catalog link into the GGUF repo its `model.yaml` names on the Hub, and a version Civitai no
longer has into the Civitai Archive's copy of it on the Hub. Either way the plan, the recipe
and every later step are an ordinary Hugging Face one; `ModelFacts.via` says where it came
from."""

from __future__ import annotations

from urllib.parse import quote

from makenomistakes.guide import Alternative, Guidance
from makenomistakes.hub import HubClient, HubError, ModelFacts, ModelRef, build_facts
from makenomistakes.hub.civitai import (
    CivitaiError,
    CivitaiGoneError,
    CivitaiRef,
    CivitaiRefusedError,
    SourcesHub,
    as_hub_model,
)
from makenomistakes.hub.client import NotFoundError
from makenomistakes.hub.facts import Format
from makenomistakes.hub.lmstudio import LmStudioError, LmStudioFamily, parse_lmstudio
from makenomistakes.hub.modelscope import ModelScopeError, ModelScopeRef
from makenomistakes.hub.ollama import OllamaError, OllamaRef, OllamaRefusedError
from makenomistakes.plan_types import PlanError, SourceReadError

MAX_LISTED = 10  # a family's models offered to pick from


def lmstudio_ref(reference: str, hub: HubClient) -> ModelRef | None:
    """The Hub repo an LM Studio model link stands for (its first GGUF build); None for any
    other reference. A family page and a model with MLX builds only are refused with what
    to paste instead."""
    try:
        found = parse_lmstudio(reference)
        if found is None:
            return None
        if not isinstance(hub, SourcesHub):
            raise PlanError("LM Studio links need the sources client (internal error).")
        resolved = hub.lmstudio.resolve(found)
    except LmStudioError as exc:
        raise PlanError(str(exc)) from exc
    if isinstance(resolved, LmStudioFamily):
        raise PlanError(
            f"{found.link} is a family of {len(resolved.models)} models: pick one.",
            Guidance(
                what_it_is=f"An LM Studio family page ({resolved.name}): one entry per size.",
                what_to_do="Paste one model's link; it is planned from its GGUF build on "
                "Hugging Face, sized for this card like any other.",
                alternatives=[
                    Alternative(repo_id=f"lmstudio.ai/models/{m}", downloads=0, note="LM Studio")
                    for m in resolved.models[:MAX_LISTED]
                ],
            ),
        )
    if not resolved.gguf_repos:
        builds = ", ".join(resolved.other_builds[:3]) or "none"
        raise PlanError(
            f"LM Studio lists no GGUF build of {resolved.name}.",
            Guidance(
                what_it_is=f"{resolved.name} on LM Studio comes as MLX ({builds}): the format "
                "for Apple silicon.",
                what_to_do="This machine runs GGUF on llama.cpp (and safetensors on vLLM): "
                "paste a Hugging Face GGUF of the same model.",
            ),
        )
    repo = resolved.gguf_repos[0]
    return ModelRef(repo, via=f"LM Studio's {resolved.name} is {repo} on Hugging Face.")


def archived_facts(
    hub: SourcesHub, ref: CivitaiRef, gone: CivitaiGoneError
) -> tuple[ModelRef, ModelFacts]:
    """The facts of a version Civitai no longer has, read from the Civitai Archive's copy on
    the Hub (`SourcesHub.archived`: its checksum matched the archive's), described as
    Civitai's own versions are (`as_hub_model`): the archive's type, and the one file."""
    try:
        found = hub.archived(ref, gone)  # ArchiveError: no usable copy
        info = as_hub_model(found.copy.model, found.copy.type, found.ref.repo_id)
    except HubError as exc:  # CivitaiError: not a checkpoint or LoRA
        raise PlanError(str(exc)) from exc
    via = (
        f"Civitai no longer has {found.copy.model}: planned from the Civitai Archive's copy "
        f"on Hugging Face, {found.ref.repo_id}/{found.ref.path} (its sha256 matches the "
        "archive's record)."
    )
    facts = build_facts(info, [found.entry], found.ref)
    return found.ref, facts.model_copy(update={"display_name": found.copy.model, "via": via})


def facts_for(
    hub: SourcesHub, ref: ModelRef | CivitaiRef | OllamaRef | ModelScopeRef
) -> tuple[ModelRef, ModelFacts]:
    """The facts and the Hub-shaped reference they were built for: a Civitai link resolves
    to its version, `civitai/<model id>` at revision `<version id>`; an
    Ollama tag to `ollama/<namespace>--<name>` at revision `<tag>`."""
    if isinstance(ref, OllamaRef):
        try:
            info, tree, hub_ref = hub.ollama.resolve(ref).as_hub()
        except OllamaRefusedError as exc:  # a projector layer, not one GGUF
            raise PlanError(str(exc)) from exc
        except OllamaError as exc:
            raise SourceReadError(str(exc)) from exc
        facts = build_facts(info, tree, hub_ref)
        return hub_ref, facts.model_copy(update={"display_name": f"{ref.name}:{ref.tag}"})
    if isinstance(ref, ModelScopeRef):
        try:
            info, tree, hub_ref = hub.modelscope.resolve(ref).as_hub()
        except ModelScopeError as exc:
            raise SourceReadError(str(exc)) from exc
        facts = build_facts(info, tree, hub_ref)
        if facts.format is not Format.GGUF:  # GGUF only from here: llama.cpp reads one file
            raise PlanError(
                f"{ref.page} is not a GGUF model: mnm takes GGUF models from ModelScope (a "
                "safetensors model: paste its Hugging Face repo)."
            )
        return hub_ref, facts.model_copy(update={"display_name": ref.repo})
    if isinstance(ref, CivitaiRef):
        try:
            resolved = hub.civitai.resolve(ref)
            info, tree = resolved.as_hub()
        except CivitaiGoneError as gone:  # deleted from Civitai: the archive may hold a copy
            return archived_facts(hub, ref, gone)
        except CivitaiRefusedError as exc:  # not a checkpoint or LoRA, no safetensors file
            raise PlanError(str(exc)) from exc
        except CivitaiError as exc:
            raise SourceReadError(str(exc)) from exc
        facts = build_facts(info, tree, resolved.ref)
        return resolved.ref, facts.model_copy(update={"display_name": resolved.name})
    try:
        facts = build_facts(hub.model_info(ref.repo_id), hub.tree(ref.repo_id, ref.revision), ref)
        return ref, facts.model_copy(update={"via": ref.via}) if ref.via else facts
    except NotFoundError as exc:  # a typo is the usual cause: point at a search for the name
        name = ref.repo_id.rsplit("/", 1)[-1]
        raise SourceReadError(
            f"{exc} Search for it: https://huggingface.co/models?search={quote(name)}"
        ) from exc
    except HubError as exc:
        raise SourceReadError(str(exc)) from exc
