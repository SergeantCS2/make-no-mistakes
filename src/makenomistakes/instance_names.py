"""An instance's name: the checks a new name passes and the default
a model's family gives."""

from __future__ import annotations

import re

from makenomistakes.hub import ModelFacts
from makenomistakes.hub.facts import CHAT_LIKE, Modality
from makenomistakes.instance_types import InstanceError
from makenomistakes.recipes import validate_instance_name
from makenomistakes.recipes.llamacpp import VISION_ID_RE, text_only_name, vision_name
from makenomistakes.recipes.model import SELFTEST_PREFIX, TUNE_PREFIX
from makenomistakes.roles import ROLES
from makenomistakes.store import InstanceStore, StoreError

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def check_name(store: InstanceStore, name: str) -> str:
    try:
        validate_instance_name(name)
    except ValueError as exc:
        raise InstanceError(str(exc)) from exc
    if name.startswith((SELFTEST_PREFIX, TUNE_PREFIX)):
        raise InstanceError(
            f"Names starting with '{SELFTEST_PREFIX}' or '{TUNE_PREFIX}' are kept for the "
            "copies the engines' self-test and the tuner run: pick another name (`--name`)."
        )
    if name in ROLES:  # the desk's roles are model ids: never an instance's
        raise InstanceError(
            f"'{name}' is a set role: the desk set's roles are model ids (`mnm sets "
            "default`), so no instance takes a role's name. Pick another (`--name "
            f"{name}-1`, or the model's own)."
        )
    if store.exists(name):
        try:
            unfinished = not store.load(name).downloaded
        except StoreError:  # unreadable: `mnm ls` says which file and why
            unfinished = False
        runs = (  # an add cut short left its recipe
            f"its download is not finished: `mnm start {name}` fetches the rest and runs it"
            if unfinished
            else f"`mnm start {name}` runs it"
        )
        raise InstanceError(
            f"An instance named '{name}' already exists: {runs}, or pick another name "
            "(`--name`), or `mnm rm` it first (its downloads stay)."
        )
    return name


def default_name(facts: ModelFacts) -> str:
    """The family's slug; a vision model's keeps a vision token, because the name is the
    model id clients see and Odysseus offers image upload by id. A vision repo
    added as chat or coding (`plan.retyped`, text-only) loses its `vl`/`vision` words, so
    no client offers it pictures it cannot see."""
    slug = _SLUG_RE.sub("-", facts.family_name.lower()).strip("-")
    if facts.modality is Modality.VISION_CHAT:
        if not VISION_ID_RE.search(slug):
            return vision_name(slug)
    elif facts.modality in CHAT_LIKE:
        slug = text_only_name(slug)
    return slug[:64].rstrip("-") or "model"
