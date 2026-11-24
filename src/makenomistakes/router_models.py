"""What the router lists (`GET /v1/models`, `/v1/models/<id>`): every loaded or parked instance
by name, and each desk role as an id of its own (`meta.alias_of`). The entry is
typed here (`Entry`) and dumped once at the edge; the one rule for which roles are listed,
`roles_listed`, serves Ollama's `/api/tags` too; `router` re-exports `entry`, `listed` and
`chat_like`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from makenomistakes.aliases import Aliases
from makenomistakes.hub.facts import CHAT_LIKE, MUSIC_MODALITIES
from makenomistakes.instance_types import InstanceStatus
from makenomistakes.model_meta import MetaCache, ModelMeta, serving_context


class Meta(BaseModel):
    engine_id: str
    type: str
    loaded: bool
    n_ctx: int | None = None
    n_ctx_train: int | None = None
    alias_of: str | None = None  # a desk role's entry: the instance that answers it


class Entry(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "makenomistakes"
    created: int
    context_length: int | None = None
    meta: Meta

    def as_role(self, role: str) -> Entry:
        """The same model under a desk role's id, `meta.alias_of` naming the instance."""
        meta = self.meta.model_copy(update={"alias_of": self.id})
        return self.model_copy(update={"id": role, "meta": meta})

    def dumped(self) -> dict[str, Any]:
        """The JSON a client reads: fields that do not apply are left out, not null."""
        return self.model_dump(exclude_none=True)


def chat_like(status: InstanceStatus) -> bool:
    return status.recipe.modality in CHAT_LIKE


def listed(status: InstanceStatus) -> bool:
    """Loaded, or parked: what a client may ask for. Not a stopped instance, nor one whose
    files are not on disk (a request never downloads)."""
    return not status.error and (status.running or status.parked)


def shown(status: InstanceStatus) -> bool:
    """Listed to clients: a listed instance that is not one of the studio's:
    no OpenAI client can use audio.cpp's task route, and Odysseus would offer a song model in
    its chat picker. The router still loads and serves it at `/upstream/<name>/v1/tasks/run`."""
    return listed(status) and status.recipe.modality not in MUSIC_MODALITIES


def entry_of(status: InstanceStatus, meta: ModelMeta | None = None) -> Entry:
    """One entry: the instance name as the id, the engine's own id beside it, and for a model
    with a context the one a request gets (`context_length`, `meta.n_ctx`) and the one it was
    trained for (`meta.n_ctx_train`, when its files say). Odysseus reads `context_length`,
    then `meta.n_ctx` (`src/model_context.py:324-347` in 0.20.19), and budgets every model
    without either at 128,000 tokens."""
    recipe = status.recipe
    serving = serving_context(recipe)
    trained = meta.trained_ctx if meta is not None and serving is not None else None
    return Entry(
        id=recipe.name,
        created=int(recipe.created_at.timestamp()),
        context_length=serving,
        meta=Meta(
            engine_id=recipe.served_name,
            type=recipe.modality.value,
            loaded=status.running,
            n_ctx=serving,
            n_ctx_train=trained or None,
        ),
    )


def entry(status: InstanceStatus, meta: ModelMeta | None = None) -> dict[str, object]:
    """`entry_of`, as the JSON a client reads."""
    return entry_of(status, meta).dumped()


def roles_listed[T](
    aliases: Aliases, listed_by_name: dict[str, T], as_role: Callable[[str, T], T]
) -> list[T]:
    """Each desk role whose member is listed, as `as_role` renames the member's entry: an
    alias of a stopped or missing member is left out, as its member is."""
    return [
        as_role(r, listed_by_name[n]) for r, n in aliases.members.items() if n in listed_by_name
    ]


def listing(
    statuses: list[InstanceStatus], meta: MetaCache, aliases: Aliases
) -> list[dict[str, object]]:
    """The entries of what may be asked for: each instance, then each listed desk role."""
    entries = {s.recipe.name: entry_of(s, meta.get(s.recipe)) for s in statuses if shown(s)}
    roles = roles_listed(aliases, entries, lambda role, found: found.as_role(role))
    return [e.dumped() for e in [*entries.values(), *roles]]
