"""The desk's card for a coding agent. The desk's roles are model ids on the router: the
agent gets one card, for the role it uses (the coder, else the chat role), and that card
names the role (`…/upstream/coder`, `ANTHROPIC_MODEL=coder`), which the router resolves to
whichever member holds it at each request; a card naming the instance
(`ANTHROPIC_BASE_URL=…/upstream/coder8b`) would break the client at a member swap. A plain
set keeps its per-member cards (`WorkflowService.cards`)."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.consumers import Consumer, HarnessUrl
from makenomistakes.harness import consumer_urls
from makenomistakes.harness_agents import AGENT_MODALITIES
from makenomistakes.recipes.model import Recipe

AGENT_ROLES = ("coder", "chat")  # the role an agent uses, in order of preference


def desk_agent_cards(
    members: dict[str, str], consumer: Consumer, find: Callable[[str], Recipe | None]
) -> list[tuple[str, Recipe, list[HarnessUrl]]]:
    """One (role, member, cards) for the agent's role; empty when the desk has neither role
    (or its member is gone: the members table says so)."""
    for role in AGENT_ROLES:
        name = members.get(role)
        recipe = find(name) if name is not None else None
        if recipe is None or recipe.modality not in AGENT_MODALITIES:
            continue
        view = recipe.model_copy(update={"name": role, "served_name": role})
        cards = consumer_urls(view, consumer)
        if cards:
            first = cards[0]
            said = (
                f"The desk's {role} (now {recipe.name}): the card names the role, so it "
                f"follows `mnm sets edit <desk> {role}=<another>` without a change. "
            )
            cards = [first.model_copy(update={"note": said + first.note}), *cards[1:]]
        return [(role, recipe, cards)]
    return []
