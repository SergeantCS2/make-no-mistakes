"""The router's settings: offered on every instance whatever its engine, saved on the
recipe itself rather than in its engine's args."""

from __future__ import annotations

from makenomistakes.controls_catalog import Control, Kind, check_controls
from makenomistakes.hub.facts import Modality
from makenomistakes.recipes.model import IDLE_SECONDS, Recipe

EVERY_MODALITY = frozenset(Modality)

ROUTER_CONTROLS: tuple[Control, ...] = (
    Control(
        key="idle",
        field="idle",
        label="Unload when idle (seconds)",
        help=(
            "The router unloads the model this long after its last reply finished, with "
            "nothing in flight, and loads it again at the next request. 0 never unloads it."
        ),
        kind=Kind.INT,
        recommended=f"{IDLE_SECONDS} (ten minutes); 0 with keep=always for a model that must "
        "answer at once",
        modalities=EVERY_MODALITY,
        on_recipe=True,
        live=True,
    ),
    Control(
        key="keep",
        field="keep",
        label="Keep loaded",
        help=(
            "always pins the model: the router never unloads it, for idleness or to make "
            "room, and a load that needs its room is refused with the reason. auto lets the "
            "router unload it."
        ),
        kind=Kind.CHOICE,
        choices=("auto", "always"),
        recommended="auto; always for the one model every request should find loaded",
        modalities=EVERY_MODALITY,
        on_recipe=True,
        live=True,
    ),
)

check_controls([(c, Recipe) for c in ROUTER_CONTROLS])
