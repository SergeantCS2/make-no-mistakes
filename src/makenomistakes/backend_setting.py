"""The backend per instance: `mnm set <name> backend=rocm|vulkan` on a box whose AMD
card runs both of llama.cpp's builds. The box's own backend (`MNM_BACKEND`, `backend.decide`)
is what new plans get: Vulkan by default (the floor), ROCm when the user chose it; this
setting moves one llama.cpp instance to the other image without moving the box, and the
tuner's backend stage tries the other one when both images are pulled (`tune.search`).

The setting is the recipe's image (`Recipe.image`, the backend's tag for the box's arch): its
device arguments, readings and tune key follow the image (`backend_now.recipe_backend`). It
is offered only where it can choose (an AMD card whose gfx target `server-rocm` carries):
everywhere else it is not listed. vLLM on ROCm is planned (`engine_images`)
and never offered here; sd.cpp, whisper.cpp and audio.cpp have Vulkan images only."""

from __future__ import annotations

from makenomistakes.backend import Backend, BackendError, label, refusal, used_cards
from makenomistakes.backend_now import box_now, recipe_backend_or_none
from makenomistakes.controls_catalog import ALL_MODALITIES, Control, Kind
from makenomistakes.engine_images import IMAGES
from makenomistakes.hardware.amd_cards import rocm_runs
from makenomistakes.hardware.fingerprint import MachineFingerprint, Vendor
from makenomistakes.hub.facts import Engine
from makenomistakes.recipes.model import Recipe

CHOICES = (Backend.VULKAN.value, Backend.ROCM.value)
BACKEND_CONTROL = Control(
    key="backend",
    field="image",
    label="Backend",
    help=(
        "Which llama.cpp build runs this instance on the AMD card: vulkan (the default: small "
        "and MoE models decode fast) or rocm (dense models over about 20B and long prompts; a "
        "7.7 GB image). Applies at the next start; `mnm tune` tries both when both are pulled."
    ),
    kind=Kind.CHOICE,
    choices=CHOICES,
    recommended="vulkan; rocm when `mnm tune` measured it faster here",
    modalities=ALL_MODALITIES,
    on_recipe=True,
)
NOT_HERE = (
    "backend= chooses between llama.cpp's ROCm and Vulkan builds for an AMD card that runs "
    "both; {why}"
)


class BackendSettingError(ValueError):
    """The backend cannot be set: why, worded for `mnm set` and the web form."""


def offered(box: MachineFingerprint | None = None) -> bool:
    """The box (as last recorded) has an AMD card that runs both builds."""
    found = box if box is not None else box_now()
    return found is not None and any(
        c.vendor is Vendor.AMD and rocm_runs(c.gfx_target) for c in found.cards
    )


def current(recipe: Recipe) -> str:
    """The instance's backend as the setting reads it (`vulkan`, `rocm`, `cuda12` …)."""
    found = recipe_backend_or_none(recipe)
    return found.value if found is not None else ""


def not_here(box: MachineFingerprint | None = None) -> str:
    """Why the setting is not offered on this box."""
    found = box if box is not None else box_now()
    if found is None:
        return NOT_HERE.format(why="this box has not been read yet (`mnm doctor`)")
    amd = [c for c in found.cards if c.vendor is Vendor.AMD]
    if not amd:
        return NOT_HERE.format(why="this box has no AMD card")
    names = ", ".join(f"{c.short} ({c.gfx_target or 'gfx not read'})" for c in amd)
    return NOT_HERE.format(why=f"llama.cpp's ROCm image is not built for {names}: Vulkan only")


def with_backend(recipe: Recipe, raw: str, gpu: str | None = None) -> Recipe:
    """The recipe on `raw`'s image; the same recipe when it is already there (the web form
    posts every field). A backend the instance or its cards cannot run is refused."""
    wanted_text = raw.strip().lower()
    if not wanted_text or wanted_text == current(recipe):
        return recipe
    if recipe.engine is not Engine.LLAMACPP:
        raise BackendSettingError(
            "backend= applies to llama.cpp instances; pictures, speech and voices run "
            "their Vulkan images on an AMD card"
        )
    if wanted_text not in CHOICES:
        raise BackendSettingError(f"backend must be one of {', '.join(CHOICES)}, not '{raw}'")
    wanted = Backend(wanted_text)
    box = box_now()
    if box is None or not offered(box):
        raise BackendSettingError(not_here(box))
    problem = refusal(used_cards(box, gpu), wanted)
    if problem is not None:
        raise BackendSettingError(f"backend={wanted.value} cannot run here: {problem}")
    tag = IMAGES[Engine.LLAMACPP].image_for(wanted, box.arch)
    if not isinstance(tag, str):
        raise BackendSettingError(tag.words)
    return recipe.model_copy(update={"image": tag})


def words(recipe: Recipe) -> str | None:
    """`ROCm` / `Vulkan` for an instance on an AMD backend (`mnm show`), else None."""
    found = recipe_backend_or_none(recipe)
    if found not in (Backend.ROCM, Backend.VULKAN):
        return None
    return label(found)


__all__ = ["BACKEND_CONTROL", "BackendError", "BackendSettingError", "offered", "with_backend"]
