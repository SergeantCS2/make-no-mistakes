"""What a picture's or clip's speed reading carries besides its seconds: the settings
that made it, the family whose speed correction it feeds, and the fit model's own seconds for
those settings then (`fit.sdcpp.recipe_seconds`), so `calibration.speed_correction` can say
how far the estimate was off on this card.

The size is the instance's own: the router drops a client's `size` unless `client-size=on`,
and then nothing here knows the size and the reading carries no estimate."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from makenomistakes.engine_args import SdCppArgs
from makenomistakes.fit.devices import device_for
from makenomistakes.fit.sdcpp import recipe_seconds
from makenomistakes.fit.video import video_of
from makenomistakes.hub.kit_types import ImageFamily
from makenomistakes.model_files import read_model_files
from makenomistakes.recipes import Recipe


class PictureSettings(BaseModel):
    """The fields a picture's or clip's reading carries beside its seconds."""

    width: int
    height: int
    steps: int
    frames: int | None
    family: ImageFamily
    estimate_seconds: float | None = None
    speed_model: int | None = None  # `fit.sdcpp.SPEED_MODEL` when the estimate is set


def picture_settings(recipe: Recipe) -> PictureSettings | None:
    """The settings a picture or clip of this instance is made at (None for another engine)."""
    args = recipe.args
    if not isinstance(args, SdCppArgs):
        return None
    video = video_of(recipe.image_family)
    return PictureSettings(
        width=args.width,
        height=args.height,
        steps=args.steps + (args.high_noise_steps or 0),
        frames=(args.frames or video.frames) if video is not None else None,
        family=recipe.image_family,
    )


def estimated_seconds(recipe: Recipe, models_dir: Path, gpu_name: str | None) -> float | None:
    """The fit model's seconds for this instance's settings, from its files on disk and the
    card's compute; None when either cannot be read or the settings make no estimate."""
    args = recipe.args
    if not isinstance(args, SdCppArgs) or gpu_name is None:
        return None
    files = read_model_files(recipe, models_dir)
    if files is None:
        return None
    device = device_for(gpu_name)
    found = recipe_seconds(
        files.weights_bytes,
        recipe.quant,
        args,
        recipe.image_family,
        recipe.high_noise is not None,
        device.tflops * 1e12,
        device.picture_speeds,
    )
    return round(found, 2) if found is not None else None
