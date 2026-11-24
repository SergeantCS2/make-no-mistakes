"""How much faster each picture family runs than the one formula's 45% of the card's peak
FLOPs says (`fit.image_memory.seconds_per_image`, `fit.video.seconds_per_video`), per speed
model (`SPEED_MODEL`; a reading carries the one its estimate came from, and
`fit.sdcpp.speed_refit` rescales an older one). A leaf, so the device table (`fit.devices`)
can carry the 4070's factors as its row's `picture_speeds`.

Model 1 was the formula alone. Model 2 is fitted to readings on an RTX 4070: each factor
is the formula's seconds over a chosen target, the target set between the
warm and cold readings (a first picture after a start also loads the weights, so it reads
slow). Re-measure before touching them (`mnm calibration export`); a family missing here keeps
the formula (1.0). The readings, s/image or s/clip, and the target:
- Z-Image (Q4_K_M, 1024x1024, 8 steps): 12.8 warm, 15.5 to 17.0 cold; formula 51.54; 14.0
- SDXL (F16 checkpoints, 1024x1024, 20 steps): pony 16.7, civ 21.6, both cold; formula
  36.26; 18.5 (between the two checkpoints)
- FLUX.1 Kontext (Q4_K_M, 1024x1024, 20 steps): 20.9 cold; formula 178.01; 19.0 (a warm
  picture is quicker than the cold one)
- Wan 2.2 TI2V 5B (Q4_K_M, 832x480, 33 frames, 20 steps): 132.9 cold; formula 153.9; 125.0
"""

from __future__ import annotations

from makenomistakes.hub.kit_types import ImageFamily

FAMILY_SPEEDS: dict[int, dict[ImageFamily, float]] = {
    1: {},
    2: {
        ImageFamily.Z_IMAGE: 51.54 / 14.0,
        ImageFamily.SDXL: 36.26 / 18.5,
        ImageFamily.FLUX1: 178.01 / 19.0,
        ImageFamily.WAN22_TI2V: 153.9 / 125.0,
    },
}
SPEED_MODEL = 2  # the speed model estimates are made with now; a key of `FAMILY_SPEEDS`
