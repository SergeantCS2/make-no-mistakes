"""This box's own speed for each music model: audio.cpp answers every task with
`timing.rtf`, the seconds of work a second of audio took, which the router's meter keeps as a
reading (`request_meter`, `SpeedUnit.AUDIO`, the model's file and the task's route named on
it). The rate every plan, every client's wait and `mnm calibration` take in place of the
estimate (`fit.music.music_time`) is the median of three or more such readings of one group:
one model file, one route, one settings key, one engine build. A first task after a start is
timed apart and never counted (its load ran 54 s of work a second of audio against about 5.5
warm), and another route (a repaint, a cover) or other settings (`mnm set threads=4`) is
another speed, so none is mixed into another's median.

An instance's own rate is its settings' group on the newest build it ran (`instance_rate`,
as `calibration.instance_figure` reads its VRAM); a plan for a model not added yet, or added
again, takes the newest group that has enough readings (`music_rates`). A separator's answer
carries no `rtf` (its time alone, not the audio's length), so it is not timed here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from makenomistakes.backend_now import box_now
from makenomistakes.calibration import every_reading
from makenomistakes.config import settings_now
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.engine_args_music import OPTIONS_KIND, AceStepOptions
from makenomistakes.fit.device_row import DeviceRow
from makenomistakes.fit.devices import device_for
from makenomistakes.fit.music import rate_key, task_timeout
from makenomistakes.hub.music import MusicFamily
from makenomistakes.measurements import MeasurementStore, Reading, SpeedUnit, settings_key
from makenomistakes.music_api import SINGING_ROUTES, asked, is_task
from makenomistakes.readings_filter import every_here
from makenomistakes.recipes.audiocpp import config_backend
from makenomistakes.recipes.cards import recipe_cards
from makenomistakes.recipes.model import Recipe

MIN_READINGS = 3  # as a picture family's correction: never one reading
RECENT = 12  # the newest a rate rests on


@dataclass(frozen=True)
class MusicRate:
    family: str
    file: str  # the GGUF's file name
    rate: float  # seconds of work a second of audio, this box's median
    readings: int
    settings: str  # the settings key its readings share
    build: str | None  # the engine build they ran on
    backend: str | None = None  # the image's backend they ran on (`Reading.backend`)

    @property
    def key(self) -> str:
        return rate_key(self.family, self.file)


def model_file(recipe: Recipe) -> str | None:
    """What a music reading names: the instance's GGUF (or checkpoint) file name; None for
    any other instance, text-to-speech included."""
    if not isinstance(recipe.args, AudioCppArgs) or not recipe.args.music:
        return None
    return PurePosixPath(recipe.files[0]).name if recipe.files else None


def configured_route(recipe: Recipe) -> str | None:
    """The route a task that names none runs: the instance's own (`default_request_options`:
    ACE-Step's `options.route`); None for a family with no route of its own."""
    args = recipe.args
    if isinstance(args, AudioCppArgs) and isinstance(args.options, AceStepOptions):
        return args.options.route
    return None  # a converter's request always names its singing route


def planned_route(family: str) -> str | None:
    """The route a plan's time is for: a song model's song (`text2music`), a converter's
    singing route; None for a family whose requests name none."""
    if OPTIONS_KIND.get(family) == "ace_step":
        return AceStepOptions().route
    return SINGING_ROUTES.get(family)


Group = tuple[str, str | None, str | None]  # (settings key, build, backend)


def _groups(
    readings: list[Reading], backend: str | None = None
) -> dict[tuple[str, str], dict[Group, list[float]]]:
    """Each model file's warm readings on its planned route, by settings, build and backend
    (only `backend`'s when given: a card's plan never takes the processor's rate), each
    group in time order and the groups in the order their newest reading came."""
    found: dict[tuple[str, str], dict[Group, list[float]]] = {}
    for reading in sorted(readings, key=lambda r: r.at):
        args = reading.args
        if (
            reading.speed_unit is not SpeedUnit.AUDIO
            or not reading.speed
            or reading.cold
            or not reading.model_file
            or not isinstance(args, AudioCppArgs)
            or args.family is None
            or reading.task_route != planned_route(args.family)
            or (backend is not None and reading.backend != backend)
        ):
            continue
        groups = found.setdefault((args.family, reading.model_file), {})
        group = (reading.settings, reading.build, reading.backend)
        speeds = groups.pop(group, [])  # moved to the end: the newest group last
        speeds.append(reading.speed)
        groups[group] = speeds
    return found


def _rate(family: str, file: str, group: Group, speeds: list[float]) -> MusicRate | None:
    newest = speeds[-RECENT:]
    if len(newest) < MIN_READINGS:
        return None
    rate = round(statistics.median(newest), 3)
    return MusicRate(family, file, rate, len(newest), group[0], group[1], group[2])


def music_rates(readings: list[Reading], backend: str | None = None) -> list[MusicRate]:
    """One per family and file: the newest group of its readings with enough of them, only
    those whose `Reading.backend` is `backend` when given: what readings name
    (`measurements.readings_backend` for a box's plans), never the box's own backend (what a
    plan takes, `plan.calibrated`, and `mnm calibration` shows)."""
    rates = []
    for (family, file), groups in sorted(_groups(readings, backend).items()):
        for group, speeds in reversed(groups.items()):
            found = _rate(family, file, group, speeds)
            if found is not None:
                rates.append(found)
                break
    return rates


def recipe_rate(readings: list[Reading], recipe: Recipe) -> MusicRate | None:
    """The instance's own rate: its settings' group on the newest build it ran; None until it
    has enough readings (another key's readings say nothing of these settings)."""
    file = model_file(recipe)
    if file is None:
        return None
    assert isinstance(recipe.args, AudioCppArgs)
    family = recipe.args.family
    key = settings_key(recipe)
    groups = _groups(readings).get((family, file), {})
    for group, speeds in reversed(groups.items()):
        if group[0] == key:  # the newest build these settings ran on
            return _rate(family, file, group, speeds)
    return None


def _readings_here(state_dir: Path) -> list[Reading]:
    return every_reading(every_here(MeasurementStore(state_dir)))


def rates_here(state_dir: Path, backend: str | None = None) -> dict[str, MusicRate]:
    """This box's learned rates by `rate_key`, of readings naming `backend` when given."""
    return {rate.key: rate for rate in music_rates(_readings_here(state_dir), backend)}


def instance_rate(state_dir: Path, recipe: Recipe) -> MusicRate | None:
    """`recipe_rate` over this box's readings."""
    return recipe_rate(_readings_here(state_dir), recipe)


def music_timeout(
    recipe: Recipe, seconds: float, state_dir: Path | None = None, *, steps: int | None = None
) -> float:
    """How long a client of the instance waits for `seconds` of audio, and the router for its
    answer (`task_wait`): the estimate for the card it runs on (or its cores on the
    processor) at the `steps` asked (else its own), longer by this box's own rate for its
    model and settings once it has one (`task_timeout`); the readings of `state_dir`, else
    the state folder the settings name now."""
    args = recipe.args
    assert isinstance(args, AudioCppArgs)
    learned = instance_rate(state_dir or settings_now().expanded_state_dir(), recipe)
    own = args.options.steps if isinstance(args.options, AceStepOptions) else None
    cpu = config_backend(recipe) == "cpu"
    return task_timeout(
        MusicFamily(args.family),
        seconds,
        cpu,
        recipe.files[0],
        learned.rate if learned is not None else None,
        cores=args.threads,  # a CPU plan's physical cores (`audiocpp_recipe(cpu=…)`)
        steps=steps or own,
        device=None if cpu else card_row(recipe),
    )


def own_length(recipe: Recipe) -> float | None:
    """The seconds a song request that names none runs (ACE-Step's `options.duration`); None
    for an instance whose task has no length of its own (it works on the audio it is sent)."""
    args = recipe.args
    if isinstance(args, AudioCppArgs) and isinstance(args.options, AceStepOptions):
        return args.options.duration
    return None


def task_wait(recipe: Recipe, path: str, body: bytes) -> float | None:
    """How long the router waits for the engine's answer to one music task: what its client
    waits (`music_timeout`) for the audio the request asks for or sends, else the instance's
    own length, at the steps it names. audio.cpp says nothing until a task ends, so a socket
    timed for a chat would cut a full-tier task on the processor (about 87 minutes for a
    3-minute take on 4 cores) while its client still waits. None for any other
    request, or a task whose audio cannot be read from the request (it sends a path)."""
    if model_file(recipe) is None or not is_task(path):
        return None
    found = asked(body)
    seconds = found.seconds or own_length(recipe)
    return music_timeout(recipe, seconds, steps=found.steps) if seconds else None


def card_row(recipe: Recipe) -> DeviceRow | None:
    """The device row of the card the instance runs on (`recipe_cards`' first), from the box
    `machine.json` last recorded; None when no box or card is recorded (the 4070's figure
    then stands)."""
    box = box_now()
    if box is None or not box.cards:
        return None
    first = (recipe_cards(recipe, listed=[c.index for c in box.cards]) or (None,))[0]
    card = next((c for c in box.cards if c.index == first), box.cards[0])
    return device_for(card.name, compute=card.compute_cap, gfx=card.gfx_target)
