"""The stages registry: every stage of a song as one data row (what it does, where it runs, its
two tiers, its form, its inputs and outputs by role, its licence and source), read from
`stages.json` beside this file at import. A better model is a changed row, never a code path:
code asks `resolve(stage, tier)`.

The user's picks (`<state>/studio/stages.json`, schema 1) name the instance that plays a
stage's tier (`mnm studio stage song fast=<instance>`); the registry's package stands until
one is picked. The picks override a row's model, never its shape.

The rows include `song` (ACE-Step 1.5 turbo Q8 to explore; the base build at Q8 for the keeper,
as the repo packages no `sft` build) and `stems` (HTDemucs 6 stems alone to explore;
BS-RoFormer's vocals, then HTDemucs for the other five, for the keeper); `score` (Audiobox
Aesthetics) and `structure` (Beat This! and a novelty curve), both `tool` rows the studio tools
image plays: a tool tier names the image's command, never a package, and there is no instance to
pick for it; `voice` (Seed-VC); and `edit` (the ACE-Step that plays every model-side edit: turbo
to explore, base for the keeper; the edits' own fields are `studio/edits.py`'s typed asks, so
the row lists none)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from makenomistakes.engine_args_music import MusicOptions
from makenomistakes.hub.modality import Modality
from makenomistakes.hub.music import Commercial, MusicFamily
from makenomistakes.locking import shared_lock

SCHEMA_VERSION = 1
REGISTRY = Path(__file__).with_name("stages.json")
StageId = Literal[
    "lyrics", "song", "stems", "score", "structure", "voice", "edit", "mix", "export", "sections",
    "tempo_key", "cleanup", "tuning", "midi", "master", "loudness",
]  # fmt: skip
Kind = Literal[
    "generate", "separate", "convert", "synthesise", "transcribe", "score", "edit", "tool",
    "export",
]  # fmt: skip
Host = Literal["audiocpp", "comfy", "tool", "trainer", "realtime"]  # realtime reserved
TierName = Literal["fast", "full"]
FieldKind = Literal[
    "text", "seconds", "range", "choice", "strength", "semitones", "percent", "upload", "stem"
]


class StudioError(ValueError):
    """A registry or picks problem, worded for the user."""


class Package(BaseModel):
    """One audio.cpp package of a tier: the repo, its folder, the quant; what it gives."""

    repo: str
    folder: str
    quant: str
    gives: list[str] = Field(default_factory=list)  # roles, for a stage of several packages

    @property
    def reference(self) -> str:
        """What `mnm add` takes."""
        return f"{self.repo}/{self.folder}:{self.quant}"


class TierRow(BaseModel):
    packages: list[Package] = Field(default_factory=list)  # run in order (the full stems: two)
    options: MusicOptions | None = Field(default=None, discriminator="kind")
    tool: str | None = None  # a `tool` row's command in the studio tools image

    @model_validator(mode="after")
    def _one_player(self) -> TierRow:
        if bool(self.packages) == bool(self.tool):
            raise ValueError("a tier names its packages or its tool, one of the two")
        return self


class FieldRow(BaseModel):
    name: str
    kind: FieldKind
    min: float | None = None
    max: float | None = None
    choices: list[str] = Field(default_factory=list)
    default: str | float | None = None


class Source(BaseModel):
    url: str = Field(pattern=r"^https://")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class StageRow(BaseModel):
    stage: StageId
    kind: Kind
    label: str
    host: Host
    tiers: dict[TierName, TierRow]
    fit_model: MusicFamily | None = None  # None for a tool row: the tools run on the CPU
    # The audio.cpp families a user may pick to play a tool row instead (MIDI:
    # MuScriptor and SheetSage2, non-commercial, so never the row's default)
    pickable: list[MusicFamily] = Field(default_factory=list)
    fields: list[FieldRow] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    licence: str
    commercial: Commercial
    source: Source


class Registry(BaseModel):
    schema_version: int
    stages: list[StageRow]


def _read() -> dict[str, StageRow]:
    registry = Registry.model_validate_json(REGISTRY.read_text(encoding="utf-8"))
    if registry.schema_version != SCHEMA_VERSION:
        raise AssertionError(f"stages.json is schema {registry.schema_version}")
    return {row.stage: row for row in registry.stages}


STAGES: dict[str, StageRow] = _read()


class Picks(BaseModel):
    """The user's choices: per stage and tier, the instance that plays it."""

    schema_version: int = SCHEMA_VERSION
    picks: dict[str, dict[TierName, str]] = Field(default_factory=dict)


WORKFLOW_PICK = "workflow:"  # a pick naming an imported ComfyUI workflow


def workflow_of(pick: str | None) -> str | None:
    """The ComfyUI workflow a pick names (`workflow:<name>`), else None: the one decode."""
    return pick.removeprefix(WORKFLOW_PICK) if pick and pick.startswith(WORKFLOW_PICK) else None


class Resolved(BaseModel):
    """What plays a stage's tier: the user's instance or ComfyUI workflow, or the
    registry's packages to add."""

    stage: str
    tier: TierName
    instance: str | None  # the user's pick as written: an instance, or `workflow:<name>`
    packages: list[Package]
    options: MusicOptions | None = None

    @property
    def workflow(self) -> str | None:
        return workflow_of(self.instance)


PICKS_LOCK = ".stages.lock"  # the picks' read-modify-write


def picks_path(state_dir: Path) -> Path:
    return state_dir / "studio" / "stages.json"


def read_picks(state_dir: Path) -> Picks:
    path = picks_path(state_dir)
    if not path.exists():
        return Picks()
    try:
        picks = Picks.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise StudioError(f"{path} cannot be read ({exc}): fix or delete it.") from None
    if picks.schema_version > SCHEMA_VERSION:
        raise StudioError(f"{path} is from a newer mnm: run `mnm update`.")
    return picks


def row(stage: str) -> StageRow:
    try:
        return STAGES[stage]
    except KeyError:
        raise StudioError(f"No stage '{stage}' (the stages: {', '.join(STAGES)}).") from None


def refuse_tool_pick(stage: str) -> None:
    """A tool row has no instance to pick (the studio tools image plays it), unless it names
    the families that may play it instead (`pickable`)."""
    found = row(stage)
    if found.host == "tool" and not found.pickable:
        raise StudioError(
            f"The {stage} stage is played by the studio tools image, not an instance: "
            "`mnm studio install` builds it; there is nothing to pick."
        )


# The instances a stage takes: what they do, and the family when one alone plays it (the
# edits are ACE-Step's routes; a voice of each engine sings on its own converter), so a chat
# model never "plays" the voice or the edit stage.
TAKES: dict[str, tuple[Modality, MusicFamily | None, str]] = {
    "song": (Modality.SONG, None, "a song"),
    "stems": (Modality.STEMS, None, "a stems"),
    "voice": (Modality.VOICE_CONVERT, None, "a voice conversion"),
    "edit": (Modality.SONG, MusicFamily.ACE_STEP, "an ACE-Step song"),
}


def pick_refusal(stage: str, name: str, family: str | None, modality: Modality) -> str | None:
    """Why an instance cannot play a stage, None when it can: the kind the stage takes
    (`TAKES`), and a tool row's families (`pickable`, MIDI's). The one rule `mnm studio
    stage`, any other pick route and the stage itself read."""
    takes = TAKES.get(row(stage).stage)
    if takes is not None and (
        modality is not takes[0] or (takes[1] is not None and family != takes[1].value)
    ):
        return (
            f"{name} is {modality.a_label} instance ({family or 'no audio.cpp model'}): the "
            f"{stage} stage takes {takes[2]} instance"
        )
    allowed = [f.value for f in row(stage).pickable]
    if not allowed or family in allowed:
        return None
    return (
        f"{name} is {family or 'no audio.cpp model'}: the {stage} stage takes the studio tools "
        f"or an instance of {' or '.join(allowed)}"
    )


def resolve(stage: str, tier: TierName, state_dir: Path) -> Resolved:
    """The instance the user picked for this stage's tier, else the registry's packages."""
    found = row(stage)
    chosen = read_picks(state_dir).picks.get(stage, {}).get(tier)
    tier_row = found.tiers[tier]
    return Resolved(
        stage=stage,
        tier=tier,
        instance=chosen,
        packages=tier_row.packages,
        options=tier_row.options,
    )


def pick(state_dir: Path, stage: str, tier: TierName, instance: str | None) -> Picks:
    """Save the user's pick for a stage's tier (`None` goes back to the registry's)."""
    row(stage)  # an unknown stage is refused before anything is written
    if instance is not None:
        refuse_tool_pick(stage)
    path = picks_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with shared_lock(path.parent, PICKS_LOCK)():  # a CLI and a web pick at once keep both
        picks = read_picks(state_dir)
        chosen = dict(picks.picks.get(stage, {}))
        if instance is None:
            chosen.pop(tier, None)
        else:
            chosen[tier] = instance
        picks.picks[stage] = chosen
        tmp = path.with_suffix(".tmp")
        tmp.write_text(picks.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    return picks


def tier_words(stage: StageRow, tier: TierName) -> str:
    """`ACE-Step1.5-GGUF/turbo Q8_0` (the packages of a tier, in order), or a tool row's
    `the studio tools: score`."""
    row = stage.tiers[tier]
    if row.tool is not None:
        return f"the studio tools: {row.tool}"
    return " then ".join(f"{p.folder} {p.quant}" for p in row.packages)
