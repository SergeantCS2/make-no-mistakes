"""Who made a song's parts: the song run's own credits, each tied to the item that made what it
is for (the lyrics' chat, a stems package's separator, the dereverb, the voice's converter, the
MIDI's MuScriptor), so a credit leaves with what it made. The one rule the disclosure and
`provenance.json` rest on (credits that only grew would keep naming a voice set to none, the
first converter after another voice, the first separator after new stems).

`give` credits an item (replacing what the same item credited before); `drop`, called by the
runner's `_undo` for every item a reset makes again, takes those items' credits away. A
candidate's or an edit's models are on its own child run and go with it.
"""

from __future__ import annotations

from makenomistakes.pipeline import song
from makenomistakes.run_parts import ModelCredit
from makenomistakes.runs import RunRecord, StepRecord

# Records saved before credits named their item kept these two by a setting:
# the instance whose stage-less credit goes when that stage is made again.
_NAMED_BY = {song.VOCALS: "vocals_dereverb", song.MIDI: "midi_instance"}


def _of(credit: ModelCredit, step: StepRecord) -> bool:
    return credit.stage == step.stage and credit.item == step.index


def give(run: RunRecord, step: StepRecord, credits: list[ModelCredit]) -> None:
    """The models that made this item, now the item's credits on the song."""
    tagged = [c.model_copy(update={"stage": step.stage, "item": step.index}) for c in credits]
    named = {c.instance for c in credits if c.instance}
    kept = [
        m for m in run.models if not _of(m, step) and not (m.stage is None and m.instance in named)
    ]  # a stage-less credit of the same instance is this one saved before
    run.models = kept + tagged


def drop(run: RunRecord, items: list[StepRecord]) -> None:
    """The credits of the items made again: they leave with what the items made."""
    run.models = [m for m in run.models if not any(_of(m, item) for item in items)]
    for stage in {item.stage for item in items} & set(_NAMED_BY):
        played = run.settings.pop(_NAMED_BY[stage], "")
        if played:
            run.models = [m for m in run.models if m.stage is not None or m.instance != played]


def same_model(credit: ModelCredit) -> ModelCredit:
    """The credit without the item it is for: one model named once in an export's list,
    whichever items it played (the separator of the stems and of the vocals' debleed)."""
    return credit.model_copy(update={"stage": None, "item": 0})
