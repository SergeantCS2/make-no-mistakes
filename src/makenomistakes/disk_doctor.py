"""The doctor's Model storage line with the disk policy in it: the doctor
runs the policy (`disk_policy.check`, as every start does) and its line says the outcome. From
the first warning the line WARNs with what will happen and how to turn it off; an eviction
or a refusal is the remedy's text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.disk_policy import Outcome, check
from makenomistakes.disk_settings import limit_words
from makenomistakes.disk_types import Level
from makenomistakes.doctor_types import Check, Status

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

NAME = "Model storage"  # `doctor._disk_check`'s line


def with_policy(checks: list[Check], service: InstanceService) -> list[Check]:
    """`checks` with the Model storage line carrying the policy's outcome."""
    if not any(c.name == NAME for c in checks):
        return checks
    outcome = check(service)
    return [policy_line(c, outcome) if c.name == NAME else c for c in checks]


def policy_line(base: Check, outcome: Outcome) -> Check:
    gauge = outcome.gauge
    said = [*outcome.lines, *([outcome.refused] if outcome.refused else [])]
    if base.status is Status.FAIL:
        return base  # no folder to measure
    if gauge is None:
        tail = (
            "the disk policy is off" if not outcome.keep.on else "the disk policy measured nothing"
        )
        return _joined(base, tail, said)
    if gauge.level is Level.OK:
        tail = f"the disk policy evicts at {limit_words(gauge)} (now {gauge.percent:.0f}%)"
        return _joined(base, tail, said)
    remedy = " ".join([*said, *([base.remedy] if base.remedy else [])]) or None
    return Check(
        name=base.name,
        status=Status.WARN,
        detail=f"{base.detail}; {outcome.warning}",
        remedy=remedy,
    )


def _joined(base: Check, tail: str, said: list[str]) -> Check:
    remedy = " ".join([*([base.remedy] if base.remedy else []), *said]) or None
    status = Status.WARN if said and base.status is Status.OK else base.status
    return base.model_copy(
        update={"detail": f"{base.detail}; {tail}", "remedy": remedy, "status": status}
    )
