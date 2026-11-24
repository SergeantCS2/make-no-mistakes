"""What `mnm doctor --deep` adds after the machine's checks, one function for the CLI, the
web's System → Doctor and the nightly (so a RAM change reaches the MoE plans' price
whichever runs it): the RAM's bandwidth measured and saved in `machine.json` for the planner,
and *Experts in RAM* against it; on a unified pool the bandwidth only.
Also the report as `--json` prints it: every card's uuid masked as `mnm machine show` masks
it (hardware identifiers are secrets)."""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.config import Settings
from makenomistakes.doctor_box import expert_need_check
from makenomistakes.doctor_types import Check, Status
from makenomistakes.doctor_unified import pool_card
from makenomistakes.hardware import ram_bandwidth
from makenomistakes.hardware.probe import HardwareReport
from makenomistakes.hardware.ram_bandwidth import RamReading
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.machine_facts import MachineFactsError, plausible
from makenomistakes.machine_facts import load as load_facts
from makenomistakes.private_ids import masked_cards
from makenomistakes.ram_check import ram_check
from makenomistakes.store import StoreError

EXPERTS = "Experts in RAM"


def deep_tail(
    settings: Settings,
    report: HardwareReport,
    measured: Callable[[], float | RamReading] | None = None,
) -> list[Check]:
    """The RAM's bandwidth (saved), then each saved MoE's need against it; `measured` the
    probe (`ram_bandwidth.measure_all`, read at call time)."""
    pool = pool_card(report) is not None  # one pool: nothing is kept in RAM beside it
    probe = measured or ram_bandwidth.measure_all
    checks = [ram_check(settings.expanded_state_dir(), probe, pool=pool)]
    if not pool:
        checks.append(experts_check(settings))
    return checks


def experts_check(settings: Settings) -> Check:
    """The bandwidth just measured (`ram_check` saved it) against each saved MoE's need."""
    try:
        facts = load_facts(settings.expanded_state_dir())
    except MachineFactsError:
        facts = None
    gbps = facts.ram_gbps if facts is not None and plausible(facts.ram_gbps) else None
    try:
        return expert_need_check(InstanceService(settings), gbps)
    except (InstanceError, StoreError) as exc:
        return Check(name=EXPERTS, status=Status.WARN, detail=f"not read: {exc}")


def json_report(report: HardwareReport) -> dict[str, object]:
    """The probe's report for `mnm doctor --json`, every card's uuid and AMD unique id masked
    (the smoke test's output holds none: `probe.SmokeResult`)."""
    return masked_cards(report.model_dump())
