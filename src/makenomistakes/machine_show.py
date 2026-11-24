"""`mnm machine show` and `history` in words: the box in full, what changed since the
box mnm saw before it, which instances were planned on another box; every box seen, one line
each. Pure: the CLI reads and records the box, these only word what was recorded."""

from __future__ import annotations

from makenomistakes.hardware.fingerprint import GIB, CardId, MachineFingerprint, differences
from makenomistakes.local_time import local
from makenomistakes.machine import FORGOTTEN, drift, planned_id
from makenomistakes.machine_facts import MachineFacts, MachineFile, SeenMachine, plausible
from makenomistakes.private_ids import masked
from makenomistakes.recipes import Recipe

WHEN = "%Y-%m-%d %H:%M"


def show_lines(
    file: MachineFile,
    recipes: list[Recipe],
    reveal: bool = False,
    readings: dict[str | None, int] | None = None,
) -> list[str]:
    found = drift(file, recipes)
    here = file.here
    if found is None or here is None:
        return ["No box recorded yet: `mnm doctor` or `mnm ls` reads this one."]
    box = here.fingerprint
    lines = [
        f"This box: {box.words()}",
        f"  id {box.id} (cards {box.cards_id}); first seen {local(here.first_seen):{WHEN}}, "
        f"last seen {local(here.last_seen):{WHEN}}",
        *detail_lines(box, here.measured, reveal),
    ]
    lines += _before(file, here)
    if found.elsewhere:
        lines.append(
            f"Planned on another box: {found.count}; since then {', '.join(found.changes)}"
        )
        for planned, names in found.elsewhere.items():
            entry = file.entry(planned)
            on = entry.fingerprint.words() if entry is not None else FORGOTTEN
            lines.append(f"  planned on {planned} ({on}): {', '.join(sorted(names))}")
        lines.append(
            "  `mnm machine adopt` re-fits them for this box (`--dry-run` shows the plans "
            "first; every download is asked)."
        )
    elif recipes:
        lines.append(f"Every saved plan ({len(recipes)}) was made on this box.")
    older = [r.name for r in recipes if r.planned_on is None]
    if older and file.unstamped_on is not None:
        first = file.entry(file.unstamped_on)
        lines.append(
            f"Planned before mnm kept fingerprints: {len(older)}, counted as planned on "
            f"{file.unstamped_on}{' (this box)' if file.unstamped_on == box.id else ''}, the "
            f"first box mnm saw{'' if first is not None else f' ({FORGOTTEN})'}."
        )
    if file.adopted is not None:
        mark = " (this box)" if file.adopted.machine_id == box.id else ""
        lines.append(
            f"Last adopted {local(file.adopted.at):{WHEN}} for {file.adopted.machine_id}{mark}: "
            f"{file.adopted.summary}"
        )
    lines.append(f"Boxes seen: {len(file.seen)} (`mnm machine history`).")
    if readings is not None:  # Each box plans from its own readings only
        lines.append("Readings per box (each box's own are the only ones it uses):")
        for seen in reversed(file.seen):
            mark = " (this box)" if seen is here else ""
            lines.append(
                f"  {seen.fingerprint.id}{mark}: {_readings(readings.get(seen.fingerprint.id, 0))}"
            )
        known = {s.fingerprint.id for s in file.seen}
        if left := sum(n for box, n in readings.items() if box not in known):
            lines.append(f"  boxes mnm no longer keeps, or unknown: {_readings(left)}")
    return lines


def _readings(count: int) -> str:
    return f"{count} reading{'s' * (count != 1)}"


def detail_lines(
    box: MachineFingerprint, measured: MachineFacts, reveal: bool = False
) -> list[str]:
    cpu, ram = box.cpu, box.ram
    lines = ["  Cards: none" if not box.cards else "  Cards:"]
    lines += [f"    {_card(card, reveal)}" for card in box.cards]
    lines.append(f"  Driver: {box.driver_version or 'unknown'}")
    wsl = box.platform.kind == "wsl" and cpu.threads  # the VM's topology is not the CPU's
    counts = [
        f"{cpu.threads} vCPUs by .wslconfig" if wsl else None,
        f"{cpu.cores} cores" if cpu.cores and not wsl else None,
        f"{cpu.threads} threads" if cpu.threads and not wsl else None,
        f"{cpu.sockets} socket{'s' if cpu.sockets != 1 else ''}" if cpu.sockets else None,
        f"{cpu.numa_nodes} NUMA node{'s' if cpu.numa_nodes != 1 else ''}"
        if cpu.numa_nodes
        else None,
        *(f"{n} {kind} cores" for kind, n in cpu.core_kinds.items()),
    ]
    vector = f"; {', '.join(cpu.vector)}" if cpu.vector else "; no AVX2, AVX-512, AMX or SVE"
    vendor = f" ({cpu.vendor})" if cpu.vendor else ""
    lines.append(
        f"  CPU: {cpu.model or 'unknown'}{vendor}: "
        f"{', '.join(c for c in counts if c) or 'cores unknown'}{vector}"
    )
    slots = (
        f"{ram.channels} channel{'s' if ram.channels != 1 else ''}"
        + (f" of {ram.kind}" if ram.kind else "")
        + (f" at {ram.speed_mts} MT/s" if ram.speed_mts else "")
        if ram.channels
        else "channels unknown (`mnm doctor` reads the slots where Linux lets it)"
    )
    lines.append(f"  RAM: {ram.total_bytes / GIB:.1f} GiB ({ram.sold_gb} GB); {slots}")
    lines.append(f"  Board: {_board(box)}")
    platform = box.platform
    where = "WSL" if platform.kind == "wsl" else "native"
    kernel = f", kernel {platform.kernel}" if platform.kernel else ""
    lines.append(f"  Platform: {where}, {platform.distro or 'unknown'}{kernel}, {box.arch}")
    lines.append(f"  Measured here: {_measured(measured)}")
    return lines


def _card(card: CardId, reveal: bool) -> str:
    size = (
        "unified memory, read as RAM"
        if card.memory_bytes is None
        else f"{card.memory_bytes / GIB:.1f} GiB"
    )
    parts = [f"GPU {card.index}: {card.name}", size]
    if card.compute_cap:
        parts.append(f"compute {card.compute_cap}")
    parts.append(f"{card.link} at most" if card.link else "link unknown")
    if card.bus_id:
        parts.append(f"bus {card.bus_id}")
    parts.append(card.stable_id if reveal else masked(card.stable_id))
    return ", ".join(parts)


def _board(box: MachineFingerprint) -> str:
    if box.board.virtual:
        return "WSL's virtual board (the PC's own is not visible from WSL)"
    words = box.board.words()
    if words is None:
        return "unknown (DMI unreadable)"
    product = box.board.product
    return f"{words} (product {product})" if product and product not in words else words


def _measured(facts: MachineFacts) -> str:
    said = []
    if plausible(facts.ram_gbps) and facts.ram_measured_at is not None:
        said.append(f"RAM {facts.ram_gbps:.1f} GB/s ({local(facts.ram_measured_at):%Y-%m-%d})")
    if facts.expert_gbps and facts.expert_learned_at is not None:
        said.append(
            f"experts in RAM {facts.expert_gbps:.1f} GB/s from {facts.expert_readings} readings "
            f"({local(facts.expert_learned_at):%Y-%m-%d})"
        )
    return "; ".join(said) or "nothing yet (`mnm doctor --deep`, `mnm calibration`)"


def _before(file: MachineFile, here: SeenMachine) -> list[str]:
    """What changed since the box seen before this one (the entry before it in `seen`)."""
    others = [s for s in file.seen if s is not here]
    if not others:
        return ["No other box seen."]
    before = others[-1]
    found = differences(before.fingerprint, here.fingerprint)
    what = ", ".join(found) or "nothing mnm names (the ids differ)"
    return [
        f"Changed since the box before ({before.fingerprint.id}, last seen "
        f"{local(before.last_seen):{WHEN}}): {what}"
    ]


def history_lines(
    file: MachineFile | None,
    recipes: list[Recipe],
    readings: dict[str | None, int] | None = None,
) -> list[str]:
    if file is None or not file.seen:
        return ["No box recorded yet: `mnm doctor` or `mnm ls` reads this one."]
    current = file.current.fingerprint.id if file.current else None
    planned: dict[str, int] = {}
    for recipe in recipes:
        if (found := planned_id(recipe, file)) is not None:
            planned[found] = planned.get(found, 0) + 1
    lines = []
    for seen in reversed(file.seen):
        box = seen.fingerprint
        mark = " (this box)" if box.id == current else ""
        plans = f"; {planned[box.id]} plans" if planned.get(box.id) else ""
        if readings and readings.get(box.id):
            plans += f"; {_readings(readings[box.id])}"
        lines.append(
            f"{box.id}{mark}: first {local(seen.first_seen):{WHEN}}, last "
            f"{local(seen.last_seen):{WHEN}}{plans}; {box.words()}"
        )
    return lines
