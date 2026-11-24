"""Section 1 of the guided setup, "This machine": what the planner found (each card
with its PCIe link and what its slot is good for, the RAM with the bandwidth `mnm doctor
--deep` measured, the CPU, the models disk, Docker), the ladder of large models this box can
hold (`ladder`, offline), and on a native box with several cards which of them mnm uses
(`MNM_GPU`: all, or a list). Nothing else is changed here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes import machine as machine_reads
from makenomistakes.doctor_lanes import roles
from makenomistakes.hardware.box_facts import BoxFacts
from makenomistakes.hardware.cards import CHIPSET_LANES
from makenomistakes.hardware.gpu_select import CardChoice, parse_choice
from makenomistakes.hardware.machine_reads import Unread
from makenomistakes.hardware.platform import Family
from makenomistakes.hardware.probe import Gpu, HardwareReport, ProbeError
from makenomistakes.ladder import LADDER, box_rows, box_words, holds, pool_words, rows
from makenomistakes.machine import drift, observe, saved_recipes
from makenomistakes.machine_facts import MachineFactsError, plausible
from makenomistakes.machine_facts import load as load_facts
from makenomistakes.setup_server import server_section
from makenomistakes.setup_view import Mark

if TYPE_CHECKING:
    from makenomistakes.setup_guide import Guide

GIB = 1024**3
CARDS_QUESTION = "Which cards should mnm use? `all`, or their numbers (such as 0,1)"
TRIES = 3


def machine(guide: Guide) -> None:
    listed = _listed(guide.report)
    several = len(listed) > 1 and not guide.platform.is_wsl
    intro = "What mnm found; nothing is changed here."
    if several:
        intro = "What mnm found, and which of the cards it uses (MNM_GPU)."
    guide.view.section(1, "This machine", intro)
    report, where = guide.report, guide.platform
    guide.view.facts(
        [
            ("System", where.label),
            *_card_rows(report),
            ("RAM", _ram_line(guide)),
            _cpu_row(guide),
            ("Models disk", _disk_line(report)),
            ("Docker", _docker_line(report)),
            ("Fingerprint", _box_line(guide)),
        ]
    )
    server_section(guide)  # A server board's BIOS checklist
    if where.distro.family is Family.OTHER:
        guide.view.say(
            f"{where.distro.label} is neither Debian nor Ubuntu: setup prints what to install "
            "instead of installing it.",
            style="yellow",
        )
        guide.summary.add(Mark.NOTE, "This machine", f"{where.distro.label}: not Debian/Ubuntu")
    else:
        guide.summary.add(Mark.DONE, "This machine", f"{where.label}; {_summary_cards(report)}")
    if several:
        cards_question(guide, listed)
    backend_question(guide)  # ROCm beside Vulkan, where an AMD card runs both
    ladder(guide)


def used_cards(guide: Guide) -> list[Gpu]:
    """The cards mnm uses (`MNM_GPU` as the guide has it now; the first card when it is not a
    card list); none without a card."""
    listed = _listed(guide.report)
    try:
        choice = parse_choice(guide.settings.gpu)
    except ValueError:
        choice = CardChoice()
    wanted = choice.resolve([c.index for c in listed])
    return [c for c in listed if c.index in wanted]


def cards_vram(guide: Guide) -> int | None:
    """The VRAM of the cards mnm uses; None without one."""
    return sum(c.vram_total_bytes for c in used_cards(guide)) or None


def ladder(guide: Guide) -> None:
    """What this box holds of the known large models (their RAM floors, `ladder`); on a
    unified pool the pool once."""
    memory, vram = guide.report.memory, cards_vram(guide)
    if memory is not None and any(c.unified or c.carve_out for c in used_cards(guide)):
        guide.view.say(
            f"Large models, by what they need in the one pool ({pool_words(memory.total_bytes)}, "
            "less 8 GiB kept for the system); holding one is not running it fast: "
            "`mnm fit <link>` plans its speed."
        )
        guide.view.facts(rows(memory.total_bytes, 0, pool=True))
        return
    if memory is None:
        return
    cards = vram or 0  # A box without a card holds them in RAM alone
    box = box_words(memory.total_bytes, cards, len(used_cards(guide)))
    guide.view.say(
        f"Large models, by what they need in RAM and on the cards together ({box}); holding "
        "one is not running it fast: `mnm fit <link>` plans its speed."
    )
    guide.view.facts(rows(memory.total_bytes, cards))
    if not holds(LADDER[-1], memory.total_bytes, cards):  # The boxes that would
        guide.view.say(
            "Some cheaper boxes, priced from vendor and review figures, and the largest each holds:"
        )
        guide.view.facts(box_rows())


def cards_question(guide: Guide, listed: list[Gpu]) -> None:
    """`MNM_GPU` on a native box with several cards: asked at a terminal (the saved value, else
    `all`, as the default); unattended, `all` when nothing is saved, a saved value kept."""
    saved = "gpu" in guide.settings.model_fields_set
    default = guide.settings.gpu if saved else "all"
    value = _ask_cards(guide, listed, default) if guide.prompt.interactive else default
    if saved and value == guide.settings.gpu:
        guide.summary.add(Mark.DONE, "Cards", f"MNM_GPU={value} (kept)")
        return
    guide.settings = guide.settings.model_copy(update={"gpu": value})
    guide.save("MNM_GPU", value)
    said = " (dry run: not saved)" if guide.dry_run else ""
    guide.view.say(f"MNM_GPU={value}: mnm plans for and runs on {_which(value)}{said}.")
    guide.summary.add(Mark.DONE, "Cards", f"MNM_GPU={value}{said}")


def _ask_cards(guide: Guide, listed: list[Gpu], default: str) -> str:
    numbers = {c.index for c in listed}
    for _ in range(TRIES):
        answer = guide.prompt.text(CARDS_QUESTION, default).strip().lower() or default
        try:
            choice = parse_choice(answer)
        except ValueError as exc:
            guide.view.say(str(exc), style="yellow")
            continue
        missing = [i for i in choice.indices if i not in numbers]
        if missing and not choice.every:
            guide.view.say(
                f"This box has cards {', '.join(str(i) for i in sorted(numbers))}; "
                f"{', '.join(str(i) for i in missing)} is not one of them.",
                style="yellow",
            )
            continue
        return "all" if choice.every else ",".join(str(i) for i in choice.indices)
    guide.view.say(f"Keeping {default}.", style="yellow")
    return default


BACKEND_QUESTION = (
    "Which llama.cpp build: vulkan (0.3 GB, every model) or rocm (a 7.7 GB image beside it: "
    "faster for dense models over about 20B and long prompts)?"
)


def backend_question(guide: Guide) -> None:
    """`MNM_BACKEND` on a native box whose AMD card runs both llama.cpp builds:
    Vulkan by default; ROCm pulls `server-rocm` (7.7 GB) at step 4, so the models disk's
    room is said. Unattended, a saved value is kept and nothing is saved otherwise."""
    if guide.platform.is_wsl or not _rocm_cards(guide.report):
        return
    saved = (guide.settings.backend or "").strip().lower()
    default = saved if saved in ("vulkan", "rocm") else "vulkan"
    disk = guide.report.disk
    room = f"{disk.free_bytes / GIB:.0f} GiB free on the models disk" if disk.exists else ""
    guide.view.say(f"{BACKEND_QUESTION} {room}".strip())
    value = default
    if guide.prompt.interactive:
        answer = guide.prompt.text("vulkan or rocm", default).strip().lower() or default
        value = answer if answer in ("vulkan", "rocm") else default
    wanted = "rocm" if value == "rocm" else ""
    if wanted == (saved if saved == "rocm" else ""):
        guide.summary.add(Mark.DONE, "Backend", f"{value} (kept)")
        return
    guide.settings = guide.settings.model_copy(update={"backend": wanted})
    guide.save("MNM_BACKEND", wanted)
    said = " (dry run: not saved)" if guide.dry_run else ""
    guide.summary.add(Mark.DONE, "Backend", f"MNM_BACKEND={wanted or 'vulkan (unset)'}{said}")


def _rocm_cards(report: HardwareReport) -> list[Gpu]:
    from makenomistakes.hardware.amd_cards import rocm_runs  # noqa: PLC0415 - AMD only

    return [g for g in _listed(report) if g.is_amd and rocm_runs(g.gfx_target)]


def _which(value: str) -> str:
    return "every card" if value == "all" else f"cards {value}" if "," in value else f"card {value}"


def _listed(report: HardwareReport) -> list[Gpu]:
    return report.gpus if isinstance(report.gpus, list) else []


def _card_rows(report: HardwareReport) -> list[tuple[str, str]]:
    listed = _listed(report)
    if len(listed) < 2:
        return [("GPU", _gpu_line(report))]
    told = {r.card.index: r for r in roles(listed)}
    found = []
    for card in sorted(listed, key=lambda c: c.index):
        line = f"{card.name}, {card.vram_total_bytes / GIB:.1f} GiB"
        if card.gfx_target:
            line += f", {card.gfx_target}"
        role = told.get(card.index)
        if role is not None:
            line += f", {role.link}: {_short(role.main, role.riser, card)}"
        found.append((f"GPU {card.index}", line))
    found.append(("Driver", listed[0].driver_version or "?"))
    if not told:
        found.append(("PCIe links", "not readable here ([N/A]): the most free VRAM decides"))
    return found


def _short(main: bool, riser: bool, card: Gpu) -> str:
    if riser:
        return "a riser: layers only"
    if main:
        return "the main card"
    if card.lanes is not None and card.lanes <= CHIPSET_LANES:
        return "a chipset slot: layers, slow prompts"
    return "the CPU's lanes"


def _summary_cards(report: HardwareReport) -> str:
    listed = _listed(report)
    if len(listed) < 2:
        return _gpu_line(report)
    total = sum(c.vram_total_bytes for c in listed) / GIB
    return f"{len(listed)} cards, {total:.0f} GiB together"


def _gpu_line(report: HardwareReport) -> str:
    if isinstance(report.gpus, ProbeError) or not report.gpus:
        return "no card's driver answers yet"
    gpu = report.gpus[0]
    if gpu.is_amd:  # Its gfx target, an APU's GTT
        what = "of GTT (an APU)" if gpu.carve_out else "VRAM"
        return (
            f"{gpu.name}, {gpu.vram_total_bytes / GIB:.1f} GiB {what}, {gpu.gfx_target or 'gfx ?'}"
        )
    return f"{gpu.name}, {gpu.vram_total_bytes / GIB:.1f} GiB, driver {gpu.driver_version or '?'}"


def _ram_line(guide: Guide) -> str:
    memory = guide.report.memory
    if memory is None:
        return "unknown"
    line = f"{memory.total_bytes / GIB:.1f} GiB"
    if guide.platform.is_wsl:
        line += " (what .wslconfig gives WSL)"
    try:
        facts = load_facts(guide.settings.expanded_state_dir())
    except MachineFactsError:
        facts = None
    if facts is not None and plausible(facts.ram_gbps):
        line += f", {facts.ram_gbps:.1f} GB/s measured (`mnm doctor --deep`)"
    else:
        line += ", bandwidth not measured yet (`mnm doctor --deep`)"
    return line


def _box_line(guide: Guide) -> str:
    """The box recorded (`machine.json`) with its id and when mnm first saw it, or what
    changed since saved plans were made."""
    box = BoxFacts(cpu=guide.cpu, wsl=guide.platform.is_wsl)
    if guide.dry_run:  # "nothing is written in a dry run": machine.json too (apps-setup 11)
        now = machine_reads.read_now(guide.report.box_cards, guide.report.memory, box)
        if isinstance(now, Unread):
            return f"not read: {now.reason}"
        return f"{now.id} (dry run: not recorded)"
    observed = observe(guide.settings, guide.report.box_cards, guide.report.memory, box)
    if observed.unread is not None or observed.file is None:
        return f"not recorded: {observed.unread or observed.error}"
    found = drift(observed.file, saved_recipes(guide.settings))
    if found is None:
        return "not recorded"
    changed = found.words()
    return changed or f"{found.here.id}, {found.here_words().removeprefix('this box: ')}"


def _cpu_row(guide: Guide) -> tuple[str, str]:
    cpu = guide.cpu
    if cpu is not None and cpu.cores:
        return ("CPU", cpu.words())
    return ("CPU threads", str(guide.report.cpu_threads or "unknown"))


def _disk_line(report: HardwareReport) -> str:
    disk = report.disk
    if not disk.exists:
        return f"{disk.path} (not there yet: section 2 makes it)"
    return f"{disk.path}: {disk.free_bytes / GIB:.0f} GiB free of {disk.total_bytes / GIB:.0f} GiB"


def _docker_line(report: HardwareReport) -> str:
    if isinstance(report.docker, ProbeError):
        return "not reachable yet (step 4 installs or fixes it)"
    amd = isinstance(report.gpus, list) and any(g.is_amd for g in report.gpus)
    if amd:  # The images reach an AMD card through its device files
        return f"engine {report.docker.server_version or '?'}, no NVIDIA runtime needed (AMD)"
    runtime = "NVIDIA runtime" if report.docker.has_nvidia_runtime else "no NVIDIA runtime yet"
    return f"engine {report.docker.server_version or '?'}, {runtime}"
