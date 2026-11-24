"""What the System page and the status bar read: the router, the two services, which settings
are set (never a value), the engines, the calibration. Every outside question goes through a
`SystemProbe` so tests answer it without a socket or a subprocess."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from makenomistakes import engines
from makenomistakes.backend import CUDA, Backend, label
from makenomistakes.calibration import (
    Correction,
    SpeedCorrection,
    corrections,
    every_reading,
    speed_corrections,
)
from makenomistakes.config import Settings
from makenomistakes.doctor import Check
from makenomistakes.doctor_deep import deep_tail
from makenomistakes.engine_check import EngineCheck, EngineReport, last_report
from makenomistakes.hardware.box_facts import read_box
from makenomistakes.hardware.probe import CommandRunner, SubprocessRunner, probe
from makenomistakes.health import HealthReport
from makenomistakes.instances import InstanceService
from makenomistakes.machine_checks import machine_checks
from makenomistakes.measurements import Reading
from makenomistakes.quality.readings import QuantScore, newest_scores
from makenomistakes.queue_figures import Waiting, waiting_words
from makenomistakes.queue_summary import QueueSummary
from makenomistakes.queue_summary import summaries as queue_summaries
from makenomistakes.readings_filter import (
    left_out,
    left_out_words,
    memory_readings,
    readings_here,
    store_here,
)
from makenomistakes.router_bind import Exposure, exposure
from makenomistakes.router_client import (
    RouterNow,
    router_answers,
    router_events,
    router_health,
    router_now,
)
from makenomistakes.router_events import Event
from makenomistakes.service import (
    ROUTER_UNIT,
    UNIT_NAME,
    ServiceError,
    SystemdState,
    checkout_revision,
    installed_environment,
    repo_root,
    systemd_state,
)
from makenomistakes.service import journal as service_journal
from makenomistakes.studio.scores import ScoreSummary, summaries
from makenomistakes.supervisor import SupervisorError

STARTED_AT = datetime.now(UTC)  # when this web UI process started: its jobs began then


def deep_checks(settings: Settings) -> list[Check]:
    """`mnm doctor --deep`: every check, the backend's container test, the RAM's bandwidth
    (saved for the planner) and Experts in RAM, as the CLI runs it (`doctor_deep`). The plain
    doctor is `app.default_doctor`, which tests replace through `app.probe`."""
    report = probe(settings.expanded_models_dir(), docker_gpu_smoke=True, host_ram=True)
    box = read_box(SubprocessRunner(), report.is_wsl)
    return [*machine_checks(settings, report, box, True), *deep_tail(settings, report)]


def mcp_tool_names(service: InstanceService) -> list[str] | None:
    """The MCP tools the router serves now (`mnm mcp tools`); None when they cannot be read
    (a recipe or workflow this version cannot read: the page says so, it never fails)."""
    from makenomistakes.bridge import bridge_tools  # noqa: PLC0415 - ComfyUI's side, on demand

    try:
        return [tool.name for tool in bridge_tools(service).tools()]
    except (RuntimeError, OSError, ValueError):  # InstanceError and ComfyRunError included
        return None


@dataclass
class SystemProbe:
    """The questions the page asks the machine. Tests replace each one."""

    router_up: Callable[[Settings], bool] = router_answers
    traffic: Callable[[Settings], RouterNow | None] = router_now  # in flight, what waits
    events: Callable[[Settings], list[Event] | None] = router_events
    health: Callable[[Settings], HealthReport | None] = router_health
    runner: CommandRunner = field(default_factory=SubprocessRunner)
    installed: Callable[[], list[str]] = installed_environment
    deep: Callable[[Settings], list[Check]] = deep_checks
    mcp_tools: Callable[[InstanceService], list[str] | None] = mcp_tool_names

    def revision(self) -> str | None:
        """The checkout's commit, as `mnm --version` prints it; None outside a checkout."""
        try:
            return checkout_revision(self.runner, repo_root())
        except ServiceError:  # not a checkout: the line is left out
            return None

    def units(self) -> list[tuple[str, SystemdState]]:
        return [(unit, systemd_state(self.runner, unit=unit)) for unit in (UNIT_NAME, ROUTER_UNIT)]

    def journal(self, lines: int) -> str:
        return service_journal(self.runner, lines)

    def installed_value(self, name: str) -> str | None:
        """A setting as the installed router unit carries it (None: no unit, or not set)."""
        for word in self.installed():
            key, _, value = word.partition("=")
            if key == name:
                return value
        return None


@dataclass(frozen=True)
class RouterView:
    up: bool
    port: int
    host: str
    exposure: Exposure | None  # None: the setting does not parse (the doctor says why)
    key_required: bool
    busy: dict[str, int]
    arrivals: int | None  # None: the router did not say
    events: list[Event] | None = None  # its newest journal lines, newest first
    queue: dict[str, Waiting] = field(default_factory=dict)  # what waits now

    @property
    def in_flight(self) -> int:
        return sum(self.busy.values())

    @property
    def waiting(self) -> int:
        return sum(now.total for now in self.queue.values())

    @property
    def waiting_lines(self) -> list[str]:
        """`2 waiting on coder (for a slot; 1 answering)`, one per instance where any wait."""
        return [waiting_words(name, now) for name, now in sorted(self.queue.items())]


def router_view(settings: Settings, probe: SystemProbe) -> RouterView:
    up = probe.router_up(settings)
    traffic = probe.traffic(settings) if up else None
    try:
        reach: Exposure | None = exposure(settings.router_host)
    except ValueError:
        reach = None
    return RouterView(
        up=up,
        port=settings.router_port,
        host=settings.router_host,
        exposure=reach,
        key_required=settings.api_key_value() is not None,
        busy={n: c for n, c in (traffic.busy if traffic else {}).items() if c},
        arrivals=traffic.arrivals if traffic else None,
        events=list(reversed(found)) if up and (found := probe.events(settings)) else None,
        queue=traffic.queue if traffic else {},
    )


@dataclass(frozen=True)
class SettingRow:
    name: str
    is_set: bool
    what: str
    shown: str | None = None  # a value safe to show (the router's address), never a secret


def setting_rows(settings: Settings, probe: SystemProbe) -> list[SettingRow]:
    """Which settings are set: tokens and the key as yes/no only."""
    installed_host = probe.installed_value("MNM_ROUTER_HOST")
    host_note = "where the router listens"
    if installed_host is not None and installed_host != settings.router_host:
        host_note += (
            f"; the installed router service still runs with {installed_host!r}: "
            "`mnm service install` applies the new one"
        )
    return [
        SettingRow("HF_TOKEN", settings.hf_token_value() is not None, "gated Hugging Face repos"),
        SettingRow(
            "CIVITAI_TOKEN", settings.civitai_token_value() is not None, "Civitai downloads"
        ),
        SettingRow("MNM_API_KEY", settings.api_key_value() is not None, "the router asks for it"),
        SettingRow("MNM_ROUTER_HOST", True, host_note, settings.router_host),
        SettingRow("MNM_ROUTER_PORT", True, "the router's port", str(settings.router_port)),
        SettingRow("MNM_PORT", True, "this web UI's port", str(settings.port)),
        SettingRow("MNM_MODELS_DIR", True, "the downloads", str(settings.expanded_models_dir())),
        SettingRow(
            "MNM_STATE_DIR", True, "recipes, logs, readings", str(settings.expanded_state_dir())
        ),
    ]


@dataclass(frozen=True)
class EnginesView:
    rows: list[engines.EngineStatus]
    error: str | None = None  # Docker could not be asked
    ledger_note: str | None = None  # the ledger could not be read: rollback unknown
    check: EngineReport | None = None  # the last `mnm engines check`
    box: str = ""  # The backend this box runs, in a sentence (`engines.box_words`)

    def upstream(self, image: str) -> EngineCheck | None:
        """The last check's row for an image."""
        if self.check is None:
            return None
        return next((row for row in self.check.rows if row.image == image), None)


def engines_view(service: InstanceService) -> EnginesView:
    state_dir = service.settings.expanded_state_dir()
    note = None
    try:
        ledger = engines.load_ledger(state_dir)
    except engines.EngineError as exc:
        ledger, note = engines.EngineLedger(), str(exc)
    check = last_report(state_dir)
    box = engines.box_words()
    try:
        rows = engines.engine_status(service.supervisor, service.all(), ledger)
    except (engines.EngineError, SupervisorError) as exc:
        return EnginesView([], str(exc), note, check, box)
    return EnginesView(rows, None, note, check, box)


@dataclass(frozen=True)
class CalibrationView:
    corrections: list[Correction]
    newest: dict[str, list[Reading]]  # each instance's last readings, newest last
    speeds: list[SpeedCorrection] = field(default_factory=list)  # picture time
    machine: str | None = None  # This box in one line (None: not read yet)
    left_out: str | None = None  # another machine's readings, kept and not used
    elsewhere: dict[str, int] = field(default_factory=dict)  # per instance, when any
    scores: list[ScoreSummary] = field(default_factory=list)  # Songs' candidates
    queues: list[QueueSummary] = field(default_factory=list)  # What waited, per instance
    qualities: list[tuple[str, QuantScore]] = field(default_factory=list)  # Per instance

    @property
    def empty(self) -> bool:
        tables = self.scores or self.queues or self.qualities
        return not any(self.newest.values()) and not tables


def calibration_view(service: InstanceService, last: int = 10) -> CalibrationView:
    """This box's readings and corrections; another machine's are only counted."""
    kept = service.measurements.every()
    here = store_here(service.measurements)
    by_name = {name: readings_here(found, here) for name, found in kept.items()}
    every = every_reading(by_name)
    elsewhere = {name: n for name, found in kept.items() if (n := left_out(found, here))}
    plain = {n: memory_readings(found) for n, found in by_name.items()}
    return CalibrationView(
        corrections(every),
        {name: readings[-last:] for name, readings in plain.items() if readings},
        speed_corrections(every),
        machine=here.words,
        left_out=left_out_words(sum(elsewhere.values())),
        elsewhere=elsewhere,
        scores=summaries(by_name),
        queues=queue_summaries(by_name),
        qualities=[(name, s) for name, found in by_name.items() for s in newest_scores(found)],
    )


@dataclass(frozen=True)
class DeepWords:
    """What the Doctor page's deep run says it does, by this box's backend (a 100 MB
    CUDA pull only on a CUDA box)."""

    button: str
    hint: str
    confirm: str
    running: str


def deep_words(backend: Backend) -> DeepWords:
    if backend in CUDA:
        return DeepWords(
            "Run the GPU container test",
            "proves Docker reaches the card (pulls ~100 MB once), measures the RAM's speed.",
            "Run a CUDA container to check the GPU from Docker? The first run pulls about 100 MB.",
            "running a CUDA container (pulls ~100 MB the first time)…",
        )
    if backend is Backend.CPU:
        return DeepWords(
            "Run the deep checks",
            "no card to test from Docker; measures the RAM's speed.",
            "Measure the RAM's speed now? It takes a few seconds and a few GB of RAM.",
            "measuring the RAM's speed…",
        )
    image = label(backend)
    return DeepWords(
        "Run the GPU container test",
        f"lists the card from llama.cpp's {image} image (pulls the image "
        "once, several GB), measures the RAM's speed.",
        f"Run llama.cpp's {image} image to list the card from Docker? The first run pulls the "
        "image, several GB.",
        f"running llama.cpp's {image} image (a pull of several GB the first time)…",
    )
