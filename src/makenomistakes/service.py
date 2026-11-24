"""The web UI as a systemd user service, and `mnm update`: one command that updates everything.

Pure helpers build the unit file and command lists; the impure parts go through
`CommandRunner` so tests never touch systemd or git.
"""

from __future__ import annotations

import getpass
import grp
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import makenomistakes
from makenomistakes.config import Settings
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import CommandRunner, FileReader, read_text_or_empty

UNIT_NAME = "makenomistakes-web"
# One URL for every model: its own unit, so a web UI restart never cuts a chat.
ROUTER_UNIT = "makenomistakes-router"
UNITS = (UNIT_NAME, ROUTER_UNIT)
UNIT_DIR = Path("~/.config/systemd/user").expanduser()
WSL_CONF = Path("/etc/wsl.conf")


class ServiceError(RuntimeError):
    pass


def repo_root() -> Path:
    """The git checkout this package runs from (an editable install under `src/`)."""
    root = Path(makenomistakes.__file__).resolve().parents[2]
    if not (root / ".git").exists() or not (root / "pyproject.toml").exists():
        raise ServiceError(
            f"{root} is not the makenomistakes checkout; `mnm update` and `mnm service` need "
            "the package installed from the repository with `uv sync`."
        )
    return root


def uv_binary() -> str:
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path("~/.local/bin/uv").expanduser()
    if candidate.exists():
        return str(candidate)
    raise ServiceError("uv was not found on PATH or in ~/.local/bin; install.sh installs it.")


def unit_file(root: Path, uv: str, settings: Settings) -> str:
    """The web UI's unit: `mnm serve` on 127.0.0.1, restarted on failure."""
    command = f"serve --host 127.0.0.1 --port {settings.port}"
    return _unit("makenomistakes web UI (mnm serve)", root, uv, command, "on-failure", settings)


def router_unit_file(root: Path, uv: str, settings: Settings) -> str:
    """The router's unit: `mnm router` where `MNM_ROUTER_HOST` says (every
    interface unless the user chose otherwise, so Odysseus in Docker reaches it), restarted
    whatever ended it: it is the one way in to every model (and `local` waits for Docker)."""
    command = f"router --port {settings.router_port}"
    return _unit("makenomistakes router (mnm router)", root, uv, command, "always", settings)


def _unit(
    description: str, root: Path, uv: str, command: str, restart: str, settings: Settings
) -> str:
    """A user unit that keeps the model dirs and the router's port in its environment (the
    web UI's cards and parks name the router too, so both must agree on it).

    No PATH line: WSL's shell PATH carries Windows entries with spaces, which systemd would
    split into bogus assignments; `uv` is absolute and `uv run` adds `.venv/bin` itself.
    A user unit cannot order itself after system units, so no `After=docker.service`
    (both tolerate Docker being down and say so per instance). `uv run` exits 143
    (128 + SIGTERM) when systemd stops it, which would log every restart as a failure.
    """
    return (
        "[Unit]\n"
        f"Description={description}\n"
        "\n"
        "[Service]\n"
        f"WorkingDirectory={specifiers(str(root))}\n"
        f"ExecStart={exec_word(uv)} run mnm {command}\n"
        f"Restart={restart}\n"
        "RestartSec=3\n"
        "SuccessExitStatus=143\n" + environment_lines(settings) + "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def environment_lines(settings: Settings) -> str:
    """The `Environment=` lines every mnm unit carries (the backup timer's too), so
    each process mnm starts reads the same folders and ports."""
    lines = [
        f"MNM_MODELS_DIR={settings.expanded_models_dir()}",
        f"MNM_STATE_DIR={settings.expanded_state_dir()}",
        f"MNM_OUTPUTS_DIR={settings.expanded_outputs_dir()}",
        f"MNM_APPS_DIR={settings.expanded_apps_dir()}",
        f"MNM_PROJECTS_DIR={settings.expanded_projects_dir()}",
        f"MNM_BACKUPS_DIR={settings.expanded_backups_dir()}",
        f"MNM_ROUTER_PORT={settings.router_port}",
        f"MNM_ROUTER_HOST={settings.router_host}",
        *_optional(settings),
    ]
    return "".join(f"Environment={_quoted(line)}\n" for line in lines)


def _optional(settings: Settings) -> list[str]:
    """The optional settings, carried only when set: where another machine opens the web UI,
    Ollama's listener, and a Hub mirror (huggingface_hub reads `HF_ENDPOINT` itself). Never
    `MNM_GPU`: the services read the card from the secrets file, where the doctor's and
    adopt's advice sends the user (`service_gpu`)."""
    found = []
    if settings.web_url:
        found.append(f"MNM_WEB_URL={settings.web_url}")
    if settings.ollama_port is not None:
        found.append(f"MNM_OLLAMA_PORT={settings.ollama_port}")
    if endpoint := os.environ.get("HF_ENDPOINT", "").strip():
        found.append(f"HF_ENDPOINT={endpoint}")
    return found


def _quoted(assignment: str) -> str:
    """One `Environment=` assignment that survives spaces, quotes and `%` specifiers."""
    escaped = assignment.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def unquoted(value: str) -> str:
    """`_quoted` undone: what `mnm update` reads back from the installed unit."""
    inner = value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value
    out: list[str] = []
    chars = iter(inner.replace("%%", "%"))
    for char in chars:
        out.append(next(chars, "") if char == "\\" else char)
    return "".join(out)


def specifiers(path: str) -> str:
    """A path systemd takes literally: `%` starts a specifier (`%h`, `%u`) unless doubled."""
    return path.replace("%", "%%")


def exec_word(word: str) -> str:
    """One `ExecStart=` word: systemd splits on spaces and expands `%` and `$`, so a uv path
    with any of them (a home folder with a space) is quoted and escaped."""
    escaped = specifiers(word).replace("$", "$$")
    if not any(c in escaped for c in " \t\"'\\;"):
        return escaped
    return '"' + escaped.replace("\\", "\\\\").replace('"', '\\"') + '"'


# What `mnm update` carries from the installed units into the new ones; never `MNM_GPU`:
# the new code's install moves a carried one to the secrets file (`service_gpu`).
CARRIED = (
    "MNM_MODELS_DIR",
    "MNM_STATE_DIR",
    "MNM_OUTPUTS_DIR",  # The CLI, the web and the router keep runs in one gallery
    "MNM_APPS_DIR",  # The web's Apps page and the manifest read the apps' folders
    "MNM_PROJECTS_DIR",
    "MNM_BACKUPS_DIR",  # The web's Back up now writes where the timer does
    "MNM_ROUTER_PORT",
    "MNM_ROUTER_HOST",
    "MNM_WEB_URL",  # the router's MCP tools link to the web UI where Odysseus can open it
    "MNM_OLLAMA_PORT",  # the router's second listener, for Ollama's API
    "HF_ENDPOINT",  # a Hub mirror (hf-mirror.com), for the web's and the router's downloads
)


def installed_environment(unit_dir: Path | None = None) -> list[str]:
    """The installed units' folders and ports as `NAME=value` words, so `mnm update` rewrites
    the units with what they run now, not with the shell's `MNM_*` variables."""
    unit_dir = unit_dir or UNIT_DIR  # read now: tests point it at a folder of their own
    found: dict[str, str] = {}  # by name: both units carry MNM_ROUTER_PORT
    for unit, port_name in ((UNIT_NAME, "MNM_PORT"), (ROUTER_UNIT, "MNM_ROUTER_PORT")):
        try:
            text = (unit_dir / f"{unit}.service").read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            key, _, value = line.partition("=")
            if key == "Environment":
                name, _, setting = unquoted(value).partition("=")
                if name in CARRIED:
                    found.setdefault(name, setting)
            elif key == "ExecStart" and " --port " in value:
                port = value.rsplit(" --port ", 1)[1].split()[0]
                if port.isdigit():
                    found[port_name] = port  # what the unit runs wins over its Environment
    return [f"{name}={value}" for name, value in found.items()]


@dataclass
class SystemdState:
    available: bool  # `systemctl --user` answers
    installed: bool  # the unit file exists
    active: bool  # the unit is running
    enabled: bool  # starts at login/boot
    detail: str = ""
    wsl_conf_has_systemd: bool | None = None  # None outside WSL or when unreadable


def journal(runner: CommandRunner, lines: int = 100) -> str:
    """The web UI and router services' newest log lines, interleaved (journalctl --user):
    `mnm service logs` prints it, the web UI's System page shows it."""
    units = [word for unit in UNITS for word in ("-u", unit)]
    code, out, err = runner.run(["journalctl", "--user", *units, "-n", str(lines), "--no-pager"])
    return out if code == 0 else err or out


# `systemctl is-system-running` states in which the user manager accepts units (exit 0 or 1).
MANAGER_UP = {"initializing", "starting", "running", "degraded", "maintenance", "stopping"}


def systemd_state(
    runner: CommandRunner,
    read_text: FileReader = read_text_or_empty,
    unit_dir: Path = UNIT_DIR,
    unit: str = UNIT_NAME,
) -> SystemdState:
    code, out, err = runner.run(["systemctl", "--user", "is-system-running"])
    answer = (out or err).strip()
    available = code in (0, 1) and answer in MANAGER_UP
    conf = read_text(WSL_CONF)
    has_systemd: bool | None = None
    if conf:
        lines = [line.strip().replace(" ", "") for line in conf.splitlines()]
        has_systemd = "systemd=true" in lines
    installed = (unit_dir / f"{unit}.service").exists()
    active = enabled = False
    if available and installed:
        active = runner.run(["systemctl", "--user", "is-active", unit])[0] == 0
        enabled = runner.run(["systemctl", "--user", "is-enabled", unit])[0] == 0
    return SystemdState(
        available=available,
        installed=installed,
        active=active,
        enabled=enabled,
        detail=answer,
        wsl_conf_has_systemd=has_systemd,
    )


SYSTEMD_MANUAL = (
    "systemd is not running in this WSL distro. Enable it once: add\n"
    "    [boot]\n    systemd=true\n"
    "to /etc/wsl.conf (sudo nano /etc/wsl.conf), then run `wsl --shutdown` from PowerShell, "
    "open Ubuntu again and run `mnm service install` again. Until then start the UI with "
    "`mnm serve`."
)

NATIVE_SYSTEMD_MANUAL = (
    "systemd's user manager does not answer (`systemctl --user`): log in on a console or "
    "over SSH as yourself (not with `su`), or reboot, then run `mnm service install` again. "
    "Until then start the UI with `mnm serve` and the router with `mnm router`."
)


def docker_group_note() -> str:
    where = platforms.current()
    return (
        "Your systemd user session is not in the `docker` group yet (mnm setup just added "
        "it; a new terminal or `newgrp docker` does not reach it). The service would inherit "
        "the old groups and every Docker call would fail with permission denied. "
        + where.services_relogin
    )


def systemd_note(state: SystemdState) -> str:
    """What to do when `systemctl --user` does not answer, given what /etc/wsl.conf says."""
    if not platforms.current().is_wsl:
        return NATIVE_SYSTEMD_MANUAL
    if state.wsl_conf_has_systemd:
        return (
            "systemd is enabled in /etc/wsl.conf but not running yet: run `wsl --shutdown` from "
            "PowerShell, open Ubuntu again and run `mnm service install` again. Until then "
            "start the UI with `mnm serve`."
        )
    return SYSTEMD_MANUAL


def docker_gid_on_box() -> int | None:
    try:
        return grp.getgrnam("docker").gr_gid
    except KeyError:
        return None


# A transient unit gets exactly the credentials the service will: the user manager's groups.
MANAGER_GROUPS = ["systemd-run", "--user", "--wait", "--pipe", "--quiet", "--collect", "id", "-G"]


def docker_group_active(
    runner: CommandRunner, docker_gid: Callable[[], int | None] = docker_gid_on_box
) -> bool:
    """Will the service's processes be in the `docker` group?

    The unit runs under the systemd user manager, which keeps the groups it started with
    (with linger, until `wsl --shutdown`), so this shell's groups prove nothing: `newgrp
    docker` or a new terminal fixes the shell only. The manager is asked instead; if it
    cannot answer, this process's groups are the best guess.
    """
    gid = docker_gid()
    if gid is None:
        return True  # no docker group on this box: nothing to wait for
    code, out, _ = runner.run(MANAGER_GROUPS)
    if code != 0:
        return gid in os.getgroups()
    return str(gid) in out.split()


# Written to the state folder when `install` turned linger on (it was off): uninstall.sh turns
# linger off again only then, never when the user had it on for their own reasons.
LINGER_MARKER = "linger-enabled-by-mnm"


def linger_on(runner: CommandRunner) -> bool | None:
    """Whether this user lingers (None: loginctl could not say)."""
    argv = ["loginctl", "show-user", getpass.getuser(), "--property=Linger", "--value"]
    code, out, _ = runner.run(argv)
    return {"yes": True, "no": False}.get(out.strip()) if code == 0 else None


def install(
    runner: CommandRunner,
    settings: Settings,
    unit_dir: Path = UNIT_DIR,
    docker_gid: Callable[[], int | None] = docker_gid_on_box,
    told: Callable[[str], None] | None = None,
) -> list[str]:
    """Write the unit, enable it, (re)start it, keep it running after logout. Returns what ran;
    what the user should read (`MNM_GPU` moved out of the units) goes to `told`, else
    with what ran."""
    from makenomistakes import config, service_gpu  # noqa: PLC0415 - service_gpu imports this

    state = systemd_state(runner, unit_dir=unit_dir)
    if not state.available:
        raise ServiceError(systemd_note(state))
    if not docker_group_active(runner, docker_gid):
        raise ServiceError(docker_group_note())
    ran: list[str] = []
    said = service_gpu.move_out(unit_dir, config.SECRETS_FILE)  # before the units are read
    if (shell := service_gpu.shell_note(config.SECRETS_FILE)) is not None:
        said.append(shell)
    for line in said:
        (told or ran.append)(line)
    unit_dir.mkdir(parents=True, exist_ok=True)
    root, uv = repo_root(), uv_binary()
    (unit_dir / f"{UNIT_NAME}.service").write_text(unit_file(root, uv, settings), "utf-8")
    (unit_dir / f"{ROUTER_UNIT}.service").write_text(router_unit_file(root, uv, settings), "utf-8")
    lingered = linger_on(runner)
    for argv in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", *UNITS],
        # `enable --now` leaves a running unit on its old command line and environment.
        ["systemctl", "--user", "restart", *UNITS],
        ["loginctl", "enable-linger"],
    ):
        code, out, err = runner.run(argv)
        ran.append(" ".join(argv))
        if code != 0 and argv[0] != "loginctl":  # linger is best effort
            raise ServiceError(f"{' '.join(argv)} failed: {(err or out).strip()}")
    if lingered is False and linger_on(runner):
        marker = settings.expanded_state_dir() / LINGER_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("mnm service install turned linger on\n", encoding="utf-8")
    return ran


def uninstall(runner: CommandRunner, unit_dir: Path = UNIT_DIR) -> list[str]:
    ran: list[str] = []
    for name in UNITS:
        unit = unit_dir / f"{name}.service"
        if unit.exists():
            argv = ["systemctl", "--user", "disable", "--now", name]
            runner.run(argv)
            ran.append(" ".join(argv))
            unit.unlink()
    if ran:
        runner.run(["systemctl", "--user", "daemon-reload"])
        ran.append("systemctl --user daemon-reload")
    return ran


def restart(runner: CommandRunner, unit_dir: Path = UNIT_DIR) -> None:
    """The installed units: a box whose units an older mnm wrote has no router unit yet."""
    units = [u for u in UNITS if (unit_dir / f"{u}.service").exists()] or [UNIT_NAME]
    code, out, err = runner.run(["systemctl", "--user", "restart", *units])
    if code != 0:
        raise ServiceError(f"could not restart the web UI and router: {(err or out).strip()}")


# --- update -----------------------------------------------------------------------


REINSTALL = ["mnm", "service", "install"]  # idempotent: rewrite, daemon-reload, enable


@dataclass
class UpdateReport:
    before: str
    after: str
    changes: list[str] = field(default_factory=list)  # one line per new commit
    service_restarted: bool = False
    notes: list[str] = field(default_factory=list)  # what the user should know or do


def checkout_revision(runner: CommandRunner, root: Path) -> str | None:
    """The checkout's short commit (what `mnm update` moves), or None outside a git checkout."""
    code, out, _ = runner.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"])
    return (out.strip() or None) if code == 0 else None


def update(
    runner: CommandRunner,
    root: Path,
    uv: str,
    service_installed: bool,
    unit_dir: Path = UNIT_DIR,
) -> UpdateReport:
    """git pull (fast-forward only), uv sync, restart the UI service when it is installed.

    A crashed unit is restarted too (it comes up on the new code); a restart failure goes
    in the notes, since the pull and sync already landed.
    """

    def git(*args: str) -> tuple[int, str, str]:
        return runner.run(["git", "-C", str(root), *args], timeout=300.0)

    code, out, err = git("rev-parse", "--short", "HEAD")
    if code != 0:
        raise ServiceError(f"git rev-parse failed in {root}: {(err or out).strip()}")
    before = out.strip()
    code, out, err = git("pull", "--ff-only")
    if code != 0:
        raise ServiceError(
            f"git pull failed: {(err or out).strip()}\nIf you edited files in {root}, stash "
            "or commit them; if the branch diverged, `git status` says so."
        )
    after = git("rev-parse", "--short", "HEAD")[1].strip()
    changes: list[str] = []
    if after != before:
        changes = [
            line for line in git("log", "--oneline", f"{before}..{after}")[1].splitlines() if line
        ]
    # Pinned to the checkout: `mnm` runs from any folder, and in ~/odysseus an unpinned sync
    # made Odysseus's own .venv. Never rewrite the lock here.
    code, out, err = runner.run([uv, "sync", "--frozen", "--project", str(root)], timeout=900.0)
    if code != 0:
        raise ServiceError(
            f"uv sync --frozen failed: {(err or out).strip()} (uv.lock no longer matches "
            "pyproject.toml: `git status` in the checkout lists the files changed by hand)"
        )
    report = UpdateReport(before=before, after=after, changes=changes)
    if service_installed:
        # This process still runs the old code, so the new code rewrites its own unit file
        # (and restarts it), with the folders and port the unit has now.
        carried = installed_environment(unit_dir)
        argv = [uv, "run", "--project", str(root), *REINSTALL]
        code, out, err = runner.run(["env", *carried, *argv] if carried else argv, timeout=120.0)
        if code == 0:
            report.service_restarted = True
        else:
            report.notes.append(f"The service unit was not rewritten: {(err or out).strip()}")
            try:
                restart(runner, unit_dir)
                report.service_restarted = True
            except ServiceError as exc:
                report.notes.append(f"{exc} Run `mnm service status`.")
    else:
        report.notes.append(
            "The web UI is not running as a service: restart `mnm serve` by hand, or run "
            "`mnm service install` once so updates restart it for you."
        )
    return report
