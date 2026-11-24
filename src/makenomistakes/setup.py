"""`mnm setup`: take a fresh WSL or native Debian/Ubuntu box to "ready", checking before
installing.

Each step has a live check and, when the check fails, the commands that fix it (run with
sudo, printed before they run). Checks go through the injected `CommandRunner`, so a dry
run shows the whole plan without touching the machine and tests drive it with a fake.
Under WSL the GPU driver is the Windows one, which only the user can install. On a native
box setup installs the distro's or NVIDIA's packaged driver, but only when
asked (`confirm`), and says to reboot.

Right after Docker is installed the current shell is not in the `docker` group yet, so the
daemon answers "permission denied". The steps treat that as "running, re-login pending" and
run the image pull with sudo instead of failing the whole bootstrap.
"""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from makenomistakes.backend import CUDA, Backend
from makenomistakes.config import SECRETS_FILE, Settings, is_system_folder
from makenomistakes.hardware import CommandRunner, FileReader, HardwareReport
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.platform import Platform, amd_pci_devices, nvidia_pci_devices
from makenomistakes.hub.facts import Engine
from makenomistakes.machine import backend_read
from makenomistakes.recipes import LLAMACPP_IMAGE, spec
from makenomistakes.service import ServiceError, repo_root, uv_binary
from makenomistakes.setup_amd import card_steps
from makenomistakes.setup_commands import (  # re-exported: tests and the CLI import here
    Command,
    apt_nvidia_toolkit_commands,
    start_docker_command,
)
from makenomistakes.setup_docker import (  # re-exported for the CLI
    Docker,
    docker_group_members,
)
from makenomistakes.setup_machine import (  # re-exported
    base_packages_step,
    docker_step,
)
from makenomistakes.setup_types import Outcome, Step, StepResult

LAUNCHER = Path("~/.local/bin/mnm").expanduser()
FIX_TIMEOUT = 1800.0


class InteractiveRunner:
    """Runs commands with the terminal attached, so `sudo` can prompt and output streams."""

    def run(self, argv: list[str], timeout: float = FIX_TIMEOUT) -> tuple[int, str, str]:
        try:
            code = subprocess.run(argv, check=False, timeout=timeout).returncode
        except FileNotFoundError:
            return 127, "", f"{argv[0]}: not found"
        except subprocess.TimeoutExpired:
            return 124, "", f"{argv[0]}: timed out"
        return code, "", ""


def build_steps(  # noqa: PLR0913 - each part can be swapped
    report: HardwareReport,
    settings: Settings,
    *,
    runner: CommandRunner,
    pull_image: bool = True,
    read_text: FileReader = lambda p: p.read_text(encoding="utf-8"),
    launcher: Launcher | None = None,
    group_members: Callable[[str], list[str]] | None = None,
    platform: Platform | None = None,
    pci_devices: Callable[[], list[str]] = nvidia_pci_devices,
    backend: Backend | None = None,
    amd_devices: Callable[[], list[str]] = amd_pci_devices,
) -> list[Step]:
    """The ordered plan for this machine. `report` is the probe `doctor` also uses;
    `launcher` is `Launcher.for_checkout()` from the CLI (None outside a git checkout).
    `backend` (the box's as `backend_read` reads it when not given): the NVIDIA runtime is a
    step only for the CUDA backends, and the engine image pulled is llama.cpp's for it; None
    (the box could not be read) keeps the CUDA steps."""
    models_dir = settings.expanded_models_dir()
    user = current_user()
    where = platform or platforms.current()
    read = backend_read(settings, report.box_cards, report.memory)
    backend = backend or read.backend
    arch = read.box.arch if read.box is not None and read.box.arch else os.uname().machine
    image = llama_image(backend, arch)
    docker = Docker(runner, read_text, user, group_members or docker_group_members, where, image)
    cuda = backend is None or backend in CUDA
    steps = [
        *([] if where.is_wsl else [base_packages_step(runner)]),
        *card_steps(
            report, where, runner, user, read_text, pci_devices, amd_devices, backend=backend
        ),
        docker_step(where, runner, user, docker),
        Step(
            "Docker daemon",
            check=docker.daemon,
            fix=[start_docker_command()],
            requires="Docker Engine",
        ),
        Step(  # a Docker that was already there never ran setup's `usermod`
            "Docker group",
            check=docker.group_membership,
            fix=[["sudo", "usermod", "-aG", "docker", user]],
            after=f"Then {where.relogin}.",
            requires="Docker daemon",
            settles_at_relogin=docker.relogin_pending,
        ),
        *(
            [
                Step(
                    "NVIDIA container runtime",
                    check=docker.nvidia_runtime,
                    fix=apt_nvidia_toolkit_commands(),
                    requires="Docker daemon",  # its fix ends in `systemctl restart docker`
                )
            ]
            if cuda
            else []
        ),
        storage_step(models_dir, user),
        storage_step(  # What runs make
            settings.expanded_outputs_dir(),
            user,
            "Output storage",
            "MNM_OUTPUTS_DIR",
            "/data/outputs, or ~/mnm-outputs",
        ),
        storage_step(  # the apps' data (LibreChat, OpenHands)
            settings.expanded_apps_dir(),
            user,
            "Apps storage",
            "MNM_APPS_DIR",
            "/data/apps, or ~/mnm-apps",
        ),
        storage_step(  # `mnm backup --auto` writes and rotates here
            settings.expanded_backups_dir(),
            user,
            "Backup storage",
            "MNM_BACKUPS_DIR",
            "/data/backups, or ~/mnm-backups",
        ),
        Step(
            "Secrets file",
            check=_secrets_file_check,
            fix=secrets_file_commands(),
            after="For gated models, add a line HF_TOKEN=hf_... to that file.",
        ),
    ]
    if launcher is not None:
        steps.append(launcher.step(read_text))
    if pull_image:
        steps.append(
            Step(
                "llama.cpp engine image",
                check=docker.image_present,
                fix=[docker.pull_command(image)],
                requires="Docker daemon",
                settles_at_relogin=docker.relogin_pending,
            )
        )
        if backend is Backend.ROCM:  # The Vulkan image beside it (the device check)
            vulkan = llama_image(Backend.VULKAN, arch)
            steps.append(
                Step(
                    "llama.cpp Vulkan image",
                    check=lambda: docker.tag_present(vulkan),
                    fix=[docker.pull_command(vulkan)],
                    requires="Docker daemon",
                    settles_at_relogin=docker.relogin_pending,
                )
            )
    return steps


def llama_image(backend: Backend | None, arch: str) -> str:
    """The llama.cpp tag setup pulls: the backend's for the box's arch (as the
    fingerprint read it, `server-cuda13` on a DGX Spark), else the default (`LLAMACPP_IMAGE`)."""
    if backend is None:
        return LLAMACPP_IMAGE
    found = spec(Engine.LLAMACPP).image_for(backend, arch)
    return found if isinstance(found, str) else LLAMACPP_IMAGE


def storage_step(
    models_dir: Path,
    user: str,
    label: str = "Model storage",
    setting: str = "MNM_MODELS_DIR",
    example: str = "/data/models, or ~/models",
) -> Step:
    """A folder mnm writes to (the models folder, the outputs folder too), checked live (a
    re-check after the fix sees the new folder). Setup creates and chowns only a folder that
    does not exist yet: an existing one may be anything (`MNM_MODELS_DIR=/home`), so it is
    never chowned for the user, only explained."""
    if is_system_folder(models_dir):  # never a `chown` of /home, /usr or a Windows drive
        return Step(
            label,
            check=lambda: (False, f"{models_dir} is a system folder"),
            manual=f"{setting} is {models_dir}, a system folder: set it to a folder of your "
            f"own (the default {example}), then run `mnm setup` again.",
        )
    if models_dir.exists() and not (models_dir.is_dir() and os.access(models_dir, os.W_OK)):
        what = "is not a folder" if not models_dir.is_dir() else f"is not writable by {user}"
        return Step(
            label,
            check=lambda: _storage_ready(models_dir),
            manual=(
                f"{models_dir} already exists and {what}. If it is meant for mnm, run "
                f"`sudo chown {shlex.quote(user)} {shlex.quote(str(models_dir))}`; otherwise "
                f"set {setting} to a folder you own, then run `mnm setup` again."
            ),
        )
    return Step(
        label,
        check=lambda: _storage_ready(models_dir),
        fix=[
            ["sudo", "mkdir", "-p", str(models_dir)],
            ["sudo", "chown", user, str(models_dir)],
        ],
    )


def _storage_ready(models_dir: Path) -> tuple[bool, str]:
    if not models_dir.is_dir():
        return False, f"{models_dir} missing"
    if not os.access(models_dir, os.W_OK):
        return False, f"{models_dir} not writable"
    return True, str(models_dir)


_MARKER = "# Written by `mnm setup`"


@dataclass(frozen=True)
class Launcher:
    """`~/.local/bin/mnm`: makenomistakes from any directory. `uv run` in another folder (say
    ~/odysseus, which has its own pyproject.toml) finds no `mnm` and creates a .venv there;
    `--project` pins the checkout. A file there that `mnm setup` did not write is left alone."""

    lines: tuple[str, ...]
    path: Path = LAUNCHER
    executable: Callable[[Path], bool] = lambda p: os.access(p, os.X_OK)
    search_path: str = field(default_factory=lambda: os.environ.get("PATH", ""))

    @classmethod
    def for_checkout(cls) -> Launcher | None:
        try:
            root, uv = repo_root(), uv_binary()
        except ServiceError:
            return None  # a wheel install puts `mnm` on PATH itself
        return cls(launcher_lines(root, uv))

    @property
    def content(self) -> str:
        return "\n".join(self.lines) + "\n"

    def commands(self) -> list[Command]:
        # printf repeats its format per argument; "$0" is the target (sh -c's first operand).
        write = ["sh", "-c", 'printf "%s\\n" "$@" > "$0"', str(self.path), *self.lines]
        return [["mkdir", "-p", str(self.path.parent)], write, ["chmod", "755", str(self.path)]]

    def check(self, read_text: FileReader) -> tuple[bool, str]:
        try:
            current = read_text(self.path)
        except OSError:
            return False, f"{self.path} missing"
        if current != self.content:
            return False, f"{self.path} points at another checkout"
        if not self.executable(self.path):
            return False, f"{self.path} is not executable"
        on_path = str(self.path.parent) in self.search_path.split(os.pathsep)
        return True, str(self.path) if on_path else f"{self.path} (not on PATH in this shell yet)"

    def step(self, read_text: FileReader) -> Step:
        try:
            foreign = not read_text(self.path).startswith(f"#!/bin/sh\n{_MARKER}")
        except OSError:
            foreign = False
        if foreign:
            return Step(
                "mnm command",
                check=lambda: (False, f"{self.path} exists and was not written by `mnm setup`"),
                manual=f"Left as is. Remove or rename {self.path}, then run `mnm setup` again; "
                "until then use `uv run mnm` inside the checkout.",
            )
        return Step(
            "mnm command",
            check=lambda: self.check(read_text),
            fix=self.commands(),
            after="Open a new terminal if `mnm` is not found yet (Ubuntu adds ~/.local/bin "
            "to PATH at login once it exists).",
        )


def launcher_lines(root: Path, uv: str) -> tuple[str, ...]:
    return (
        "#!/bin/sh",
        f"{_MARKER}: runs makenomistakes from its checkout, {root}.",
        f'exec {shlex.quote(uv)} run --quiet --project {shlex.quote(str(root))} mnm "$@"',
    )


def secrets_file_commands() -> list[Command]:
    content = (
        "# makenomistakes secrets. Never commit this file.\\n"
        "# HF_TOKEN=hf_...   uncomment and fill in for gated or private Hugging Face repos\\n"
        "# CIVITAI_TOKEN=...   uncomment and fill in for Civitai downloads that need a login\\n"
    )
    return [
        ["mkdir", "-p", str(SECRETS_FILE.parent)],
        ["chmod", "700", str(SECRETS_FILE.parent)],
        # an existing file (a token in it, or a 0644 copy of .env.example) is kept: chmod only
        ["sh", "-c", f"[ -e '{SECRETS_FILE}' ] || printf '{content}' > '{SECRETS_FILE}'"],
        ["chmod", "600", str(SECRETS_FILE)],
    ]


Echo = Callable[[str], None]


Confirm = Callable[[str], bool]


def run_setup(
    steps: list[Step],
    runner: CommandRunner,
    dry_run: bool,
    echo: Echo | None = None,
    confirm: Confirm | None = None,
) -> list[StepResult]:
    """`confirm` answers a step's question (the GPU driver); without one such a step is not
    run and its commands are printed instead."""
    results: list[StepResult] = []
    outcomes: dict[str, Outcome] = {}
    for step in steps:
        result = _run_step(step, runner, dry_run, outcomes, echo, confirm)
        outcomes[step.name] = result.outcome
        results.append(result)
    return results


def _run_step(  # noqa: PLR0911 - one return per outcome
    step: Step,
    runner: CommandRunner,
    dry_run: bool,
    outcomes: dict[str, Outcome],
    echo: Echo | None = None,
    confirm: Confirm | None = None,
) -> StepResult:
    commands = [shlex.join(c) for c in step.fix]  # each one pastes into a shell as printed
    blocked = {Outcome.FAILED, Outcome.SKIPPED}
    if step.requires is not None and outcomes.get(step.requires) in blocked:
        return StepResult(
            name=step.name, outcome=Outcome.SKIPPED, detail=f"needs '{step.requires}' first"
        )
    ok, detail = step.check()
    if ok:
        return StepResult(name=step.name, outcome=Outcome.OK, detail=detail)
    if step.manual:
        return StepResult(
            name=step.name,
            outcome=Outcome.MANUAL,
            detail=detail,
            note=step.manual,
            reboot=step.reboot,
        )
    if dry_run:
        return StepResult(name=step.name, outcome=Outcome.PLANNED, detail=detail, commands=commands)
    if step.confirm is not None and (confirm is None or not confirm(step.confirm)):
        return StepResult(
            name=step.name,
            outcome=Outcome.MANUAL,
            detail=detail,
            commands=commands,
            note="Not installed: run `mnm setup --install-driver`, or the commands above "
            "yourself, then reboot.",
        )
    failed = _run_all(step.fix, runner, echo)
    if failed is not None:
        return StepResult(
            name=step.name,
            outcome=Outcome.FAILED,
            detail=f"command failed: {failed}",
            commands=commands,
        )
    return _rechecked(step, commands)


def _rechecked(step: Step, commands: list[str]) -> StepResult:
    """What is there now ("…/mnm" rather than "…/mnm missing"). A re-check that still fails
    is a failure, unless the step only waits for a re-login (the docker group)."""
    fixed, after = step.check()
    if not fixed and not step.settles_at_relogin():
        return StepResult(
            name=step.name,
            outcome=Outcome.FAILED,
            detail=f"the commands ran, but the check still fails: {after}",
            commands=commands,
        )
    return StepResult(
        name=step.name,
        outcome=Outcome.FIXED,
        detail="installed; it loads at the next boot" if step.reboot else after,
        commands=commands,
        note=step.after,
        reboot=step.reboot,
    )


def _run_all(commands: list[Command], runner: CommandRunner, echo: Echo | None) -> str | None:
    """Each command is printed before it runs (the user meets sudo's prompt knowing why)."""
    for argv in commands:
        if echo is not None:
            echo(shlex.join(argv))
        code, _, err = runner.run(argv, timeout=FIX_TIMEOUT)
        if code != 0:
            return f"{' '.join(argv)} ({err.strip() or f'exit {code}'})"
    return None


def current_user() -> str:
    """The login user, even under `sudo mnm setup` (never chown /data/models to root)."""
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and sudo_user != "root":
        return sudo_user
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "root"


def _secrets_file_check() -> tuple[bool, str]:
    """There, and readable by its owner only (mode 600)."""
    if not SECRETS_FILE.exists():
        return False, f"{SECRETS_FILE} is missing"
    mode = SECRETS_FILE.stat().st_mode & 0o777
    if mode != 0o600:
        return False, f"{SECRETS_FILE} is mode {mode:o}, not 600"
    return True, str(SECRETS_FILE)


def write_hf_token(path: Path, token: str) -> None:
    """Store HF_TOKEN in the secrets file (mode 600), replacing any placeholder or old value."""
    write_setting(path, "HF_TOKEN", token)


def write_setting(path: Path, name: str, value: str) -> None:
    """One `NAME=value` line in the secrets file (mode 600, its folder 700), replacing the
    name's old line or commented placeholder; every other line is kept as it was. Settings
    (`MNM_MODELS_DIR`, `MNM_ROUTER_HOST`) live there beside the tokens: the services and a
    terminal read the same file."""
    kept: list[str] = []
    if path.exists():
        kept = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith((f"{name}=", f"# {name}="))
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.exists():
        path.chmod(0o600)  # before the value goes in, never after
    content = "\n".join([*kept, f"{name}={value}"]) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(0o600)


def looks_like_hf_token(token: str) -> bool:
    return token.startswith("hf_") and len(token) > 10 and " " not in token
