"""Section 2 of the guided setup: where models live and where runs put what they make;
`folder_problem` is also how section 8 checks the projects folder."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from makenomistakes.config import is_system_folder
from makenomistakes.setup_view import Mark

if TYPE_CHECKING:
    from makenomistakes.setup_guide import Guide

GIB = 1024**3
LOW_DISK_GIB = 50


def models_folder(guide: Guide) -> None:
    guide.view.section(
        2,
        "Where models live",
        "Downloads go here (the Hugging Face cache layout); plan 100 GB for a full set.",
    )
    current = guide.settings.expanded_models_dir()
    chosen = current
    for _ in range(3):
        answer = guide.prompt.text("Models folder", str(current))
        problem, path = folder_problem(answer)
        if problem is None and path is not None:
            chosen = path
            break
        guide.view.say(problem or "", style="yellow")
    free = _free_gib(chosen)
    guide.view.say(
        f"{chosen}: {free:.0f} GiB free" if free is not None else f"{chosen}: free space unknown"
    )
    if free is not None and free < LOW_DISK_GIB:
        guide.view.say(f"Under {LOW_DISK_GIB} GiB free: a picture model alone is ~10 GB.", "yellow")
    if chosen != current:
        guide.settings = guide.settings.model_copy(update={"models_dir": chosen})
        guide.save("MNM_MODELS_DIR", str(chosen))
    guide.summary.add(Mark.DONE, "Models folder", f"{chosen} ({_gib(free)} free)")
    outputs_folder(guide)


def outputs_folder(guide: Guide) -> None:
    """Where Runs, pipelines and workflows put what they make."""
    current = guide.settings.expanded_outputs_dir()
    chosen = current
    for _ in range(3):
        answer = guide.prompt.text("Outputs folder (pictures, clips, films)", str(current))
        problem, path = folder_problem(answer)
        if problem is None and path is not None:
            chosen = path
            break
        guide.view.say(problem or "", style="yellow")
    if chosen != current:
        guide.settings = guide.settings.model_copy(update={"outputs_dir": chosen})
        guide.save("MNM_OUTPUTS_DIR", str(chosen))
    guide.summary.add(Mark.DONE, "Outputs folder", f"{chosen} (`mnm clean` empties it)")
    apps_folder(guide)


def apps_folder(guide: Guide) -> None:
    """Where LibreChat and OpenHands keep their data; the
    prerequisites (section 4) make it and give it to you, as they do the models folder."""
    current = guide.settings.expanded_apps_dir()
    chosen = current
    for _ in range(3):
        answer = guide.prompt.text("Apps folder (LibreChat's and OpenHands' data)", str(current))
        problem, path = folder_problem(answer)
        if problem is None and path is not None:
            chosen = path
            break
        guide.view.say(problem or "", style="yellow")
    if chosen != current:
        guide.settings = guide.settings.model_copy(update={"apps_dir": chosen})
        guide.save("MNM_APPS_DIR", str(chosen))
    guide.summary.add(Mark.DONE, "Apps folder", f"{chosen} (`mnm apps uninstall` empties it)")
    backups_folder(guide)


def backups_folder(guide: Guide) -> None:
    """Where `mnm backup --auto` keeps its archives. Not asked (one question fewer):
    the prerequisites make it like the others, and MNM_BACKUPS_DIR in the secrets file moves
    it."""
    folder = guide.settings.expanded_backups_dir()
    guide.summary.add(
        Mark.DONE,
        "Backups folder",
        f"{folder} (`mnm backup schedule --at 03:30` backs up every night; MNM_BACKUPS_DIR "
        "moves it)",
    )


def folder_problem(answer: str) -> tuple[str | None, Path | None]:
    path = Path(answer.strip()).expanduser()
    if not answer.strip() or not path.is_absolute():
        return "Give a full path, such as /data/models or ~/models.", None
    if is_system_folder(path):
        return f"{path} is a system folder; pick a folder of its own, such as /data/models.", None
    if path.exists() and not path.is_dir():
        return f"{path} is a file, not a folder.", None
    return None, path


def _free_gib(path: Path) -> float | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / GIB
    except OSError:
        return None


def _gib(value: float | None) -> str:
    return "?" if value is None else f"{value:.0f} GiB"
