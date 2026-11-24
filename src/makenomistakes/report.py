"""`mnm report`: everything a diagnostic report needs, in one folder of the current one (the
journals, engine logs, doctor, ls, engines and calibration). Every secret value from the
secrets file, and anything shaped like a Hugging Face token, is replaced before a file is
written."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

from makenomistakes import config
from makenomistakes.config import TOKEN_NAMES, Settings
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.links import NAME_PART
from makenomistakes.private_ids import without_card_ids
from makenomistakes.service import ROUTER_UNIT, UNIT_NAME

JOURNAL_LINES = 3000
SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD", re.IGNORECASE)
SECRET_SETTINGS = frozenset({"MNM_NTFY_URL"})  # secret by what they hold, not their name
HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")
# A MAC address identifies the user's hardware, as a card's uuid does (`private_ids`):
# secrets like a token wherever a report or a log leaves the box.
MAC = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])")
BEARER = re.compile(r"(Authorization:\s*Bearer\s+)\S+", re.IGNORECASE)
API_KEY_HEADER = re.compile(r"(x-api-key:\s*)\S+", re.IGNORECASE)
# a query with any parameter whose name is, or has a `_`/`-` part that is, a secret word
# (`access_token`, `api_key`, `X-Amz-Signature`, `Expires`; never `designId`): the query goes
SIGNED_URL = re.compile(
    rf"(https?://[^\s?\"'<>]+)\?(?=(?:[^\s\"'<>]*&)?{NAME_PART}=)[^\s\"'<>]*",
    re.IGNORECASE,
)
SHORTEST = 4  # a shorter value would be replaced inside ordinary words
MNM = [sys.executable, "-m", "makenomistakes"]
WIDE = {"COLUMNS": "160", "NO_COLOR": "1", "TERM": "dumb"}


@dataclass(frozen=True)
class Item:
    file: str  # its name in the report folder
    argv: list[str]


def items(instances: list[str]) -> list[Item]:
    """What the report holds, in the order it is collected."""
    journal = ["journalctl", "--user", "-n", str(JOURNAL_LINES), "--no-pager", "-u"]
    fixed = [
        Item("version.txt", [*MNM, "--version"]),
        Item("doctor.txt", [*MNM, "doctor"]),
        Item("ls.txt", [*MNM, "ls"]),
        Item("machine.txt", [*MNM, "machine", "show"]),  # The box and what changed
        Item("engines.txt", [*MNM, "engines"]),
        Item("calibration.txt", [*MNM, "calibration"]),
        Item("calibration-export.txt", [*MNM, "calibration", "export"]),
        Item("outputs.txt", [*MNM, "outputs", "ls"]),
        Item("backups.txt", [*MNM, "backup", "ls"]),  # The folder and the last test
        Item("nightly-ls.txt", [*MNM, "nightly", "ls"]),  # Every night's word
        Item("nightly.txt", [*MNM, "nightly", "show"]),  # The last night's page
        Item("sets.txt", [*MNM, "sets", "ls"]),
        Item("comfyui-status.txt", [*MNM, "comfyui", "status"]),
        Item("comfyui-log.txt", [*MNM, "comfyui", "logs", "--tail", "5000"]),
        Item("mcp-tools.txt", [*MNM, "mcp", "tools"]),
        # The studio: its images, stages, songs and libraries
        Item("studio-status.txt", [*MNM, "studio", "status"]),
        Item("studio-stages.txt", [*MNM, "studio", "stage"]),
        Item("studio-trainer.txt", [*MNM, "studio", "trainer", "status"]),
        Item("studio-style-trainer.txt", [*MNM, "studio", "trainer", "status", "--for", "styles"]),
        Item("studio-trainings.txt", [*MNM, "studio", "train", "ls"]),
        Item("songs.txt", [*MNM, "song", "ls"]),
        Item("voices.txt", [*MNM, "voices", "ls"]),
        Item("styles.txt", [*MNM, "studio", "styles", "ls"]),
        Item("service-status.txt", [*MNM, "service", "status"]),
        Item("nvidia-smi.txt", ["nvidia-smi"]),
        Item("journal-web.txt", [*journal, UNIT_NAME]),
        Item("journal-router.txt", [*journal, ROUTER_UNIT]),
    ]
    logs = [Item(f"logs/{name}.txt", [*MNM, "logs", name, "--tail", "0"]) for name in instances]
    return fixed + logs


@dataclass(frozen=True)
class Secrets:
    known: list[tuple[str, str]]  # (name, value), each replaced by its name
    too_short: list[str]  # names whose value is too short to replace safely: said, not scrubbed


def secrets(settings: Settings, path: Path | None = None) -> Secrets:
    """Every secret mnm knows: the tokens and key as `Settings` reads them (the environment and
    the secrets file both), every token name in the environment, and the file's other names
    that look secret (TOKEN, KEY, SECRET, PASSWORD). The file is `config.SECRETS_FILE` as it
    is when called, never a default bound at import."""
    path = config.SECRETS_FILE if path is None else path
    found: dict[str, str] = {}
    for name, value in (
        ("HF_TOKEN", settings.hf_token_value()),
        ("CIVITAI_TOKEN", settings.civitai_token_value()),
        ("MNM_API_KEY", settings.api_key_value()),
        # whoever knows an ntfy topic's URL reads every notice sent to it (`mnm report`'s
        # nightly.txt would carry it)
        ("MNM_NTFY_URL", settings.ntfy_url or None),
    ):
        if value:
            found[name] = value
    for names in TOKEN_NAMES.values():
        for name in names:
            if value := os.environ.get(name):
                found.setdefault(name, value)
    try:
        listed = dotenv_values(path)
    except OSError:
        listed = {}
    for name, value in listed.items():
        if value and (SECRET_NAME.search(name) or name in SECRET_SETTINGS):
            found.setdefault(name, value)
    pairs = sorted(found.items())
    return Secrets(
        known=[(n, v) for n, v in pairs if len(v) >= SHORTEST],
        too_short=[n for n, v in pairs if len(v) < SHORTEST],
    )


def redact(text: str, known: list[tuple[str, str]], hardware: bool = True) -> str:
    """Each known secret's value replaced by its name, then whatever only looks like one: a
    bearer token or an x-api-key a log printed, a Hugging Face token, and the query of a
    signed download link (Civitai's CDN carries `token=` and `signature=`), and, with
    `hardware` (the default: whatever leaves the box), a card's uuid (an AMD card's unique id
    too) and a MAC address. A backup passes `hardware=False`: it stays on the box, and a
    restored `machine.json` must still match the cards it names."""
    for name, value in sorted(known, key=lambda pair: -len(pair[1])):
        text = text.replace(value, f"<{name} removed>")
    text = BEARER.sub(r"\1<removed>", text)
    text = API_KEY_HEADER.sub(r"\1<removed>", text)
    text = SIGNED_URL.sub(r"\1?<query removed>", text)
    if hardware:
        text = without_card_ids(text)
        text = MAC.sub("<mac removed>", text)
    return HF_TOKEN.sub("<hf token removed>", text)


class FolderRunner:
    """Each command run inside the report folder (`calibration export` writes there), wide
    and plain (no colour codes in a text file)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def run(self, argv: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
        try:
            done = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
                cwd=self.folder,
                env={**os.environ, **WIDE},
            )
        except FileNotFoundError:
            return 127, "", f"{argv[0]}: not found"
        except subprocess.TimeoutExpired:
            return 124, "", f"{argv[0]}: timed out after {timeout:.0f}s"
        return done.returncode, done.stdout, done.stderr


@dataclass(frozen=True)
class Written:
    folder: Path
    files: list[tuple[str, int]]  # (name, exit code)


def report_folder(where: Path, now: Callable[[], datetime] = datetime.now) -> Path:
    folder = where / f"report-{now():%Y%m%d-%H%M%S}"
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    return folder


def write_report(
    folder: Path,
    instances: list[str],
    runner: CommandRunner,
    known: list[tuple[str, str]],
    said: Callable[[str], None] = lambda _line: None,
    kept: list[Path] | None = None,
) -> Written:
    """Run each item, redact, write; a failing item is written with its error and the rest
    go on. Files a command left in the folder (the calibration export) are redacted too, and
    so are the `kept` files copied in (`mnm machine adopt`'s tables, if one ran)."""
    done: list[tuple[str, int]] = []
    for item in items(instances):
        said(item.file)
        code, out, err = runner.run(item.argv, timeout=180.0)
        shown = ["mnm", *item.argv[len(MNM) :]] if item.argv[: len(MNM)] == MNM else item.argv
        body = f"$ {' '.join(shown)}\n{out}"
        if err.strip():
            body += f"\n--- stderr ---\n{err}"
        body += f"\n[exit {code}]\n"
        (folder / item.file).write_text(redact(body, known), encoding="utf-8")
        done.append((item.file, code))
    for left in folder.glob("mnm-calibration-*.json"):
        left.write_text(redact(left.read_text(encoding="utf-8"), known), encoding="utf-8")
        done.append((left.name, 0))
    for path in kept or []:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            text = f"{path.name} could not be read: {exc}\n"
        (folder / path.name).write_text(redact(text, known), encoding="utf-8")
        done.append((path.name, 0))
    return Written(folder, done)
