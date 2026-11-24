"""`mnm apps connect odysseus`: Odysseus is the user's own install, never mnm's, so mnm only
links it: the `.env` lines its card gives (`EMBEDDING_URL`, `EMBEDDING_MODEL`) merged into
`~/odysseus/.env` by `files.write` (a `.before-mnm` copy kept), shown as a diff before
anything is written, then `docker compose up -d` in its folder, which
recreates only the containers whose settings changed. The endpoints and the MCP Tool Server
are forms in Odysseus's Settings (it reads them from its database, not a file): the card
prints them. Goose's connect is `mnm url --for goose --write` (its one file).

`write_card_file` and `apply_odysseus` are the one path the CLI (`mnm url --write`,
`mnm apps connect`), the guided setup and the web's Get started all write through."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from makenomistakes import report
from makenomistakes.apps.card import AppCard, AppFile
from makenomistakes.apps.files import AppFileError, Ledger, WriteResult, merged_text, shown_diff
from makenomistakes.apps.files import write as write_file
from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import App
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import CommandRunner

COMPOSE_NAMES = ("compose.yaml", "compose.yml", "docker-compose.yml", "docker-compose.yaml")


@dataclass(frozen=True)
class Preview:
    file: AppFile
    path: Path
    diff: str  # empty: the file already says it


def odysseus_folder(given: Path | None = None) -> Path:
    return (given or Path.home() / "odysseus").expanduser()


def preview(card: AppCard, folder: Path, secrets: list[tuple[str, str]]) -> Preview:
    """What connecting would change in Odysseus's `.env`; AppsError when there is nothing to
    connect (no Odysseus there, or no embedding model to name)."""
    if not folder.is_dir():
        raise AppsError(
            f"No Odysseus at {folder}: mnm never installs it. Install it as its README says, "
            "then connect it; `--dir` names another folder."
        )
    if not card.files:
        raise AppsError(
            "Odysseus's .env gets nothing from mnm without an embedding model: add one and "
            "make it the desk's memory. Its endpoints and MCP entry are below to enter."
        )
    file = card.files[0]
    path = folder / ".env"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    new = merged_text(file, old, path) if old else file.text()
    return Preview(file, path, shown_diff(old, new, path, secrets) if new != old else "")


def compose_file(folder: Path) -> Path:
    found = next((folder / n for n in COMPOSE_NAMES if (folder / n).is_file()), None)
    if found is None:
        raise AppsError(f"{folder} holds no compose file: Odysseus was not restarted.")
    return found


def restart(folder: Path, runner: CommandRunner) -> str:
    """`docker compose up -d` in Odysseus's folder: its containers take the new `.env`."""
    argv = [
        "docker",
        "compose",
        "--project-directory",
        str(folder),
        "-f",
        str(compose_file(folder)),
    ]
    code, out, err = runner.run([*argv, "up", "-d"], timeout=600.0)
    if code != 0:
        raise AppsError(f"`docker compose up -d` in {folder} failed: {(err or out).strip()}")
    return f"Odysseus restarted (`docker compose up -d` in {folder})."


def write_card_file(
    card: AppCard, settings: Settings, force: bool = False, to: Path | None = None
) -> WriteResult:
    """The card's one file at its own place (or `to`), by `files.write`'s rules: merged into
    the user's, a `.before-mnm` copy kept, refused where it cannot merge without `force`.
    AppFileError says why nothing was written."""
    if card.no_write:
        raise AppFileError(card.no_write)
    if not card.files:
        raise AppFileError(f"{card.label} has no file to write now (the card says why).")
    file = card.files[0]
    target = to or file.default
    if target is None:
        raise AppFileError(f"{file.label} has no place of its own: `--write --to PATH`.")
    known = report.secrets(settings).known  # hidden in a merge's diff
    ledger = Ledger(settings.expanded_state_dir())
    try:
        return write_file(file, target, card.app, ledger, force, known)
    except OSError as exc:
        raise AppFileError(f"{target}: {exc.strerror or exc}") from exc


SAID = {
    "wrote": "Wrote {path}.",
    "replaced": "Replaced {path} (mnm's own file).",
    "unchanged": "{path} already says this: nothing written.",
    "merged": "Merged mnm's part into {path}; the file from before mnm first changed it is "
    "{backup}.",
    "overwrote": "Replaced {path} (--force); the file from before mnm first changed it is "
    "{backup} (its lines are not shown: they are yours).",
}


def written_line(done: WriteResult) -> str:
    """One line for what a write did (the CLI prints a merge's diff after it)."""
    return SAID[done.action].format(path=done.path, backup=done.backup)


def apply_odysseus(
    seen: Preview, folder: Path, settings: Settings, runner: CommandRunner, force: bool = False
) -> list[str]:
    """Write the previewed `.env` lines, then `docker compose up -d` in Odysseus's folder; the
    lines said. AppsError when either step fails (the `.env` stays written then)."""
    known = report.secrets(settings).known
    ledger = Ledger(settings.expanded_state_dir())
    try:
        done = write_file(seen.file, seen.path, App.ODYSSEUS, ledger, force, known)
    except AppFileError as exc:
        raise AppsError(str(exc)) from exc
    except OSError as exc:
        raise AppsError(f"{seen.path}: {exc.strerror or exc}") from exc
    return [written_line(done), restart(folder, runner)]
