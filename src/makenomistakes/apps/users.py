"""`mnm apps user add librechat EMAIL`: an account in mnm's LibreChat, whose
registration is off. LibreChat's own script makes it (`npm run create-user -- <email> <name>
<username> --email-verified=true` inside the `api` container; librechat.ai/docs/configuration/
authentication and config/create-user.js at v0.8.7, read 2026-10-23). The password is never on
a command line (where `ps` shows it, on the host and in the container): the script asks for
whatever it was not given, and reads the answer from its stdin, which mnm writes. mnm never
keeps it."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from typing import Protocol

from makenomistakes.apps.ledger import AppsError
from makenomistakes.apps.names import CONTAINERS, App

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
MIN_PASSWORD = 8  # LibreChat's own minimum (its registration schema)
ASKS = 3  # a password that does not match, or is too short, is asked again this often


class Asks(Protocol):
    """The part of the setup's `Prompter` a password needs (`setup_prompt.TerminalPrompter`
    and any stand-in)."""

    def text(self, question: str, default: str, secret: bool = False) -> str: ...


def ask_password(prompt: Asks, say: Callable[[str], None]) -> str | None:
    """The password asked twice, up to `ASKS` tries; None when every try failed
    (`mnm apps user add` and the setup's first account share it). The first answer is checked
    before it is asked again (else a 4-digit password is asked twice, then refused as
    "under 8 characters"): an empty one is said as empty, a short one at once."""
    for left in range(ASKS - 1, -1, -1):
        password = prompt.text(f"Its password ({MIN_PASSWORD} characters or more)", "", True)
        if not password:
            problem = "Nothing was typed"
        elif len(password) < MIN_PASSWORD:
            problem = f"That password is under {MIN_PASSWORD} characters"
        elif prompt.text("The password again", "", True) != password:
            problem = "The two passwords did not match"
        else:
            return password
        say(f"{problem}; {f'again ({left} more)' if left else 'no account made'}.")
    return None


class InputRunner(Protocol):
    def run_input(self, argv: list[str], stdin: str, timeout: float) -> tuple[int, str, str]: ...


class SubprocessInputRunner:
    def run_input(self, argv: list[str], stdin: str, timeout: float) -> tuple[int, str, str]:
        try:
            done = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return 127, "", f"{argv[0]}: not found"
        except subprocess.TimeoutExpired:
            return 124, "", f"{argv[0]}: timed out after {timeout:.0f}s"
        return done.returncode, done.stdout, done.stderr


def add_librechat_user(
    email: str, name: str | None, username: str | None, password: str, runner: InputRunner
) -> str:
    """The account, or AppsError with LibreChat's own reason."""
    if not EMAIL.match(email):
        raise AppsError(f"{email!r} is not an email address.")
    default = email.split("@", 1)[0]
    name = name or default
    username = username or re.sub(r"[^A-Za-z0-9._-]", "", default) or "owner"
    for label, value in (("name", name), ("username", username)):
        if not NAME.match(value):
            raise AppsError(f"The {label} {value!r} holds characters LibreChat's script splits on.")
    if len(password) < MIN_PASSWORD:
        raise AppsError(f"LibreChat wants a password of {MIN_PASSWORD} characters or more.")
    argv = [
        "docker",
        "exec",
        "-i",
        CONTAINERS[App.LIBRECHAT],
        "npm",
        "run",
        "create-user",
        "--",
        email,
        name,
        username,
        "--email-verified=true",
    ]
    code, out, err = runner.run_input(argv, password + "\n", timeout=120.0)
    said = _plain(out + "\n" + err)
    if code != 0 or "User created successfully" not in said:
        reason = next(
            (line for line in said.splitlines() if line.startswith("Error")),
            said.strip().splitlines()[-1] if said.strip() else f"exit {code}",
        )
        if "no such container" in said.lower() or "is not running" in said.lower():
            reason = "LibreChat is not running: `mnm apps start librechat` first"
        raise AppsError(f"LibreChat made no account: {reason}")
    return f"LibreChat account {email} (username {username}) made; sign in at the URL above."


def _plain(text: str) -> str:
    """The script's colours taken out."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)
