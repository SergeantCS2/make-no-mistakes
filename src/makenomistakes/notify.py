"""Notifications: a word when something long ends, so the user need not watch a terminal.

Two channels, each optional: an ntfy topic (`MNM_NTFY_URL`: ntfy.sh, or a server of your own)
and, under WSL, a Windows toast through `powershell.exe`
(`MNM_TOAST`, on there by default, off elsewhere). Sent when `mnm run`, a ComfyUI workflow
run, a film's steps, `mnm add`'s download, `mnm engines update` and `mnm comfyui install`
end, and when a start fails, from the CLI and the web UI's jobs alike.

Best effort: a notice is sent off a thread with a 5 s timeout and never raises; what failed
is said only by `mnm notify`, the test. ntfy's headers are ASCII (Odysseus issue #5207 was
a non-ASCII title, which ntfy refuses); the message body is UTF-8 as it is.
"""

from __future__ import annotations

import base64
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from makenomistakes.config import Settings
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import CommandRunner, SubprocessRunner
from makenomistakes.jobs import Job, JobKind, JobState

TIMEOUT_SECONDS = 5.0
LONG_SECONDS = 20.0  # what ends sooner was watched as it happened: no notice for it
TITLE_LIMIT = 120
MESSAGE_LIMIT = 2000
# PowerShell's own app id: Windows shows a toast only for an app id it knows (an unregistered
# one such as "mnm" is dropped without a word), and every Windows has this one.
TOAST_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
# The script never holds a notice's text: an engine's error is not PowerShell's to read
# (PowerShell also ends a single-quoted string at U+2018-U+201B, so a typographic
# apostrophe would end it and what follows would run). Title and message arrive as base64
# of their UTF-8, only [A-Za-z0-9+/=], and the script goes whole as `-EncodedCommand`
# (UTF-16LE base64), so WSL interop's re-quoting of the argv never sees a quote either.
_DECODE = "[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{}'))"
_TOAST = (
    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
    "ContentType = WindowsRuntime] > $null; "
    "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
    "[Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
    "$x = $t.GetElementsByTagName('text'); "
    "$x.Item(0).AppendChild($t.CreateTextNode({title})) > $null; "
    "$x.Item(1).AppendChild($t.CreateTextNode({message})) > $null; "
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{app}')"
    ".Show([Windows.UI.Notifications.ToastNotification]::new($t))"
)


@dataclass(frozen=True)
class Notice:
    title: str  # what ended: "mnm run qwen3-8b"
    message: str  # how: "done in 41 s", or the error's own words
    failed: bool = False

    @property
    def priority(self) -> str:
        return "high" if self.failed else "default"

    @property
    def tags(self) -> tuple[str, ...]:
        return ("warning",) if self.failed else ("white_check_mark",)


def ascii_text(text: str, limit: int = TITLE_LIMIT) -> str:
    """One header-safe line: accents dropped to their letters (`é` → `e`), anything else
    outside printable ASCII left out, whitespace (a line break included) folded."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    line = "".join(c for c in " ".join(plain.split()) if c.isprintable())
    return line if len(line) <= limit else line[: limit - 1] + "~"


def toast_on(settings: Settings) -> bool:
    """`MNM_TOAST`, else on under WSL (Windows shows it) and off on a native box."""
    if settings.toast:
        return settings.toast == "on"
    return platforms.current().is_wsl


def toast_script(notice: Notice) -> str:
    """The PowerShell that shows a notice as a Windows toast (ToastText02: a bold title, a
    line under it); its texts only as base64 the script decodes (`_DECODE`)."""

    def carried(text: str, limit: int) -> str:
        line = " ".join(text.split())[:limit]  # one line: whitespace and breaks folded
        return _DECODE.format(base64.b64encode(line.encode("utf-8")).decode("ascii"))

    return (
        _TOAST.replace("{title}", carried(notice.title, TITLE_LIMIT))
        .replace("{message}", carried(notice.message, 300))
        .replace("{app}", TOAST_APP_ID)
    )


def toast_argv(notice: Notice) -> list[str]:
    """`powershell.exe -EncodedCommand`: the script as base64 of its UTF-16LE, PowerShell's
    own way to take a script that no quoting can break."""
    encoded = base64.b64encode(toast_script(notice).encode("utf-16-le")).decode("ascii")
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


@dataclass
class Notifier:
    """Sends a notice on every channel the settings turn on."""

    settings: Settings
    client: Callable[[], httpx.Client] = field(
        default=lambda: httpx.Client(timeout=TIMEOUT_SECONDS)  # ntfy.sh is out there: env proxy
    )
    runner: CommandRunner = field(default_factory=SubprocessRunner)

    def channels(self) -> list[str]:
        """The channels on, in words."""
        on = [f"ntfy ({self.settings.ntfy_url})"] if self.settings.ntfy_url else []
        return on + (["a Windows toast"] if toast_on(self.settings) else [])

    def send(self, notice: Notice) -> list[str]:
        """Each channel's outcome in words (`mnm notify` prints them); never raises."""
        said = []
        if self.settings.ntfy_url:
            said.append(self._ntfy(notice))
        if toast_on(self.settings):
            said.append(self._toast(notice))
        return said

    def post(self, notice: Notice) -> None:
        """`send` off a thread: what ended goes on at once."""
        if self.channels():
            threading.Thread(target=self.send, args=(notice,), daemon=True).start()

    def _ntfy(self, notice: Notice) -> str:
        url = self.settings.ntfy_url
        headers = {
            "X-Title": ascii_text(notice.title),
            "X-Priority": notice.priority,
            "X-Tags": ",".join(notice.tags),
        }
        token = self.settings.ntfy_token_value()
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        body = notice.message[:MESSAGE_LIMIT].encode("utf-8")
        try:
            with self.client() as http:
                answer = http.post(url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            return f"ntfy: {url} did not answer ({type(exc).__name__}: {exc})"
        if answer.status_code in (401, 403):
            return (
                f"ntfy: {url} answered {answer.status_code}: the topic wants a token "
                "(MNM_NTFY_TOKEN in the secrets file) or this one is wrong"
            )
        if answer.is_error:
            return f"ntfy: {url} answered {answer.status_code}: {answer.text[:200].strip()}"
        return f"ntfy: sent to {url}"

    def _toast(self, notice: Notice) -> str:
        code, out, err = self.runner.run(toast_argv(notice), timeout=TIMEOUT_SECONDS * 3)
        if code == 0:
            return "toast: shown on Windows"
        if code == 127:
            return (
                "toast: powershell.exe is not reachable from here (WSL's Windows interop is "
                "off?); `mnm notify --toast off` turns the toast off"
            )
        return f"toast: powershell.exe exited {code}: {(err or out).strip()[:200]}"


def notifier(settings: Settings | None = None) -> Notifier:
    return Notifier(settings or Settings())


# The web UI's jobs that tell when they end: what the user waits for. A start only when it
# fails (a start that worked is seen on the page that asked for it).
TOLD = frozenset({JobKind.DOWNLOAD, JobKind.PIPELINE, JobKind.RUN, JobKind.ENGINES})


def job_notice(job: Job) -> Notice | None:
    """The notice a finished web job sends, or None (short, or not one the user waits on)."""
    if job.finished_at is None or job.seconds is None:
        return None
    took = job.seconds  # monotonic: the wall clock can step back mid-job
    failed = job.state is JobState.FAILED
    if took < LONG_SECONDS or not (job.kind in TOLD or (failed and job.kind is JobKind.START)):
        return None
    title = f"mnm {job.title}"
    if failed:
        return Notice(title, f"failed after {took:.0f} s: {job.error or 'see the web UI'}", True)
    return Notice(title, f"done in {took:.0f} s")


def job_notices(settings: Settings) -> Callable[[Job], None]:
    """`JobRegistry`'s `on_finish` for the web UI: each notice off a thread."""
    sender = Notifier(settings)

    def tell(job: Job) -> None:
        notice = job_notice(job)
        if notice is not None:
            sender.post(notice)

    return tell
