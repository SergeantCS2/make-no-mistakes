"""What the templates show for a state, a verdict or a link, decided once in Python: a word,
a glyph and a tone each (colour never carries a state alone), and the page a repository
lives on (Hugging Face, or Civitai for `civitai/<id>`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from markupsafe import Markup, escape

from makenomistakes.capacity import FitNow, Verdict
from makenomistakes.doctor_types import Check, Status
from makenomistakes.fit import FitLevel
from makenomistakes.hub.civitai import API_URL, civitai_page, is_civitai
from makenomistakes.hub.elsewhere import elsewhere
from makenomistakes.hub.facts import Modality
from makenomistakes.instance_types import InstanceStatus
from makenomistakes.jobs import Job, JobKind
from makenomistakes.router_bind import Exposure
from makenomistakes.service import SystemdState
from makenomistakes.supervisor import ContainerState, Health
from makenomistakes.tool_check import Verdict as ToolsVerdict


class Tone(StrEnum):
    """The CSS class suffix: ok bright, warn normal, busy blinking, dim, off faint, bad on the
    alert colour, inv inverse video."""

    OK = "ok"
    WARN = "warn"
    BUSY = "busy"
    DIM = "dim"
    OFF = "off"
    BAD = "bad"
    INV = "inv"


@dataclass(frozen=True)
class Badge:
    word: str
    glyph: str
    tone: Tone
    title: str = ""

    @property
    def css(self) -> str:
        return f"st st-{self.tone.value}" + (" st-warn" if self.tone is Tone.BUSY else "")


def state_badge(status: InstanceStatus, job: Job | None = None) -> Badge:
    """The State column: Docker's answer, the user's stop, the router's parking, and a
    job in progress (a start or a download says so before Docker or the files can)."""
    if status.error:
        return Badge("Docker unavailable", "?", Tone.BAD, status.error)
    if status.state is ContainerState.RUNNING:
        if status.health is Health.READY:
            return Badge("ready", "●", Tone.OK, "Running and answering /health")
        word = status.health.value if status.health else "unknown"
        return Badge(word, "◔", Tone.BUSY, "Running, not answering /health yet")
    if job is not None and job.running and job.kind in _JOB_BADGES:
        word, glyph = _JOB_BADGES[job.kind]
        return Badge(word, glyph, Tone.BUSY, job.message)
    return _idle_badge(status)


_JOB_BADGES = {JobKind.START: ("starting", "◔"), JobKind.DOWNLOAD: ("downloading", "↓")}


def _idle_badge(status: InstanceStatus) -> Badge:
    """No container running and no job for it: crashed, not downloaded, off, or parked."""
    if status.state is ContainerState.EXITED:
        return Badge(f"exited ({status.exit_code})", "✖", Tone.BAD, "Its engine stopped by itself")
    if status.evicted:
        why = "The disk policy deleted its files: the next start or request downloads them"
        return Badge("evicted", "↓", Tone.WARN, why)
    if not status.files_present:
        return Badge("not downloaded", "↓", Tone.WARN, "Start downloads the files first")
    if status.recipe.off:
        return Badge("stopped", "○", Tone.OFF, "Switched off: Start starts it again")
    return Badge("parked", "◐", Tone.DIM, "Not loaded: the next request loads it")


_VERDICTS = {  # the word is `Verdict.words` (a running one's says what it holds)
    FitNow.FITS: ("✓", Tone.OK),
    FitNow.NEAR_LIMIT: ("!", Tone.WARN),
    FitNow.NO_FIT: ("✕", Tone.BAD),
    FitNow.TAKES_TURNS: ("⇄", Tone.DIM),
    FitNow.ON_REQUEST: ("⇄", Tone.DIM),
    FitNow.RUNNING: ("▶", Tone.DIM),
    FitNow.UNKNOWN: ("?", Tone.OFF),
}


def verdict_badge(verdict: Verdict) -> Badge:
    glyph, tone = _VERDICTS[verdict.level]
    return Badge(verdict.words, glyph, tone, verdict.note)


_FITS = {
    FitLevel.COMFORTABLE: ("✓", Tone.OK),
    FitLevel.NO_FIT: ("✕", Tone.BAD),
}


def fit_badge(level: FitLevel) -> Badge:
    """A profile's fit on the add page: comfortable, no fit, or the levels between."""
    glyph, tone = _FITS.get(level, ("!", Tone.WARN))
    return Badge(level.value, glyph, tone)


_TOOLS = {  # `mnm try --tools`' verdict, the instance page's Tools line
    ToolsVerdict.WORKS: ("✓", Tone.OK, "Every step an agent depends on passed"),
    ToolsVerdict.PARTIAL: ("!", Tone.WARN, "A call works, but a step an agent depends on failed"),
    ToolsVerdict.NO: ("✕", Tone.BAD, "No step got a call an agent could run"),
}


def tools_badge(verdict: ToolsVerdict) -> Badge:
    glyph, tone, title = _TOOLS[verdict]
    return Badge(verdict.value, glyph, tone, title)


_CHECKS = {Status.OK: ("✓", Tone.OK), Status.WARN: ("!", Tone.INV), Status.FAIL: ("✕", Tone.BAD)}


def check_badge(check: Check) -> Badge:
    glyph, tone = _CHECKS[check.status]
    return Badge(check.status.value.upper(), glyph, tone)


def unit_badge(state: SystemdState) -> Badge:
    """A systemd user unit: no manager, not installed, or active/inactive and enabled."""
    if not state.available:
        return Badge("no systemd", "?", Tone.OFF, state.detail)
    if not state.installed:
        return Badge("not installed", "○", Tone.DIM)
    enabled = "enabled" if state.enabled else "not enabled"
    if state.active:
        return Badge("active", "●", Tone.OK, enabled)
    return Badge("inactive", "✕", Tone.BAD, enabled)


def meter_tone(percent_used: int) -> str:
    """The VRAM meter's class: past 60% it warns, past 85% it is nearly full."""
    if percent_used >= 85:
        return "bad"
    return "warn" if percent_used >= 60 else ""


EXPOSURE_WORDS = {
    Exposure.EVERYWHERE: "every network this machine is on",
    Exposure.LOCAL: "this machine and Docker's bridge only",
    Exposure.LISTED: "the listed addresses",
}


def repo_page(repo_id: str, version: str | None = None) -> str:
    """Where a repository lives: Civitai's model page (at a version when known) or the Hub."""
    if is_civitai(repo_id):
        if version and version != "main":
            return civitai_page(repo_id, version)
        return f"{API_URL}/models/{repo_id.removeprefix('civitai/')}"
    if (where := elsewhere(repo_id)) is not None:
        return where.page(repo_id, version or "latest")
    return f"https://huggingface.co/{repo_id}"


def repo_site(repo_id: str) -> str:
    if (where := elsewhere(repo_id)) is not None:
        return where.name
    return "Civitai" if is_civitai(repo_id) else "Hugging Face"


def percent(part: int | None, whole: int | None) -> int:
    if not part or not whole:
        return 0
    return max(0, min(100, round(100 * part / whole)))


# What a test of each type brings back (`smoke_replies.summarise`), shown before the first run.
_TRY_SHOWS = {
    Modality.CHAT: "The reply, its time and tok/s show here.",
    Modality.CODING: "The reply, its time and tok/s show here.",
    Modality.VISION_CHAT: "The reply, its time and tok/s show here.",
    Modality.EMBEDDING: "The vectors' count and size, and the time, show here.",
    Modality.RERANK: "The two scores and the time show here.",
    Modality.IMAGE_GEN: "The picture and its time show here.",
    Modality.IMAGE_EDIT: "The edited picture and its time show here.",
    Modality.VIDEO: "The clip and its time show here.",
    Modality.SPEECH_TO_TEXT: "The transcript and its time show here.",
    Modality.TEXT_TO_SPEECH: "The WAV and its time show here.",
}


def try_shows(modality: Modality) -> str:
    return _TRY_SHOWS.get(modality, "What came back, and its time, show here.")


_CODE_SPAN = re.compile(r"`([^`\n]+)`")


def ticks(text: object) -> Markup:
    """Words with `code` spans, as the CLI writes them, shown as code on a page: the text is
    escaped first, so only a pair of backticks becomes markup."""
    return Markup(_CODE_SPAN.sub(r"<code>\1</code>", str(escape(text))))
