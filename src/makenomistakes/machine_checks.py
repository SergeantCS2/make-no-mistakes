"""The checks `mnm doctor` and the web UI's Doctor page show: the hardware report mapped
by `doctor`, with what mnm itself knows (the token, its running instances, the router)."""

from __future__ import annotations

import os

from makenomistakes import __version__
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.doctor_section import app_checks
from makenomistakes.apps.hub import AppsService
from makenomistakes.comfy.doctor import comfy_checks
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.config import Settings
from makenomistakes.disk_doctor import with_policy
from makenomistakes.doctor import Check, run_checks
from makenomistakes.doctor_backend import backend_checks
from makenomistakes.doctor_machine import machine_check
from makenomistakes.doctor_network import UFW_MARKER, ollama_check, router_hosts_check
from makenomistakes.doctor_server import with_server_lines
from makenomistakes.doctor_types import Status
from makenomistakes.doctor_unified import unified_checks
from makenomistakes.engine_check import doctor_line as engines_line
from makenomistakes.folder_hint import folder_hint
from makenomistakes.hardware.box_facts import BoxFacts
from makenomistakes.hardware.probe import HardwareReport, SubprocessRunner
from makenomistakes.instances import InstanceService
from makenomistakes.machine import saved_recipes
from makenomistakes.manifest import Manifest
from makenomistakes.nightly import timer as nightly_timer
from makenomistakes.nightly.report import NightlyStore, Outcome
from makenomistakes.router_client import (
    not_answering_note,
    ollama_version,
    router_answers,
    router_health,
)
from makenomistakes.service import installed_environment
from makenomistakes.service_watch import services_check


def machine_checks(
    settings: Settings, report: HardwareReport, box: BoxFacts | None = None, deep: bool = False
) -> list[Check]:
    """Every check for this machine, the same in the terminal and the web UI; `box` the CPU
    and RAM slots (`hardware.box_facts.read_box`); `deep` runs the backend's own device
    check on a box that does not run CUDA (CUDA's is the probe's)."""
    service = InstanceService(settings)
    answered = router_answers(settings)
    checks = run_checks(
        report,
        hf_token_present=settings.hf_token_value() is not None,
        instances=service.running_vram(in_ram=False),  # the cards'
        router=None if answered else not_answering_note(settings),
        key_set=(in_file := settings.api_key_value()) is not None,
        # A shell's own is never read (the router takes the secrets file's): say so.
        key_differs=(shell := os.environ.get("MNM_API_KEY") or None) is not None
        and shell != in_file,
        router_host=settings.router_host,
        router_port=settings.router_port,
        ufw_rule_noted=(settings.expanded_state_dir() / UFW_MARKER).exists(),
        router_host_running=_installed("MNM_ROUTER_HOST"),
        box=box,
        gpu_set="gpu" in settings.model_fields_set,  # Unset on several cards is said
        health=health_words(settings) if answered else None,
    )
    checks = with_policy(checks, service)  # The disk policy runs and says so here
    gpu = next((i + 1 for i, c in enumerate(checks) if c.name == "NVIDIA GPU"), 0)
    checks.insert(gpu, machine_check(settings, report, box, saved_recipes(settings)))
    checks = backend_checks(settings, report, checks, SubprocessRunner(), deep)
    checks = unified_checks(report, checks, SubprocessRunner())  # The pool's lines
    checks = with_server_lines(settings, report, checks)  # The RAM platform, NUMA
    token = next((i + 1 for i, c in enumerate(checks) if c.name == HF_TOKEN), len(checks))
    checks.insert(token, civitai_token_check(settings.civitai_token_value() is not None))
    at = next((i + 1 for i, c in enumerate(checks) if c.name == "API key"), len(checks))
    checks.insert(at, router_hosts_check(settings.router_hosts))  # beside the key
    comfy = comfy_checks(ComfyEngine(settings), report, SubprocessRunner())
    port = settings.ollama_port
    ollama = (
        [] if port is None else [ollama_check(port, ollama_version(settings, port), __version__)]
    )
    nightly = nightly_check(settings)
    services = services_check(SubprocessRunner())  # The units' restarts
    return [
        *checks,
        *ollama,
        services,
        engines_line(settings.expanded_state_dir()),  # The last `mnm engines check`
        outputs_check(settings),
        nightly,
        *comfy,
        *apps_section(settings),
    ]


def health_words(settings: Settings) -> str | None:
    """What the router's `/__mnm/health` says, for the Router line; None when the
    router does not answer at all (the line says that)."""
    found = router_health(settings)
    if found is None:
        return "/__mnm/health does not answer (a router older than this mnm: `mnm update`)"
    why = f" ({'; '.join(found.reasons)})" if found.reasons else ""
    return f"/__mnm/health answers {found.status.value}{why}"


def apps_section(settings: Settings) -> list[Check]:
    """The workspace apps mnm installed (states only: no manifest is built)."""
    ctx = AppContext(settings, SubprocessRunner(), _no_manifest, say=lambda _line: None)
    return app_checks(AppsService(ctx).statuses(), settings.api_key_value() is not None)


def _no_manifest() -> Manifest:
    raise AssertionError("the doctor's apps section reads states only")


def nightly_check(settings: Settings) -> Check:
    """The nightly's timer (its unit file, read without systemd) and its last word."""
    at = nightly_timer.installed_at()
    latest = NightlyStore(settings.expanded_state_dir()).latest()
    timer = f"every day at {at}" if at else "not installed"
    last = f"last {latest.id}: {latest.outcome.value}" if latest else "no report yet"
    detail = f"{timer}; {last}"
    if latest is not None and latest.outcome in (
        Outcome.REGRESSION,
        Outcome.FAILED,
        Outcome.TO_READ,
    ):
        remedy = "`mnm nightly show` says what regressed and what mnm did or offers"
        return Check(name="Nightly", status=Status.WARN, detail=detail, remedy=remedy)
    if at is None:
        remedy = f"`mnm nightly install` runs it every night at {nightly_timer.UNIT.default_at}"
        return Check(name="Nightly", status=Status.OK, detail=detail, remedy=remedy)
    return Check(name="Nightly", status=Status.OK, detail=detail)


def outputs_check(settings: Settings) -> Check:
    """Where runs keep what they make. A box updated from an older mnm has no such folder until
    `mnm setup` makes it, and its first Run would say so; the doctor says it first."""
    folder = settings.expanded_outputs_dir()
    name = "Outputs folder"
    hint = folder_hint(folder, "MNM_OUTPUTS_DIR")
    if hint is None:
        return Check(name=name, status=Status.OK, detail=str(folder))
    return Check(name=name, status=Status.WARN, detail=f"{folder} cannot be written", remedy=hint)


def _installed(name: str) -> str | None:
    """A setting as the installed router unit carries it (None: no unit, or not there)."""
    for word in installed_environment():
        key, _, value = word.partition("=")
        if key == name:
            return value
    return None


HF_TOKEN = "Hugging Face token"  # the doctor's line the Civitai one follows


def civitai_token_check(present: bool) -> Check:
    """The Civitai key beside the Hugging Face token. OK either way: most Civitai files
    download without it."""
    detail = "set" if present else "not set (optional; some Civitai files need it)"
    return Check(name="Civitai token", status=Status.OK, detail=detail)
