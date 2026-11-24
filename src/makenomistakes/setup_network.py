"""Section 3 of the guided setup: who may reach the router, its key, and a firewall
that would drop what an app in Docker sends it. The rule for a setup nobody answers
(`decides`, `unattended_host`) is the plain setup's too, so a script's install
(`MNM_SETUP_YES=1` without a terminal) leaves the router where the guided one would."""

from __future__ import annotations

import ipaddress
import secrets
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import dotenv_values

from makenomistakes.doctor_network import (
    DOCKER_NETWORKS,
    KEY_REMEDY,
    LOCAL_REMEDY,
    UFW_MARKER,
    keyless_lan_detail,
    on_the_lan,
    ufw_allows,
)
from makenomistakes.router_bind import EVERYWHERE, LOCAL, Exposure, check_setting, exposure
from makenomistakes.setup_view import Mark

if TYPE_CHECKING:
    from makenomistakes.config import Settings
    from makenomistakes.hardware import HardwareReport
    from makenomistakes.hardware.platform import Platform
    from makenomistakes.setup_guide import Guide

__all__ = ["UFW_MARKER", "decides", "file_host", "moved_words", "network", "unattended_host"]

ROUTER_HOST = "MNM_ROUTER_HOST"


def network(guide: Guide) -> None:
    guide.view.section(
        3,
        "Network",
        "Every model answers on one port (7801, the router). An app in Docker (LibreChat, "
        "OpenHands, Odysseus) reaches it as host.docker.internal.",
    )
    said = _decide(guide) if decides(guide.platform, guide.report) else ""
    host = guide.settings.router_host
    lan = on_the_lan(guide.report, host)
    reach = reach_words(guide.report, host)
    guide.summary.add(Mark.NOTE if lan else Mark.DONE, "Network", f"the router: {reach}{said}")
    _key(guide, lan)
    _firewall(guide)


def decides(where: Platform, report: HardwareReport) -> bool:
    """Setup decides who reaches the router on a native box, and under WSL's mirrored
    networking, where every interface is the whole network; under WSL's NAT it stays on
    the PC whatever the address."""
    return not where.is_wsl or on_the_lan(report, EVERYWHERE)


def chosen_before(settings: Settings) -> bool:
    """The router is where it is on purpose: an address set in the secrets file or the
    environment, or a key (setup makes one only at a terminal, and an earlier setup saved no
    address when the answer was the default, the network)."""
    return "router_host" in settings.model_fields_set or settings.api_key_value() is not None


def unattended_host(settings: Settings, report: HardwareReport) -> str:
    """Where a setup nobody answers leaves the router: where it is when it was chosen or does
    not reach the network; else, from the state the doctor FAILs (on the network, no key,
    nothing chosen), this machine and its containers."""
    host = settings.router_host
    if chosen_before(settings) or not on_the_lan(report, host):
        return host
    return LOCAL


def moved_words(dry_run: bool) -> str:
    """Said when setup keeps the router off the network, guided or plain."""
    if dry_run:
        return (
            "Setup would keep the router to this machine and its containers "
            f"({ROUTER_HOST}={LOCAL}): on your network it would have no key."
        )
    return (
        f"The router listens on this machine and its containers only ({ROUTER_HOST}={LOCAL}): "
        "on your network it would have no key. To reach it from other machines, run "
        "`mnm setup` at a terminal and choose your network, with a key."
    )


def file_host(secrets_file: Path) -> str | None:
    """`MNM_ROUTER_HOST` as the secrets file holds it, written as the router reads it (None:
    not there, or not an address the router takes)."""
    if not secrets_file.is_file():
        return None
    try:
        held = dotenv_values(secrets_file).get(ROUTER_HOST)
        return check_setting(held) if held else None
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def reach_words(report: HardwareReport, host: str) -> str:
    if on_the_lan(report, host):
        return "on your network"
    if exposure(host) is Exposure.LISTED and all(
        ipaddress.ip_address(a).is_loopback for a in check_setting(host).split(",")
    ):
        return "this machine only"  # a container reaches the bridge's gateway, not 127.0.0.1
    return "this machine and its containers"


def _decide(guide: Guide) -> str:
    """Who reaches the router: asked at a terminal, else `unattended_host`; saved unless the
    secrets file holds it already, so a later run (an unattended one too) starts from it.
    The summary row's tail comes back."""
    if guide.prompt.interactive:
        host = _asked(guide)
    else:
        host = unattended_host(guide.settings, guide.report)
        if host != guide.settings.router_host:
            guide.view.say(moved_words(guide.dry_run))
    guide.settings = guide.settings.model_copy(update={"router_host": host})
    if file_host(guide.secrets_file) == host:
        return ""
    guide.save(ROUTER_HOST, host)
    return " (dry run: not saved)" if guide.dry_run else ""


def _asked(guide: Guide) -> str:
    """The answer at a terminal. Its default is what was chosen before, and either answer
    keeps an address chosen before that already means it ("2": reaches the network; "1":
    does not, such as 127.0.0.1), so Enter changes nothing."""
    host = guide.settings.router_host
    chosen = chosen_before(guide.settings)
    lan = on_the_lan(guide.report, host)
    choice = guide.prompt.choose(
        "Who may reach the router?",
        {
            "1": "this machine and its Docker containers only (recommended)",
            "2": "every machine on your network (set a key below)",
        },
        "2" if chosen and lan else "1",
    )
    if choice == "2":
        return host if chosen and lan else EVERYWHERE
    return host if chosen and not lan else LOCAL


def _key(guide: Guide, lan: bool) -> None:
    if guide.settings.api_key_value() is not None:
        guide.view.say("A router key is already set (MNM_API_KEY in the secrets file).")
        guide.summary.add(Mark.DONE, "Router key", "set")
        return
    if not guide.prompt.interactive or not guide.prompt.yes(
        "Protect the router with a key? Clients then send it as their API key."
        + (" (Required: the router is on your network.)" if lan else ""),
        lan,
    ):
        if lan:  # the doctor's FAIL, said the same way
            guide.summary.add(
                Mark.FAIL, "Router key", keyless_lan_detail(not guide.platform.is_wsl)
            )
            guide.summary.todo.append(f"{KEY_REMEDY} {LOCAL_REMEDY}")
        else:
            guide.summary.add(Mark.SKIP, "Router key", "none (the router is open)")
        return
    key = secrets.token_hex(24)
    guide.save("MNM_API_KEY", key)
    guide.view.say(
        f"Key: {key}\nPaste it into each client's API key field (Odysseus: each endpoint). "
        f"It is kept in {guide.secrets_file}; `mnm url` never prints it.",
        style="bold",
    )
    guide.summary.add(Mark.DONE, "Router key", f"made; kept in {guide.secrets_file}")


def _firewall(guide: Guide) -> None:
    """ufw drops what Docker's containers send to the host: Odysseus could not reach 7801."""
    if guide.platform.is_wsl or not guide.report.ufw_enabled:
        return
    port = guide.settings.router_port
    marker = guide.settings.expanded_state_dir() / UFW_MARKER
    if _noted(marker, port):  # added by an earlier run: nothing to ask
        guide.summary.add(Mark.DONE, "Firewall", f"ufw lets Docker's networks reach {port}")
        return
    rule = ["sudo", "ufw", "allow", "from", DOCKER_NETWORKS, "to", "any", "port", str(port)]
    rule += ["proto", "tcp"]
    guide.view.say(f"ufw is on: it would drop Odysseus's requests to port {port}.", "yellow")
    if guide.dry_run or not guide.prompt.yes(
        f"Allow Docker's networks ({DOCKER_NETWORKS}) to reach port {port}?", True
    ):
        guide.summary.todo.append(f"Let Docker's containers through ufw: `{' '.join(rule)}`")
        guide.summary.add(Mark.NOTE, "Firewall", "ufw blocks containers from the router")
        return
    # sudo asks once at the terminal; the status is then read quietly, so a rule the user
    # already had is never noted as mnm's (uninstall.sh deletes only what the note names).
    if guide.fix_runner.run(["sudo", "-v"])[0] != 0:
        guide.summary.add(Mark.FAIL, "Firewall", "sudo refused; ufw left as it is")
        return
    code, out, _ = guide.runner.run(["sudo", "-n", "ufw", "status"])
    if code == 0 and ufw_allows(out, port):
        guide.summary.add(Mark.DONE, "Firewall", f"ufw already lets Docker's networks reach {port}")
        return
    guide.view.command(" ".join(rule))
    code, _, err = guide.fix_runner.run(rule)
    if code != 0:
        guide.summary.add(Mark.FAIL, "Firewall", f"ufw refused: {err.strip() or code}")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{DOCKER_NETWORKS} {port}\n", encoding="utf-8")
    guide.summary.add(Mark.DONE, "Firewall", f"ufw lets Docker's networks reach {port}")


def _noted(marker: Path, port: int) -> bool:
    """Setup's note names this very rule (a note for another port does not cover this one)."""
    try:
        return marker.read_text(encoding="utf-8").split() == [DOCKER_NETWORKS, str(port)]
    except OSError:
        return False
