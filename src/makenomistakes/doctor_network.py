"""The doctor's network checks: who can reach the router, its key, and a firewall that keeps
Docker's containers from it. Under WSL's NAT networking a 0.0.0.0 bind
stays on this machine; on a native box or under mirrored networking it is on the LAN."""

from __future__ import annotations

import ipaddress

from makenomistakes.doctor_types import Check, Status
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import HardwareReport, NetworkingMode
from makenomistakes.router_bind import Exposure, check_setting, exposure

KEY_REMEDY = (
    "Add MNM_API_KEY=<a long random string> to ~/.config/makenomistakes/secrets.env "
    "(`mnm setup` makes one); clients then send it as their API key."
)
# What a keyless router refuses, and who needs the key.
CROSS_SITE = (
    "Without a key the router refuses requests from web pages on other sites (a browser "
    "client served from elsewhere needs the key; a page of your own on another name goes in "
    "MNM_ROUTER_HOSTS)."
)
LOCAL_REMEDY = (
    "Or keep the router to this machine and its Docker containers: MNM_ROUTER_HOST=local in "
    "the secrets file, then `mnm service install`."
)
# Docker's bridge networks (the default 172.17.0.0/16 and the ones compose creates).
DOCKER_NETWORKS = "172.16.0.0/12"
UFW_MARKER = "ufw-rule-by-mnm"  # in the state folder: setup added the rule (uninstall deletes it)


def ufw_allows(status: str, port: int) -> bool:
    """`ufw status` lists the rule that lets Docker's networks reach the port."""
    return any(
        f"{port}/tcp" in line.split() and DOCKER_NETWORKS in line.split() and "ALLOW" in line
        for line in status.splitlines()
    )


def on_the_lan(report: HardwareReport, router_host: str) -> bool:
    """Can other machines reach the router's port (before any firewall)?"""
    return reaches_lan(report.wsl_networking_mode, router_host)


def reaches_lan(wsl_mode: NetworkingMode | None, router_host: str) -> bool:
    """`on_the_lan` from WSL's networking mode alone (the router's manifest asks it without
    probing the whole machine)."""
    if exposure(router_host) is Exposure.LOCAL:
        return False
    if exposure(router_host) is Exposure.LISTED and all(
        ipaddress.ip_address(a).is_loopback for a in check_setting(router_host).split(",")
    ):
        return False  # 127.0.0.1 (and its kin) only
    if platforms.current().is_wsl:
        return wsl_mode is NetworkingMode.MIRRORED
    return True


def networking_check(
    report: HardwareReport, router_host: str = "0.0.0.0", router_port: int = 7801
) -> Check:
    if not platforms.current().is_wsl:
        return _native_networking(router_host, router_port)
    name = "WSL networking"
    match report.wsl_networking_mode:
        case NetworkingMode.NAT | None:
            return Check(
                name=name,
                status=Status.OK,
                detail="nat: 0.0.0.0 binds stay off the LAN; Odysseus uses host.docker.internal",
            )
        case NetworkingMode.MIRRORED:
            if exposure(router_host) is Exposure.LOCAL:
                return Check(
                    name=name,
                    status=Status.OK,
                    detail="mirrored; the router listens on this machine and Docker only",
                )
            return Check(
                name=name,
                status=Status.WARN,
                detail="mirrored: ports bound on 0.0.0.0 are visible on your LAN",
                remedy="Set networkingMode=nat in .wslconfig, or accept LAN exposure and add "
                f"a key. {LOCAL_REMEDY}",
            )
        case NetworkingMode.UNKNOWN:
            return Check(
                name=name,
                status=Status.WARN,
                detail="wslinfo did not report a mode",
                remedy="Update WSL (`wsl --update`) so `wslinfo --networking-mode` works.",
            )


def _native_networking(router_host: str, router_port: int) -> Check:
    name = "Network"
    match exposure(router_host):
        case Exposure.LOCAL:
            # Linux answers for its own addresses on any interface: a machine that routes
            # Docker's subnet to this one could reach the bridge address too.
            return Check(
                name=name,
                status=Status.OK,
                detail="the router listens on this machine and Docker's bridge only (another "
                "machine reaches that only with a route to Docker's network; a key closes it)",
            )
        case Exposure.LISTED:
            return Check(name=name, status=Status.OK, detail=f"the router listens on {router_host}")
        case Exposure.EVERYWHERE:
            return Check(
                name=name,
                status=Status.OK,
                detail="the router listens on every interface: other machines on your network "
                f"can reach port {router_port} unless a firewall stops them",
            )


def router_host_check(running: str | None, configured: str) -> Check | None:
    """The router's unit carries the address it was installed with: a later edit of the
    secrets file applies only at `mnm service install`."""
    if running is None or running == configured:
        return None
    return Check(
        name="Router address",
        status=Status.WARN,
        detail=f"the router runs with MNM_ROUTER_HOST={running}; the secrets file says "
        f"{configured}",
        remedy="`mnm service install` restarts it with the secrets file's.",
    )


def keyless_lan_detail(native: bool) -> str:
    """The one wording for a router on the LAN without a key: the doctor's FAIL and
    the guided setup's summary say it alike."""
    where = (
        "the router's port is on your network"
        if native
        else "mirrored networking puts the router's port on your LAN"
    )
    return f"not set, and {where}: anyone there reaches every model"


def api_key_check(key_set: bool, lan: bool, differs: bool = False) -> Check:
    """The router's key: optional on this machine and its containers, required where
    the router is on the LAN (a FAIL there, never a mere warning)."""
    name = "API key"
    if differs:
        return Check(
            name=name,
            status=Status.WARN,
            detail=(
                "this shell's MNM_API_KEY is ignored: the router reads the secrets file's, "
                "which differs"
                if key_set
                else "this shell's MNM_API_KEY is ignored: the router reads the secrets file, "
                "which has none, so the router is open"
            ),
            remedy="Put the key in ~/.config/makenomistakes/secrets.env (the router reads it "
            "from there only); export that same value for the snippets that read the shell.",
        )
    if key_set:
        return Check(name=name, status=Status.OK, detail="set: the router asks every request")
    if lan:
        return Check(
            name=name,
            status=Status.FAIL,
            detail=keyless_lan_detail(not platforms.current().is_wsl),
            remedy=f"{KEY_REMEDY} {LOCAL_REMEDY} {CROSS_SITE}",
        )
    return Check(
        name=name,
        status=Status.OK,
        detail=(
            "not set (optional; the router is open to this machine and its containers, "
            "never to web pages on other sites)"
        ),
    )


def router_hosts_check(setting: str) -> Check:
    """The names the router answers to: a page whose own name points at this machine
    (DNS rebinding) gets 421, whatever the key says."""
    from makenomistakes.router_hosts import SETTING, listed  # noqa: PLC0415 - no cycle

    extra = listed(setting)
    added = f"; and {', '.join(extra)} ({SETTING})" if extra else ""
    return Check(
        name="Router hosts",
        status=Status.OK,
        detail="an allow-list: localhost, 127.0.0.1, host.docker.internal and the router's own "
        f"addresses{added}; any other name gets 421",
    )


def firewall_check(
    report: HardwareReport, router_port: int = 7801, rule_noted: bool = False
) -> Check | None:
    """ufw drops what containers send to the host by default, Odysseus's requests to the
    router included (only on a native box: WSL's own firewall is Windows'). `rule_noted`:
    setup added the rule that lets them in."""
    if platforms.current().is_wsl or not report.ufw_enabled:
        return None
    if rule_noted:
        return Check(
            name="Firewall",
            status=Status.OK,
            detail=f"ufw is on; mnm's rule lets Docker's networks reach {router_port}",
        )
    return Check(
        name="Firewall",
        status=Status.WARN,
        detail="ufw is on: it drops what Docker's containers send to the router unless a rule "
        "allows them (ignore this if you added one yourself)",
        remedy=f"`sudo ufw allow from {DOCKER_NETWORKS} to any port {router_port} proto tcp` "
        "(Docker's networks only; `mnm setup` offers it). Clients on this machine itself are "
        "not affected.",
    )


def ollama_check(port: int, answered: str | None, ours: str) -> Check:
    """Ollama's API on its own port (`MNM_OLLAMA_PORT`): the router there, nothing, or
    another program (most likely Ollama itself)."""
    name = "Ollama API"
    if answered == ours:
        return Check(
            name=name,
            status=Status.OK,
            detail=f"port {port} (MNM_OLLAMA_PORT): the router, with its addresses and key",
        )
    if answered is None:
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"port {port} (MNM_OLLAMA_PORT): nothing answers there",
            remedy="`mnm service install` writes the port into the router's unit and restarts "
            "it; a router in a terminal needs a restart too.",
        )
    who = f"Ollama {answered}" if answered else "something that is not the router"
    return Check(
        name=name,
        status=Status.WARN,
        detail=f"port {port} (MNM_OLLAMA_PORT) is held by {who}",
        remedy="Stop it (Ollama: `sudo systemctl disable --now ollama`), or set another "
        "MNM_OLLAMA_PORT, then `mnm service install`.",
    )
