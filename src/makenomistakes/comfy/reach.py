"""What else on this machine ComfyUI's container can reach. As read from the internal network,
with the sandbox's flags (Docker 29.8, its default iptables rules and userland proxy): the host
answers only at the network's gateway, and there only what listens on every address (or on that
one), over TCP and UDP alike; a 127.0.0.1 listener refuses, the host's other addresses and
every other container have no route, and another container's published port (Odysseus's, even
on 0.0.0.0) is dropped by Docker. mnm's router and web UI refuse ComfyUI's network themselves
(`guard.py`), which this check proves by asking them from the gateway's address rather than
trusting a port number; anything else listening on every address is named. One such reading
named a VM's own control process on 0.0.0.0:2024, which the container could open."""

from __future__ import annotations

import ipaddress
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import httpx

from makenomistakes.comfy.engine import NETWORK
from makenomistakes.comfy.guard import SANDBOXED, Address, ComfyGuard, Network, unmapped
from makenomistakes.config import Settings
from makenomistakes.doctor_types import Check, Status
from makenomistakes.hardware.probe import CommandRunner, FileReader, read_text_or_empty

NAME = "ComfyUI's reach"
# each table and the state its sockets wait in: TCP LISTEN, UDP unconnected
TABLES = (
    ("tcp", Path("/proc/net/tcp"), "0A"),
    ("tcp", Path("/proc/net/tcp6"), "0A"),
    ("udp", Path("/proc/net/udp"), "07"),
    ("udp", Path("/proc/net/udp6"), "07"),
)
_PUBLISHED = re.compile(r":(\d+)(?:-(\d+))?->[\d-]+/(tcp|udp)")
Probe = Callable[[str, int], bool]


class Socket(NamedTuple):
    address: Address
    port: int
    proto: str

    def __str__(self) -> str:
        return f"{self.port}/{self.proto}"


def listeners(read: FileReader = read_text_or_empty) -> set[Socket] | None:
    """Every socket waiting for a peer; None when neither TCP table could be read (tcp6 alone
    may be missing, on a kernel without IPv6) or neither UDP table could."""
    found: set[Socket] = set()
    readable: set[str] = set()
    for proto, table, waiting in TABLES:
        text = read(table)
        if text:  # a readable table always has its header line
            readable.add(proto)
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4 or fields[3] != waiting or ":" not in fields[1]:
                continue
            hex_address, hex_port = fields[1].rsplit(":", 1)
            try:
                found.add(Socket(_address(hex_address), int(hex_port, 16), proto))
            except ValueError:
                continue
    return found if readable == {"tcp", "udp"} else None


def _address(hex_address: str) -> Address:
    """The kernel prints each 32-bit word in the machine's own byte order."""
    raw = bytes.fromhex(hex_address)
    if len(raw) not in (4, 16):
        raise ValueError(hex_address)
    words = [raw[i : i + 4] for i in range(0, len(raw), 4)]
    ordered = b"".join(word[::-1] for word in words) if sys.byteorder == "little" else raw
    if len(ordered) == 4:
        return ipaddress.IPv4Address(ordered)
    return unmapped(ipaddress.IPv6Address(ordered))


def published(runner: CommandRunner) -> set[tuple[int, str]]:
    """The host ports Docker publishes for containers: their docker-proxy listens on every
    address, but Docker drops what ComfyUI's network sends to them (measured). When Docker does
    not answer, none are left out."""
    code, out, _err = runner.run(["docker", "ps", "--format", "{{.Ports}}"], timeout=30)
    if code != 0:
        return set()
    ports: set[tuple[int, str]] = set()
    for first, last, proto in _PUBLISHED.findall(out):
        ports.update((port, proto) for port in range(int(first), int(last or first) + 1))
    return ports


def in_reach(sockets: Iterable[Socket], networks: Iterable[Network]) -> list[Socket]:
    """The sockets ComfyUI's container can open: on every address, or on an address of its
    network (the gateway)."""
    nets = tuple(networks)
    return sorted(
        {s for s in sockets if s.address.is_unspecified or any(s.address in n for n in nets)},
        key=lambda s: (s.port, s.proto),
    )


def gateways(runner: CommandRunner) -> list[str]:
    template = "{{range .IPAM.Config}}{{.Gateway}} {{end}}"
    code, out, _err = runner.run(
        ["docker", "network", "inspect", "--format", template, NETWORK], timeout=30
    )
    return out.split() if code == 0 else []


def refuses_comfyui(gateway: str, port: int) -> bool:
    """Asked from the gateway's address (the host connects to its own address there, so that is
    the source the server sees): mnm's router and web UI answer 403 with SANDBOXED."""
    host = f"[{gateway}]" if ":" in gateway else gateway
    try:
        reply = httpx.get(f"http://{host}:{port}/", timeout=3, trust_env=False)
    except httpx.HTTPError:
        return False
    return reply.status_code == 403 and SANDBOXED in reply.text


def check(
    settings: Settings,
    runner: CommandRunner,
    read: FileReader | None = None,
    probe: Probe | None = None,
) -> Check:
    """The doctor's line. The defaults are looked up at the call, so tests replace them."""
    read, probe = read or read_text_or_empty, probe or refuses_comfyui
    networks = ComfyGuard(settings.expanded_state_dir()).networks()
    if not networks:
        return Check(name=NAME, status=Status.OK, detail="read after its first start")
    sockets = listeners(read)
    if sockets is None:
        return Check(
            name=NAME,
            status=Status.WARN,
            detail="could not read /proc/net/tcp or /proc/net/udp: what it reaches is unknown",
            remedy="Check by hand, or " + _rule(networks),
        )
    dropped = published(runner)
    open_ = [s for s in in_reach(sockets, networks) if (s.port, s.proto) not in dropped]
    # the router, the web UI, and the router again on `MNM_OLLAMA_PORT`
    mnm_ports = {settings.router_port, settings.port} | ({settings.ollama_port} - {None})
    asked = [s for s in open_ if s.proto == "tcp" and s.port in mnm_ports]
    found = gateways(runner) if asked else []
    proven = {s for s in asked if any(probe(gateway, s.port) for gateway in found)}
    left = [s for s in open_ if s not in proven]
    if not left:
        return Check(name=NAME, status=Status.OK, detail="nothing on this machine answers it")
    words = ", ".join(str(s) for s in left)
    udp = (
        " (a UDP one may be a program's own client socket)"
        if any(s.proto == "udp" for s in left)
        else ""
    )
    return Check(
        name=NAME,
        status=Status.WARN,
        detail=f"its container can open {words} on this machine (listening on every address); "
        f"a node pack could try them{udp}",
        remedy=f"Bind those services to 127.0.0.1 if nothing else needs them, or {_rule(networks)}",
    )


def _rule(networks: tuple[Network, ...]) -> str:
    rules = " and ".join(
        f"`sudo {'ip6tables' if net.version == 6 else 'iptables'} -I INPUT -s {net} "
        "-m conntrack --ctstate NEW -j DROP`"
        for net in networks
    )
    return (
        f"drop what ComfyUI's network starts towards this machine: {rules} (mnm still reaches "
        "ComfyUI; the rule lasts until a reboot). With ufw on, only the ports it allows are in "
        "reach."
    )
