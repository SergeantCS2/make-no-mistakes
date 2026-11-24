"""Where the router listens: `MNM_ROUTER_HOST`.

- `0.0.0.0` (the default): every interface. Under WSL's NAT networking that is this machine
  and its Docker containers only; on a native box or under mirrored networking it is the
  whole network, so the router's key is advised (`mnm doctor`).
- `local`: 127.0.0.1 and the gateway of Docker's default bridge, which is what
  `host.docker.internal` (`host-gateway`) names inside a container. A container on another
  bridge network reaches it too: the address is the host's own. So does another machine
  that routes Docker's subnet to this one (Linux answers for its own addresses on every
  interface); a key closes that too.
- A comma list of addresses; 127.0.0.1 is always added, since mnm asks the router there.
"""

from __future__ import annotations

import ipaddress
from enum import StrEnum

from makenomistakes.hardware import CommandRunner

EVERYWHERE = "0.0.0.0"
LOCAL = "local"
LOOPBACK = "127.0.0.1"
BRIDGE_GATEWAY = [
    "docker",
    "network",
    "inspect",
    "bridge",
    "--format",
    "{{range .IPAM.Config}}{{.Gateway}} {{end}}",
]


class Exposure(StrEnum):
    EVERYWHERE = "everywhere"  # every interface
    LOCAL = "local"  # this machine and its containers
    LISTED = "listed"  # the addresses the user named


class BindError(RuntimeError):
    pass


def check_setting(value: str) -> str:
    """The setting as written, or ValueError naming what is wrong (Settings' validator)."""
    value = value.strip()
    if value == LOCAL:
        return value
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("MNM_ROUTER_HOST is empty: use 0.0.0.0, local or addresses")
    for part in parts:
        try:
            address = ipaddress.ip_address(part)
        except ValueError:
            raise ValueError(
                f"MNM_ROUTER_HOST: {part!r} is not an IPv4 address (use 0.0.0.0, local, or "
                "addresses separated by commas)"
            ) from None
        if address.version != 4:
            raise ValueError(f"MNM_ROUTER_HOST: {part} is IPv6; the router binds IPv4 only")
    return ",".join(parts)


def exposure(value: str) -> Exposure:
    if value.strip() == LOCAL:
        return Exposure.LOCAL
    parts = {part.strip() for part in value.split(",")}
    return Exposure.EVERYWHERE if EVERYWHERE in parts else Exposure.LISTED


def bridge_gateway(runner: CommandRunner) -> str | None:
    """Docker's default bridge gateway (172.17.0.1 unless daemon.json moves it)."""
    code, out, _ = runner.run(BRIDGE_GATEWAY)
    if code != 0:
        return None
    for word in out.split():
        try:
            if ipaddress.ip_address(word).version == 4:
                return word
        except ValueError:
            continue
    return None


def bind_addresses(value: str, runner: CommandRunner) -> list[str]:
    """The addresses to listen on, in order (the loopback first)."""
    setting = check_setting(value)
    if setting == LOCAL:
        gateway = bridge_gateway(runner)
        if gateway is None:
            raise BindError(
                "MNM_ROUTER_HOST=local needs Docker's bridge gateway, and Docker did not "
                "answer (is it running?)"
            )
        return [LOOPBACK, gateway]
    parts = setting.split(",")
    if EVERYWHERE in parts:
        return [EVERYWHERE]
    return list(dict.fromkeys([LOOPBACK, *parts]))
