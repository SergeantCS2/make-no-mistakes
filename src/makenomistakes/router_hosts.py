"""The names the router answers to: a request whose `Host` is not one of them gets 421
Misdirected Request, whatever its route (`/v1/…`, `/mcp`, `/api/…`, `/metrics`, `/__mnm/…`) or
port.

A keyless router (WSL's default) would answer any web page in the user's browser, and DNS
rebinding (a page's own name pointed at 127.0.0.1) would let that page read `/__mnm/manifest`
and `/__mnm/events`: the browser sends that page's name as `Host`, which no allow-list holds.
Allowed: `localhost`, 127.0.0.1, `[::1]`, `host.docker.internal` (what Odysseus and the apps
use from their containers), every address the router listens on (Docker's bridge gateway
for `local`, every interface address for 0.0.0.0, read at the router's start), and the names
or addresses in `MNM_ROUTER_HOSTS` (a Tailscale name, a LAN DNS name). A request without a
`Host` passes: a browser always sends one.

The same names are the web pages a keyless router answers (`router_origins`): a page's
`Origin` on any other site is refused."""

from __future__ import annotations

import ipaddress
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.hardware import CommandRunner
from makenomistakes.hardware.probe import read_text_or_empty
from makenomistakes.router_bind import EVERYWHERE

ALWAYS = ("localhost", "127.0.0.1", "::1", "host.docker.internal")
SETTING = "MNM_ROUTER_HOSTS"
INTERFACES = ["ip", "-o", "-4", "addr", "show"]
FIB_TRIE = Path("/proc/net/fib_trie")


def host_name(header: str) -> str:
    """`Host`'s name without its port, lower-cased: `[::1]:7801` → `::1`."""
    value = header.strip().lower()
    if value.startswith("["):
        return value[1 : value.find("]")] if "]" in value else value
    name, colon, port = value.rpartition(":")
    return name if colon and port.isdigit() else value


def listed(setting: str) -> list[str]:
    """`MNM_ROUTER_HOSTS`'s entries (commas, spaces allowed), each as `host_name` reads it."""
    return [host_name(part) for part in setting.split(",") if part.strip()]


def interface_addresses(
    runner: CommandRunner, read_text: Callable[[Path], str] = read_text_or_empty
) -> list[str]:
    """Every IPv4 address of this machine: the kernel's own list (`/proc/net/fib_trie`'s
    `/32 host LOCAL` entries, nothing to install), else `ip -o -4 addr show` (some
    containers have no `ip`); none when neither answers."""
    found: list[str] = []
    last = ""
    for line in read_text(FIB_TRIE).splitlines():
        words = line.split()
        if line.lstrip().startswith("|--") and len(words) == 2:
            last = words[1]
        elif words[:3] == ["/32", "host", "LOCAL"] and last and last not in found:
            found.append(last)
    if found:
        return found
    code, out, _ = runner.run(INTERFACES)
    for line in out.splitlines() if code == 0 else []:
        words = line.split()
        if "inet" in words and words.index("inet") + 1 < len(words):
            address = words[words.index("inet") + 1].split("/", 1)[0]
            try:
                found.append(str(ipaddress.ip_address(address)))
            except ValueError:
                continue
    return found


@dataclass(frozen=True)
class HostList:
    names: frozenset[str]

    @classmethod
    def of(
        cls, bound: Iterable[str], setting: str = "", interfaces: Iterable[str] = ()
    ) -> HostList:
        """The allow-list for a router listening on `bound` (`interfaces` when one of them is
        0.0.0.0), with `MNM_ROUTER_HOSTS`' entries."""
        bound = list(bound)
        everywhere = list(interfaces) if EVERYWHERE in bound else []
        own = [a for a in bound if a != EVERYWHERE]
        return cls(frozenset([*ALWAYS, *own, *everywhere, *listed(setting)]))

    def allows(self, header: str | None) -> bool:
        return header is None or host_name(header) in self.names

    def allows_origin(self, origin: str) -> bool:
        """A web page's `Origin` on one of these names, on any port (a page served from this
        machine, or from a name the user listed, is the user's own); `null` (a
        sandboxed frame, a file) and every other site are not."""
        try:
            name = urllib.parse.urlsplit(origin.strip()).hostname
        except ValueError:
            return False
        return name is not None and name in self.names

    def origin_refusal(self, origin: str) -> str:
        return (
            f"mnm's router has no key (MNM_API_KEY), so it refuses requests a web page on "
            f"another site sends ({origin}): any page in your browser could otherwise drive "
            f"the models. A browser client served from there sends the key once MNM_API_KEY "
            f"is set in the secrets file; a page of your own on another name goes in {SETTING} "
            f"there, then `mnm service install`."
        )

    def refusal(self, header: str) -> str:
        return (
            f"mnm's router does not answer to the name {host_name(header)!r}: it answers to "
            f"localhost, host.docker.internal and its own addresses. Another name (a Tailscale "
            f"or LAN name) goes in {SETTING} in the secrets file, then `mnm service install`."
        )


def default_route_addresses() -> list[str]:
    """Without a command to ask: the default route's address (a router
    on 0.0.0.0 without `listen`'s interface list)."""
    from makenomistakes.manifest import default_route_address  # noqa: PLC0415 - no cycle

    found = default_route_address()
    return [found] if found else []
