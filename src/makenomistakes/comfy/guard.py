"""The router keeps ComfyUI's container out. Its network is `--internal`, which Docker enforces
on the way *out* of the host only: the bridge's gateway is an address of the host itself, so
the container still reaches whatever listens on every address, the router included (on
`0.0.0.0` by default, with no key). A node pack, approved or not, must never load models, start
runs or read run outputs through it. ComfyUI's engine writes its network's subnets beside the
ledger at every start; the router answers 403 to any request from them, key or no key, reading
the file again only when it changes (one stat per request). The web UI does the same (on
127.0.0.1, its default, ComfyUI cannot reach it at all; `mnm serve --host 0.0.0.0` would put it
in reach). What else on the machine is in reach is `reach.py`'s doctor line."""

from __future__ import annotations

import ipaddress
from pathlib import Path

SUBNET_FILE = "subnet"  # in <state>/comfyui/, beside the ledger (never mounted)
SANDBOXED = "mnm answers nothing from ComfyUI's container (its network is mnm's sandbox)."
Network = ipaddress.IPv4Network | ipaddress.IPv6Network
Address = ipaddress.IPv4Address | ipaddress.IPv6Address


def unmapped(ip: Address) -> Address:
    """`::ffff:172.18.0.2` (a dual-stack socket's view of an IPv4 peer) as 172.18.0.2."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


class ComfyGuard:
    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / "comfyui" / SUBNET_FILE
        self._stamp: tuple[int, int] | None = None
        self._networks: tuple[Network, ...] = ()

    def networks(self) -> tuple[Network, ...]:
        """ComfyUI's network as its last start found it; none before the first start."""
        return self._read()

    def blocks(self, address: str | None) -> bool:
        """The request comes from ComfyUI's network. Once ComfyUI has started, a peer whose
        address is unknown or unreadable is refused too: nothing of the user's arrives so."""
        networks = self._read()
        if not networks:
            return False
        if address is None:
            return True
        try:
            ip = unmapped(ipaddress.ip_address(address.split("%", 1)[0]))
        except ValueError:
            return True
        return any(ip in network for network in networks)

    def is_gateway(self, address: str) -> bool:
        """The peer is ComfyUI's network's gateway: the host itself, as the doctor's reach
        check asks from it (Docker gives a network's first address to its gateway)."""
        try:
            ip = unmapped(ipaddress.ip_address(address.split("%", 1)[0]))
        except ValueError:
            return False
        return any(ip == network.network_address + 1 for network in self._read())

    def _read(self) -> tuple[Network, ...]:
        try:
            stat = self.path.stat()
        except OSError:
            self._stamp, self._networks = None, ()
            return ()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self._stamp:
            found = []
            for word in self.path.read_text(encoding="ascii", errors="replace").split():
                try:
                    found.append(ipaddress.ip_network(word, strict=False))
                except ValueError:
                    continue
            self._stamp, self._networks = stamp, tuple(found)
        return self._networks


def write_subnets(state_dir: Path, subnets: str) -> None:
    """What `docker network inspect` says the network's subnets are, for the router."""
    path = state_dir / "comfyui" / SUBNET_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(" ".join(subnets.split()) + "\n", encoding="ascii")
    tmp.replace(path)


def forget_subnets(state_dir: Path) -> None:
    """ComfyUI's network is gone: its subnets are no longer ComfyUI's. Docker hands a freed
    subnet to the next network it makes (Odysseus's compose, say), which the router would
    otherwise refuse as ComfyUI."""
    (state_dir / "comfyui" / SUBNET_FILE).unlink(missing_ok=True)
