"""OpenCode's release binary: the asset of the pinned tag for this CPU, downloaded,
checked against the sha256 in the pin (a mismatch is refused and deleted, never run), and its
one `opencode` executable unpacked under `<state>/apps/opencode/<version>/`. The version
before it stays beside it for `mnm apps rollback opencode`.

The asset is chosen as OpenCode's own installer (opencode.ai/install, read 2026-10-23) chooses
it on Linux: `x64`, or `x64-baseline` for a CPU without AVX2, and `arm64` on an Arm box (a
DGX Spark); other machines are refused."""

from __future__ import annotations

import hashlib
import platform
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path

import httpx

from makenomistakes.apps.defs import Asset, BinaryPin
from makenomistakes.apps.ledger import AppsError

EXECUTABLE = "opencode"
MAX_BYTES = 512 * 2**20  # the 1.19.4 asset is 80 MB, its executable 215 MB
Download = Callable[[str, Path], None]
CPUINFO = Path("/proc/cpuinfo")


def target(machine: str | None = None, cpuinfo: str | None = None) -> str:
    """`x64` or `x64-baseline`, `arm64` on an Arm machine; AppsError on another."""
    machine = machine or platform.machine()
    if machine in ("aarch64", "arm64"):
        return "arm64"  # OpenCode's installer: no baseline split on arm64
    if machine not in ("x86_64", "amd64", "AMD64"):
        raise AppsError(
            f"mnm installs OpenCode's Linux x86_64 and arm64 builds only; this machine is "
            f"{machine}. OpenCode's own installer covers the others (opencode.ai/docs)."
        )
    if cpuinfo is None:
        try:
            cpuinfo = CPUINFO.read_text(encoding="utf-8", errors="replace")
        except OSError:
            cpuinfo = ""
    flags = {
        word for line in cpuinfo.splitlines() if line.startswith("flags") for word in line.split()
    }
    return "x64" if "avx2" in flags or not cpuinfo else "x64-baseline"


def http_download(url: str, dest: Path) -> None:
    """The asset into `dest` (GitHub redirects to its CDN), bounded."""
    got = 0
    with (
        httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as answer,
        dest.open("wb") as out,
    ):
        if answer.status_code != 200:
            raise AppsError(f"{url} answered {answer.status_code}: nothing was installed.")
        for chunk in answer.iter_bytes():
            got += len(chunk)
            if got > MAX_BYTES:
                raise AppsError(f"{url} is larger than any OpenCode release: refused.")
            out.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def folder(apps_state: Path, version: str) -> Path:
    return apps_state / "opencode" / version


def executable(apps_state: Path, version: str) -> Path:
    return folder(apps_state, version) / EXECUTABLE


def fetch(
    pin: BinaryPin,
    apps_state: Path,
    download: Download = http_download,
    choose: Callable[[], str] = target,
) -> Path:
    """The pinned executable, downloaded and checked once; the path of the one on disk."""
    found = executable(apps_state, pin.version)
    if found.is_file():
        return found
    asset = pin.asset(choose())
    if asset is None:
        raise AppsError(f"OpenCode {pin.version}'s pin has no asset for this CPU.")
    where = folder(apps_state, pin.version)
    where.mkdir(parents=True, exist_ok=True)
    archive = where / f"{asset.name}.part"
    try:
        download(pin.url(asset), archive)
        _check(archive, asset)
        _unpack(archive, found)
    except (OSError, httpx.HTTPError, tarfile.TarError) as exc:
        remove_version(apps_state, pin.version)
        raise AppsError(f"OpenCode {pin.version} could not be installed: {exc}") from exc
    except AppsError:
        remove_version(apps_state, pin.version)  # nothing of a refused download stays
        raise
    finally:
        archive.unlink(missing_ok=True)
    return found


def _check(archive: Path, asset: Asset) -> None:
    actual = sha256(archive)
    if actual != asset.sha256:
        raise AppsError(
            f"{asset.name} does not match mnm's pin (sha256 {actual}, pinned "
            f"{asset.sha256}): it was deleted and nothing was installed. A release asset "
            "never changes, so this is a damaged download or not the file GitHub publishes."
        )


def _unpack(archive: Path, dest: Path) -> None:
    """Only the one regular file named `opencode`, bounded; written beside and moved in."""
    with tarfile.open(archive, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.lstrip("./") == EXECUTABLE and m.isfile()),
            None,
        )
        if member is None or member.size > MAX_BYTES:
            raise AppsError(f"{archive.name} holds no `{EXECUTABLE}` executable mnm expects.")
        source = tar.extractfile(member)
        if source is None:
            raise AppsError(f"{archive.name}'s `{EXECUTABLE}` cannot be read.")
        partial = dest.with_name(dest.name + ".part")
        with source, partial.open("wb") as out:
            shutil.copyfileobj(source, out)
        partial.chmod(0o755)
        partial.replace(dest)


def remove_version(apps_state: Path, version: str) -> None:
    shutil.rmtree(folder(apps_state, version), ignore_errors=True)
