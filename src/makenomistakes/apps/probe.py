"""Where `mnm doctor --app` asks from: this machine for a native app, the app's own
container for a Docker one, and when that container is not running a one-off container on
Docker's default bridge with `host.docker.internal` mapped as the apps' own are (said so).

The key never reaches a command line (where `ps` shows it): it rides in the probe's own
environment as `MNM_PROBE_KEY`, which `docker exec -e MNM_PROBE_KEY` / `docker run -e
MNM_PROBE_KEY` pass on by name, and a small POSIX script inside the container sends it with
whichever client it finds: curl, which reads the header from its stdin (`-H @-`, written by
the shell's builtin printf), else node or python3, which read it from their environment.

The one-off image is `curlimages/curl` (the curl project's own image on Docker Hub, not a
Docker Official Image), pinned by tag and by its index digest (it is handed the router's key,
so a moved tag must not run); Docker pulls it at the first one-off run, and `mnm doctor --app`
says so before."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Protocol

import httpx

from makenomistakes.hardware import CommandRunner

KEY_ENV = "MNM_PROBE_KEY"
# The one-off probe's image: the curl project's own on Docker Hub, pinned, ~25 MB. The digest
# is the tag's multi-arch index as Docker Hub's tags API gave it on 2026-10-23
# (hub.docker.com/v2/repositories/curlimages/curl/tags/8.11.1, pushed 2024-12-11).
CURL_IMAGE = (
    "curlimages/curl:8.11.1@sha256:c1fe1679c34d9784c1b0d1e5f62ac0a79fca01fb6377cdd33e90473c6f9f9a69"
)
HOST_GATEWAY = ["--add-host", "host.docker.internal:host-gateway"]
NO_CLIENT = 3  # the script's exit code when the container has no HTTP client at all
TIMEOUT = 600.0  # a parked model loads first; on a CPU build that is minutes

# $1 method, $2 url, $3 body (may be empty), $4 seconds. Prints the body, a newline, the
# status code (curl's `000` when nothing answered); errors on stderr.
SCRIPT = r"""
m=$1; u=$2; b=$3; t=$4
if command -v curl >/dev/null 2>&1; then
  set --
  if [ -n "$b" ]; then set -- --data-raw "$b"; fi
  if [ -n "$MNM_PROBE_KEY" ]; then
    # the header from curl's stdin (`-H @-`): printf is the shell's builtin, so the key is in
    # no process's command line
    printf 'Authorization: Bearer %s\n' "$MNM_PROBE_KEY" | curl -sS -m "$t" -X "$m" \
      -H 'Content-Type: application/json' -H @- "$@" -o - -w '\n%{http_code}' "$u"
  else
    exec curl -sS -m "$t" -X "$m" -H 'Content-Type: application/json' "$@" \
      -o - -w '\n%{http_code}' "$u"
  fi
elif command -v node >/dev/null 2>&1; then
  exec node -e '
const [m, u, b, t] = process.argv.slice(1);
const h = {"Content-Type": "application/json"};
if (process.env.MNM_PROBE_KEY) h.Authorization = "Bearer " + process.env.MNM_PROBE_KEY;
fetch(u, {method: m, headers: h, body: b || undefined, signal: AbortSignal.timeout(t * 1000)})
  .then(async r => process.stdout.write((await r.text()) + "\n" + r.status))
  .catch(e => { process.stderr.write(String(e)); process.stdout.write("\n000"); });
' "$m" "$u" "$b" "$t"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 -c '
import os, sys, urllib.error, urllib.request
m, u, b, t = sys.argv[1:5]
h = {"Content-Type": "application/json"}
if os.environ.get("MNM_PROBE_KEY"): h["Authorization"] = "Bearer " + os.environ["MNM_PROBE_KEY"]
r = urllib.request.Request(u, data=b.encode() or None, headers=h, method=m)
try:
    with urllib.request.urlopen(r, timeout=float(t)) as a: s, x = a.status, a.read()
except urllib.error.HTTPError as e: s, x = e.code, e.read()
except Exception as e: sys.stderr.write(str(e)); s, x = 0, b""
sys.stdout.write(x.decode("utf-8", "replace") + "\n" + str(s).zfill(3))
' "$m" "$u" "$b" "$t"
else
  echo "no curl, node or python3 in this container" >&2; exit 3
fi
"""


@dataclass(frozen=True)
class Answer:
    status: int  # 0: nothing answered (`error` says why)
    body: str
    error: str = ""


class Probe(Protocol):
    where: str  # "from this machine", "from inside mnm-librechat-api-1", …

    def ask(self, method: str, url: str, body: str = "", timeout: float = TIMEOUT) -> Answer: ...


class EnvRunner(Protocol):
    def run(self, argv: list[str], env: dict[str, str], timeout: float) -> tuple[int, str, str]: ...


class SubprocessEnvRunner:
    def run(self, argv: list[str], env: dict[str, str], timeout: float) -> tuple[int, str, str]:
        try:
            done = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
                env={**os.environ, **env},
            )
        except FileNotFoundError:
            return 127, "", f"{argv[0]}: not found"
        except subprocess.TimeoutExpired:
            return 124, "", f"no answer within {timeout:.0f} s"
        return done.returncode, done.stdout, done.stderr


class HostProbe:
    """An app on this machine: the request from here."""

    where = "from this machine"

    def __init__(self, key: str | None, client: httpx.Client | None = None) -> None:
        self._key = key
        self._client = client

    def ask(self, method: str, url: str, body: str = "", timeout: float = TIMEOUT) -> Answer:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        try:
            if self._client is not None:
                reply = self._client.request(
                    method, url, content=body or None, headers=headers, timeout=timeout
                )
            else:  # a client of its own per request, closed after it (a few per doctor run)
                reply = httpx.request(
                    method, url, content=body or None, headers=headers, timeout=timeout
                )
        except httpx.HTTPError as exc:
            return Answer(0, "", str(exc) or type(exc).__name__)
        return Answer(reply.status_code, reply.text)


class DockerProbe:
    """The request from inside a container: `docker exec` into the app's, or `docker run` of
    a one-off on the default bridge."""

    def __init__(self, prefix: list[str], where: str, key: str | None, runner: EnvRunner) -> None:
        self._prefix = prefix
        self.where = where
        self._key = key
        self._runner = runner

    def ask(self, method: str, url: str, body: str = "", timeout: float = TIMEOUT) -> Answer:
        argv = [*self._prefix, "sh", "-c", SCRIPT, "sh", method, url, body, f"{timeout:.0f}"]
        env = {KEY_ENV: self._key} if self._key else {}
        code, out, err = self._runner.run(argv, env, timeout + 30)
        if code == NO_CLIENT:
            return Answer(0, "", err.strip() or "no HTTP client in the container")
        text, _, status = out.rpartition("\n")
        if not status.strip().isdigit():
            return Answer(0, "", (err or out).strip()[-400:] or f"exit {code}")
        return Answer(int(status), text, err.strip()[-400:] if int(status) == 0 else "")


def running(container: str, runner: CommandRunner) -> bool | None:
    """Whether the container runs (None: Docker could not be asked)."""
    code, out, err = runner.run(["docker", "inspect", "-f", "{{.State.Running}}", container])
    if code == 0:
        return out.strip() == "true"
    said = err.lower()  # Docker 29 says "error: no such object", older ones "Error: No such …"
    if "no such object" in said or "no such container" in said:
        return False
    return None


def inside(container: str, key: str | None, runner: EnvRunner) -> DockerProbe:
    prefix = ["docker", "exec", *(["-e", KEY_ENV] if key else []), container]
    return DockerProbe(prefix, f"from inside {container}", key, runner)


def one_off(key: str | None, runner: EnvRunner, why: str) -> DockerProbe:
    prefix = [
        "docker", "run", "--rm", *HOST_GATEWAY, *(["-e", KEY_ENV] if key else []),
        "--entrypoint", "", CURL_IMAGE,
    ]  # fmt: skip
    return DockerProbe(
        prefix,
        f"from a one-off container on Docker's default bridge ({why}; Docker pulls {CURL_IMAGE}, "
        "about 25 MB, unless it is here already)",
        key,
        runner,
    )
