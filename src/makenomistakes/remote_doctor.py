"""`mnm doctor --from here|docker|lan`: the router as a client sees it from
that place. Five requests, each a `doctor.Check`: the model list holds the desk's roles, a
short chat with `coder` (else the first chat id), MCP's `initialize`, the key (a request
without it refused wherever one is set; on the LAN one must be), and Ollama's port when it
has one of its own. `remote_script` prints the same five for another machine to run.

From `here` and `lan` the requests leave this process (`apps.probe.HostProbe`); on the box
`lan` asks the router's own LAN address (the bind and the key, not a firewall in between:
only another machine can tell that, which is what the script is for). From `docker` they
leave a one-off `curlimages/curl` container on Docker's default bridge, the key in its
environment only (`apps.probe.one_off`). On a client install (`MNM_ROUTER_URL`) `lan` is the
box's router at that URL."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from makenomistakes.apps.answers import McpHello, ModelList, error_text
from makenomistakes.apps.probe import Answer, EnvRunner, HostProbe, Probe, one_off
from makenomistakes.bridge.protocol import PATH as MCP_PATH
from makenomistakes.client_view import From, Seen
from makenomistakes.doctor_types import Check, Status
from makenomistakes.engine_answers import ChatAnswer, MalformedAnswerError, parse
from makenomistakes.hub.facts import CHAT_LIKE
from makenomistakes.manifest import Manifest
from makenomistakes.roles import CHAT_ROLES

CHAT_TYPES = frozenset(m.value for m in CHAT_LIKE)
USABLE = frozenset({"loaded", "parked", "unknown"})  # what a request may reach
HELLO = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "mnm-doctor", "version": "1"},
    },
}


def chat_body(model: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in two words."}],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},  # 16 tokens spent thinking: ""
    }


class _Tags(BaseModel):
    model_config = ConfigDict(extra="ignore")
    models: list[dict[str, object]] = []


@dataclass(frozen=True)
class Target:
    """What a client at one place asks, read from the manifest as that place sees it."""

    where: From
    base: str  # the router's origin there (no `/v1`)
    ollama: str | None  # Ollama's own port there, when MNM_OLLAMA_PORT gives it one
    roles: tuple[str, ...]  # the desk's roles: ids `/v1/models` must list
    chat: str | None  # the id the chat goes to
    keyed: bool


def target(manifest: Manifest, seen: Seen) -> Target:
    """The router's doors at `seen`'s place: the manifest's own for this machine, as
    `seen` re-points them for a container or another machine."""
    base = seen.url(manifest.openai.loopback).removesuffix("/v1")
    own = seen.url(manifest.ollama.loopback)
    ollama = own if own.rstrip("/") != base else None
    roles = tuple(manifest.aliases)
    chat_roles = [r for r in CHAT_ROLES if r in manifest.aliases]  # coder first
    names = [m.name for m in manifest.models if m.type in CHAT_TYPES and m.state in USABLE]
    chat = next(iter(chat_roles or names), None)
    return Target(seen.where, base, ollama, roles, chat, manifest.key_required)


def _nothing(target: Target, error: str) -> tuple[str, str]:
    detail = f"nothing answered: {error}"
    match target.where:
        case From.DOCKER:
            return detail, (
                "the router must listen on Docker's bridge (MNM_ROUTER_HOST 0.0.0.0 or "
                "local; `mnm doctor` says where it listens), and a native box's ufw must let "
                "Docker's networks in"
            )
        case From.LAN:
            return detail, (
                "is the router on the network (MNM_ROUTER_HOST; `mnm doctor` says), is a "
                "firewall in the way, and, from a client install, is MNM_ROUTER_URL the box's?"
            )
        case From.HERE:
            return detail, "is the router running? `mnm service status`"


def _failed(name: str, what: str, answer: Answer, target: Target) -> Check:
    if answer.status == 0:
        detail, remedy = _nothing(target, answer.error)
    elif answer.status == 403:  # the router's own refusal of this peer (the answer says why)
        detail = f"403: {error_text(answer.body)[:300]}"
        remedy = "`mnm mcp` says who may call the router without a key"
    elif answer.status == 401:
        detail = f"401: {error_text(answer.body)[:300]}"
        remedy = "this machine's MNM_API_KEY (its secrets file) must be the router's"
    else:
        detail = f"{answer.status}: {error_text(answer.body)[:300]}"
        remedy = "`mnm ls` on the box says what is stopped; the router's journal says more"
    return Check(name=name, status=Status.FAIL, detail=f"{what}: {detail}", remedy=remedy)


def models(target: Target, probe: Probe) -> tuple[Check, bool]:
    """The list, and whether anything answered at all (the rest wait on it)."""
    url = f"{target.base}/v1/models"
    answer = probe.ask("GET", url, timeout=30)
    if answer.status != 200:
        return _failed("Models", f"GET {url}", answer, target), answer.status != 0
    try:
        ids = {m.id for m in parse(ModelList, answer.body, "model list").data}
    except MalformedAnswerError as exc:
        return Check(name="Models", status=Status.FAIL, detail=f"GET {url}: {exc}"), True
    missing = [r for r in target.roles if r not in ids]
    if missing:
        detail = f"{url} lists {len(ids)} ids, not the desk's {', '.join(missing)}"
        return Check(
            name="Models",
            status=Status.WARN,
            detail=detail,
            remedy="`mnm sets show <desk>` on the box: a member that is off is not listed",
        ), True
    roles = f"the desk's {', '.join(target.roles)}" if target.roles else "no desk roles"
    return Check(name="Models", status=Status.OK, detail=f"{url}: {len(ids)} ids, {roles}"), True


def chat(target: Target, probe: Probe) -> Check:
    if target.chat is None:
        return Check(
            name="Chat",
            status=Status.WARN,
            detail="no chat model to ask",
            remedy="`mnm sets default <set>` with a coder, or `mnm add` a chat model",
        )
    url = f"{target.base}/v1/chat/completions"
    answer = probe.ask("POST", url, json.dumps(chat_body(target.chat)))
    what = f"POST {url} ({target.chat})"
    if answer.status != 200:
        return _failed("Chat", what, answer, target)
    try:
        found = parse(ChatAnswer, answer.body, "chat answer")
    except MalformedAnswerError as exc:
        return Check(name="Chat", status=Status.FAIL, detail=f"{what}: {exc}")
    said = (found.choices[0].message.content or "") if found.choices else ""
    shown = " ".join(said.split())[:60] or "(an empty reply)"
    return Check(name="Chat", status=Status.OK, detail=f"{target.chat} answered: {shown}")


def mcp(target: Target, probe: Probe) -> Check:
    url = f"{target.base}{MCP_PATH}"
    answer = probe.ask("POST", url, json.dumps(HELLO), timeout=30)
    if answer.status != 200:
        return _failed("MCP", f"initialize at {url}", answer, target)
    try:
        server = parse(McpHello, answer.body, "MCP answer").result.serverInfo.name
    except MalformedAnswerError as exc:
        return Check(name="MCP", status=Status.FAIL, detail=f"initialize at {url}: {exc}")
    return Check(name="MCP", status=Status.OK, detail=f"initialize at {url}: {server or 'ok'}")


def key(target: Target, bare: Probe) -> Check:
    """A request without the key: refused wherever one is set; from another machine there
    must be one (the doctor's API key line says the same from the box's side)."""
    url = f"{target.base}/v1/models"
    if not target.keyed:
        if target.where is From.LAN:
            return Check(
                name="Key",
                status=Status.FAIL,
                detail="the router answers this LAN address without a key",
                remedy="MNM_API_KEY in the box's secrets file (`mnm setup`, section 3)",
            )
        return Check(name="Key", status=Status.OK, detail="no MNM_API_KEY: keyless here")
    answer = bare.ask("GET", url, timeout=30)
    if answer.status == 401:
        return Check(name="Key", status=Status.OK, detail="a request without MNM_API_KEY: 401")
    return Check(
        name="Key",
        status=Status.FAIL if target.where is From.LAN else Status.WARN,
        detail=f"a request without MNM_API_KEY got {answer.status or answer.error}",
        remedy="the router reads the key from the box's secrets file: `mnm service install`",
    )


def ollama(target: Target, probe: Probe) -> Check | None:
    if target.ollama is None:
        return None
    url = f"{target.ollama}/api/tags"
    answer = probe.ask("GET", url, timeout=30)
    if answer.status != 200:
        return _failed("Ollama API", f"GET {url}", answer, target)
    try:
        count = len(parse(_Tags, answer.body, "Ollama's tags").models)
    except MalformedAnswerError as exc:
        return Check(name="Ollama API", status=Status.FAIL, detail=f"GET {url}: {exc}")
    return Check(name="Ollama API", status=Status.OK, detail=f"{url}: {count} models")


def run(target: Target, probe: Probe, bare: Probe) -> list[Check]:
    """The checks in order; when nothing answers the list, only the key line follows."""
    listed, answered = models(target, probe)
    done = [listed]
    if answered:
        done += [chat(target, probe), mcp(target, probe)]
        if (tags := ollama(target, probe)) is not None:
            done.append(tags)
    return [*done, key(target, bare)]


def probes(where: From, key_value: str | None, env_runner: EnvRunner) -> tuple[Probe, Probe]:
    """Where the requests leave from: this process, or a one-off container here."""
    if where is From.DOCKER:
        why = "a container on this machine"
        return one_off(key_value, env_runner, why), one_off(None, env_runner, why)
    return HostProbe(key_value), HostProbe(None)
