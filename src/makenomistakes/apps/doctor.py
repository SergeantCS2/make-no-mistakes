"""`mnm doctor --app <app>`: the router as the app sees it, from where the app sits
(`probe.py`), with what its file names (`AppCard.uses`): the model list and the id it picks,
a chat that calls a tool and reads its result, an embedding, a rerank at the exact URL, MCP's
`initialize`, the context floor, the key, and whether a file mnm wrote is still current.
Each line is a `doctor.Check` (PASS/WARN/FAIL with a remedy), as `mnm doctor` prints them.
Every answer is typed at the edge (`apps.answers`, `engine_answers`): one that does not fit
is a FAIL line saying where, never a traceback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from makenomistakes.apps.answers import McpHello, ModelList, error_text
from makenomistakes.apps.card import AppCard
from makenomistakes.apps.files import up_to_date
from makenomistakes.apps.floors import floor_for, floor_met, floor_note
from makenomistakes.apps.names import Vantage
from makenomistakes.apps.probe import Answer, Probe
from makenomistakes.doctor_types import Check, Status
from makenomistakes.engine_answers import (
    ChatAnswer,
    EmbeddingAnswer,
    MalformedAnswerError,
    ToolCallPart,
    parse,
)
from makenomistakes.manifest import Manifest

TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two whole numbers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}
ASK = "What is 2 + 3? Use the add tool."
NO_THINKING = {"enable_thinking": False}  # a thinking model spends a short reply thinking
UNREADABLE = "the router's answer is not what an OpenAI client reads: `mnm logs <instance>`"


def _why(answer: Answer, vantage: Vantage) -> tuple[str, str]:
    """A failed answer in words, and what to do about it."""
    if answer.status == 0:
        reach = (
            "the container must map host.docker.internal (`--add-host host.docker.internal:"
            "host-gateway`) and the router must listen on Docker's bridge (MNM_ROUTER_HOST "
            "0.0.0.0 or local)"
            if vantage is Vantage.DOCKER
            else "is the router running? `mnm service status`"
        )
        return f"nothing answered: {answer.error}", reach
    text = error_text(answer.body)[:600]
    if answer.status == 401:
        return f"401: {text}", "the router asks for MNM_API_KEY: the app must send it"
    if answer.status == 403:  # the router's own refusal of this peer (the answer says why)
        return f"403: {text}", "`mnm mcp` says who may call the router without a key"
    if answer.status in (404, 409):
        return f"{answer.status}: {text}", "`mnm ls` shows what is stopped or gone"
    return f"{answer.status}: {text}", "`mnm logs <instance>` and the router's journal say more"


def _failed(name: str, what: str, answer: Answer, vantage: Vantage) -> Check:
    detail, remedy = _why(answer, vantage)
    return Check(name=name, status=Status.FAIL, detail=f"{what}: {detail}", remedy=remedy)


def _malformed(name: str, what: str, exc: MalformedAnswerError) -> Check:
    return Check(name=name, status=Status.FAIL, detail=f"{what}: {exc}", remedy=UNREADABLE)


def models_check(
    card: AppCard, probe: Probe, vantage: Vantage
) -> tuple[Check, Answer, ModelList | None]:
    """The model list as the app asks for it, and whether it holds the id the file picks."""
    url = f"{card.uses.base_url}/models"
    answer = probe.ask("GET", url, timeout=30)
    name = "Model list"
    if answer.status != 200:
        return _failed(name, f"GET {url}", answer, vantage), answer, None
    try:
        listing = parse(ModelList, answer.body, "model list")
    except MalformedAnswerError as exc:
        return _malformed(name, f"GET {url}", exc), answer, None
    ids = [m.id for m in listing.data]
    wanted = card.uses.chat_model
    if wanted is None or wanted not in ids:
        found = Check(
            name=name,
            status=Status.WARN,
            detail=f"GET {url} lists {', '.join(ids) or 'nothing'}, not {wanted or 'a chat model'}",
            remedy="`mnm sets default <set>` makes a set's roles ids; `mnm ls` shows what is "
            "stopped",
        )
        return found, answer, listing
    ok = Check(name=name, status=Status.OK, detail=f"{url} lists {wanted} ({len(ids)} ids)")
    return ok, answer, listing


def _call_back(call: ToolCallPart) -> dict[str, Any]:
    """The call as the next request carries it back (arguments as JSON text, as sent)."""
    arguments = call.function.arguments
    text = arguments if isinstance(arguments, str) else json.dumps(arguments or {})
    return {
        "id": call.id or "call_0",
        "type": "function",
        "function": {"name": call.function.name or "add", "arguments": text},
    }


def tool_check(card: AppCard, probe: Probe, vantage: Vantage) -> Check:
    """One call of the `add` tool, then its result sent back for the answer: what an agent
    does on every turn (a template that does not declare tools fails the first half)."""
    name, model = "Tool call", card.uses.chat_model
    url = f"{card.uses.base_url}/chat/completions"
    ask: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": ASK}],
        "tools": [TOOL],
        "tool_choice": "required",
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": NO_THINKING,
    }
    first = probe.ask("POST", url, json.dumps(ask))
    if first.status != 200:
        return _failed(name, f"POST {url}", first, vantage)
    try:
        answer = parse(ChatAnswer, first.body, "chat answer")
    except MalformedAnswerError as exc:
        return _malformed(name, f"POST {url}", exc)
    message = answer.choices[0].message if answer.choices else None
    calls = (message.tool_calls or []) if message is not None else []
    if message is None or not calls:
        said = "with no choices at all" if message is None else "without calling the tool"
        return Check(
            name=name,
            status=Status.WARN,
            detail=f"{model} answered {said}",
            remedy="its chat template may not declare tools (`mnm router manifest`: tools); "
            "an agent needs a model that calls them",
        )
    call = _call_back(calls[0])
    back = {
        **ask,
        "tool_choice": "none",
        "messages": [
            *ask["messages"],
            {"role": "assistant", "content": message.content or "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "5"},
        ],
    }
    second = probe.ask("POST", url, json.dumps(back))
    if second.status != 200:
        return _failed(name, "the tool's result", second, vantage)
    try:
        parse(ChatAnswer, second.body, "chat answer")
    except MalformedAnswerError as exc:
        return _malformed(name, "the tool's result", exc)
    called = call["function"]["name"]
    return Check(name=name, status=Status.OK, detail=f"{model} called {called} and read its result")


def embed_check(card: AppCard, probe: Probe, vantage: Vantage) -> Check:
    url = f"{card.uses.embed_base or card.uses.base_url}/embeddings"
    model = card.uses.embed_model
    answer = probe.ask("POST", url, json.dumps({"model": model, "input": "a lighthouse"}))
    if answer.status != 200:
        return _failed("Embedding", f"POST {url}", answer, vantage)
    try:
        found = parse(EmbeddingAnswer, answer.body, "embeddings answer")
    except MalformedAnswerError as exc:
        return _malformed("Embedding", f"POST {url}", exc)
    size = len(found.data[0].embedding) if found.data else 0
    status = Status.OK if size else Status.FAIL
    return Check(name="Embedding", status=status, detail=f"{model}: {size} numbers from {url}")


def rerank_check(manifest: Manifest, probe: Probe, vantage: Vantage, base: str) -> Check:
    url = f"{base}/rerank"  # the exact URL a reranking client is given
    body = {"model": "rerank", "query": "a lighthouse", "documents": ["a lamp by the sea", "a cat"]}
    answer = probe.ask("POST", url, json.dumps(body))
    if answer.status != 200:
        return _failed("Rerank", f"POST {url}", answer, vantage)
    member = manifest.aliases.get("rerank", "rerank")
    return Check(name="Rerank", status=Status.OK, detail=f"rerank ({member}) at {url}")


def mcp_check(card: AppCard, probe: Probe, vantage: Vantage) -> Check:
    """MCP's `initialize` as the app sends it: `probe` carries the key only when the app's MCP
    client can (Odysseus's and Goose's cannot: then a key set refuses them, said so)."""
    url = card.uses.mcp_url or ""
    hello = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "mnm-doctor", "version": "1"},
        },
    }
    answer = probe.ask("POST", url, json.dumps(hello), timeout=30)
    what = f"initialize at {url}"
    if answer.status == 401 and not card.uses.mcp_takes_key:
        return Check(
            name="MCP",
            status=Status.FAIL,
            detail=f"{what}: 401: {card.label} cannot send MNM_API_KEY to /mcp",
            remedy="its MCP tools work only with no key set (`mnm mcp` says why)",
        )
    if answer.status != 200:
        return _failed("MCP", what, answer, vantage)
    try:
        server = parse(McpHello, answer.body, "MCP answer").result.serverInfo.name
    except MalformedAnswerError as exc:
        return _malformed("MCP", what, exc)
    return Check(name="MCP", status=Status.OK, detail=f"{what}: {server or 'answered'}")


def floor_check(card: AppCard, listing: ModelList | None, manifest: Manifest) -> Check | None:
    """The context a request gets, as the app's model list says, against the app's floor."""
    model = card.uses.chat_model
    have = listing.context(model) if listing is not None and model else None
    entry = next((m for m in manifest.models if model in (m.name, *m.aliases)), None)
    if have is None or entry is None:
        return None
    floor = floor_for(card.app)
    note = floor_note(entry.name, have, entry.slots, card.label, floor)
    if floor_met(have, floor):
        return Check(name="Context", status=Status.OK, detail=note)
    return Check(
        name="Context", status=Status.FAIL if floor and floor.hard else Status.WARN, detail=note
    )


def key_check(card: AppCard, manifest: Manifest, probe_without: Probe) -> Check:
    """With a key set: the router refuses a request without it (so the app must send it, and
    its file says how); without one: nothing to send."""
    if not manifest.key_required:
        return Check(
            name="Key", status=Status.OK, detail="No MNM_API_KEY: the router answers keyless here"
        )
    bare = probe_without.ask("GET", f"{card.uses.base_url}/models", timeout=30)
    if bare.status == 401:
        return Check(
            name="Key",
            status=Status.OK,
            detail=f"the router refuses a request without MNM_API_KEY; {card.label}'s file "
            "names the key (the app's environment holds it)",
        )
    return Check(
        name="Key",
        status=Status.WARN,
        detail=f"MNM_API_KEY is set, yet a request without it got {bare.status or bare.error}",
        remedy="the router reads the key from the secrets file: `mnm service install` restarts it",
    )


def file_check(card: AppCard, written: list[Path]) -> list[Check]:
    """Each file mnm wrote for the app (the ledger's) that no longer says what mnm would write
    now (a model added, the key set, the desk changed). A card has one file at most."""
    checks: list[Check] = []
    # Only the card's own file: `mnm apps` also records the files it writes beside it (an app's
    # `.env`, its compose file), which the card does not render.
    name = Path(card.files[0].label).name if card.files else None
    own = card.files[0].default if card.files else None
    # The app's own place when it has one: a copy `--write --to` put elsewhere is kept in the
    # ledger for backups, never the app's file (else a librechat.yaml written into
    # another folder reads as a second App file).
    mine = [p for p in written if p == own.expanduser().resolve()] if own is not None else None
    for path in mine if mine is not None else [p for p in written if p.name == name]:
        current = up_to_date(card.files[0], path)
        if current is None:
            continue
        if current:
            checks.append(Check(name="App file", status=Status.OK, detail=f"{path} is current"))
        else:
            checks.append(
                Check(
                    name="App file",
                    status=Status.WARN,
                    detail=f"{path} differs from what mnm writes now",
                    remedy=f"`mnm url --for {card.app.value} --write` updates it (merged, "
                    "a copy kept)",
                )
            )
    return checks
