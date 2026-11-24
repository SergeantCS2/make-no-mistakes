"""Ollama's API on the router's HTTP side: `/api/version`, `/api/tags`, `/api/ps`,
`/api/show`, `/api/chat`, `/api/generate` and `/api/embed`, on the router's port and, with
`MNM_OLLAMA_PORT` set, on a second listener (Ollama's 11434 is what clients assume), whose root
answers "Ollama is running" as Ollama's does. The router's key applies as everywhere else.
Errors are Ollama's shape, `{"error": "…"}`, and each is a journal line. `ollama_api` holds the
shapes; this module asks the router and the engine."""

from __future__ import annotations

import http.client
import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from makenomistakes.engine_answers import (
    EmbeddingAnswer,
    ErrorAnswer,
    MalformedAnswerError,
    parse,
    said_error,
)
from makenomistakes.forward import (
    JSON_BODY_LIMIT,
    REPLY_TAIL,
    UPSTREAM_TIMEOUT_SECONDS,
    Forwarder,
    TooLarge,
)
from makenomistakes.hub.facts import CHAT_LIKE
from makenomistakes.instance_types import InstanceError, InstanceStatus
from makenomistakes.model_meta import serving_context
from makenomistakes.ollama_api import (
    NEVER,
    KeepAliveError,
    OllamaChat,
    OllamaEmbed,
    OllamaGenerate,
    OllamaShow,
    chat_body,
    embed_answer,
    embed_body,
    generate_body,
    keep_alive_seconds,
    running,
    serves,
    show,
    tag,
    version,
    wanted_name,
)
from makenomistakes.ollama_stream import Kind, Reply, load_only
from makenomistakes.recipes import Recipe
from makenomistakes.request_meter import Tap
from makenomistakes.router_models import listed, roles_listed
from makenomistakes.router_refusals import NO_DOCKER, Refusal, not_loadable
from makenomistakes.routing import GoneMemberError, RouteError, find

if TYPE_CHECKING:
    from makenomistakes.router_http import RouterServer

PREFIX = "/api"
RUNNING = "Ollama is running"
NDJSON = [("Content-Type", "application/x-ndjson"), ("Access-Control-Allow-Origin", "*")]
CHAT_PATH = "/v1/chat/completions"
EMBED_PATH = "/v1/embeddings"
NOT_SERVED = (
    "{name} ({kind}) is not a model the Ollama API serves: it serves chat, coding, vision and "
    "embedding models. Ask the router's OpenAI API for it (/upstream/{name}/v1)."
)
NOT_HERE = (
    "mnm's router does not serve {path}: models are added with `mnm add <link>` and removed "
    "with `mnm rm`; it serves /api/tags, /api/ps, /api/show, /api/chat, /api/generate, "
    "/api/embed and /api/version"
)
LEFT_OUT = "{name}: Ollama options not applied: {options}"
NO_SUCH_OPTION = "the engines take no such option"
OWN_CONTEXT = "the context is the instance's own: `mnm set {name} ctx=…`"
FORMAT_WHY = 'only "json" or a JSON schema becomes a reply format'


class OllamaRoutes(Forwarder):
    """The `/api/…` answers; `router_http._Handler` is one."""

    server: RouterServer
    journaled = False  # the journal line is written (or left out): the reply adds none
    asked_for: str | None = None

    def _refusal_note(self, status: int, why: str) -> None:  # the handler's journal line
        raise NotImplementedError

    def _ollama(self) -> bool:
        """Answered here when it is Ollama's: True once the client has its answer."""
        path = self.path.split("?", 1)[0].rstrip("/")
        looking = self.command in ("GET", "HEAD")
        if path == "" and looking and self.server.ollama:
            self._text(RUNNING)
            return True
        if path != PREFIX and not path.startswith(PREFIX + "/"):
            return False
        routes: dict[tuple[bool, str], Callable[[], None]] = {
            (True, "/api/version"): lambda: self.json_reply(200, version(), cors=True),
            (True, "/api/tags"): self._tags,
            (True, "/api/ps"): self._ps,
            (False, "/api/show"): self._show,
            (False, "/api/chat"): lambda: self._work("chat", OllamaChat),
            (False, "/api/generate"): lambda: self._work("generate", OllamaGenerate),
            (False, "/api/embed"): self._embed,
        }
        answer = routes.get((looking, path)) if looking or self.command == "POST" else None
        if answer is None:
            self._drop()
            self.ollama_error(404, NOT_HERE.format(path=path))
        else:
            answer()
        return True

    # --- the lists ---------------------------------------------------------------------

    def _listed(self) -> list[tuple[Any, Any]] | None:
        """(status, files' facts) of every loaded or parked instance the API serves; None
        once the client has been told Docker could not be asked."""
        router = self.server.router
        try:
            statuses = router.listed_statuses()  # reused for a moment
        except InstanceError as exc:
            self.ollama_error(503, str(exc))
            return None
        broken = next((s.error for s in statuses if s.error), None)
        if broken is not None:
            self.ollama_error(503, NO_DOCKER.format(error=broken))
            return None
        return [
            (s, router.meta.get(s.recipe))
            for s in statuses
            if (s.running or s.parked) and serves(s.recipe)
        ]

    def _tags(self) -> None:
        """Every model the API serves, then each desk role whose member it lists."""
        if (found := self._listed()) is None:
            return
        tags = {s.recipe.name: tag(s, meta) for s, meta in found}
        aliases = self.server.router.aliases()
        roles = roles_listed(aliases, tags, lambda role, t: {**t, "name": role, "model": role})
        self.json_reply(200, {"models": [*tags.values(), *roles]}, cors=True)

    def _ps(self) -> None:
        if (found := self._listed()) is None:
            return
        idle = self.server.router.idle
        models = [running(s, meta, _when(idle.expires_at(s))) for s, meta in found if s.running]
        self.json_reply(200, {"models": models}, cors=True)

    def _show(self) -> None:
        request = self._parsed(OllamaShow)
        if request is None:
            return
        if not (request.model or request.name):
            self.ollama_error(400, "the request names no model (`model`)")
            return
        recipe = self._recipe(request.model or request.name)
        if recipe is not None:
            status = self.server.router.instances.status(recipe)
            self.json_reply(200, show(status, self.server.router.meta.get(recipe)), cors=True)

    # --- the work ------------------------------------------------------------------------

    def _work(self, kind: Kind, shape: type[OllamaChat] | type[OllamaGenerate]) -> None:
        request = self._parsed(shape)
        recipe = self._recipe(request.model) if request is not None else None
        if request is None or recipe is None:
            return
        empty = not (request.messages if isinstance(request, OllamaChat) else request.prompt)
        if recipe.modality not in CHAT_LIKE and not empty:  # loading or unloading one is fine
            self.ollama_error(400, f"{recipe.name} is an embedding model: ask /api/embed")
            return

        def ask(load_ns: int) -> bytes:
            if empty:  # Ollama's way to load (or, with keep_alive 0, unload) a model
                reply = load_only(request.model, kind, "load", load_ns)
                self.json_reply(200, reply, cors=True)
                return b""
            built = chat_body if isinstance(request, OllamaChat) else generate_body
            body, left = built(request, recipe.served_name)  # type: ignore[arg-type]
            left = [o for o in left if o != "num_ctx" or _more_context(request.options, recipe)]
            if left:
                said = left_out(recipe.name, left)
                self.server.note(LEFT_OUT.format(name=recipe.name, options=said))
            return self._ask_engine(recipe, CHAT_PATH, body, Reply(request.model, kind, load_ns))

        self._served(recipe, request.keep_alive, ask, unload=empty, kind=kind)

    def _embed(self) -> None:
        request = self._parsed(OllamaEmbed)
        recipe = self._recipe(request.model) if request is not None else None
        if request is None or recipe is None:
            return
        if recipe.modality in CHAT_LIKE:
            self.ollama_error(
                400, f"{recipe.name} is a chat model: it has no embedding output; ask /api/chat"
            )
            return

        def ask(load_ns: int) -> bytes:
            body = embed_body(request, recipe.served_name)
            return self._ask_engine(recipe, EMBED_PATH, body, None, request.model, load_ns)

        self._served(recipe, request.keep_alive, ask)

    def _served(
        self,
        recipe: Recipe,
        keep_alive: str | int | float | None,
        ask: Callable[[int], bytes],
        unload: bool = False,
        kind: Kind = "chat",
    ) -> None:
        """The request with its instance counted in flight and loaded first; then what its
        `keep_alive` asks: 0 parks it once the reply is done (an empty chat with 0 only
        unloads), negative or a duration is its idle limit from now on. An empty chat only
        loads: it is no work, and never counted in the queue."""
        try:
            keep = keep_alive_seconds(keep_alive)
        except KeepAliveError as exc:
            self.ollama_error(400, str(exc))
            return
        router = self.server.router
        self.asked_for = recipe.name
        if unload and keep == 0:
            self._unload(recipe, kind)
            return
        with router.request(recipe, not unload) as ticket:
            began = time.monotonic_ns()
            refused = router.ready(recipe, "POST", False)
            if refused is not None:
                self.ollama_error(refused.status, refused.message)
                return
            if keep:
                router.idle.keep(recipe, keep)
            router.queue.forwarded(ticket, recipe)
            ticket.tail = ask(time.monotonic_ns() - began)
        if keep == 0:
            self._park(recipe)

    def _unload(self, recipe: Recipe, kind: Kind) -> None:
        self._park(recipe)
        self.json_reply(200, load_only(recipe.name, kind, "unload", 0), cors=True)

    def _park(self, recipe: Recipe) -> None:
        """A client's `keep_alive: 0`, honoured unless the router keeps the model (a journal
        line either way; the client's answer is Ollama's either way)."""
        try:
            kept = self.server.router.park(recipe)
        except InstanceError as exc:  # after the reply: the journal is the only place to say
            self.server.note(f"{recipe.name} could not be parked for keep_alive 0: {exc}")
            return
        if kept is None:
            self.server.note(f"Parked {recipe.name}: a client's keep_alive 0.")
        elif kept.loaded:
            self.server.note(f"{recipe.name} stays loaded despite keep_alive 0: {kept.why}.")

    def _ask_engine(
        self,
        recipe: Recipe,
        path: str,
        body: dict[str, Any],
        reply: Reply | None,
        model: str = "",
        load_ns: int = 0,
    ) -> bytes:
        """The engine's answer in Ollama's words: a chat stream as NDJSON, a whole chat or
        an embedding as one object, an engine error as Ollama's error. The end of what was
        passed on (the queue reads it)."""
        upstream = http.client.HTTPConnection(
            "127.0.0.1", recipe.port, timeout=UPSTREAM_TIMEOUT_SECONDS
        )
        meter = self.server.router.meter
        try:
            measuring = meter.measure(recipe, "POST", path) if meter else None
            with measuring or nullcontext(Tap()) as tap:
                data = json.dumps(body).encode()
                began = time.monotonic_ns()
                try:
                    upstream.request("POST", path, data, {"Content-Type": "application/json"})
                    answer = upstream.getresponse()
                except (OSError, http.client.HTTPException):
                    tap.status = 0
                    self.ollama_error(502, f"{recipe.name} is not answering (`mnm logs`)")
                    return b""
                tap.status = answer.status
                if answer.status != 200:
                    tap.status = 0 if self._engine_error(recipe, answer) else answer.status
                elif reply is not None and body.get("stream"):
                    self._ndjson(answer, reply)
                else:
                    took = load_ns + time.monotonic_ns() - began
                    self._whole(answer, reply, model, took, load_ns)
                tap.tail = self.reply_tail
        finally:
            upstream.close()
        return tap.tail

    def _ndjson(self, answer: http.client.HTTPResponse, reply: Reply) -> None:
        """The engine's stream as NDJSON as it comes; an error line (the engine's, or a chunk
        mnm could not read) ends it, with a journal line."""
        chunked = self.open_stream(200, "OK", NDJSON)
        raw = b""  # the engine's own stream's end: llama.cpp's timings, for the meter and queue
        try:
            while reply.error is None and (chunk := answer.read1(1 << 16)):
                raw = (raw + chunk)[-REPLY_TAIL:]
                for line in reply.feed(chunk):
                    self.write_piece(line, chunked)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
            self.reply_tail = raw
            return
        for line in reply.finish():
            self.write_piece(line, chunked)
        self.end_stream(chunked)
        self.reply_tail = raw
        if reply.error is not None:
            self._refusal_note(502, reply.error)

    def _whole(
        self,
        answer: http.client.HTTPResponse,
        reply: Reply | None,
        model: str,
        took_ns: int,
        load_ns: int,
    ) -> None:
        """A whole chat or embeddings answer as Ollama's one object; one mnm cannot read is
        a 502 saying where."""
        try:
            raw = answer.read()
            if reply is not None:
                result = reply.whole(raw)
            else:
                embedded = parse(EmbeddingAnswer, raw, "embeddings answer")
                result = embed_answer(model, embedded, took_ns, load_ns)
        except (OSError, http.client.HTTPException):
            self.ollama_error(502, f"{model}: the engine's answer was cut off (`mnm logs`)")
            return
        except MalformedAnswerError as exc:
            self.ollama_error(502, f"{model}: {exc}")
            return
        self.reply_tail = raw[-REPLY_TAIL:]  # llama.cpp's timings, for the meter and queue
        self.json_reply(200, result, cors=True)

    def _engine_error(self, recipe: Recipe, answer: http.client.HTTPResponse) -> bool:
        """The engine's refusal in Ollama's shape; True when nothing it said could be read."""
        try:
            text = answer.read().decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException):
            text = ""
        try:
            said = said_error(parse(ErrorAnswer, text, "error").error)
        except MalformedAnswerError:
            said = ""
        message = " ".join((said or text).split())[:400]
        self.ollama_error(answer.status, f"{recipe.name} answered {answer.status}: {message}")
        return not text

    # --- the pieces ------------------------------------------------------------------------

    def _recipe(self, model: str) -> Recipe | None:
        """The instance a request names (its name, `name:latest`, or the one engine id),
        loaded or parked; None once the client has been told why not."""
        router = self.server.router
        try:
            recipes = router.instances.all()
            found = find(wanted_name(model), recipes, router.loadable, router.aliases())
            recipe = found.recipe
            if found.alias:  # a desk role
                router.answered_as(found.alias, recipe.name)
        except GoneMemberError as exc:  # a desk role whose instance is gone: say which
            self.ollama_error(404, str(exc))
            return None
        except RouteError:
            self.ollama_error(404, f"model '{model}' not found (`mnm ls` lists them)")
            return None
        except InstanceError as exc:
            self.ollama_error(503, str(exc))
            return None
        if not serves(recipe):
            kind = recipe.modality.value.replace("_", " ")
            self.ollama_error(400, NOT_SERVED.format(name=recipe.name, kind=kind))
            return None
        if (refused := _refused(router.instances.status(recipe))) is not None:
            self.ollama_error(refused.status, refused.message)
            return None
        return recipe

    def _parsed[T: BaseModel](self, shape: type[T]) -> T | None:
        try:
            return shape.model_validate_json(self.read_body(JSON_BODY_LIMIT) or b"{}")
        except TooLarge as exc:  # drained (a few MiB), then the connection ends
            self.drain_body()
            self.close_connection = True
            self.ollama_error(413, f"Request refused: {exc}.")
            return None
        except (ValueError, ValidationError) as exc:
            first = exc.errors()[0] if isinstance(exc, ValidationError) else None
            where = ".".join(str(p) for p in first["loc"]) if first else ""
            said = f"{where}: {first['msg']}" if first else str(exc)
            self.ollama_error(400, f"the request could not be read ({said})")
            return None

    def _drop(self) -> None:
        try:
            self.read_body(1 << 20)
        except ValueError:
            self.close_connection = True

    def _text(self, text: str) -> None:
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def ollama_error(self, status: int, message: str) -> None:
        """Ollama's error shape, with the router's journal line for it."""
        self._refusal_note(status, message)
        self.journaled = True  # the line is written: the reply's own would say less
        self.json_reply(status, {"error": message}, cors=True)


def _when(epoch: float) -> str:
    if epoch == float("inf"):
        return NEVER
    return datetime.fromtimestamp(epoch, UTC).isoformat().replace("+00:00", "Z")


def left_out(name: str, left: list[str]) -> str:
    """The options left out, each with why: "num_ctx (the context is …); mirostat, seed (the
    engines take no such option)"."""
    said = []
    if "num_ctx" in left:
        said.append(f"num_ctx ({OWN_CONTEXT.format(name=name)})")
    if "format" in left:
        said.append(f"format ({FORMAT_WHY})")
    if others := [o for o in left if o not in ("num_ctx", "format")]:
        said.append(f"{', '.join(others)} ({NO_SUCH_OPTION})")
    return "; ".join(said)


def _more_context(sent: dict[str, Any] | None, recipe: Recipe) -> bool:
    """A client's `num_ctx` above what the instance serves (worth a journal line; at or below
    it, as Home Assistant's card sets it, the request fits and nothing is said)."""
    asked = (sent or {}).get("num_ctx")
    have = serving_context(recipe) or 0
    return not isinstance(asked, int) or asked > have


def _refused(status: InstanceStatus) -> Refusal | None:
    """The OpenAI routes' words for an instance a request may not load: Docker that cannot
    be asked is a 503, never "not downloaded"; stopped or not
    downloaded a 409; None when it is loaded or parked."""
    if status.error:
        return Refusal(503, NO_DOCKER.format(error=status.error))
    if listed(status):
        return None
    return not_loadable(status, "POST", probe=False)
