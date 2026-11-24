"""The router's HTTP side: `RouterServer` and its handler. Every request is
answered: the router's own answers (a preflight, the model lists, health, the busy count),
a refusal (no key, a stopped or unloadable instance), or the engine's reply, forwarded."""

from __future__ import annotations

import contextlib
import http.client
import json
import sys
import urllib.parse
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from http.client import HTTPResponse

from makenomistakes import keep_alive, manifest, router_metrics
from makenomistakes.bridge.peers import LocalPeers
from makenomistakes.bridge.protocol import Toolbox
from makenomistakes.comfy.guard import SANDBOXED, ComfyGuard
from makenomistakes.forward import (
    UPLOAD_BODY_LIMIT,
    UPSTREAM_TIMEOUT_SECONDS,
    QuickBindServer,
    TooLarge,
    error_body,
    streamed,
    with_model,
)
from makenomistakes.hardware.probe import SubprocessRunner
from makenomistakes.health_now import HealthReader
from makenomistakes.instance_types import InstanceError
from makenomistakes.journal import Quieter, error_message
from makenomistakes.loading_notes import (
    SSE_FIELDS,
    Frames,
    Opened,
    noted_model,
    recipe_note,
    wait_with_notes,
)
from makenomistakes.music_speed import task_wait
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.registry import spec_for
from makenomistakes.request_meter import Tap, measured_route
from makenomistakes.router import Router, rewrites
from makenomistakes.router_events import EVENTS_PATH
from makenomistakes.router_hosts import HostList, default_route_addresses
from makenomistakes.router_own import OWN_POSTS, ROUTER_ONLY_LOOKS, WELL_KNOWN, OwnRoutes
from makenomistakes.router_refusals import NOT_ANSWERING, SLOTS_REFUSED, Refusal, slot_action
from makenomistakes.routing import (
    BUSY_PATH,
    CHAT,
    HEALTH,
    MODELS,
    PROBE_HEADER,
    UPSTREAM,
    RouteError,
    named,
    route,
    scoped,
)
from makenomistakes.thinking_notes import answer_edit

NO_KEY = (
    "This router asks for its key (MNM_API_KEY): send it as `Authorization: Bearer <key>` "
    "(or `x-api-key`), the API key field of most clients."
)


@dataclass
class _Noted:
    """A stream of loading notes the reply continues."""

    frames: Frames
    chunked: bool = True
    opened: bool = False  # the stream's head went out: a refusal is its last chunk
    gone: bool = False  # the client left while the notes went out


def _said(text: str) -> str:
    """An engine's error answer in a few words: its JSON message, else the text, cut."""
    try:
        message = error_message(json.loads(text))
    except ValueError:
        message = ""
    return " ".join((message or text).split())[:400]


@dataclass(frozen=True)
class _Found:
    recipe: Recipe
    path: str
    body: bytes


ROOT_WORDS = "mnm's router: every model at /v1 (GET /v1/models lists them)."
FIXED_ANSWERS: dict[str, dict[str, str]] = {HEALTH: {"status": "ok"}, "": {"message": ROOT_WORDS}}


class RouterServer(QuickBindServer):
    def __init__(  # noqa: PLR0913 - the swappable parts
        self,
        listen: tuple[str, int],
        router: Router,
        tools: Toolbox | None = None,
        guard: ComfyGuard | None = None,
        peers: LocalPeers | None = None,
        note: Callable[[str], None] | None = None,
        ollama: bool = False,
        *,
        hosts: HostList | None = None,
    ) -> None:
        super().__init__(listen, _Handler)
        # the names it answers to; `cli.router_cmd.listen` passes every bound address
        self._hosts = hosts
        self.misdirected: set[tuple[str, str]] = set()  # (peer, host) journaled once
        self.cross_site: set[tuple[str, str]] = set()  # (peer, origin) journaled once
        self.ollama = ollama  # `MNM_OLLAMA_PORT`'s listener: its root says "Ollama is running"
        self.router = router
        self.counts = router.counts  # what `/metrics` counts, refusals of its own too
        self.tools = tools  # mnm's runs as MCP tools at /mcp; None: not served
        self.guard = guard  # ComfyUI's network, refused whatever it sends (comfy/guard.py)
        self.peers = peers or LocalPeers()  # who calls /mcp without a key (bridge/peers.py)
        # One line per /mcp call to the router's journal: an MCP client (Odysseus) shows a
        # refusal only as "unhandled errors in a TaskGroup".
        self._note = note
        self.quiet = Quieter(self.note)  # a peer's repeated refusals, collapsed
        self.well_known: set[str] = set()  # peers told once there is no OAuth metadata
        self._health: HealthReader | None = None  # `/__mnm/health`'s reads

    @property
    def hosts(self) -> HostList:
        """The allow-list given, else this listener's address, the settings' and the route's."""
        if self._hosts is None:
            setting = self.router.instances.settings.router_hosts
            bound = [str(self.server_address[0])]
            self._hosts = HostList.of(bound, setting, default_route_addresses())
        return self._hosts

    @property
    def health(self) -> HealthReader:
        """What `/__mnm/health` reads: the one `listen` gave every listener, else this
        box's, made at the first look (nothing is asked until a monitor calls)."""
        if self._health is None:
            self._health = HealthReader(self.router.instances.settings, SubprocessRunner())
        return self._health

    @health.setter
    def health(self, reader: HealthReader) -> None:
        self._health = reader

    def note(self, line: str) -> None:
        """The router's journal and its ring of recent lines, unless told otherwise."""
        (self._note or self.router.note)(line)

    def handle_error(self, request: object, client_address: object) -> None:
        """A client that drops its connection (an MCP client closing its GET stream, OpenCode
        at exit) is one journal line, not socketserver's traceback;
        anything else is still printed whole."""
        failed = sys.exc_info()[1]
        if isinstance(failed, ConnectionError):  # reset, aborted, a broken pipe
            peer = client_address[0] if isinstance(client_address, tuple) else client_address
            self.note(f"{peer} dropped the connection ({type(failed).__name__}).")
            return
        super().handle_error(request, client_address)  # type: ignore[arg-type]

    def service_actions(self) -> None:
        """Between requests (every half second): a quiet window's count is written."""
        super().service_actions()
        self.quiet.flush()


class _Handler(OwnRoutes):
    server: RouterServer

    def _serve(self) -> None:
        """Every request gets an answer: the router is the one way in, so a fault of its own
        is a 500 that names it, never a dropped connection (or, once a reply has begun, a
        closed one)."""
        self.responded = False  # one handler serves every request on a kept-alive connection
        self.reply_tail = b""  # this request's reply's end only (the meter's, the queue's)
        self.asked_for: str | None = None  # the instance it was routed to, for the journal
        self.journaled = False  # its journal line written, or left out (`ROUTER_ONLY_LOOKS`)
        try:
            self._answer()
        except Exception as exc:
            if self.responded:
                self.close_connection = True
            else:
                with contextlib.suppress(OSError):
                    message = f"mnm's router failed on this request: {exc!r}"
                    self.json_reply(500, error_body(message), cors=True)

    def _answer(self) -> None:
        guard = self.server.guard
        if guard is not None and guard.blocks(self.client_address[0]):
            self._drop_body()
            self.json_reply(403, error_body(SANDBOXED), close=True)
            return
        host = self.headers.get("Host")
        if host is not None and not self.server.hosts.allows(host):
            self._misdirected(host)
            return
        if self._health():  # keyless: an outside monitor reads it
            return
        if self.command != "OPTIONS" and not self._key_given():
            # Its body is read and dropped (up to a cap) so the client sees the 401 rather than
            # a broken pipe, and the connection ends: on a kept-alive one the next request
            # would otherwise be parsed out of what is left.
            self._drop_body()
            if self._is_mcp():  # the line first: the reply may end the connection
                self._mcp_note(401, "no key (MNM_API_KEY is set)")
            self.json_reply(401, error_body(NO_KEY), cors=True, close=True)
            return
        if self._is_mcp():
            self._mcp()
        elif self._cross_site():  # a web page on another site, a keyless router
            return
        elif not self._ollama() and not self._answered_here():  # Ollama's API: `/api/…`
            self._routed()

    def _routed(self) -> None:
        """A request for one instance: routed, loaded if parked, relayed; then its keep_alive."""
        found = self._route()
        if found is None:
            return
        if slot_action(self.command, found.path):  # The slots' files are mnm's own
            self.json_reply(403, error_body(SLOTS_REFUSED), cors=True)
            return
        self.asked_for = self.asked_for or found.recipe.name
        try:
            keep, body = keep_alive.asked(found.path, found.body, self.headers)
        except keep_alive.KeepAliveError as exc:
            self.json_reply(400, error_body(f"{exc} (0 unloads it after the reply)"), cors=True)
            return
        found = replace(found, body=body)
        router = self.server.router
        work = measured_route(self.command, found.path)  # Who waits, and for how long
        with router.request(found.recipe, work) as ticket:  # in flight before the load
            refused, noted = self._ready(found)
            if noted is not None and noted.gone:
                return  # the client left during the notes; its load was waited for
            if refused is None:
                if keep:  # a duration, or -1: its idle limit from now on
                    router.idle.keep(found.recipe, keep)
                router.queue.forwarded(ticket, found.recipe)
                ticket.tail = self._relay(found.recipe, found.path, found.body, noted).tail
            elif noted is None:
                self.json_reply(refused.status, error_body(refused.message), cors=True)
            else:
                self._end_noted(noted, refused.status, refused.message)
        if keep == 0 and refused is None:  # parked after the reply unless the router keeps it
            self._park(found.recipe)

    def _answered_here(self) -> bool:
        """The router's own answers, which never load anything; a saved file mnm cannot
        read (a newer or broken recipe) is the 503 `/v1/models` gives, never a 500 repr."""
        try:
            return self._own_answer()
        except InstanceError as exc:
            self.json_reply(503, error_body(str(exc)), cors=True)
            return True

    def _own_answer(self) -> bool:
        """A CORS preflight, the model list and one model, a parked instance's own model
        list, its health, the router's own routes."""
        chat = self.path.startswith(CHAT + "/")
        own = self.path.removeprefix(CHAT if chat else "").split("?", 1)[0].rstrip("/")
        looking = self.command in ("GET", "HEAD")
        if self.command == "OPTIONS":
            self.preflight()  # a browser's look first: its request follows
        elif looking and own == MODELS:
            self._models(chat)
        elif looking and own.startswith(MODELS + "/"):
            self._one(urllib.parse.unquote(own.removeprefix(MODELS + "/")), chat)
        elif looking and own in FIXED_ANSWERS:  # health; the root a browser opens
            self.json_reply(200, FIXED_ANSWERS[own], cors=True)
        elif looking and (own in ROUTER_ONLY_LOOKS or own.startswith(WELL_KNOWN)):
            self._not_here(own)
        elif looking and own == router_metrics.PATH and not chat:
            self._metrics()
        elif self.command == "POST" and own.startswith(OWN_POSTS):  # saved, tune
            self._own_post(own)
        elif looking and own == BUSY_PATH:
            self._busy()
        elif looking and own == manifest.PATH and not chat:  # what an app needs
            self.json_reply(200, manifest.of_router(self.server.router).model_dump(mode="json"))
        elif looking and own == EVENTS_PATH:  # its newest journal lines
            self.json_reply(200, self.server.router.events.recent().model_dump(mode="json"))
        elif looking and own.startswith(UPSTREAM) and own.endswith(MODELS):
            return self._parked_models(own.removeprefix(UPSTREAM).removesuffix(MODELS))
        else:
            return False
        return True

    def json_reply(
        self, status: int, value: object, cors: bool = False, close: bool = False
    ) -> None:
        """Every answer the router gives itself; one that is not a success is a journal line
        (`/mcp` has its own, `_mcp_note`), before the reply: it may end the connection."""
        if status >= 400 and not self._is_mcp() and not self.journaled:
            self._refusal_note(status, error_message(value))
        super().json_reply(status, value, cors, close)

    def _models(self, chat_only: bool) -> None:
        try:
            listing = self.server.router.models(chat_only)
        except InstanceError as exc:
            self.json_reply(503, error_body(str(exc)), cors=True)
            return
        self.json_reply(200, listing, cors=True)

    def _one(self, name: str, chat_only: bool) -> None:
        found = self.server.router.model(name, chat_only)
        if found is None:
            self.json_reply(404, error_body(f"no model named {name!r}"), cors=True)
        else:
            self.json_reply(200, found, cors=True)

    def _parked_models(self, name: str) -> bool:
        """`/upstream/<name>/v1/models` of a parked instance, answered without loading it (a
        client's model refresh); a loaded one's engine answers for itself, unless it has no
        such route (whisper-server): then the router does, so a client sees one model
        whether the instance is loaded or not. `<name>` may be a desk role, as anywhere."""
        router = self.server.router
        wanted = urllib.parse.unquote(name.strip("/"))
        named_as = named(wanted, router.instances.all(), router.aliases())
        if named_as is None:
            return False
        recipe = named_as.recipe
        if spec_for(recipe).lists_models and router.instances.status(recipe).running:
            return False
        found = router.model(wanted)
        if found is None:
            return False  # stopped or not downloaded: the routed request says so
        self.json_reply(200, {"object": "list", "data": [found]}, cors=True)
        return True

    def _route(self) -> _Found | None:
        """The instance this request is for, or None once the client has been told why not."""
        router = self.server.router
        try:
            body = self.read_body(UPLOAD_BODY_LIMIT)
            path, recipes = scoped(self.path, router.instances.all())
            kind = self.headers.get("Content-Type", "")
            found = route(path, body, kind, recipes, router.loadable, router.aliases())
        except TooLarge as exc:  # drained (a few MiB), then the connection ends
            self.drain_body()
            self._refusal_note(413, str(exc))
            self.journaled = True
            self.json_reply(413, error_body(f"Request refused: {exc}."), cors=True, close=True)
        except ValueError:  # JSON like every other refusal: an OpenAI client reads it
            self.json_reply(400, error_body("malformed request body"), cors=True, close=True)
        except RouteError as exc:
            self.json_reply(exc.status, error_body(str(exc)), cors=True)
        except InstanceError as exc:  # the saved instances could not be read
            self.json_reply(503, error_body(str(exc)), cors=True)
        else:
            if found.alias is not None:  # a desk role: the engine sees the instance's id
                router.answered_as(found.alias, found.recipe.name)
                self.asked_for = f"{found.recipe.name} (as {found.alias})"
                body = with_model(body, found.alias, found.recipe.name)
            return _Found(found.recipe, found.path, body)
        return None

    def _ready(self, found: _Found) -> tuple[Refusal | None, _Noted | None]:
        """Whether the instance may take the request now (`Router.ready`). A streamed chat
        request that waits past the grace gets loading notes meanwhile (`loading_notes`), and
        the stream they opened comes back to carry the reply."""
        router = self.server.router
        probe = bool(self.headers.get(PROBE_HEADER))
        model = None if probe else noted_model(self.command, found.path, found.body, self.headers)
        if model is None:
            return router.ready(found.recipe, self.command, probe), None
        noted = _Noted(Frames(model or found.recipe.name))

        def open_stream() -> None:
            noted.chunked = self.open_stream(200, "OK", SSE_FIELDS)
            noted.opened = True

        def send(seconds: float) -> None:
            text = recipe_note(found.recipe, seconds)
            self.write_piece(noted.frames.note(text), noted.chunked)

        try:
            refused, opened = wait_with_notes(
                lambda: router.ready(found.recipe, self.command, probe), open_stream, send
            )
        except Exception as exc:
            if not noted.opened:
                raise  # nothing sent yet: `_serve`'s 500 says it
            # The stream's head is out, so no 500 can follow: its last chunk says it.
            return Refusal(500, f"mnm's router failed on this request: {exc!r}"), noted
        if opened is Opened.GONE:
            noted.gone = True
            self.close_connection = True
        return refused, noted if opened is not Opened.NO else None

    def _end_noted(self, noted: _Noted, status: int, message: str) -> None:
        """A stream of loading notes that ends without the model's answer: the reason as its
        last chunk (and a journal line, as a refusal's reply would have)."""
        self._refusal_note(status, message)
        self.write_piece(noted.frames.error(message), noted.chunked)
        self.end_stream(noted.chunked)

    def _after_notes(self, noted: _Noted, recipe: Recipe, answer: HTTPResponse | None) -> int:
        """The engine's reply inside the stream the notes opened: its own stream as it comes,
        any other answer as the stream's error. The status the meter keeps (0: unknown)."""
        if answer is not None and answer.status == 200 and streamed(answer):
            if self.stream_body(answer, noted.chunked):
                self.end_stream(noted.chunked)
            return 200
        message = NOT_ANSWERING.format(name=recipe.name)
        if answer is not None:
            try:
                text = answer.read().decode("utf-8", errors="replace")
            except (OSError, http.client.HTTPException):
                text = ""
            said = _said(text)
            how = "without a stream" if answer.status == 200 else str(answer.status)
            message = f"{recipe.name} answered {how}" + (f": {said}" if said else ".")
        self._end_noted(noted, 502 if answer is None else answer.status, message)
        return 0 if answer is None else answer.status

    def _relay(self, recipe: Recipe, path: str, body: bytes, noted: _Noted | None = None) -> Tap:
        """To the engine on 127.0.0.1, rewritten as the instance asks (`rewrites`); after
        loading notes, into the stream they opened. A music task, silent until it ends, is
        waited for as long as its client waits (`task_wait`). What the engine answered: its
        status, and the end of its reply (the meter's, the queue's)."""
        wait = max(UPSTREAM_TIMEOUT_SECONDS, task_wait(recipe, path, body) or 0.0)
        upstream = http.client.HTTPConnection("127.0.0.1", recipe.port, timeout=wait)
        meter = self.server.router.meter
        try:
            measuring = meter.measure(recipe, self.command, path, body) if meter else None
            with measuring or nullcontext(Tap()) as tap:
                try:
                    answer = self.ask(upstream, rewrites(recipe, path, body, self.headers), path)
                except (OSError, http.client.HTTPException):
                    answer = None
                # What the engine answered, before the client is given it: a client gone
                # while the reply is passed on still leaves the engine's work done (a first
                # picture loaded the weights), and the meter must know so.
                tap.status = answer.status if answer is not None else 0
                if noted is not None:
                    tap.status = self._after_notes(noted, recipe, answer)
                    tap.tail = self.reply_tail
                elif answer is None or not self.pass_on(
                    answer, answer_edit(recipe, self.server.router, self.server.note)
                ):
                    # The engine cut a whole reply: nothing it did is known. (A stream cut
                    # midway is passed on as far as it went and keeps its status.)
                    tap.status = 0
                    message = NOT_ANSWERING.format(name=recipe.name)
                    self.json_reply(502, error_body(message), cors=True)
                else:
                    tap.tail = self.reply_tail
        finally:
            upstream.close()
        return tap

    # http.server dispatches on these names
    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _serve  # noqa: N815
