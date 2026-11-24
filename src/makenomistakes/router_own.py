"""The router's own small answers:
`POST /__mnm/saved/<name>`, `GET /metrics`, `POST /__mnm/tune/<name>` (`mnm tune` asks the
tuner's loop in this process), `POST /__mnm/tune-hold` (the nightly holds the tuner off the
card), `POST /__mnm/adopted` (`mnm machine adopt`'s journal line), `GET /__mnm/health`, the
421 for a name it does not answer to, the 404s for what only an engine has, and the journal
line of every refusal the router answers itself."""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable
from typing import TYPE_CHECKING

from makenomistakes import router_metrics
from makenomistakes.adopt_words import ADOPT_BODY_LIMIT, AdoptCounts, journal_line
from makenomistakes.comfy.sandbox import NETWORK
from makenomistakes.forward import TooLarge, error_body
from makenomistakes.health import HEALTH_PATH
from makenomistakes.health_now import for_router
from makenomistakes.journal import reason, shown
from makenomistakes.keep_alive import SAVED_PATH
from makenomistakes.readings_filter import here_of
from makenomistakes.router_hosts import host_name
from makenomistakes.router_mcp import McpRoutes
from makenomistakes.routing import ADOPTED_PATH, TRAIN_PATH, TUNE_HOLD_PATH, TUNE_PATH

if TYPE_CHECKING:
    from makenomistakes.router import TuneHook
    from makenomistakes.router_http import RouterServer

# `{"action": "tune", "mode": …, "minutes": …}` (a quality tune's plan is in the queue, its
# ask names it by fingerprint), a hold's `{"seconds": …, "who": …}`, a training's ask
JOB_BODY_LIMIT = 4096
NO_TUNER = "this router runs no tuner (an older `mnm router`?): `mnm update`, then ask again"
NO_TRAINER = "this router runs no trainer (an older `mnm router`?): `mnm update`, then ask again"


OWN_POSTS = (SAVED_PATH, TUNE_PATH, TUNE_HOLD_PATH, ADOPTED_PATH, TRAIN_PATH)

# llama-server's own pages, which Odysseus asks at the endpoint's root before the model list
# (`src/model_context.py:425-454` in 0.20.19); the router serves many engines, so it has none.
ROUTER_ONLY_LOOKS = frozenset({"/slots", "/props"})
ONE_ENGINE_ONLY = (
    "The router serves many models, so it has no {path} of its own: `GET /v1/models` gives "
    "each one's context (context_length), and /upstream/<name>{path} asks one engine."
)
WELL_KNOWN = "/.well-known/"  # an MCP client's OAuth look (LibreChat's); mnm has none
NO_OAUTH = (
    "mnm serves no OAuth metadata: its /mcp needs no sign-in (when MNM_API_KEY is set, the key "
    "goes in the Authorization header as `Bearer <key>`)."
)


class OwnRoutes(McpRoutes):
    server: RouterServer
    journaled: bool  # this request's journal line written, or left out on purpose
    asked_for: str | None  # the instance it was routed to, for the journal

    def _own_post(self, own: str) -> None:
        if own == ADOPTED_PATH:
            self._adopted()
        elif own == TUNE_HOLD_PATH:
            self._tune_hold()
        elif own.startswith(TUNE_PATH):
            self._tune(urllib.parse.unquote(own.removeprefix(TUNE_PATH)))
        elif own.startswith(TRAIN_PATH):
            self._train(urllib.parse.unquote(own.removeprefix(TRAIN_PATH)))
        else:
            self._saved(urllib.parse.unquote(own.removeprefix(SAVED_PATH)))

    def _busy(self) -> None:
        """`GET /__mnm/busy`: in flight, arrivals, what waits, the tune running."""
        router = self.server.router
        busy: dict[str, object] = {"busy": router.in_flight(), "arrivals": router.arrivals}
        if waiting := router.queue.waiting():  # only while something waits
            busy["queue"] = {name: now.model_dump() for name, now in waiting.items()}
        if router.tuner is not None and (tuning := router.tuner.line()) is not None:
            busy["tuning"] = tuning  # only while a tune runs
        if router.trainer is not None and (training := router.trainer.line()) is not None:
            busy["training"] = training  # only while a training runs
        self.json_reply(200, busy)

    def _tune(self, name: str) -> None:
        """Queue, run now or cancel a tune; the tuner answers in words."""
        self._ask_tuner(lambda tuner, body: tuner.ask(name, body), JOB_BODY_LIMIT)

    def _tune_hold(self) -> None:
        """Hold every tune off the card, or let go; the tuner answers in words."""
        self._ask_tuner(lambda tuner, body: tuner.hold(body), JOB_BODY_LIMIT)

    def _ask_tuner(self, ask: Callable[[TuneHook, bytes], tuple[int, str]], limit: int) -> None:
        tuner = self.server.router.tuner
        self._ask_job(None if tuner is None else lambda body: ask(tuner, body), NO_TUNER, limit)

    def _train(self, run_id: str) -> None:
        """Wake the trainer for a queued training, or cancel one; it answers in words."""
        trainer = self.server.router.trainer
        asked = None if trainer is None else lambda b: trainer.ask(run_id, b)
        self._ask_job(asked, NO_TRAINER, JOB_BODY_LIMIT)

    def _ask_job(
        self, ask: Callable[[bytes], tuple[int, str]] | None, absent: str, limit: int
    ) -> None:
        """A tune or training route's body, asked of its job. One the router cannot read is
        refused in words (413 over `limit`, 400 malformed) and the connection ends with the
        reply, so what is left of an unread body never reads as the next request; never
        asked as an empty body, which would queue a default speed tune for the idle minute."""
        try:
            body = self.read_body(limit)
        except TooLarge as exc:
            self.drain_body()  # a few MiB, so the 413 reaches the client
            self.json_reply(413, error_body(f"Request refused: {exc}."), close=True)
            return
        except ValueError:
            self.json_reply(400, error_body("malformed request body"), close=True)
            return
        status, said = (409, absent) if ask is None else ask(body)
        answer = {"message": said} if status < 400 else error_body(said)
        self.json_reply(status, answer)

    def _adopted(self) -> None:
        """`POST /__mnm/adopted`: `mnm machine adopt` ended; its counts become one
        journal line, worded here (a body of numbers, never text to print)."""
        try:
            counts = AdoptCounts.model_validate_json(self.read_body(ADOPT_BODY_LIMIT) or b"{}")
        except (ValueError, TooLarge):
            self.json_reply(400, error_body("the body is not adopt's counts"))
            return
        state = self.server.router.instances.settings.expanded_state_dir()
        self.server.note(journal_line(counts, here_of(state).words))
        self.json_reply(200, {"adopted": True})

    def _saved(self, name: str) -> None:
        """`POST /__mnm/saved/<name>` (an instance's name, else 404): a client's keep_alive goes."""
        router = self.server.router
        self._drop_body()  # it carries none; what a client sent is never the next request
        if "/" in name or name not in {r.name for r in router.instances.all()}:
            self.json_reply(404, error_body(f"no instance named {name!r}"))
            return
        router.idle.forget_keep(name)  # the user's word wins
        self.server.note(f"{name}: idle/keep saved by the user; a client's keep_alive goes.")
        self.json_reply(200, {"saved": name})

    def _metrics(self) -> None:
        """`GET /metrics`: the router's figures and each loaded engine's, relabelled."""
        router = self.server.router
        queue = router_metrics.QueueNow(router.queue.waiting(), router.queue.totals())
        page = router_metrics.page(router.instances, self.server.counts, queue)
        self.text_reply(200, page, router_metrics.CONTENT_TYPE)

    def _misdirected(self, host: str) -> None:
        """421 for a name it does not answer to, one journal line per peer and name."""
        self._drop_body()
        seen, why = (self.client_address[0], host_name(host)), self.server.hosts.refusal(host)
        if seen in self.server.misdirected:
            self.server.counts.refused(421)
        else:
            self.server.misdirected.add(seen)
            self._refusal_note(421, f"Host {shown(host)}: not a name it answers to")
        self.journaled = True  # `/mcp` included: its own note would not say it
        self.json_reply(421, error_body(why), close=True)

    def _not_here(self, own: str) -> None:
        """What only an engine has (`/slots`, `/props`: asked before every message, never
        journaled: a line each would drown it) or nobody here (`/.well-known/…`: journaled
        once per peer while the router runs, the first look saying a client expected
        OAuth; the rest are the same client connecting again): a 404 that says why."""
        if own in ROUTER_ONLY_LOOKS:
            self.journaled = True
            self.json_reply(404, error_body(ONE_ENGINE_ONLY.format(path=own)), cors=True)
            return
        peer = self.client_address[0]
        self.journaled = peer in self.server.well_known
        self.server.well_known.add(peer)
        self.json_reply(404, error_body(NO_OAUTH), cors=True)

    def _refusal_note(self, status: int, why: str) -> None:
        """`/v1/chat/completions POST for tiny from 127.0.0.1: 503 (tiny does not fit …)`, and
        the count `/metrics` reports, fed here with the journal's line."""
        self.server.counts.refused(status)
        path = shown(self.path.split("?", 1)[0])
        model = f" for {self.asked_for}" if self.asked_for else ""
        peer = self.client_address[0]
        said = f" ({reason(why)})" if why else ""
        line = f"{path} {shown(self.command)}{model} from {self._who(peer)}: {status}{said}"
        self.server.quiet.refusal(line, peer, status, path)

    def _who(self, peer: str) -> str:
        """The peer as the journal names it: ComfyUI's network's gateway is the host itself,
        the doctor's reach check asking (never a refusal "from ComfyUI's
        container")."""
        guard = self.server.guard
        if guard is not None and guard.is_gateway(peer):
            return f"the {NETWORK} gateway {peer} (the doctor's reach check)"
        return peer

    def _health(self) -> bool:
        """`GET /__mnm/health`, keyless and never journaled: the router's own part from
        memory, the rest read at most once a minute (`health_now`). True when answered."""
        if self.command not in ("GET", "HEAD") or self.path.split("?", 1)[0] != HEALTH_PATH:
            return False
        data = for_router(self.server.router, self.server.health).model_dump_json().encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        return True
