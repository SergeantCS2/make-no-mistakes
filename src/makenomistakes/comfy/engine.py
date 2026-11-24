"""ComfyUI as an optional engine: an image mnm builds on the box, one container
on an internal Docker network (no internet; mnm reaches it at the container's address), the
models folder read-only, ComfyUI's own folders under `<state>/comfyui/`.

Nothing here runs until the user installs it (`mnm comfyui install`): `installed()` reads
the ledger only, so the router's start path costs nothing on a box without ComfyUI.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

import httpx

from makenomistakes import comfyui
from makenomistakes.comfy import build_stream
from makenomistakes.comfy.guard import forget_subnets, write_subnets
from makenomistakes.comfy.http_side import FREED_STAMP, HttpSide
from makenomistakes.comfy.image import (
    COMFYUI_COMMIT,
    COMFYUI_VERSION,
    DOCKER_DIR,
    IMAGE_GIB,
    IMAGE_REPO,
    TORCH_INDEX,
    ComfyBusyError,
    ComfyError,
    Ledger,
    Variant,
    build_options,
    extra_kinds,
    image_tag,
    read_ledger,
    resolved_versions,
)
from makenomistakes.comfy.object_info import ObjectInfo, parse_file
from makenomistakes.comfy.observe import (
    NO_SUCH_CONTAINER,
    ComfyDockerError,
    InspectState,
    State,
)
from makenomistakes.comfy.sandbox import CONTAINER, FOLDERS, INSIDE, NETWORK, PORT, run_argv
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import LineRunner, SubprocessRunner
from makenomistakes.locking import shared_lock
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR

__all__ = [  # the image's, the sandbox's, the observation's and the HTTP side's names,
    # re-exported: callers have always found them here
    "COMFYUI_VERSION",
    "CONTAINER",
    "FOLDERS",
    "FREED_STAMP",
    "IMAGE_GIB",
    "INSIDE",
    "NETWORK",
    "NO_SUCH_CONTAINER",
    "PORT",
    "ComfyBusyError",
    "ComfyDockerError",
    "ComfyEngine",
    "ComfyError",
    "InspectState",
    "Ledger",
    "State",
    "Variant",
    "extra_kinds",
    "image_tag",
]

BUILD_SECONDS = 3600.0
READY_SECONDS = 180.0
GIB = 2**30
Progress = Callable[[str], None]


def _quiet(_message: str) -> None:
    """No one listening: an install from a caller that does not show progress."""


NOT_INSTALLED = (
    "ComfyUI is not installed: `mnm comfyui install` (or the web's System → ComfyUI) builds it."
)
NOT_STARTED = "Start ComfyUI once (`mnm comfyui start`): mnm reads its nodes then."
NOT_ASKED = (
    "Docker could not be asked whether ComfyUI is installed or running (`mnm comfyui status` "
    "says why): mnm reads its nodes once it answers."
)


def _no_such_network(text: str) -> bool:
    """Docker's answer for a network that is not there ("No such network", or "network X not
    found"); anything else (Docker silent) keeps the sandbox's subnets."""
    lowered = text.lower()
    return "no such network" in lowered or "not found" in lowered


def unread_nodes(state: State) -> str:
    """What to do first when ComfyUI's nodes have not been read: install it, or start it
    once. Every surface says it this way (the CLI, the web, a film's clips, MCP)."""
    if state is State.NOT_INSTALLED:
        return NOT_INSTALLED
    return NOT_ASKED if state is State.UNKNOWN else NOT_STARTED


class ComfyEngine(HttpSide):
    def __init__(
        self,
        settings: Settings,
        runner: LineRunner | None = None,
        http: Callable[[], httpx.Client] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.base = settings.expanded_state_dir() / "comfyui"
        self.runner = runner or SubprocessRunner()
        self._http = http or (lambda: httpx.Client(timeout=10.0, trust_env=False))
        self.sleep = sleep  # also how a run waits to read how the container ended
        self.clock = clock  # also how long a build has been quiet
        self._lock = shared_lock(self.base)  # the CLI, the web, the router and a packs switch

    def locked(self) -> AbstractContextManager[None]:
        """The ledger's and the container's one lock: a change of either, or a start, waits for
        the one under way (two runs arriving while ComfyUI is stopped start it once)."""
        return self._lock()

    def try_locked(self) -> AbstractContextManager[bool]:
        """The same lock only when nobody holds it now (True), else False at once: a click
        never hangs behind a start or a packs switch, which hold it for minutes (a packs
        build holds `pack_build.packs_building` instead)."""
        return self._lock.attempt()

    # -- the ledger ----------------------------------------------------------------------

    @property
    def ledger_path(self) -> Path:
        return self.base / "engine.json"

    def ledger(self) -> Ledger | None:
        """What `mnm comfyui install` built; None before that. A file this cannot read, or a
        newer mnm's, is a ComfyError naming it (never "not installed")."""
        return read_ledger(self.ledger_path)

    def _ledger_or_none(self) -> Ledger | None:
        """For the questions that must never fail a model's start (freeing the card): an
        unreadable ledger answers as no ComfyUI here; the doctor says why."""
        try:
            return self.ledger()
        except ComfyError:
            return None

    def save(self, ledger: Ledger) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.ledger_path)

    # -- building ------------------------------------------------------------------------

    def build_argv(self, variant: Variant, tag: str, ca: Path | None = None) -> list[str]:
        argv = [
            "docker", "build", build_stream.PLAIN, "--tag", tag,
            "--build-arg", f"TORCH_INDEX={TORCH_INDEX[variant]}",
            "--build-arg", f"COMFYUI_REF={COMFYUI_VERSION}",
            "--build-arg", f"COMFYUI_COMMIT={COMFYUI_COMMIT}",
            "--label", "mnm.engine=comfyui",
        ]  # fmt: skip
        return [*argv, *build_options(ca), str(DOCKER_DIR)]

    def install(
        self, variant: Variant, progress: Progress = _quiet, ca: Path | None = None
    ) -> Ledger:
        """Build the image (minutes; the old one stays for `rollback`) and record it. The
        build's own output reaches `progress` as it comes, each line a `BuildLine`."""
        tag = image_tag(variant)
        progress(f"Building {tag} (about {IMAGE_GIB[variant.value]:.0f} GB, 10-30 minutes)")
        argv = self.build_argv(variant, tag, ca)
        code, tail = build_stream.build(self.runner, argv, progress, self.clock, BUILD_SECONDS)
        if code != 0:
            raise ComfyError(f"The ComfyUI image did not build:\n{tail}")
        code, freeze, _err = self._docker(
            ["run", "--rm", "--network", "none", "--entrypoint", "pip", tag, "freeze"]
        )
        versions = resolved_versions(freeze) if code == 0 else {}
        with self.locked():  # the build took minutes: the ledger is read as it is now
            old = self._ledger_or_none()
            same = old is not None and old.image == tag
            previous = (old.previous if same else old.image) if old is not None else None
            ledger = Ledger(
                image=tag,
                variant=variant,
                previous=previous,
                built_at=datetime.now(UTC),
                # a running container keeps its address (and the image it started from, until
                # `stop` and `start`): mnm never loses sight of what holds the card
                address=old.address if old is not None and self._running() else None,
                versions=versions,
            )
            if same and old is not None:  # the same base: its packs layer still fits
                ledger = ledger.model_copy(
                    update={"packs_image": old.packs_image, "packs": old.packs}
                )
            self.save(ledger)
        if old is not None and not same:
            gone = [old.previous, old.packs_image, old.packs_previous]  # layers fit one base
            for image in {g for g in gone if g and g not in (ledger.image, ledger.previous)}:
                self._docker(["image", "rm", image])  # in use or out of reach: best effort
        return ledger

    def _running(self) -> bool:
        found = self._inspect()
        return found is not None and found.State.Running

    def rollback(self) -> Ledger:
        with self.locked():
            return self._rollback()

    def _rollback(self) -> Ledger:
        old = self.ledger()
        if old is None or old.previous is None:
            raise ComfyError("There is no earlier ComfyUI image to go back to.")
        ledger = old.model_copy(
            update={
                "image": old.previous,
                "previous": old.image,
                "packs_image": None,  # a packs layer is built on one base: built again
                "packs": [],
                "packs_previous": None,
                "previous_packs": [],
            }
        )
        self.save(ledger)  # a running container keeps its address until `stop` and `start`
        for image in {old.packs_image, old.packs_previous} - {None}:
            self._docker(["image", "rm", str(image)])  # its base is not in use: best effort
        return ledger

    # -- running -------------------------------------------------------------------------

    def run_argv(self, ledger: Ledger) -> list[str]:
        return run_argv(ledger, self.settings.expanded_models_dir(), self.base, self.paths_file)

    def _prepare(self) -> None:
        """ComfyUI's folders, mnm's model links and the block that names them."""
        for name in FOLDERS:
            (self.base / name).mkdir(parents=True, exist_ok=True)
        models = self.settings.expanded_models_dir()
        from makenomistakes.comfy.models import ModelError, Models  # noqa: PLC0415 - a cycle
        from makenomistakes.instances import InstanceService  # noqa: PLC0415 - a heavy import

        try:  # an unreadable recipe, a broken ledger, a link mnm did not make: said as such
            recipes = InstanceService(self.settings).all()
            layout = Models(models, self.settings.expanded_state_dir()).layout(recipes)
            comfyui.write_links(models, layout)
        except (ModelError, ValueError, RuntimeError) as exc:  # InstanceError is a RuntimeError
            raise ComfyError(f"ComfyUI's model links could not be made: {exc}") from exc
        block = comfyui.yaml_block(Path(CONTAINER_MODELS_DIR), extra_kinds(layout))
        self.paths_file.unlink(missing_ok=True)  # never written through a link
        self.paths_file.write_text(block, encoding="utf-8")

    def _network(self) -> None:
        """The internal network, made once; its subnets written for the router, which
        refuses every request from them (`comfy/guard.py`)."""
        code, out, _err = self._docker(["network", "inspect", "--format", "{{.Internal}}", NETWORK])
        if code == 0 and out.strip() != "true":  # made by hand: packs could reach out
            raise ComfyError(
                f"The Docker network {NETWORK} exists and is not internal, so ComfyUI could "
                f"reach the internet: `docker network rm {NETWORK}`, then start again."
            )
        if code != 0:
            code, out, err = self._docker(["network", "create", "--internal", NETWORK])
            if code != 0:
                raise ComfyError(f"Could not make the network {NETWORK}: {(err or out).strip()}")
        fmt = "{{range .IPAM.Config}}{{.Subnet}} {{end}}"
        code, out, err = self._docker(["network", "inspect", "--format", fmt, NETWORK])
        if code != 0 or not out.split():
            raise ComfyError(f"Could not read the network {NETWORK}: {(err or out).strip()}")
        write_subnets(self.settings.expanded_state_dir(), out)

    def start(self, progress: Progress | None = None) -> str:
        """Start the container and wait until it answers; returns its address. One start at
        a time: a second caller waits, then finds it running."""
        with self.locked():
            return self._start(progress)

    def _start(self, progress: Progress | None) -> str:
        ledger = self.ledger()
        if ledger is None:
            raise ComfyError(NOT_INSTALLED)
        try:  # never `rm --force` a container Docker did not say is gone (a run in it)
            running = self.observe() is State.RUNNING
        except ComfyDockerError as exc:
            raise ComfyDockerError(
                f"Docker could not be asked whether ComfyUI runs, so it was not started: {exc}"
            ) from exc
        if running and ledger.address:
            return ledger.address
        self._prepare()
        self._network()
        self._docker(["rm", "--force", CONTAINER])
        if progress:
            progress("Starting ComfyUI")
        try:
            argv = self.run_argv(ledger)
        except ValueError as exc:  # MNM_GPU is not a card number
            raise ComfyError(str(exc)) from exc
        code, out, err = self._docker(argv[1:], timeout=120.0)
        if code != 0:
            raise ComfyError(f"docker run failed: {(err or out).strip()}")
        found = self._inspect()
        if found is None or not found.IPAddress:
            raise ComfyError("ComfyUI started without an address on its network.")
        address = f"http://{found.IPAddress}:{PORT}"
        self.save(ledger.model_copy(update={"address": address}))
        self._wait_ready(address)
        self.refresh_object_info(address)
        return address

    @property
    def paths_file(self) -> Path:
        """`extra_model_paths.yaml`, mounted read-only as one file."""
        return self.base / "extra_model_paths.yaml"

    @property
    def object_info_path(self) -> Path:
        return self.base / "object_info.json"

    def refresh_object_info(self, address: str) -> None:
        """Every node class this ComfyUI knows, kept so a report reads it while it is off."""
        try:
            with self._http() as http:
                response = http.get(f"{address}/object_info", timeout=60.0)
        except httpx.HTTPError:
            return
        if response.is_success:
            self.object_info_path.write_bytes(response.content)

    def object_info(self) -> ObjectInfo | None:
        """What the last start's ComfyUI knew; None before its first start. Parsed once per
        version of the file (a page asks for it once per workflow)."""
        path = self.object_info_path
        try:
            stat = path.stat()
            return parse_file(path, stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError):
            return None

    def _wait_ready(self, address: str) -> None:
        deadline = self.clock() + READY_SECONDS
        while self.clock() < deadline:
            if self.stats(address) is not None:
                return
            found = self._inspect()
            if found is None or not found.State.Running:
                raise ComfyError(f"ComfyUI exited while starting:\n{self.logs(40)}")
            self.sleep(2.0)
        raise ComfyError(f"ComfyUI did not answer in {READY_SECONDS:.0f} s: `mnm comfyui logs`.")

    def stop(self) -> None:
        """Remove the container; ComfyDockerError when Docker cannot say it is gone (a shell
        whose docker group has not reached the login): the address stays, so the router keeps
        freeing ComfyUI's VRAM before a load, never "stopped" on a guess."""
        with self.locked():
            self._remove_container()
            ledger = self._ledger_or_none()
            if ledger is not None and ledger.address:
                self.save(ledger.model_copy(update={"address": None}))

    def _remove_container(self) -> bool:
        """`docker rm --force`: True when it removed one, False when there was none; anything
        else Docker says (permission denied, the daemon down) raises."""
        code, out, err = self._docker(["rm", "--force", CONTAINER])
        if code == 0:
            return True
        why = (err or out).strip()
        if any(word in why.lower() for word in NO_SUCH_CONTAINER):
            return False
        raise ComfyDockerError(f"Docker could not remove {CONTAINER}: {why or f'exit {code}'}")

    def uninstall(self) -> list[str]:
        """The container, every image mnm built, the network and the ledger; the workflows'
        folders and the outputs stay. Under the engine's lock; a Docker that cannot remove the
        container raises before the ledger or the sandbox's subnets are touched (the guard
        keeps refusing a container that may still run)."""
        removed = []
        with self.locked():
            if self._remove_container():
                removed.append(f"container {CONTAINER}")
            code, out, _err = self._docker(
                ["image", "ls", "--format", "{{.Repository}}:{{.Tag}}", IMAGE_REPO]
            )
            for image in out.split() if code == 0 else []:
                if (
                    image.startswith(f"{IMAGE_REPO}:")
                    and self._docker(["image", "rm", image])[0] == 0
                ):
                    removed.append(image)
            if self._docker(["network", "rm", NETWORK])[0] == 0:
                removed.append(f"network {NETWORK}")
            code, out, err = self._docker(["network", "inspect", NETWORK])
            if code != 0 and _no_such_network(err or out):  # gone, by now or before
                forget_subnets(self.settings.expanded_state_dir())
            self.ledger_path.unlink(missing_ok=True)
        return removed

    def logs(self, tail: int = 200) -> str:
        """ComfyUI's log, or a line saying why Docker gave none (`mnm comfyui logs`)."""
        try:
            return self.read_log(tail)
        except ComfyDockerError as exc:
            return f"(no log: {exc})"

    def read_log(self, tail: int = 200) -> str:
        """The container's log; ComfyDockerError when Docker gives none, so a check that reads
        it (every pack imported) never takes no log for a clean one."""
        code, out, err = self._docker(["logs", "--tail", str(tail), CONTAINER])
        if code != 0:
            raise ComfyDockerError((err or out).strip() or f"docker logs exited {code}")
        return out + err


def release_for(settings: Settings) -> Callable[[Progress | None], bool]:
    """The router's hook: `ComfyEngine(settings).release_vram`."""
    return ComfyEngine(settings).release_vram
