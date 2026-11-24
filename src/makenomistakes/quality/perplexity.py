"""One `llama-perplexity` run in llama.cpp's official `full` image for the
instance's backend (`engine_images.QUALITY_IMAGES`).

The line is built from the candidate's recipe as its server line is (the recipe is the source
of truth): the same cards (`device_args`), the same placement of the weights and the KV cache
(`-ngl`, `--n-cpu-moe`, the cards' `-ts`/`-mg`/`-ot`, `-ctk`/`-ctv`, the threads, the NUMA
binding, the load mode), never a server's own flag (llama.cpp's parser refuses one a tool does
not take: read at b12359) nor the batch the tool sets itself. The tool's own: four 512-token
sequences at once (`-c 512 -b 2048`), the base run reading the text and saving the reference's
log-probabilities, a KLD run scoring another file against them.

The container takes the tune's copy name (`mnm-tune-<name>`): the router's cut stops it as it
stops a trial's copy (`TuneRunner.stop_job`) and its start removes a leftover
(`tune.wiring.clear_copies`). It runs as the user, with no network, reading the models folder
and writing only the tune's work folder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from makenomistakes.backend import Backend
from makenomistakes.backend_now import box_now, recipe_backend_or_none
from makenomistakes.diagnose import diagnose
from makenomistakes.engine_images import QUALITY_IMAGES, NoImage
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.hub.facts import Engine
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.cards import device_args, split_argv
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR, container_name

# Left out of the tool's line: the batch flags the tool sets itself (`-b`, `-ub`), and the
# server's own, which `llama-perplexity` refuses (read in the b12359 `full` image's `--help`)
LEFT_OUT = frozenset(
    {"-b", "-ub", "--sleep-idle-seconds", "--pooling", "--image-min-tokens", "--cache-ram",
     "--slot-save-path"}
)  # fmt: skip
CTX = 512  # the tool's own context: the one quants are compared at
BATCH = 2048  # four sequences at once (its own default)
WORK = "/work"
BASE_FILE = "base.kld"
TEXT_FILE = "text.txt"
WAIT_SECONDS = 3600.0  # one `docker wait`; asked again until the run ends
STOP_SECONDS = 10


class RunKind(StrEnum):
    BASE = "base"  # the reference: reads the text, saves its log-probabilities
    KLD = "kld"  # another quant against them


class PerplexityError(RuntimeError):
    """The run could not be started or followed (Docker said no)."""


@dataclass(frozen=True)
class Ran:
    code: int  # the container's exit code (143/137 after a stop)
    log: str  # both streams, as `docker logs` gives them
    cause: str | None = None  # `diagnose`'s code when it failed (`out-of-memory`, …)
    why: str | None = None  # one line for the user when it failed


def quality_image(recipe: Recipe) -> str | NoImage:
    """The `full` image of the instance's own backend (a CUDA 13 box on amd64 runs the CUDA 12
    build, as its server image does) and this box's arch."""
    box = box_now()
    arch = box.arch if box is not None else "x86_64"
    backend = recipe_backend_or_none(recipe)
    if backend is None:
        why = "the backend `MNM_BACKEND` names is refused on this box (`mnm doctor` says why)"
        return NoImage(Engine.LLAMACPP, Backend.CPU, arch, why)
    return QUALITY_IMAGES.image_for(backend, arch)


def tool_flags(recipe: Recipe) -> list[str]:
    """Where the weights and the KV cache go, as the server places them: the flags every
    llama.cpp tool takes, in the server line's order, and the cards' split."""
    args = recipe.llama_args
    kv = args.kv_type.value
    argv = ["--fit", "off", "-ngl", str(args.gpu_layers), "-fa", args.flash_attn,
            "-ctk", kv, "-ctv", kv]  # fmt: skip
    for flag, value in args.optional_flags():
        if value is not None and flag not in LEFT_OUT:
            argv += [flag, str(value)]
    if args.lazy_tensors:  # a placement's `-ot` carries them first (`split_argv`)
        argv += ["--lazy-mode", "on"]
        if args.placement is None:
            argv += ["-ot", ",".join(f"{t}=CPU" for t in args.lazy_tensors)]
    return argv + split_argv(recipe)


def run_argv(
    recipe: Recipe,
    models_dir: Path,
    work: Path,
    image: str,
    kind: RunKind,
    copy: str,
    chunks: int,
) -> list[str]:
    """`docker run` of one run: `copy` the tune's copy name (`tune.trial.copy_name`)."""
    user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else None
    tool = [
        "--perplexity",
        "--model", recipe.container_path(recipe.files[0]),
        "-c", str(CTX), "-b", str(BATCH), *tool_flags(recipe),
        "--kl-divergence-base", f"{WORK}/{BASE_FILE}",
    ]  # fmt: skip
    if kind is RunKind.BASE:
        tool += ["-f", f"{WORK}/{TEXT_FILE}", "--chunks", str(chunks)]
    else:
        tool += ["--kl-divergence"]
    return [
        "docker", "run", "--detach", "--pull", "never", "--name", container_name(copy),
        "--label", f"mnm.instance={copy}", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "512",
        *(["--user", user] if user else []),
        *device_args(recipe), *recipe.llama_args.cpuset_argv(),
        "--volume", f"{models_dir}:{CONTAINER_MODELS_DIR}:ro",
        "--volume", f"{work}:{WORK}:{'rw' if kind is RunKind.BASE else 'ro'}",
        image, *tool,
    ]  # fmt: skip


class DockerUnreachableError(PerplexityError):
    """Docker itself did not answer (its daemon down, the socket refused): the run waits for
    it, unlike a run Docker refused, which ends its step."""


# No `docker` on PATH (127), or a `docker run` that did not return in its time (124: the
# runner's timeout, a daemon hanging): Docker itself silent, the run waits for it
SILENT_CODES = (124, 127)
# What the client says when its daemon does not answer (or the socket is not the user's)
UNREACHABLE = (
    "Cannot connect to the Docker daemon",
    "Is the docker daemon running",
    "permission denied while trying to connect",
)


class PerplexityRunner:
    """Start a run detached, wait for it, read it and remove it; a cut (`cut` set, the
    container stopped by the router's `stop_job`) ends the wait with Docker's stop code."""

    def __init__(self, runner: CommandRunner) -> None:
        self._runner = runner

    def _docker(self, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        return self._runner.run(["docker", *args], timeout=timeout)

    def start(self, argv: list[str], copy: str) -> None:
        """Start the run detached (one left behind removed first); PerplexityError when Docker
        says no. Called under the box's start lock, which the wait does not hold."""
        name = container_name(copy)
        self._docker(["rm", "--force", name])
        code, out, err = self._runner.run(argv, timeout=180)
        if code != 0:
            self._docker(["rm", "--force", name])  # a `run` that timed out may have started it
            said = (err or out).strip()
            if code in SILENT_CODES or any(word in said for word in UNREACHABLE):
                raise DockerUnreachableError(f"Docker could not be asked: {said[:300]}")
            if "No such image" in said or "Unable to find image" in said:
                raise PerplexityError(
                    "llama.cpp's `full` image is no longer on this box (the router never "
                    "pulls): planning the quality tune again pulls it"
                )
            raise PerplexityError(f"Docker did not start llama-perplexity: {said[:300]}")

    def stop(self, copy: str) -> None:
        """Remove a run that a cut caught while Docker started it (the router's stop found no
        container yet): it never runs on unwatched."""
        self._docker(["rm", "--force", container_name(copy)])

    def finish(self, copy: str, backend: Backend | None = None) -> Ran:
        """Wait for the run to end (a cut ends it with Docker's stop code), read both streams,
        remove the container; a failure diagnosed from its log."""
        name = container_name(copy)
        try:
            exit_code = self._wait(name)
            _code, out, err = self._docker(["logs", name], timeout=120)
        finally:
            self._docker(["rm", "--force", name])
        log = f"{err}\n{out}"
        if exit_code == 0:
            return Ran(code=0, log=log)
        found = diagnose(log, Engine.LLAMACPP, backend)
        cause = found[0].cause_code.value if found else None
        why = found[0].cause if found else f"llama-perplexity exited {exit_code}"
        return Ran(code=exit_code, log=log, cause=cause, why=why)

    def _wait(self, name: str) -> int:
        """The container's exit code; a `docker wait` that outlasts its timeout (exit 124,
        nothing said) is the client's, never the run's end: asked again."""
        while True:
            code, out, err = self._docker(["wait", name], timeout=WAIT_SECONDS)
            if code != 124 or out.strip():
                break
        if code != 0 or not out.strip().lstrip("-").isdigit():
            raise PerplexityError(f"Docker lost llama-perplexity: {(err or out).strip()[:300]}")
        return int(out.strip())
