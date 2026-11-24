"""How an engine's container ended, in words: `mnm try` after a request that found it gone,
and the router's healer after a death. ComfyUI's own wording is `comfy/exits.py`
(`why_stopped`); this is the engines', on the same two signs: Docker's exit code and its
OOMKilled flag."""

from __future__ import annotations

from makenomistakes.diagnose import Diagnosis
from makenomistakes.supervisor import ContainerState, ContainerStatus

KILLED_BY_SOMEONE = "a `docker kill`, the kernel, or something else outside the engine"
"""Exit 137 without Docker's OOMKilled flag, never worded as the machine running out of RAM (a
`docker kill` reads the same). The kernel's OOM killer, global or the cgroup's,
counts its kill in the container's own cgroup, which is what Docker's flag reads: 137 without
the flag is a SIGKILL from somewhere else."""


def killed_words(oom_killed: bool) -> str:
    """Exit 137 in words, the one wording the engines, ComfyUI and the healer share."""
    if oom_killed:
        return "killed out of memory (Docker says OOMKilled)"
    return f"killed by {KILLED_BY_SOMEONE}"


def exit_meaning(code: int | None, oom_killed: bool = False) -> str:
    """Docker's exit code in words: 139 a crash inside the engine, 137 killed (out of memory
    only when Docker's OOMKilled flag says so)."""
    match code:
        case 139:
            return "code 139: the engine crashed, a segmentation fault"
        case 137:
            return f"code 137: {killed_words(oom_killed)}"
        case _:
            return f"code {code}"


def death_cause(status: ContainerStatus, diagnoses: list[Diagnosis]) -> str:
    """One line: how the container ended (Docker's OOMKilled first, then the exit code), and
    the first cause its log names (`diagnose`), when it names one."""
    if status.state is ContainerState.ABSENT:
        how = "its container was removed outside mnm (`docker rm`?)"
    elif status.oom_killed:
        how = f"killed out of memory (Docker says OOMKilled; exit {status.exit_code})"
    else:
        how = f"its engine exited ({exit_meaning(status.exit_code)})"
    if diagnoses:
        how += f"; its log says: {diagnoses[0].cause.rstrip('.')}"
    return " ".join(how.split())
