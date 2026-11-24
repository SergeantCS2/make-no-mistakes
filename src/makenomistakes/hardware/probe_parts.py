"""The probe's Docker and disk reads (`probe()` calls them): `docker info`, the NVIDIA
smoke test `--deep` runs, and the models folder's disk."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from makenomistakes.hardware.probe import (
    GPU_SMOKE_IMAGE,
    CommandRunner,
    DiskInfo,
    DockerInfo,
    ProbeError,
    SmokeResult,
    parse_docker_info,
)


def probe_docker(runner: CommandRunner) -> DockerInfo | ProbeError:
    code, out, err = runner.run(["docker", "info", "--format", "{{json .}}"])
    if code != 0:
        return ProbeError(message=(err or out).strip() or f"docker info exited with {code}")
    try:
        return parse_docker_info(out)
    except (ValueError, TypeError) as exc:
        return ProbeError(message=f"could not parse docker info: {exc}")


def docker_gpu_smoke(runner: CommandRunner) -> SmokeResult:
    argv = ["docker", "run", "--rm", "--gpus", "all", GPU_SMOKE_IMAGE, "nvidia-smi", "-L"]
    code, out, err = runner.run(argv, timeout=180.0)
    return SmokeResult(ok=code == 0, output=(out if code == 0 else err or out).strip())


def probe_disk(models_dir: Path) -> DiskInfo:
    path = models_dir.expanduser()
    if not path.exists():
        return DiskInfo(path=str(path), exists=False, writable=False)
    usage = shutil.disk_usage(path)
    return DiskInfo(
        path=str(path),
        exists=True,
        writable=os.access(path, os.W_OK),
        total_bytes=usage.total,
        free_bytes=usage.free,
    )
