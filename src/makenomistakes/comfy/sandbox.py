"""ComfyUI's sandbox: the container's name, its internal network and port, the
folders it may write, and the `docker run` that holds it to them (`engine` re-exports the
names)."""

from __future__ import annotations

import os
from pathlib import Path

from makenomistakes.backend import Backend, device_argv
from makenomistakes.backend_now import host_devices
from makenomistakes.comfy.image import CUDA_VARIANTS, Ledger, Variant
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import chosen_card, gpu_device
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR

CONTAINER = "mnm_comfyui"  # an underscore: no instance name can make it (`mnm-<name>`)
NETWORK = "mnm-comfyui"  # internal: no route out
PORT = 8188
INSIDE = "/comfy"  # `<state>/comfyui` in the container
FOLDERS = ("input", "output", "temp", "user", "home")


def run_argv(ledger: Ledger, models: Path, base: Path, paths_file: Path) -> list[str]:
    """The `docker run` that holds ComfyUI's container to its sandbox."""
    uid, gid = os.getuid(), os.getgid()
    argv = [
        "docker", "run", "--detach", "--name", CONTAINER,
        "--network", NETWORK,
        "--user", f"{uid}:{gid}",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",  # its code and packs as built; only its own folders take writes
        "--tmpfs", "/tmp:rw,size=2g",
        "--pids-limit", "512",  # a pack that forks without end stops there
        "--restart", "no",
        "--label", "mnm.engine=comfyui",
        "--label", "mnm.instance=comfyui",  # uninstall.sh removes it with the instances
        "--volume", f"{models}:{CONTAINER_MODELS_DIR}:ro",
        "--volume", f"{paths_file}:{INSIDE}/extra_model_paths.yaml:ro",
        "--env", f"HOME={INSIDE}/home",
    ]  # fmt: skip
    # Only ComfyUI's own folders, each on its own: mnm's ledger and its copy of
    # /object_info stay where the container cannot reach (a node could otherwise turn
    # one into a link and have mnm write through it).
    for name in FOLDERS:
        argv += ["--volume", f"{base / name}:{INSIDE}/{name}"]
    if ledger.variant in CUDA_VARIANTS:
        argv += ["--gpus", gpu_device()]  # MNM_GPU: the card the readings and doctor read
    elif ledger.variant is Variant.ROCM:  # The first chosen card's render node
        argv += device_argv(Backend.ROCM, (chosen_card(),), host_devices(Backend.ROCM))
    argv += [
        ledger.running_image,
        "--listen", "0.0.0.0",  # the container's own address, on a network with no way out
        "--port", str(PORT),
        "--disable-auto-launch",
        "--disable-partner-nodes",  # cloud nodes: never
        "--preview-method", "auto",
        "--extra-model-paths-config", f"{INSIDE}/extra_model_paths.yaml",
        "--input-directory", f"{INSIDE}/input",
        "--output-directory", f"{INSIDE}/output",
        "--temp-directory", f"{INSIDE}/temp",
        "--user-directory", f"{INSIDE}/user",
    ]  # fmt: skip
    if platforms.current().is_wsl:
        argv.append("--disable-pinned-memory")  # WSL's VAE-decode crashes (ComfyUI #11531)
    if ledger.variant is Variant.CPU:
        argv.append("--cpu")
    if ledger.variant is Variant.CUDA_LEGACY:  # no cudaMallocAsync before Turing
        argv.append("--disable-cuda-malloc")
    return argv
