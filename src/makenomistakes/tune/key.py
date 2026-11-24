"""What a tune was measured on, read now: the engine build the
tag names, each card's UUID and name, the driver, the RAM the kernel reports and
`mnm doctor --deep`'s bandwidth (`machine.json`), every model file's size and modification
time, the tuner's version, and the box's fingerprint id and CPU model as
`machine.json` last recorded them, and the backend the instance's image is built for
(`backend_now.recipe_backend`: a Vulkan reading never stands for a ROCm plan). Files are not
hashed: sha256 over 100 GB of weights would take longer than the tune, and a new download
changes the size or the time anyway."""

from __future__ import annotations

import contextlib

from makenomistakes import machine_facts
from makenomistakes.backend_now import recipe_backend_or_none
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.instances import InstanceService
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.tuned import TuneKey


def key_now(
    service: InstanceService, recipe: Recipe, snapshot: MemorySnapshot | None = None
) -> TuneKey:
    """The key the box gives `recipe` now (one nvidia-smi read unless `snapshot` is passed)."""
    found = snapshot or service.memory()
    cards = [f"{card.uuid or card.unique_id or card.index} {card.name}" for card in found.cards]
    drivers = {card.driver_version for card in found.cards if card.driver_version}
    gbps, machine, cpu = None, None, None
    with contextlib.suppress(machine_facts.MachineFactsError):
        state = service.settings.expanded_state_dir()
        file = machine_facts.load_file(state)
        facts = file.facts() if file is not None else None
        gbps = facts.ram_gbps if facts is not None else None
        current = file.current.fingerprint if file is not None and file.current else None
        machine = current.id if current is not None else None
        cpu = current.cpu.model if current is not None else None
    return TuneKey(
        build=service.build_of(recipe.image),
        cards=cards,
        driver=", ".join(sorted(drivers)) or None,
        ram_bytes=found.memory.total_bytes if found.memory is not None else None,
        ram_gbps=gbps,
        files=_files(service, recipe),
        machine=machine,
        cpu=cpu,
        backend=_backend(recipe),
    )


def _backend(recipe: Recipe) -> str | None:
    """The backend the tune runs on; None when `MNM_BACKEND` is refused (the doctor says)."""
    found = recipe_backend_or_none(recipe)
    return found.value if found is not None else None


def _files(service: InstanceService, recipe: Recipe) -> dict[str, str]:
    found = {}
    for rel in sorted(set(recipe.local_paths.values())):
        try:
            stat = (service.models_dir / rel).stat()
        except OSError:
            found[rel] = "missing"
            continue
        found[rel] = f"{stat.st_size}:{stat.st_mtime_ns}"
    return found
