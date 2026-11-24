"""Node classes from third-party packs that send their work to a service outside mnm's
sandboxed ComfyUI: it has no internet and nothing listening beside it, so they cannot
work here. ComfyUI's own partner nodes are `partner_nodes.json`; these are read from each
pack's code into `cloud_nodes.json`. A workflow that uses one is refused (`report.analyze`),
and so is approving a pack whose only use is one (`packs.assess`)."""

from __future__ import annotations

import io
import zipfile
from functools import cache
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

CLOUD_FILE = Path(__file__).parent / "cloud_nodes.json"
MAX_SOURCE = 4 * 1024 * 1024  # a Python file larger than this is no node's registration


class CloudNode(BaseModel):
    pack: str  # its Registry id
    service: str  # where it sends the work, in words
    read: str  # the version and date its code was read
    note: str = ""  # what else the pack is, read with it ("its other nodes only feed that one")


class _CloudNodes(BaseModel):
    nodes: dict[str, CloudNode]


@cache
def cloud_nodes() -> dict[str, CloudNode]:
    """Each such class by name."""
    return _CloudNodes.model_validate_json(CLOUD_FILE.read_text(encoding="utf-8")).nodes


def cloud_pack(pack: str) -> list[tuple[str, CloudNode]]:
    """A pack's classes that send their work out, by its Registry id (any case)."""
    return [(name, row) for name, row in cloud_nodes().items() if row.pack.lower() == pack.lower()]


def cloud_classes_in(data: bytes) -> list[tuple[str, CloudNode]]:
    """The listed classes a pack's own code registers (its zip's Python files name them as
    strings): what a GitHub copy or fork of a listed pack carries under any name."""
    with zipfile.ZipFile(io.BytesIO(data)) as packed:
        code = "\n".join(
            packed.read(item).decode("utf-8", "replace")
            for item in packed.infolist()
            if PurePosixPath(item.filename).suffix == ".py" and item.file_size < MAX_SOURCE
        )
    return [
        (name, row)
        for name, row in cloud_nodes().items()
        if f'"{name}"' in code or f"'{name}'" in code
    ]


def words(name: str, row: CloudNode) -> str:
    """`AceStepText2MusicServer (ACE-Step-ComfyUI) sends the work to …`, one sentence for
    the workflow report and the approval page alike."""
    note = f"; {row.note}" if row.note else ""
    return (
        f"{name} ({row.pack}) sends the work to {row.service}, which mnm's ComfyUI cannot "
        f"reach (no internet, nothing beside it){note}"
    )
