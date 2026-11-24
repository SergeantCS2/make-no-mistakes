"""Checks of the vetted packs list and of a new ComfyUI pin, run when either changes: ask the
Registry about every vetted row, check each row's zip still hashes as recorded, and refresh
what a new pin changes (the classes each pack registers, the partner nodes) from a running
ComfyUI's `/object_info`. No command of mnm's runs them on a user's box."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from makenomistakes.comfy.packs import VETTED, VETTED_FILE, VettedPack, assess
from makenomistakes.comfy.registry import PackError, Registry, folder_of


@dataclass
class RowCheck:
    """One vetted row against the Registry today: refusals fail the job, news is said."""

    pack: str
    refusals: list[str] = field(default_factory=list)
    news: list[str] = field(default_factory=list)


def check(registry: Registry, rows: tuple[VettedPack, ...] = VETTED) -> list[RowCheck]:
    """Each row as the user's approval would see it now: banned, deleted, flagged or gone
    is a refusal; a newer version is news (the pin moves only with a new mnm)."""
    found = []
    for row in rows:
        result = RowCheck(row.pack)
        try:
            seen = assess(registry, row.pack, row.version)
        except PackError as exc:
            result.refusals.append(str(exc))
            found.append(result)
            continue
        result.refusals += seen.refusals
        latest = seen.node.latest_version if seen.node is not None else None
        if latest is not None and latest.version != row.version:
            result.news.append(f"the Registry's latest is {latest.version} (vetted {row.version})")
        found.append(result)
    return found


def rehash(registry: Registry, rows: tuple[VettedPack, ...] = VETTED) -> list[str]:
    """Rows whose zip no longer hashes as recorded (or cannot be fetched): never rewritten
    here, since a changed zip needs a review, not a new number."""
    problems = []
    for row in rows:
        release = registry.version(row.pack, row.version)
        if release is None or not release.downloadUrl:
            problems.append(f"{row.pack} {row.version}: the Registry has no zip for it")
            continue
        digest = hashlib.sha256(registry.download(release.downloadUrl)).hexdigest()
        if digest != row.sha256:
            problems.append(f"{row.pack} {row.version}: sha256 {digest}, recorded {row.sha256}")
    return problems


def classes_by_folder(object_info: dict[str, Any]) -> dict[str, list[str]]:
    """The node classes each custom-node folder registers (`python_module`)."""
    found: dict[str, list[str]] = defaultdict(list)
    for name, node in object_info.items():
        module = str(node.get("python_module", "")) if isinstance(node, dict) else ""
        head, _, rest = module.partition(".")
        if head == "custom_nodes" and rest:
            found[rest.split(".")[0]].append(name)
    return {folder: sorted(names) for folder, names in found.items()}


def partner_classes(object_info: dict[str, Any]) -> list[str]:
    """The paid cloud nodes (`api_node`), from a ComfyUI started with its API nodes on."""
    return sorted(n for n, v in object_info.items() if isinstance(v, dict) and v.get("api_node"))


def with_classes(object_info: dict[str, Any], path: Path = VETTED_FILE) -> tuple[str, list[str]]:
    """vetted.json with each row's classes as this ComfyUI registers them; the rows it did not
    load (a CUDA-only pack on a CPU build) keep theirs and are named."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_folder = classes_by_folder(object_info)
    kept = []
    for row in doc["packs"]:
        found = by_folder.get(folder_of(row["pack"]))
        if found:
            row["classes"] = found
        else:
            kept.append(row["pack"])
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n", kept


PARTNERS_CONTAINER = "mnm_comfyui_partners"  # the pin check's second start, never the box's


def partners_argv(argv: list[str]) -> list[str]:
    """The sandbox's `docker run` with the partner (API) nodes left on, under another name:
    only to list their classes (`partners`) for a new pin. mnm's own start always passes
    `--disable-partner-nodes`, so its /object_info lists none and the step could never end."""
    out = [a for a in argv if a != "--disable-partner-nodes"]
    at = out.index("--name") + 1
    out[at] = PARTNERS_CONTAINER
    return out
