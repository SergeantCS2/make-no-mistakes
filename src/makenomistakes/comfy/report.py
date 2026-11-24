"""What an imported workflow needs, read before anything is installed: its
node packs (core, or which pack at which version), the model files it names, partner (cloud)
nodes, nodes that run code they are given, and whether this ComfyUI can convert it now.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path, PurePosixPath

from makenomistakes.comfy import cloud
from makenomistakes.comfy.cloud import cloud_nodes
from makenomistakes.comfy.convert import VIRTUAL
from makenomistakes.comfy.object_info import ObjectInfo
from makenomistakes.comfy.packs import vetted, vetted_for_class
from makenomistakes.comfy.ui_format import BYPASSED, MUTED, ModelFile, Node, Workflow

CORE = "comfy-core"
MODEL_SUFFIXES = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
PICKLES = (".ckpt", ".pt", ".pth", ".bin")  # can run code when loaded
# Where ComfyUI's core loaders look, by node class (the folder a missing file goes to).
LOADER_FOLDERS = {
    "CheckpointLoaderSimple": "checkpoints",
    "CheckpointLoader": "checkpoints",
    "ImageOnlyCheckpointLoader": "checkpoints",
    "UNETLoader": "diffusion_models",
    "UnetLoaderGGUF": "diffusion_models",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "TripleCLIPLoader": "text_encoders",
    "CLIPLoaderGGUF": "text_encoders",
    "DualCLIPLoaderGGUF": "text_encoders",
    "VAELoader": "vae",
    "LoraLoader": "loras",
    "LoraLoaderModelOnly": "loras",
    "CLIPVisionLoader": "clip_vision",
    "UpscaleModelLoader": "upscale_models",
    "ControlNetLoader": "controlnet",
    # kijai's WanVideoWrapper (on the vetted list)
    "WanVideoModelLoader": "diffusion_models",
    "WanVideoVAELoader": "vae",
    "WanVideoTinyVAELoader": "vae_approx",
    "LoadWanVideoT5TextEncoder": "text_encoders",
    "LoadWanVideoClipTextEncoder": "clip_vision",
    "WanVideoLoraSelect": "loras",
    "WanVideoLoraSelectMulti": "loras",
    # the other vetted packs' loaders
    "MMAudioModelLoader": "mmaudio",
    "MMAudioVoCoderLoader": "mmaudio",
    "MMAudioFeatureUtilsLoader": "mmaudio",
    "SeedVR2LoadDiTModel": "seedvr2",
    "SeedVR2LoadVAEModel": "seedvr2",
    "RIFE VFI": "packs/comfyui-frame-interpolation/rife",
}
# A node class that loads weights by its name (core's StyleModelLoader, GLIGENLoader, …; a
# pack's `DownloadAndLoad…Model`), beside the classes `LOADER_FOLDERS` knows.
_LOADS = re.compile(r"(Loader|LoadModel|ModelLoad|Load\w*Model)", re.I)
# Node names that suggest running code or commands they are given: asked about by name before
# a pack is approved, never installed silently.
_RISKY = re.compile(r"(exec|eval|python|shell|subprocess|command|script|terminal)", re.I)
PARTNER_FILE = Path(__file__).parent / "partner_nodes.json"

_HEX64 = re.compile(r"[0-9a-fA-F]{64}")


@cache
def partner_nodes() -> frozenset[str]:
    """ComfyUI v0.42.1's partner (cloud, paid) node classes, read from its `/object_info`."""
    return frozenset(json.loads(PARTNER_FILE.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class PackUse:
    pack: str  # Registry id (`comfyui-videohelpersuite`) or GitHub `owner/repo`
    version: str | None
    nodes: tuple[str, ...]  # the classes this workflow uses from it
    installed: bool = False  # every one of them is in this ComfyUI now

    @property
    def reference(self) -> str:
        """What `packs add` takes for it: the Registry id, or `github:owner/repo@<commit>`
        (ComfyUI saves a GitHub pack's commit as its nodes' `ver`)."""
        if "/" not in self.pack:
            return self.pack
        return f"github:{self.pack}" + (f"@{self.version}" if self.version else "")


@dataclass(frozen=True)
class ModelNeed:
    name: str  # the file name the node's widget names
    folder: str  # where ComfyUI looks (`diffusion_models`), "" when unknown
    url: str = ""  # from the workflow's own metadata, when it has one
    node: str = ""  # the class that loads it
    pickle: bool = False
    sha256: str = ""  # from the workflow's metadata, when it records one (lowercase hex)


@dataclass
class Report:
    packs: list[PackUse] = field(default_factory=list)
    models: list[ModelNeed] = field(default_factory=list)
    partner: list[str] = field(default_factory=list)  # refused: cloud nodes
    cloud: list[str] = field(default_factory=list)  # refused: a pack's own cloud nodes
    risky: list[str] = field(default_factory=list)  # named before any approval
    missing: list[str] = field(default_factory=list)  # classes this ComfyUI does not know
    problems: list[str] = field(default_factory=list)  # what stops it outright
    classes: list[str] = field(default_factory=list)  # every node class that would run

    @property
    def reads_weights(self) -> bool:
        """It may load model weights onto the card: a model file it names, a loader node, a
        pack with weights of its own (`rife`) or a pack mnm has not vetted (anything). Only a
        workflow that reads none needs no more than ComfyUI's margin (`room.run_need`)."""
        if self.models or any(c in LOADER_FOLDERS or _LOADS.search(c) for c in self.classes):
            return True
        for use in self.packs:
            row = vetted(use.pack)
            if row is None or row.weights or row.weights_dir or row.model_folders:
                return True
        return False

    @property
    def runnable(self) -> bool:
        """It converts and runs once its model files are there: no cloud node, no unknown
        node, every pack it uses installed."""
        installed = all(p.installed for p in self.packs)
        return installed and not (self.partner or self.missing or self.problems)


def _nodes(workflow: Workflow) -> list[Node]:
    """Every node that would run: the workflow's and its subgraphs', muted ones left out."""
    graphs = [workflow, *workflow.subgraphs().values()]
    return [n for g in graphs for n in g.nodes if n.mode not in (MUTED, BYPASSED)]


def analyze(workflow: Workflow, info: ObjectInfo | None = None) -> Report:
    report = Report()
    subgraphs = workflow.subgraphs()
    nodes = [n for n in _nodes(workflow) if n.type not in subgraphs and n.type not in VIRTUAL]
    packs: dict[str, tuple[str | None, set[str]]] = {}
    for node in nodes:
        props = node.properties
        pack = (
            props.cnr_id
            or props.aux_id
            or _installed_pack(info, node.type)
            or _vetted_pack(node.type)  # saved without its pack's id, and not installed
        )
        if node.type in partner_nodes():
            report.partner.append(node.type)
        elif node.type in cloud_nodes():
            report.cloud.append(node.type)
        elif pack and pack != CORE:
            _version, classes = packs.setdefault(pack, (props.ver, set()))
            classes.add(node.type)
        if _RISKY.search(node.type):
            report.risky.append(node.type)
        unknown = info is not None and info.get(node.type) is None
        outside = node.type in partner_nodes() or node.type in cloud_nodes()
        if unknown and not outside and (not pack or pack == CORE):
            report.missing.append(node.type)
    report.packs = [
        PackUse(
            pack,
            version,
            tuple(sorted(classes)),
            installed=info is not None and all(info.get(c) is not None for c in classes),
        )
        for pack, (version, classes) in sorted(packs.items())
    ]
    report.models = _models(workflow, nodes)
    report.classes = sorted({n.type for n in nodes})
    report.partner = sorted(set(report.partner))
    report.risky = sorted(set(report.risky))
    report.missing = sorted(set(report.missing))
    if workflow.group_nodes:
        report.problems.append(
            "it uses old group nodes: open it once in ComfyUI's editor (it turns them into "
            "subgraphs) and import what it saves"
        )
    if report.partner:
        report.problems.append(
            "it uses partner nodes, which run on Comfy's paid cloud: " + ", ".join(report.partner)
        )
    report.cloud = sorted(set(report.cloud))
    report.problems += [f"its {cloud.words(name, cloud_nodes()[name])}" for name in report.cloud]
    return report


def _vetted_pack(class_type: str) -> str | None:
    row = vetted_for_class(class_type)
    return row.pack if row is not None else None


def _installed_pack(info: ObjectInfo | None, class_type: str) -> str | None:
    """The pack folder this ComfyUI loaded a class from, for a node saved without its pack
    (`python_module` is `custom_nodes.<folder>`)."""
    found = info.get(class_type) if info is not None else None
    module = (found.python_module or "") if found is not None else ""
    head, _, rest = module.partition(".")
    if head != "custom_nodes" or not rest:
        return None
    return rest.split(".")[0]


def _models(workflow: Workflow, nodes: list[Node]) -> list[ModelNeed]:
    """The model files a workflow names: from its metadata (`properties.models`, the top-level
    `models`) first, then any widget value that is a model file's name."""
    known: dict[str, ModelFile] = {m.name: m for m in workflow.models}
    for node in nodes:
        for model in node.properties.models:
            known.setdefault(model.name, model)
    found: dict[str, ModelNeed] = {}
    for node in nodes:
        values = node.widgets_values
        strings = values.values() if isinstance(values, dict) else values or []
        for value in strings:
            if not isinstance(value, str) or not value.lower().endswith(MODEL_SUFFIXES):
                continue
            name = PurePosixPath(value.replace("\\", "/")).name
            meta = known.get(name) or known.get(value)
            folder = (meta.directory if meta else "") or LOADER_FOLDERS.get(node.type, "")
            found.setdefault(
                value,
                ModelNeed(
                    name=value,
                    folder=folder,
                    url=meta.url if meta else "",
                    node=node.type,
                    pickle=value.lower().endswith(PICKLES),
                    sha256=_sha256(meta),
                ),
            )
    return sorted(found.values(), key=lambda m: (m.folder, m.name))


def _sha256(meta: ModelFile | None) -> str:
    """The file's sha256 when the metadata records one (`hash_type` sha256, or none given)."""
    if meta is None or meta.hash is None or (meta.hash_type or "sha256").lower() != "sha256":
        return ""
    return meta.hash.lower() if _HEX64.fullmatch(meta.hash) else ""
