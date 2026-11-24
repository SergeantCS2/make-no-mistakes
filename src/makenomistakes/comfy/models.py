"""A workflow's model files: where each one is, where it can come from, and the
download mnm makes of it, one copy on disk, linked where ComfyUI looks.

A file is *in place* (a link under `<models>/comfyui/<folder>/` reaches it), *on disk* (an
instance's file of that name: linked, nothing downloaded), *fetchable* (the workflow names a
Hugging Face or Civitai link, or the user gave one, or a known repository holds a file of
exactly that name, which the user confirms), or *unknown* (a link is needed). Downloads go
through mnm's own fetchers: the Hugging Face cache under `<models>/hf`, Civitai's files under
`<models>/civitai/<version>/`. The links mnm made for workflows are kept in
`<state>/comfy/models.json`, so `mnm comfyui`'s own link sync keeps them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from makenomistakes import comfyui
from makenomistakes.comfy import civitai_find
from makenomistakes.comfy.civitai_find import Match
from makenomistakes.comfy.packs import VETTED
from makenomistakes.comfy.report import ModelNeed
from makenomistakes.hub.civitai import CivitaiClient, CivitaiError
from makenomistakes.hub.client import HUB_URL
from makenomistakes.hub.refs import CIVITAI_DOMAINS, HUB_HOSTS, hosts
from makenomistakes.links import without_secrets
from makenomistakes.locking import shared_lock
from makenomistakes.recipes.model import HF_CACHE_SUBDIR, Recipe

# Repositories that hold the files ComfyUI workflows name, by their exact names: a file a
# workflow names without a link is looked for here first.
KNOWN_REPOS = (
    "Kijai/WanVideo_comfy",
    "Kijai/WanVideo_comfy_fp8_scaled",
    # Renamed from Wan_2.2_ComfyUI_Repackaged and Wan_2.1_ComfyUI_repackaged (seen
    # 2026-10-26); the old names still answer with a 307 to these, which templates' links use.
    "Comfy-Org/Wan-2.2",
    "Comfy-Org/Wan-2.1",
    "Comfy-Org/HunyuanVideo_1.5_repackaged",
    "Comfy-Org/z_image_turbo",
    "Comfy-Org/Qwen-Image_ComfyUI",
    "QuantStack/Wan2.2-TI2V-5B-GGUF",
    "QuantStack/Wan2.2-T2V-A14B-GGUF",
    "QuantStack/Wan2.2-I2V-A14B-GGUF",
    "city96/umt5-xxl-encoder-gguf",
    "marduk191/rife",  # the RIFE weights Frame-Interpolation would download itself
    # The audio templates' files. ACE-Step 1.5's was `ace_step_1.5_ComfyUI_files`
    # until 2026-10 (a 307 to this name, which the template's note still links)
    "Comfy-Org/ACE-Step-1.5",
    "Comfy-Org/YuE2",
)
HF_HOSTS = HUB_HOSTS  # hf-mirror.com included: a user on HF_ENDPOINT pastes its links
CIVITAI_HOSTS = hosts(*CIVITAI_DOMAINS)
# A node pack's weights folder (`packs/<pack>`).
_PACK_FOLDER = re.compile(r"^packs/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
# The folders ComfyUI loads model files from (v0.42.1's `folder_paths`, with the old names it
# still maps: `clip`, `unet`), and the vetted packs' own. A workflow names the folder (its
# metadata's `directory`), so it is untrusted: mnm links nowhere else, never `custom_nodes`.
MODEL_FOLDERS = frozenset(
    {
        *comfyui.KINDS,
        *(folder for row in VETTED for folder in row.model_folders),
        "audio_encoders", "classifiers", "clip", "controlnet", "diffusers", "embeddings",
        "gligen", "hypernetworks", "latent_upscale_models", "model_patches", "photomaker",
        "style_models", "t2i_adapter", "unet", "vae_approx",
    }
)  # fmt: skip


def model_folder(folder: str) -> bool:
    return folder in MODEL_FOLDERS or bool(_PACK_FOLDER.match(folder))


class ModelError(ValueError):
    """A model file mnm cannot place, worded for the user."""


class Where(StrEnum):
    IN_PLACE = "in place"
    ON_DISK = "on disk"
    FETCHABLE = "fetchable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Source:
    """One downloadable file: a Hugging Face repo file, or a Civitai version's file."""

    url: str
    hf_repo: str | None = None
    hf_revision: str = "main"
    hf_path: str = ""
    civitai_version: str | None = None
    civitai_file: str = ""  # the file's name among the version's (`file=` in the link)

    @property
    def label(self) -> str:
        if self.hf_repo:
            return f"{self.hf_repo}/{self.hf_path}"
        named = f", {self.civitai_file}" if self.civitai_file else ""
        return f"Civitai version {self.civitai_version}{named}"


@dataclass(frozen=True)
class Status:
    need: ModelNeed
    where: Where
    link: str  # its name under `<models>/comfyui/<folder>/`
    source: Source | None = None
    size: int | None = None
    target: str | None = None  # on disk: the instance's file, relative to the models folder
    refused: str = ""  # unknown: why the workflow's own link is not one mnm downloads


class LinkedModel(BaseModel):
    folder: str
    name: str  # the link's name inside the folder (subfolders kept: `WanVideo/2_2/x`)
    target: str  # relative to the models folder
    source: str
    # The workflows this link is for, and whether mnm downloaded the file
    # for them (an instance's file, or a link from a schema-1 ledger, is never theirs to delete)
    owners: list[str] = Field(default_factory=list)
    fetched: bool = False

    @field_validator("source")
    @classmethod
    def _no_secret(cls, value: str) -> str:
        """Never a token in the ledger, a link an older mnm saved included."""
        return without_secrets(value)


LEDGER_SCHEMA = 2  # 2: owners and fetched (a v1 file reads as no owners, nothing fetched)


class Ledger(BaseModel):
    schema_version: int = LEDGER_SCHEMA
    links: list[LinkedModel] = Field(default_factory=list)


def link_name(need: ModelNeed) -> str:
    """The name ComfyUI is asked for, with `/` (a workflow saved on Windows uses `\\`)."""
    parts = [p for p in need.name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts) or PurePosixPath(need.name).name


def parse_source(url: str) -> Source:
    """A Hugging Face `resolve`/`blob` link or a Civitai download link; ModelError otherwise.
    Its query loses any secret (`?token=…`) first: the source is saved (`links`)."""
    url = without_secrets(url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host in HF_HOSTS:
        for section in ("resolve", "blob"):
            if section in parts:
                at = parts.index(section)
                if at == 2 and len(parts) > at + 2:
                    return Source(
                        url=url,
                        hf_repo="/".join(parts[:2]),
                        hf_revision=parts[at + 1],
                        hf_path="/".join(parts[at + 2 :]),
                    )
    if host in CIVITAI_HOSTS:
        if parts[:3] == ["api", "download", "models"] and len(parts) > 3 and parts[3].isdigit():
            return Source(url=url, civitai_version=parts[3])
        query = parse_qs(parsed.query)
        version = query.get("modelVersionId", [""])[0]
        if version.isdigit():
            return Source(url=url, civitai_version=version, civitai_file=query.get("file", [""])[0])
    raise ModelError(
        f"{url} is not a file link mnm downloads: use a Hugging Face file link "
        "(…/resolve/main/<file>) or a Civitai version (…?modelVersionId=<id>)."
    )


def default_http() -> httpx.Client:
    """The client a `Models` asks the Hub and Civitai with (sizes, the tree in `find`)."""
    return httpx.Client(timeout=30.0, follow_redirects=True)


class Models:
    def __init__(
        self,
        models_dir: Path,
        state_dir: Path,
        http: Callable[[], httpx.Client] | None = None,
        civitai: Callable[[], CivitaiClient] | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.ledger_path = state_dir / "comfy" / "models.json"
        self._http = http or default_http  # read at construction, so a caller can swap it
        self._civitai = civitai  # None: Civitai is not searched

    # -- the ledger ----------------------------------------------------------------------

    def ledger(self) -> Ledger:
        """The workflows' links; none before the first fetch. A file this cannot read, or a
        newer mnm's, is a ModelError: read as empty, `mnm rm --files` would think no workflow
        uses a file, and the next fetch would write over every link."""
        try:
            text = self.ledger_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Ledger()
        except OSError as exc:
            raise ModelError(f"{self.ledger_path} cannot be read: {exc}") from exc
        try:
            ledger = Ledger.model_validate_json(text)
        except ValidationError as exc:
            raise ModelError(
                f"{self.ledger_path} cannot be read ({exc.errors()[0]['msg']}); move it aside "
                "and fetch the workflows' files again to rebuild it."
            ) from None
        if ledger.schema_version > LEDGER_SCHEMA:
            raise ModelError(f"{self.ledger_path} is from a newer mnm: run `mnm update`.")
        return ledger

    def _save(self, ledger: Ledger) -> None:
        ledger.schema_version = LEDGER_SCHEMA  # an older file is written back as this one's
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.ledger_path)

    def layout(self, recipes: list[Recipe]) -> comfyui.Layout:
        """mnm's instances' links and the workflows' downloads, as one layout to sync."""
        base = comfyui.layout(recipes)
        links = {kind: dict(names) for kind, names in base.links.items()}
        for linked in self.ledger().links:
            links.setdefault(linked.folder, {})[linked.name] = linked.target
        return comfyui.Layout(links, base.skipped)

    # -- where each file is ----------------------------------------------------------------

    def status(
        self, needs: list[ModelNeed], recipes: list[Recipe], given: dict[str, str] | None = None
    ) -> list[Status]:
        """Each file the workflow names: in place, on disk, fetchable, or unknown. `given`:
        links the user pasted, by file name."""
        on_disk = _instance_files(recipes)
        found = []
        for need in needs:
            name = link_name(need)
            folder = need.folder or "checkpoints"
            link = self.models_dir / comfyui.SUBDIR / folder / name
            base = PurePosixPath(name).name
            url = (given or {}).get(need.name) or (given or {}).get(base) or need.url
            if link.exists():
                found.append(Status(need, Where.IN_PLACE, name))
            elif base in on_disk:
                found.append(Status(need, Where.ON_DISK, name, target=on_disk[base]))
            elif url:
                try:  # the workflow's own link is untrusted: one bad link is that file's note
                    found.append(Status(need, Where.FETCHABLE, name, parse_source(url)))
                except ModelError as exc:
                    found.append(Status(need, Where.UNKNOWN, name, refused=str(exc)))
            else:
                found.append(Status(need, Where.UNKNOWN, name))
        return found

    def find(self, need: ModelNeed) -> list[Source]:
        """Files of exactly this name in the known repositories (the user picks one)."""
        base = PurePosixPath(link_name(need)).name.lower()
        found = []
        with self._http() as http:
            for repo in KNOWN_REPOS:
                for path in _tree(http, repo):
                    if PurePosixPath(path).name.lower() == base:
                        url = f"https://huggingface.co/{repo}/resolve/main/{path}"  # a link
                        found.append(Source(url, repo, "main", path))
        return found

    def on_civitai(self, need: ModelNeed) -> list[Match]:
        """The same bytes (a recorded sha256) or files of exactly this name on Civitai."""
        if self._civitai is None:
            return []
        client = self._civitai()
        try:
            return civitai_find.find(client, need)
        finally:
            client.close()

    def size(self, source: Source, name: str = "") -> int | None:
        """The file's size from the server, when it says: a Hub file's `x-linked-size`, or
        Civitai's listing of the version's file (`name`: the file, when the link names none)."""
        if source.civitai_version is not None:
            return self._civitai_size(source.civitai_version, source.civitai_file or name)
        if source.hf_repo is None:
            return None
        url = f"{HUB_URL}/{source.hf_repo}/resolve/{source.hf_revision}/{source.hf_path}"
        try:
            with self._http() as http:
                return _hub_size(http, httpx.URL(url))
        except httpx.HTTPError:
            return None

    def _civitai_size(self, version: str, name: str) -> int | None:
        if self._civitai is None or not name:
            return None
        client = self._civitai()
        try:
            return client.file(version, name).size_bytes
        except CivitaiError:
            return None
        finally:
            client.close()

    # -- placing a file ----------------------------------------------------------------------

    def place(
        self,
        status: Status,
        fetch_hf: Callable[[str, str, str, Path], Path],
        fetch_civitai: Callable[[str, str], Path],
        owner: str = "",
    ) -> LinkedModel:
        """Link a file already on disk, or download it first; recorded in the ledger with the
        workflow it was fetched for (`owner`; none for a pack's weights)."""
        folder = status.need.folder or "checkpoints"
        if not model_folder(folder):
            raise ModelError(
                f"{status.need.name}: the workflow puts it in '{folder}', which is not a folder "
                "ComfyUI loads model files from; nothing was downloaded."
            )
        fetched = True
        if status.where is Where.ON_DISK and status.target is not None:
            target, source, fetched = status.target, "an instance's file", False
        elif status.source is not None and status.source.hf_repo is not None:
            path = fetch_hf(
                status.source.hf_repo,
                status.source.hf_path,
                status.source.hf_revision,
                self.models_dir / HF_CACHE_SUBDIR,
            )
            target, source = _relative(path, self.models_dir), status.source.url
        elif status.source is not None and status.source.civitai_version is not None:
            listed = status.source.civitai_file or PurePosixPath(status.link).name
            path = fetch_civitai(status.source.civitai_version, listed)
            target, source = _relative(path, self.models_dir), status.source.url
        else:
            raise ModelError(f"{status.need.name}: give its link first (`--set`).")
        linked = LinkedModel(
            folder=folder,
            name=status.link,
            target=target,
            source=source,
            owners=[owner] if owner else [],
            fetched=fetched,
        )
        with shared_lock(self.ledger_path.parent)():  # two fetches (web, CLI) never undo each other
            ledger = self.ledger()
            before = next(
                (i for i in ledger.links if (i.folder, i.name) == (folder, status.link)), None
            )
            if before is not None and before.target == target:  # the same file: owners add up
                linked.owners = sorted({*before.owners, *linked.owners})
                linked.fetched = linked.fetched or before.fetched
            ledger.links = [i for i in ledger.links if i is not before] + [linked]
            self._save(ledger)
        return linked

    def claim(self, owner: str, links: list[tuple[str, str]]) -> None:
        """The workflow uses these links already in place (another's download): it owns them
        too, so removing the first workflow keeps them."""
        with shared_lock(self.ledger_path.parent)():
            ledger = self.ledger()
            changed = False
            for item in ledger.links:
                if (item.folder, item.name) in links and owner not in item.owners:
                    item.owners = sorted([*item.owners, owner])
                    changed = True
            if changed:
                self._save(ledger)

    def release(self, owner: str, needed: dict[tuple[str, str], list[str]]) -> list[str]:
        """The workflow is going: it leaves every link's owners. A link another workflow still
        names (`needed`, by folder and link name) passes to it; a file mnm fetched that nobody
        owns now leaves the ledger, and its place (relative to the models folder) is returned
        for deletion; a link to an instance's file leaves the ledger, the file stays. Links from
        before owners were recorded stay as they are."""
        orphans: list[str] = []
        with shared_lock(self.ledger_path.parent)():
            ledger = self.ledger()
            kept = []
            for item in ledger.links:
                if owner in item.owners:
                    item.owners = [o for o in item.owners if o != owner]
                    if not item.owners:
                        item.owners = sorted(set(needed.get((item.folder, item.name), [])))
                    if not item.owners:  # its last workflow went: the link goes
                        if item.fetched:  # and the download with it (an instance's file stays)
                            orphans.append(item.target)
                        continue
                kept.append(item)
            ledger.links = kept
            self._save(ledger)
        still = {i.target for i in ledger.links}
        return [t for t in orphans if t not in still]


def _relative(path: Path, models_dir: Path) -> str:
    """A downloaded file's place inside the models folder (a cache link stays a link)."""
    absolute = Path(path) if Path(path).is_absolute() else models_dir / path
    try:
        return absolute.relative_to(models_dir).as_posix()
    except ValueError as exc:
        raise ModelError(f"{path} is outside the models folder.") from exc


def _instance_files(recipes: list[Recipe]) -> dict[str, str]:
    """Every downloaded instance file by its base name (an instance's Wan VAE is a workflow's)."""
    found: dict[str, str] = {}
    for recipe in recipes:
        for key, local in recipe.local_paths.items():
            found.setdefault(PurePosixPath(key).name, local)
    return found


MAX_TREE_PAGES = 20  # the Hub pages a tree at 1,000 entries (Kijai's repos run past one)


MAX_SIZE_HOPS = 5


def _hub_size(http: httpx.Client, url: httpx.URL) -> int | None:
    """A Hub file's size: the `x-linked-size` its resolve answer carries (the 302 to the CDN,
    never followed), past a renamed repo's 307 to the new name. A redirect's own
    `content-length` is its plain-text body's ("Temporary Redirect. Redirecting to …"), never
    the file's (Wan's 9.3 GiB file would read as 127 bytes). Only a
    final 2xx answer's length counts (a small file the Hub serves itself)."""
    for _hop in range(MAX_SIZE_HOPS):
        response = http.head(url, follow_redirects=False)
        linked = response.headers.get("x-linked-size", "")
        if linked.isdigit():
            return int(linked)
        if response.is_redirect and (location := response.headers.get("location")):
            url = url.join(location)
            continue
        length = response.headers.get("content-length", "")
        return int(length) if response.is_success and length.isdigit() else None
    return None


class TreeEntry(BaseModel):
    """One entry of the Hub's tree listing (typed at the edge; the rest of it ignored)."""

    type: str
    path: str


_TREE = TypeAdapter(list[TreeEntry])


def _tree(http: httpx.Client, repo: str) -> list[str]:
    """Every file path of a repo, following the Hub's `Link: rel="next"` pages."""
    url: str | None = f"{HUB_URL}/api/models/{repo}/tree/main?recursive=true"
    paths: list[str] = []
    for _page in range(MAX_TREE_PAGES):
        if url is None:
            break
        try:
            response = http.get(url)
            entries = _TREE.validate_json(response.content) if response.is_success else []
        except (httpx.HTTPError, ValidationError):
            return paths
        paths += [e.path for e in entries if e.type == "file"]
        url = response.links.get("next", {}).get("url") if response.is_success else None
    return paths
