"""Fetching a workflow's model files, shared by `mnm comfyui models --fetch` and
the web's Fetch button: each file downloaded through mnm's own fetchers (or linked when an
instance has it), then every link synced where ComfyUI looks."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from makenomistakes import comfyui
from makenomistakes.comfy.models import LinkedModel, ModelError, Models, Status, Where, link_name
from makenomistakes.comfy.report import analyze
from makenomistakes.comfy.store import Suggestion, WorkflowMeta, WorkflowStore, WorkflowStoreError
from makenomistakes.config import Settings
from makenomistakes.download import hub_fetcher
from makenomistakes.download_civitai import civitai_fetcher
from makenomistakes.hub.civitai import CivitaiError
from makenomistakes.locking import shared_lock
from makenomistakes.model_cleanup import Cleanup, delete_files
from makenomistakes.recipes import Recipe
from makenomistakes.sizes import size_words

Progress = Callable[[str], None]


def to_fetch(statuses: list[Status]) -> list[Status]:
    return [s for s in statuses if s.where in (Where.FETCHABLE, Where.ON_DISK)]


def fetch_all(
    models: Models,
    statuses: list[Status],
    settings: Settings,
    recipes: list[Recipe],
    progress: Progress,
    owner: str = "",
) -> list[LinkedModel]:
    """Each file fetched (or linked) for the workflow `owner` (none for a pack's weights), and
    the files it finds in place already claimed for it too."""
    token = settings.hf_token_value()
    room_for(models, to_fetch(statuses), settings.expanded_models_dir())
    civitai = civitai_fetcher(settings.expanded_models_dir(), settings.civitai_token_value())
    placed = []
    for status in to_fetch(statuses):
        progress(f"{status.link} ({status.source.label if status.source else 'on disk'})")
        placed.append(
            models.place(
                status,
                lambda repo, path, rev, cache: hub_fetcher(repo, path, rev, cache, token),
                # a Fetcher's (repo, file, revision, cache, token): Civitai reads the version
                lambda version, file: civitai("", file, version, Path(), None),
                owner,
            )
        )
    if owner:
        in_place = [s for s in statuses if s.where is Where.IN_PLACE]
        models.claim(owner, [(s.need.folder or "checkpoints", s.link) for s in in_place])
    comfyui.write_links(settings.expanded_models_dir(), models.layout(recipes))
    return placed


@dataclass
class Findings:
    """What a Find turned up: links to use, Civitai offers to confirm, and what to say. Worked
    out first, then applied to the workflow's meta in one locked write (`apply`)."""

    said: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)
    offers: dict[str, list[Suggestion]] = field(default_factory=dict)

    def apply(self, meta: WorkflowMeta) -> None:
        """Into the meta as it is now: a file the user linked while the job ran keeps the
        user's link, and gets no offer back."""
        for name, url in self.links.items():
            if name not in meta.links:
                meta.link(name, url)
        meta.found.update({n: o for n, o in self.offers.items() if n not in meta.links})


SPARE_BYTES = 2 * 2**30  # the partial file beside the finished one, and some room left


def room_for(models: Models, todo: list[Status], models_dir: Path) -> None:
    """Refuse a fetch the models folder's disk cannot hold, with the numbers (a size a server
    does not give counts as nothing: the check is never stricter than it can know)."""
    need = sum(
        models.size(s.source, PurePosixPath(s.link).name) or 0
        for s in todo
        if s.source is not None and s.where is Where.FETCHABLE
    )
    try:
        free = shutil.disk_usage(models_dir).free
    except OSError:
        return  # a folder not made yet: the download itself says what is wrong
    if need and free < need + SPARE_BYTES:
        raise ModelError(
            f"The files need {size_words(need)}, and mnm keeps {size_words(SPARE_BYTES)} free "
            f"beside them; {models_dir} has {size_words(free)} free: make room (`mnm clean`, "
            "`mnm rm --files`), then fetch again."
        )


def find_links(models: Models, statuses: list[Status]) -> Findings:
    """Unknown files looked up: the known repositories by exact name, then Civitai. A single
    repository match, or Civitai's file of the same sha256, becomes the file's link; Civitai
    files that only share the name are offers for the user to confirm. Civitai refusing (a
    key, a timeout) is said as such, never as "not found"."""
    found = Findings()
    for status in statuses:
        if status.where is not Where.UNKNOWN:
            continue
        name = status.need.name
        matches = models.find(status.need)
        if len(matches) == 1:
            found.links[name] = matches[0].url
            found.said.append(f"Found {status.link} in {matches[0].label}.")
            continue
        if matches:
            options = ", ".join(m.url for m in matches)
            found.said.append(f"{status.link} is in several places; paste one link: {options}")
            continue
        try:
            civitai = models.on_civitai(status.need)
        except CivitaiError as exc:
            found.said.append(f"{status.link}: Civitai did not answer the search: {exc}")
            continue
        same = [m for m in civitai if m.same_bytes]
        if same:
            found.links[name] = same[0].url
            found.said.append(f"Found {status.link} by its sha256: {same[0].label}.")
        elif civitai:
            found.offers[name] = [
                Suggestion(url=m.url, label=m.label, size=m.size) for m in civitai
            ]
            found.said.append(
                f"{status.link}: {len(civitai)} file(s) of that name on Civitai; "
                "confirm one (Use, or `--set`) if it is the right model."
            )
        else:
            found.said.append(f"{status.link}: not found; paste its link.")
    return found


@dataclass
class Removed:
    """What `remove_workflow` did with the files: deleted, kept as shared, or why it kept all."""

    cleanup: Cleanup | None = None
    kept: str = ""
    present: bool = True  # False: removed before, only its files were freed now
    stays: int = 0  # files it owned that are still on disk: another workflow's, an instance's


def remove_workflow(
    store: WorkflowStore,
    models: Models,
    name: str,
    recipes: list[Recipe],
    models_dir: Path,
    files: bool,
) -> Removed:
    """`mnm comfyui workflow rm [--files]` and the web's Remove: the workflow goes; with
    `files`, so do the files mnm fetched for it that no other workflow names and no instance
    uses. Another workflow that names one takes it over. If another workflow cannot be read,
    every file stays (it may use them).

    Without `files` the workflow keeps owning its files after it is gone (so does `mnm clean
    imports`, which never touches models): `rm --files` with the gone name frees them later.
    The files go first and the workflow last, under the ledger's lock, so a failure keeps the
    workflow to try again. A fetch for a workflow imported meanwhile finds its file outside
    that lock (only its ledger write waits), so in that narrow window a file it is linking can
    go; its next fetch downloads it again."""
    present = name in store.names()
    if not files:
        store.remove(name)
        return Removed()
    with shared_lock(models.ledger_path.parent)():
        if not present and not any(name in i.owners for i in models.ledger().links):
            store.remove(name)  # nothing here and nothing owned: the store's own refusal
        needed: dict[tuple[str, str], list[str]] = {}
        unreadable = []
        for other in store.names():
            if other == name:
                continue
            try:
                needs = analyze(store.workflow(other), None).models
            except (WorkflowStoreError, ValueError, OSError):
                unreadable.append(other)
                continue
            for need in needs:
                needed.setdefault((need.folder or "checkpoints", link_name(need)), []).append(other)
        if unreadable:
            if present:
                store.remove(name)
            return Removed(
                kept=f"its files stay: {', '.join(unreadable)} cannot be read and may use them",
                present=present,
            )
        owned = {item.target for item in models.ledger().links if name in item.owners}
        orphans = models.release(name, needed)
        in_use = {rel for r in recipes for rel in r.local_paths.values()}
        in_use |= {item.target for item in models.ledger().links}
        cleanup = delete_files(orphans, in_use, models_dir)
        comfyui.write_links(models_dir, models.layout(recipes))
        if present:
            store.remove(name)
    # files, not links (two links can share one), and only those still there (one deleted by
    # hand is gone already): what the user can still find on disk
    stays = sum((models_dir / target).exists() for target in owned)
    return Removed(cleanup, present=present, stays=stays)
