"""What System → Studio and System → Trainer share for their built images (the tools
image and the trainers'; `studio/built_image.BuiltImage`): one job at a time per image, said when
another is under way, the image's state for the page, and the rollback and uninstall routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.studio.built_image import BuiltImage, ImageError, ImageMissingError

Report = Callable[[str], None]
Work = Callable[[Report], str]
Job = Callable[[str, Work, int], None]


def image_job(jobs: JobRegistry, key: str, what: str) -> Job:
    """Start `work` as the image's one job; a second while one runs is refused in words."""

    def job(title: str, work: Work, lines: int = 0) -> None:
        busy = jobs.active_for(key)
        if busy is not None:  # one job at a time on the image: said, never a silent no-op
            raise InstanceError(f"{what} job '{busy.label}' is under way; {title} after it.")

        def run(report: Report) -> list[str]:
            try:
                return [work(report)]
            except ImageError as exc:
                raise InstanceError(str(exc)) from exc

        jobs.start_unless_active(JobKind.ENGINES, key, run, label=title, keep_lines=lines)

    return job


def image_state(image: BuiltImage) -> dict[str, object]:
    """The page's `ledger`, `missing` (not installed, in words) and `problem` (unreadable)."""
    state: dict[str, object] = {"ledger": None, "missing": None, "problem": None}
    try:
        state["ledger"] = image.installed()
    except ImageMissingError as exc:
        state["missing"] = str(exc)
    except ImageError as exc:
        state["problem"] = str(exc)
    return state


def register_image_routes(
    app: FastAPI,
    path: str,
    image: Callable[[], BuiltImage],
    job: Job,
    kept: str,
    back: str | None = None,
) -> None:
    """`POST <path>/rollback` and `<path>/uninstall`, each a job; `kept` says what stays;
    `back` is the page they return to (`path` itself unless said)."""
    label = image().spec.label
    page = back or path

    @app.post(f"{path}/rollback")
    def rollback() -> RedirectResponse:
        job(f"roll back the {label}", lambda _r: f"back to {image().rollback().image}", 0)
        return RedirectResponse(page, status_code=303)

    @app.post(f"{path}/uninstall")
    def uninstall() -> RedirectResponse:
        def work(_report: Report) -> str:
            removed = image().uninstall()
            return f"removed {', '.join(removed) or 'the record'}; {kept}"

        job(f"uninstall the {label}", work, 0)
        return RedirectResponse(page, status_code=303)
