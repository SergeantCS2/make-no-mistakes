#!/opt/venv/bin/python
"""One RVC voice trained by Applio inside `mnm/studio-trainer`.
A training is one `docker run --detach --init --network none --read-only` (mnm waits
for it, stops it for a request and removes it) over one job folder mounted at /job:

    /job/job.json        what to train (written by mnm: `Job` below)
    /job/dataset/        the recordings, copied in by mnm (WAV or FLAC)
    /job/applio/<name>/  Applio's work: sliced audio, features, checkpoints (kept for a resume)
    /job/train.log       every line Applio's scripts printed
    /job/progress.json   the stage and epoch, rewritten as they move (mnm reads it)
    /job/out/            the result: model.pth, model.index when the index is IVF, result.json

    mnm-train versions   what the image carries (the ledger records it)
    mnm-train train      run every stage not done yet, then the result

Each stage is Applio's own script run as Applio runs it (its working folder /opt/applio,
`APPLIO_LOGS_DIR` the job's); a stage done is marked so a training cut by a request (a
`docker stop`) resumes where it was: Applio itself resumes from its latest `G_`/`D_`
checkpoint. Nothing here reaches the network: the weights were fetched and checked by sha256
when the image was built, and the image runs as the user's uid. One JSON line on stdout at
the end (`{"ok": …}` or `{"error": …}`); progress lines before it. Runs with the image's
Python only; mnm's own tests never import it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

APPLIO = Path("/opt/applio")
APPLIO_COMMIT_FILE = Path("/opt/applio-commit")
JOB = Path("/job")
PRETRAINED = APPLIO / "rvc" / "models" / "pretraineds" / "hifi-gan"
STAGES = ("preprocess", "extract", "train", "index")
EPOCH_LINE = re.compile(r"\|\s*epoch=(\d+)\s*\|\s*step=(\d+)")
FINAL = re.compile(r"_(\d+)e_(\d+)s\.pth$")
IVF = b"IwFl"  # FAISS's IndexIVFFlat, the only index audio.cpp's RVC reads
_RUNNING: list[subprocess.Popen[bytes]] = []  # the stage's script, for `_stop`


class StageError(RuntimeError):
    """A stage that did not finish, worded with the last lines it printed."""


def versions() -> dict[str, object]:
    found = {}
    for name in ("torch", "torchaudio", "librosa", "faiss-cpu", "transformers", "numpy"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            found[name] = metadata.version(name)
    commit = APPLIO_COMMIT_FILE.read_text().strip() if APPLIO_COMMIT_FILE.is_file() else ""
    found["applio"] = commit or "unknown"
    import torch

    found["cuda"] = torch.version.cuda or "none"
    return {"tool": "versions", "versions": found}


def _load_job() -> dict[str, object]:
    job = json.loads((JOB / "job.json").read_text(encoding="utf-8"))
    name = str(job["name"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", name):
        raise ValueError(f"not a voice name: {name!r}")
    return job


def _progress(**fields: object) -> None:
    """`progress.json`, written whole, and the same as a line on stdout."""
    fields["at"] = time.time()
    tmp = JOB / ".progress.tmp"
    tmp.write_text(json.dumps(fields), encoding="utf-8")
    tmp.replace(JOB / "progress.json")
    print(json.dumps({"tool": "progress", **fields}), flush=True)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["APPLIO_LOGS_DIR"] = str(JOB / "applio")
    env["APPLIO_CONFIG_FILE"] = "/tmp/applio-config.json"
    return env


def _run(stage: str, argv: list[str], on_line: object = None) -> None:
    """One of Applio's scripts; its output appended to the log, its epochs to the progress."""
    tail: list[str] = []
    with (JOB / "train.log").open("ab") as log:
        log.write(f"\n== {stage}: {' '.join(argv)}\n".encode())
        log.flush()
        script = subprocess.Popen(
            [sys.executable, *argv],
            cwd=APPLIO,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _RUNNING[:] = [script]
        assert script.stdout is not None
        for raw in script.stdout:
            log.write(raw)
            log.flush()
            line = raw.decode("utf-8", "replace").rstrip()
            tail = [*tail[-19:], line]
            if callable(on_line):
                on_line(line)
        code = script.wait()
        _RUNNING.clear()
    if code != 0:
        shown = "\n".join(t for t in tail if t.strip())[-1500:]
        raise StageError(f"Applio's {stage} stopped (exit {code}):\n{shown}")


def _done(work: Path, stage: str) -> bool:
    return (work / f".mnm-{stage}").is_file()


def _mark(work: Path, stage: str) -> None:
    (work / f".mnm-{stage}").write_text(str(time.time()), encoding="utf-8")


def _checkpoint_loads(work: Path) -> None:
    """A checkpoint a `docker stop` cut while it was written does not load: training then
    starts again from the pretrained model, and the log says so."""
    import torch

    for path in sorted(work.glob("[GD]_*.pth")):
        try:
            torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            for stale in work.glob("[GD]_*.pth"):
                stale.unlink()
            with (JOB / "train.log").open("a", encoding="utf-8") as log:
                log.write(
                    f"\n== {path.name} does not load ({exc}): it was cut while it was saved; "
                    "training starts again from the pretrained model.\n"
                )
            return


def train() -> dict[str, object]:
    job = _load_job()
    name = str(job["name"])
    rate = int(job["sample_rate"])  # type: ignore[call-overload]
    epochs = int(job["epochs"])  # type: ignore[call-overload]
    cores = str(job["cpu_cores"])
    gpu = str(job["gpu"])  # "-" for the CPU, else the card's index inside the container
    work = JOB / "applio" / name
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    kilo = f"{rate // 1000}k"

    if not _done(work, "preprocess"):
        _progress(stage="preprocess")
        _run(
            "preprocess",
            ["rvc/train/preprocess/preprocess.py", str(work), str(JOB / "dataset"), str(rate),
             cores, "Automatic", "False", "3.0", "0.3", "none"],
        )  # fmt: skip
        _mark(work, "preprocess")
    if not _done(work, "extract"):
        _progress(stage="extract")
        _run(
            "extract",
            ["rvc/train/extract/extract.py", str(work), "rmvpe", cores, gpu, str(rate),
             "contentvec", "None", "2"],
        )  # fmt: skip
        _mark(work, "extract")
    if not _done(work, "train"):
        _checkpoint_loads(work)

        every = int(job["save_every"])  # type: ignore[call-overload]
        saved: list[int] = [_saved_epoch(work)]

        def epoch(line: str) -> None:
            """A batch's line (`| batch=2/7`) is the epoch under way; the epoch's own closing
            line ends it, and Applio saves a checkpoint after every `save_every`-th."""
            if (found := EPOCH_LINE.search(line)) is None:
                return
            number = int(found[1])
            if "| batch=" not in line and number % every == 0:
                saved[0] = number
            _progress(stage="train", epoch=number, epochs=epochs, step=int(found[2]),
                      saved=saved[0] or None)  # fmt: skip

        under_way = saved[0] + 1 if saved[0] else None  # a resume goes on after the saved one
        _progress(stage="train", epoch=under_way, epochs=epochs, saved=saved[0] or None)
        _run(
            "train",
            ["rvc/train/train.py", name, str(job["save_every"]), str(epochs),
             str(PRETRAINED / f"f0G{kilo}.pth"), str(PRETRAINED / f"f0D{kilo}.pth"), gpu,
             str(job["batch_size"]), str(rate), "True", "False", "False", "False", "HiFi-GAN",
             "False"],
            epoch,
        )  # fmt: skip
        if _final(work, name) is None:
            raise StageError(
                f"Applio's training ended without a model file ({name}_<epochs>e_<steps>s.pth)"
            )
        _mark(work, "train")
    if not _done(work, "index"):
        _progress(stage="index")
        _run("index", ["rvc/train/process/extract_index.py", str(work), "Auto"])
        _mark(work, "index")
    return _finish(job, work, name, time.monotonic() - started)


def _saved_epoch(work: Path) -> int:
    """The epoch Applio's latest checkpoint was saved after (0: none yet); a resume goes on
    from the epoch after it."""
    try:
        progress = json.loads((JOB / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    saved = progress.get("saved")
    return int(saved) if isinstance(saved, int) and any(work.glob("G_*.pth")) else 0


def _final(work: Path, name: str) -> Path | None:
    """The model file of the last epoch Applio wrote."""
    found = [(int(m[1]), p) for p in work.glob(f"{name}_*e_*s.pth") if (m := FINAL.search(p.name))]
    return max(found)[1] if found else None


def _finish(job: dict[str, object], work: Path, name: str, seconds: float) -> dict[str, object]:
    final = _final(work, name)
    assert final is not None
    match = FINAL.search(final.name)
    assert match is not None
    out = JOB / "out"
    out.mkdir(exist_ok=True)
    shutil.copyfile(final, out / "model.pth")
    index = work / f"{name}.index"
    ivf = index.is_file() and index.read_bytes()[:4] == IVF
    if ivf:
        shutil.copyfile(index, out / "model.index")
    info: dict[str, object] = {}
    with contextlib.suppress(OSError, ValueError):
        info = json.loads((work / "model_info.json").read_text(encoding="utf-8"))
    result = {
        "tool": "trained",
        "epochs": int(match[1]),
        "steps": int(match[2]),
        "sample_rate": int(job["sample_rate"]),  # type: ignore[call-overload]
        "index": "ivf" if ivf else ("flat" if index.is_file() else "none"),
        "dataset_seconds": _seconds(info.get("total_dataset_duration")),
        "seconds": round(seconds, 1),
    }
    (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    _progress(stage="done", epoch=result["epochs"], epochs=job["epochs"])
    return result


def _seconds(said: object) -> float | None:
    """Applio's dataset length, which it writes as "HH:MM:SS" (or a number)."""
    if isinstance(said, int | float):
        return float(said)
    parts = str(said or "").split(":")
    try:
        return float(sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))) if said else None
    except ValueError:
        return None


def _stop(_signum: int, _frame: object) -> None:
    """`docker stop`: the stage's script goes with this process (a checkpoint written is
    kept; the next start resumes from it)."""
    for script in _RUNNING:
        with contextlib.suppress(ProcessLookupError):
            script.terminate()
    sys.exit(143)


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGTERM, _stop)
    try:
        if argv == ["versions"]:
            answer = versions()
        elif argv == ["train"]:
            answer = train()
        else:
            raise ValueError("usage: mnm-train versions | train")
    except Exception as exc:  # one worded line for mnm, never a traceback it must parse
        print(json.dumps({"tool": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
