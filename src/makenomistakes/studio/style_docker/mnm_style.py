#!/opt/venv/bin/python
"""mnm's style trainer, run inside `mnm/studio-style-trainer`: ACE-Step 1.5's own trainer
(`acestep.training_v2`, the commit in /opt/acestep-commit) over one job folder, never imported
by mnm itself.

`mnm-style train` reads `/job/job.json` (the variant, the weights' folders as mounted, the
training's settings) and runs, each stage resuming what an earlier start did:

1. **ckpt**: `/job/ckpt`, the checkpoint folder ACE-Step's loaders read, built of links to the
   mounted weights and the trainer's own copy of the model's code (ACE-Step's downloader
   replaces the Hub's copy with it the same way).
2. **preprocess**: the songs in `dataset.json` to tensors (`/job/tensors`), a song whose tensor
   is written skipped.
3. **train**: `FixedLoRATrainer` in this process from the newest checkpoint that loads, one saved
   every epoch and the two newest kept; `progress.json` says the epoch under way and the last
   saved. Upstream reloads a LoRA checkpoint with `strict=False`, which drops PEFT's keys and
   trains on from fresh weights while saying it resumed: the adapter is loaded again here by
   PEFT's own rule, and a checkpoint that does not load stops the training in words.
4. **finish**: the adapter (`adapter_model.safetensors`, `adapter_config.json`) in
   `/job/result`, and one JSON line with what was trained.

SIGTERM (mnm's `docker stop`, a request taking the card) asks the trainer to stop at its next
batch; the checkpoint of the last whole epoch is where the next start goes on from.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import signal
import struct
import sys
import time
import zipfile
from collections.abc import Generator
from importlib import metadata
from pathlib import Path
from typing import Any

ACESTEP = Path("/opt/acestep")
COMMIT_FILE = Path("/opt/acestep-commit")
JOB = Path("/job")
CKPT = JOB / "ckpt"
TENSORS = JOB / "tensors"
OUT = JOB / "out"
RESULT = JOB / "result"
KEEP_CHECKPOINTS = 2
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
LOAD_FAILED = "[WARN] Checkpoint load failed"  # upstream's word when a resume raised
CHECKPOINT = re.compile(r"^epoch_(\d+)_loss_")
VARIANT_DIRS = {"base": "acestep-v15-base", "turbo": "acestep-v15-turbo"}
# each variant's own noise schedule (upstream's defaults are turbo's): `--shift`, steps
SCHEDULES = {"base": ("1.0", "50"), "turbo": ("3.0", "8")}
STOP = {"should_stop": False}  # the trainer's own stop flag, read before every batch


class StageError(RuntimeError):
    """A stage that did not finish, worded for the user."""


def versions() -> dict[str, object]:
    found = {}
    for name in ("torch", "torchaudio", "transformers", "diffusers", "peft", "lycoris-lora",
                 "lightning", "bitsandbytes", "torchcodec"):  # fmt: skip
        with contextlib.suppress(metadata.PackageNotFoundError):
            found[name] = metadata.version(name)
    commit = COMMIT_FILE.read_text().strip() if COMMIT_FILE.is_file() else ""
    found["ace-step"] = commit or "unknown"
    import torch

    found["cuda"] = torch.version.cuda or "none"
    return {"tool": "versions", "versions": found}


def _progress(**fields: object) -> None:
    """`progress.json`, written whole, and the same as a line on stdout."""
    fields["at"] = time.time()
    tmp = JOB / ".progress.tmp"
    tmp.write_text(json.dumps(fields), encoding="utf-8")
    tmp.replace(JOB / "progress.json")
    print(json.dumps({"tool": "progress", **fields}), flush=True)


def _log(line: str) -> None:
    with (JOB / "train.log").open("a", encoding="utf-8") as log:
        log.write(line.rstrip() + "\n")


def _load_job() -> dict[str, object]:
    try:
        return dict(json.loads((JOB / "job.json").read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise StageError(f"job.json cannot be read: {exc}") from exc


# --- 1. the checkpoint folder -------------------------------------------------------------


def build_ckpt(job: dict[str, object]) -> str:
    """`/job/ckpt` for the job's variant; the variant's folder name."""
    variant = str(job["variant"])
    name = VARIANT_DIRS.get(variant)
    if name is None:
        raise StageError(f"no ACE-Step variant {variant!r} (base or turbo)")
    shutil.rmtree(CKPT, ignore_errors=True)
    model = CKPT / name
    model.mkdir(parents=True)
    dit = Path(str(job["dit_dir"]))
    for file in ("model.safetensors", "config.json", "silence_latent.pt"):
        if not (dit / file).is_file():
            raise StageError(f"the {variant} checkpoint has no {file} in {dit}")
        (model / file).symlink_to(dit / file)
    code = ACESTEP / "acestep" / "models" / variant
    for source in sorted(code.glob("*.py")):  # the trainer's own model code, as ACE-Step syncs it
        if source.name != "__init__.py":
            shutil.copyfile(source, model / source.name)
    for link, key in (("vae", "vae_dir"), ("Qwen3-Embedding-0.6B", "text_encoder_dir")):
        target = Path(str(job[key]))
        if not target.is_dir():
            raise StageError(f"{target} (the {link} weights) is not mounted")
        (CKPT / link).symlink_to(target)
    return name


# --- 2. preprocessing -----------------------------------------------------------------------


def preprocess(job: dict[str, object]) -> None:
    from acestep.training_v2.preprocess import preprocess_audio_files

    songs = json.loads((JOB / "dataset.json").read_text(encoding="utf-8"))["samples"]
    TENSORS.mkdir(exist_ok=True)
    left = [s for s in songs if not (TENSORS / f"{Path(s['audio_path']).stem}.pt").is_file()]
    if not left:
        return
    _progress(stage="preprocess", done=len(songs) - len(left), songs=len(songs))
    result = preprocess_audio_files(
        audio_dir=None,
        output_dir=str(TENSORS),
        checkpoint_dir=str(CKPT),
        variant=str(job["variant"]),
        max_duration=float(job["max_seconds"]),  # type: ignore[arg-type]
        dataset_json=str(JOB / "dataset.json"),
        device=str(job["device"]),
        precision=str(job["precision"]),
        cancel_check=lambda: bool(STOP["should_stop"]),
    )
    _log(f"== preprocess: {json.dumps(result, default=str)[:2000]}")
    missing = [s for s in songs if not (TENSORS / f"{Path(s['audio_path']).stem}.pt").is_file()]
    if STOP["should_stop"]:
        raise SystemExit(143)
    if missing:
        raise StageError(
            f"{len(missing)} of {len(songs)} songs did not preprocess "
            f"({Path(missing[0]['audio_path']).name} first): train.log says why"
        )


# --- 3. training ----------------------------------------------------------------------------


def checkpoints() -> list[tuple[int, Path]]:
    """Every checkpoint folder, newest epoch first."""
    found = []
    for folder in (OUT / "checkpoints").glob("epoch_*"):
        if (match := CHECKPOINT.match(folder.name)) is not None and folder.is_dir():
            found.append((int(match.group(1)), folder))
    return sorted(found, reverse=True)


def adapter_file(folder: Path) -> Path:
    """A checkpoint's adapter: flat in its folder, as upstream's `save_checkpoint` writes it
    (`save_adapter_flat`), or under `adapter/`, the layout its loader looks for first."""
    nested = folder / "adapter" / "adapter_model.safetensors"
    return nested if nested.is_file() else folder / "adapter_model.safetensors"


def _loads(folder: Path) -> bool:
    """The adapter's safetensors header parses and the training state is a whole zip (torch
    saves one; every member's CRC checked): a checkpoint cut while it was written fails one.
    Upstream's own loader swallows a state that does not load and resumes at epoch 0, so a
    broken state is found here first."""
    weights = adapter_file(folder)
    state = folder / "training_state.pt"
    if not state.is_file() or not weights.is_file():
        return False
    try:
        with weights.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            json.loads(handle.read(length))
        with zipfile.ZipFile(state) as archive:
            if archive.testzip() is not None:
                return False
    except (OSError, ValueError, struct.error, zipfile.BadZipFile):
        return False
    return weights.stat().st_size > 8 + length


def _epoch_of(folder: Path) -> int:
    match = CHECKPOINT.match(folder.name)
    return int(match.group(1)) if match is not None else 0


def resume_from() -> tuple[int, Path | None]:
    """The newest checkpoint that loads; a newer one that does not (cut while written) is
    removed, said in the log. Checkpoints none of which loads stop the training in words:
    starting again from the pretrained model would throw the epochs away without a word."""
    found = checkpoints()
    for at, (epoch, folder) in enumerate(found):
        if _loads(folder):
            for _newer, broken in found[:at]:
                _log(f"== {broken.name} does not load (cut while written?): removed")
                shutil.rmtree(broken, ignore_errors=True)
            return epoch, folder
    if found:
        raise StageError(
            f"none of the {len(found)} checkpoints in {OUT / 'checkpoints'} loads "
            f"({found[0][1].name} first): the training stops rather than start again"
        )
    return 0, None


def prune() -> None:
    for _epoch, folder in checkpoints()[KEEP_CHECKPOINTS:]:
        shutil.rmtree(folder, ignore_errors=True)


def _patch_resume() -> None:
    """Upstream's LoRA resume loads the adapter with `strict=False`, where PEFT's saved names
    (`…lora_A.weight`) never meet the live ones (`…lora_A.default.weight`): load it again by
    PEFT's own rule and refuse a checkpoint that matches nothing."""
    from acestep.training_v2 import trainer_fixed

    original = trainer_fixed.resume_checkpoint

    def resume(
        trainer: Any, resume_path: str, optimizer: Any, scheduler: Any
    ) -> Generator[Any, None, Any]:
        resumed = yield from original(trainer, resume_path, optimizer, scheduler)
        folder = Path(resume_path)
        held = _epoch_of(folder)
        if resumed is None or int(resumed[0]) != held:
            raise StageError(  # upstream would train on from epoch 0 and say it resumed
                f"{folder.name} holds epoch {held}, and the trainer resumed "
                f"{'nothing' if resumed is None else f'at epoch {resumed[0]}'} from it"
            )
        if getattr(trainer, "adapter_type", "lora") != "lora":
            return resumed
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        weights = load_file(str(adapter_file(folder)))
        decoder = trainer.module.model.decoder
        decoder = getattr(decoder, "_forward_module", decoder)
        outcome = set_peft_model_state_dict(decoder, weights)
        unexpected = list(getattr(outcome, "unexpected_keys", []) or [])
        if unexpected or not weights:
            raise StageError(
                f"{folder.name}'s adapter does not fit the model ({len(unexpected)} of "
                f"{len(weights)} weights unknown, {unexpected[:1]} first)"
            )
        _log(f"== resumed {len(weights)} adapter weights from {folder.name}")
        return resumed

    trainer_fixed.resume_checkpoint = resume


def _argv(job: dict[str, object], resume: Path | None) -> list[str]:
    variant = str(job["variant"])
    shift, steps = SCHEDULES[variant]
    argv = [
        "--yes", "--plain", "--checkpoint-dir", str(CKPT), "--model-variant", variant,
        "--shift", shift, "--num-inference-steps", steps,
        "--dataset-dir", str(TENSORS), "--output-dir", str(OUT),
        "--adapter-type", "lora", "--rank", str(job["rank"]), "--alpha", str(job["alpha"]),
        "--dropout", "0.1", "--target-modules", "q_proj", "k_proj", "v_proj", "o_proj",
        "--attention-type", "both", "--optimizer-type", str(job["optimizer"]),
        "--lr", str(job["lr"]), "--scheduler-type", "cosine", "--warmup-steps", "100",
        "--batch-size", "1", "--gradient-accumulation", str(job["grad_accum"]),
        "--precision", str(job["precision"]), "--device", str(job["device"]),
        "--num-workers", "0", "--save-every", "1", "--epochs", str(job["epochs"]),
        "--seed", str(job.get("seed", 42)),
    ]  # fmt: skip
    if job.get("checkpointing", True):
        argv.append("--gradient-checkpointing")
    if job.get("offload_encoder"):
        argv.append("--offload-encoder")
    if resume is not None:
        argv += ["--resume-from", str(resume)]
    return argv


def train(job: dict[str, object]) -> dict[str, object]:
    from acestep.training_v2.cli.args import build_fixed_standalone_parser
    from acestep.training_v2.cli.common import build_configs
    from acestep.training_v2.cli.validation import validate_paths
    from acestep.training_v2.model_loader import load_decoder_for_training
    from acestep.training_v2.trainer_fixed import FixedLoRATrainer

    epochs = int(job["epochs"])  # type: ignore[call-overload]
    saved, resume = resume_from()
    progress: dict[str, object] = {"stage": "train", "epoch": saved + 1, "epochs": epochs,
                                   "saved": saved}  # fmt: skip
    if saved >= epochs:
        return {"epochs": epochs, "steps": None, "loss": None}
    _progress(**progress)
    _log(f"== train from {resume.name if resume else 'the pretrained model'}")
    args = build_fixed_standalone_parser().parse_args(_argv(job, resume))
    args.subcommand = "fixed"
    if not validate_paths(args):
        raise StageError("the trainer refused its paths (train.log)")
    adapter_cfg, train_cfg = build_configs(args)
    model = load_decoder_for_training(
        checkpoint_dir=train_cfg.checkpoint_dir, variant=train_cfg.model_variant,
        device=train_cfg.device, precision=train_cfg.precision,
    )  # fmt: skip
    _patch_resume()
    last: dict[str, object] = {"steps": None, "loss": None}
    for update in FixedLoRATrainer(model, adapter_cfg, train_cfg).train(training_state=STOP):
        _log(str(getattr(update, "msg", update)))
        kind = getattr(update, "kind", "")
        if kind == "fail" or str(getattr(update, "msg", "")).startswith(LOAD_FAILED):
            said = str(update.msg).removeprefix("[FAIL] ").removeprefix("[WARN] ")
            raise StageError(said.removesuffix(" -- starting fresh"))
        if kind in ("step", "epoch"):
            last = {"steps": update.step, "loss": round(float(update.loss), 4)}
            progress.update(epoch=update.epoch, step=update.step, loss=last["loss"])
            if kind == "epoch":
                progress["epoch_seconds"] = round(float(update.epoch_time), 1)
            _progress(**progress)
        elif kind == "checkpoint":
            prune()
            progress.update(saved=update.epoch, epoch=update.epoch + 1)
            _progress(**progress)
    if STOP["should_stop"]:
        raise SystemExit(143)
    return {"epochs": epochs, **last}


# --- 4. the adapter -------------------------------------------------------------------------


def finished_adapter(epochs: int) -> Path:
    """The trained adapter's folder: the trainer's `out/final`, or, when the training was
    stopped after its last epoch's checkpoint and before `final` was written, that
    checkpoint's adapter (it holds the same weights)."""
    final = OUT / "final"
    if all((final / f).is_file() for f in ADAPTER_FILES):
        return final
    last = next(((e, f) for e, f in checkpoints() if e >= epochs and _loads(f)), None)
    if last is not None:
        folder = adapter_file(last[1]).parent
        if all((folder / f).is_file() for f in ADAPTER_FILES):
            _log(f"== {final} is missing: the adapter of {last[1].name} is the trained one")
            return folder
    raise StageError(f"the trainer wrote no adapter in {final}")


def finish(job: dict[str, object], trained: dict[str, object], began: float) -> dict[str, object]:
    final = finished_adapter(int(job["epochs"]))  # type: ignore[call-overload]
    files = ADAPTER_FILES
    RESULT.mkdir(exist_ok=True)
    for file in files:
        shutil.copyfile(final / file, RESULT / file)
    config = json.loads((RESULT / "adapter_config.json").read_text(encoding="utf-8"))
    answer = {
        "tool": "trained",
        **trained,
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "variant": job["variant"],
        "seconds": round(time.time() - began, 1),
    }
    (RESULT / "result.json").write_text(json.dumps(answer), encoding="utf-8")
    return answer


def run() -> dict[str, object]:
    began = time.time()
    job = _load_job()
    build_ckpt(job)
    preprocess(job)
    trained = train(job)
    answer = finish(job, trained, began)
    _progress(stage="done", epoch=answer["epochs"], epochs=answer["epochs"],
              saved=answer["epochs"])  # fmt: skip
    return answer


def _stop(_signum: int, _frame: object) -> None:
    STOP["should_stop"] = True  # the trainer stops at its next batch, the last epoch kept


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGTERM, _stop)
    try:
        if argv == ["versions"]:
            answer = versions()
        elif argv == ["train"]:
            answer = run()
        else:
            raise ValueError("usage: mnm-style versions | train")
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # one worded line for mnm, never a traceback it must parse
        _log(f"== failed: {type(exc).__name__}: {exc}")
        print(json.dumps({"tool": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
