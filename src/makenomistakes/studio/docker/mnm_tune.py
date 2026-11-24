"""The studio tools' vocal tuning, run by `mnm-tools tune` in the tools image (never
imported by mnm): CREPE's pitch (torchcrepe, its wheel carries the weights) and a TD-PSOLA in
numpy, which moves each sung period's grain and keeps the voice's formants. Checked on a voice
34 and 37 cents sharp: it landed within 1.2 cents of the note."""

from __future__ import annotations

import contextlib
import math
import sys
import time
from importlib import metadata
from pathlib import Path

TUNE_HOP = 0.01  # CREPE's frame: 10 ms
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
VOICED = 0.5  # CREPE's periodicity above which a frame is sung


def _marks(x: object, rate: int, f0: object, hop: int) -> tuple[object, object]:
    """Pitch marks one period apart (at the waveform's peak near each), 100 a second where
    the frame is not sung."""
    import numpy as np

    out: list[int] = []
    periods: list[float] = []
    n, t = len(x), 0.0
    while t < n:
        k = min(int(t // hop), len(f0) - 1)
        period = rate / f0[k] if f0[k] > 0 else rate / 100.0
        if f0[k] > 0 and out:
            lo = max(int(t - period / 2), out[-1] + int(period / 2))
            hi = min(int(t + period / 2), n - 1)
            if hi > lo:
                t = float(lo + int(np.argmax(x[lo:hi])))
        out.append(int(t))
        periods.append(period)
        t += period
    return np.array(out), np.array(periods)


def _psola(x: object, rate: int, f0: object, ratio: object, hop: int) -> object:
    """TD-PSOLA: each two-period Hann grain of the voice moved to marks `ratio` closer."""
    import numpy as np

    marks, periods = _marks(x, rate, f0, hop)
    size = len(x) + 4 * int(periods.max())
    y, weight = np.zeros(size), np.zeros(size)
    ts = float(marks[0])
    while ts < len(x):
        i = int(np.clip(np.searchsorted(marks, ts), 1, len(marks) - 1))
        i = i if abs(marks[i] - ts) < abs(marks[i - 1] - ts) else i - 1
        half = int(periods[i])
        lo, hi = marks[i] - half, marks[i] + half
        start = round(ts) - half
        if lo >= 0 and hi <= len(x) and start >= 0:
            window = np.hanning(hi - lo)
            y[start : start + (hi - lo)] += x[lo:hi] * window
            weight[start : start + (hi - lo)] += window
        ts += periods[i] / ratio[min(int(ts // hop), len(ratio) - 1)]
    y, weight = y[: len(x)], weight[: len(x)]
    return np.where(weight > 1e-3, y / np.maximum(weight, 1e-3), 0.0)


def tune(
    path: str, tonic: str, mode: str, strength: str, speed: str, *, out: Path
) -> dict[str, object]:
    """The voice moved toward the nearest note of the key (`tonic` 0 = C … 11, `mode` major
    or minor) as `/out/vocals-tuned.wav`: CREPE's pitch every 10 ms, each sung frame's
    distance to the note smoothed over `speed` ms (vibrato and slides stay when it is slow)
    and scaled by `strength` (0 to 100), applied to each channel by TD-PSOLA."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchcrepe

    started = time.monotonic()
    audio, rate = sf.read(path, always_2d=True)
    hop = int(rate * TUNE_HOP)
    mono = torch.tensor(audio.mean(axis=1)[None], dtype=torch.float32)
    if mode not in SCALES:
        raise ValueError(f"mode {mode}: major or minor")
    with contextlib.redirect_stdout(sys.stderr):
        pitch, periodicity = torchcrepe.predict(
            mono, rate, hop, fmin=65.0, fmax=1100.0, model="full", batch_size=512,
            device="cpu", return_periodicity=True,
        )  # fmt: skip
    periodicity = torchcrepe.filter.median(periodicity, 3)
    f0 = torchcrepe.filter.mean(pitch, 3)[0].numpy()
    sung = periodicity[0].numpy() > VOICED
    f0 = np.where(sung, f0, 0.0)
    scale = [(int(tonic) + step) % 12 for step in SCALES[mode]]
    cents = np.zeros(len(f0))
    midi = 69 + 12 * np.log2(np.where(sung, f0, 440.0) / 440.0)
    for k in np.flatnonzero(sung):
        near = min((round(midi[k]) + d for d in range(-2, 3) if (round(midi[k]) + d) % 12 in scale),
                   key=lambda m: abs(m - midi[k]))  # fmt: skip
        cents[k] = (near - midi[k]) * 100
    alpha = math.exp(-TUNE_HOP * 1000 / max(float(speed), 1.0))
    smooth, held = np.zeros(len(cents)), 0.0
    for k, c in enumerate(cents):
        held = alpha * held + (1 - alpha) * c if sung[k] else 0.0
        smooth[k] = held
    applied = smooth * float(strength) / 100
    ratio = 2 ** (applied / 1200)
    tuned = np.stack([_psola(audio[:, c], rate, f0, ratio, hop) for c in range(audio.shape[1])], 1)
    sf.write(str(out / "vocals-tuned.wav"), tuned, rate, subtype="PCM_24")
    moved = np.abs(applied[sung])
    return {
        "tool": "tune",
        "model": f"torchcrepe {metadata.version('torchcrepe')} (full)",
        "file": "vocals-tuned.wav",
        "sung": round(float(sung.mean()), 3),
        "moved_cents": round(float(np.median(moved)), 1) if moved.size else 0.0,
        "seconds": time.monotonic() - started,
    }
