#!/opt/venv/bin/python
"""The studio's tools as one command inside `mnm/studio-tools`:
every job is one `docker run --rm --network none` over a run's folder mounted at /work,
and it prints one JSON line on stdout, which mnm reads into a typed answer
(`studio/tools_answers.py`). Nothing here reaches the network: the weights were fetched and
checked by sha256 when the image was built, and the image runs as the user's uid.

    mnm-tools versions                  what the image carries (the ledger records it)
    mnm-tools score /work/song.wav      Audiobox Aesthetics' four axes over the whole file
    mnm-tools structure /work/song.wav  beats, downbeats, tempo, bars and the parts of the song
    mnm-tools similarity /work/ref.wav /work/a.wav …  how alike each file's voice is to the
                                        reference's (SpeechBrain's ECAPA embeddings, cosine)
    mnm-tools midi /work/a.wav …        each file's notes as `/out/<name>.mid` (Basic Pitch,
                                        its ONNX model)
    mnm-tools master /work/mix.wav /work/ref.wav  the mix matched to the reference's level,
                                        tone and width as `/out/master.wav` (Matchering)

A job that writes files gets one empty folder mounted writable at /out; every other path is
read-only.

The score's one number and every rule that decides what a reading means live in mnm, never
here: this prints what the models measured. Runs with the image's Python only (torch,
audiobox_aesthetics, beat_this, librosa, soundfile, speechbrain, basic_pitch, matchering);
mnm's own tests never import it.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from importlib import metadata
from itertools import pairwise
from pathlib import Path

WEIGHTS = Path("/opt/weights")
SCORER = WEIGHTS / "audiobox-aesthetics"
BEATS = WEIGHTS / "beat_this" / "final0.ckpt"
ECAPA = WEIGHTS / "ecapa" / "embedding_model.ckpt"
SPEAKER_RATE = 16_000  # what the ECAPA model was trained at
QUIET_DB = 35  # below the loudest part by this much is silence, left out of an embedding
ANALYSIS_RATE = 22050
HOP = 512
KERNEL_BARS = 4  # the novelty kernel's half width: a change shows against four bars each
#                   side (two bars found every chord change of a four-bar loop)
MIN_BARS = 4  # the shortest part a boundary may leave (two boundaries this close: the stronger)
SAME_PART = 0.6  # a part repeats another when their mean features' cosine is at least this
#                  (a synthetic A-B-A song: A with A 0.83, A with B -0.46 and -0.56)
OUT = Path("/out")
MASTER_RATE = 48_000  # the house format's rate: Matchering works at 44.1 kHz unless told
# Temperley's key profiles (Kostka-Payne corpus, as music21 carries them), C first: the time-
# mean chroma of the take's harmonic part correlated with each of the 24 rotations
# (Temperley and Albrecht-Shanahan found 9 of 10 synthetic keys, Krumhansl-Kessler 6)
MAJOR = (0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400)
MINOR = (0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330)
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
TOOLS = (
    "torch",
    "audiobox_aesthetics",
    "beat_this",
    "librosa",
    "soundfile",
    "numpy",
    "speechbrain",
    "basic_pitch",
    "matchering",
    "onnxruntime",
)


def _load(path: str, rate: int | None = None) -> tuple[object, int]:
    """The file as float32 (channels, samples) at its own rate, or mono at `rate`."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    signal = np.ascontiguousarray(data.T)
    if rate is None:
        return signal, int(sr)
    import librosa

    mono = librosa.to_mono(signal)
    return librosa.resample(mono, orig_sr=sr, target_sr=rate), rate


def versions() -> dict[str, object]:
    found = {}
    for name in TOOLS:
        try:
            found[name] = metadata.version(name.replace("_", "-"))
        except metadata.PackageNotFoundError:
            found[name] = metadata.version(name)
    return {"tool": "versions", "versions": found}


def score(path: str) -> dict[str, object]:
    """Audiobox Aesthetics over the whole file (its own 10 s windows, weighted)."""
    import torch
    from audiobox_aesthetics.infer import AesPredictor, Normalize
    from audiobox_aesthetics.model.aes import AXES_NAME, AesMultiOutput

    class Local(AesPredictor):  # the image's weights, never the Hub
        def setup_model(self) -> None:
            self.device = torch.device("cpu")
            model = AesMultiOutput.from_pretrained(str(SCORER))
            model.to(self.device).eval()
            self.model = model
            self.target_transform = {
                axis: Normalize(
                    mean=model.target_transform[axis]["mean"],
                    std=model.target_transform[axis]["std"],
                )
                for axis in AXES_NAME
            }

    started = time.monotonic()
    signal, sr = _load(path)
    predictor = Local(checkpoint_pth=None, data_col="path")
    row = predictor.forward([{"path": torch.from_numpy(signal), "sample_rate": sr}])[0]
    return {
        "tool": "score",
        "scorer": f"audiobox-aesthetics {metadata.version('audiobox-aesthetics')}",
        "axes": {axis: float(row[axis]) for axis in AXES_NAME},
        "seconds": time.monotonic() - started,
    }


def _beats(path: str) -> tuple[list[float], list[float]]:
    from beat_this.inference import Audio2Beats

    signal, sr = _load(path)
    import librosa

    mono = librosa.to_mono(signal)
    tracker = Audio2Beats(checkpoint_path=str(BEATS), device="cpu", dbn=False)
    beats, downbeats = tracker(mono, sr)
    return [float(b) for b in beats], [float(d) for d in downbeats]


def _bar_length(beats: list[float], downbeats: list[float]) -> int:
    """Beats per bar: the commonest count of beats between two downbeats (4 when unsure)."""
    counts: dict[int, int] = {}
    for first, second in pairwise(downbeats):
        inside = sum(1 for b in beats if first - 0.02 <= b < second - 0.02)
        if 2 <= inside <= 7:
            counts[inside] = counts.get(inside, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else 4


def _features(y: object, beats: list[float], beat_frames: object) -> tuple[object, list[float]]:
    """Per beat: chroma (harmony), MFCC (timbre) and loudness, each scaled to unit length,
    and each column's start in seconds. `sync` cuts at the beats with the file's first and
    last frame added (merged when the first beat is at 0), so a take with a pickup before
    its first beat has one column more than beats: the starts come from the same bounds."""
    import librosa
    import numpy as np

    chroma = librosa.feature.chroma_cqt(y=y, sr=ANALYSIS_RATE, hop_length=HOP)
    bounds = librosa.util.fix_frames(beat_frames, x_min=0, x_max=chroma.shape[1])
    at_beat = {int(f): b for f, b in zip(beat_frames, beats, strict=True)}  # a beat's own time
    starts = [
        at_beat.get(int(f), float(librosa.frames_to_time(f, sr=ANALYSIS_RATE, hop_length=HOP)))
        for f in bounds[:-1]
    ]
    mfcc = librosa.feature.mfcc(y=y, sr=ANALYSIS_RATE, hop_length=HOP, n_mfcc=13)[1:]
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    parts = []
    for feature in (chroma, mfcc, rms):
        synced = librosa.util.sync(feature, beat_frames, aggregate=np.mean)
        synced = (synced - synced.mean(axis=1, keepdims=True)) / (
            synced.std(axis=1, keepdims=True) + 1e-6
        )
        parts.append(synced / (np.linalg.norm(synced, axis=0, keepdims=True) + 1e-6))
    return np.vstack(parts), starts


def _novelty(similar: object, half: int) -> object:
    """Foote's novelty: a Gaussian-tapered checkerboard kernel along the diagonal, each point
    divided by the share of the kernel that lies inside the song (near either end the
    kernel runs off it, and zero padding would raise the curve there: a pickup before the
    first beat had moved the last change a bar late)."""
    import numpy as np

    size = 2 * half
    axis = np.arange(size) - half + 0.5
    taper = np.exp(-0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2) / (0.5 * half) ** 2)
    kernel = taper * np.sign(axis[:, None]) * np.sign(axis[None, :])
    weight = np.abs(kernel)
    padded = np.pad(similar, half, mode="constant")
    inside = np.pad(np.ones_like(similar), half, mode="constant")
    count = similar.shape[0]
    curve = np.zeros(count)
    for i in range(count):
        window = (slice(i, i + size), slice(i, i + size))
        share = float((inside[window] * weight).sum() / weight.sum())
        curve[i] = (padded[window] * kernel).sum() / max(share, 0.25)
    curve -= curve.min()
    return curve / (curve.max() + 1e-9)


def _boundaries(curve: object, on_bars: set[int], bar: int) -> list[int]:
    """Peaks of the novelty curve (beat columns), strongest first, each moved to the nearest
    downbeat within half a bar (a change heard a beat early or late is the bar's), kept at
    least MIN_BARS apart and two bars from either end (the kernel runs off the edge there)."""
    import numpy as np

    count = len(curve)
    threshold = float(np.mean(curve) + 0.5 * np.std(curve))
    peaks = [
        i
        for i in range(1, count - 1)
        if curve[i] >= threshold and curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1]
    ]
    gap, edge, reach = MIN_BARS * bar, 2 * bar, max(1, bar // 2)
    kept: list[int] = []
    for peak in sorted(peaks, key=lambda p: -curve[p]):
        near = [d for d in on_bars if abs(d - peak) <= reach]
        if not near:
            continue
        cut = min(near, key=lambda d: (abs(d - peak), d))
        if edge <= cut <= count - edge and all(abs(cut - k) >= gap for k in kept):
            kept.append(cut)
    return sorted(kept)


def _labels(features: object, spans: list[tuple[int, int]]) -> list[str]:
    """A letter per part; a part whose mean features point the same way as an earlier one's
    (cosine at least SAME_PART) takes that one's letter."""
    import numpy as np

    centres = []
    for a, b in spans:
        centre = features[:, a:b].mean(axis=1)
        centres.append(centre / (np.linalg.norm(centre) + 1e-9))
    letters: list[str] = []
    for i in range(len(spans)):
        best, best_value = None, SAME_PART
        for j in range(i):
            value = float(centres[i] @ centres[j])
            if value >= best_value:
                best, best_value = j, value
        if best is not None:
            letters.append(letters[best])
        else:
            used = len(set(letters))
            letters.append(chr(ord("A") + used) if used < 26 else "Z")
    return letters


def _key(y: object) -> tuple[str, float]:
    """The take's key and how far it stands out from the next best (0 to 1): its harmonic
    part's mean chroma against Temperley's 24 profiles."""
    import librosa
    import numpy as np

    harmonic = librosa.effects.harmonic(y, margin=3)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=ANALYSIS_RATE, hop_length=HOP)
    mean = chroma.mean(axis=1)
    found = []
    for minor, profile in ((False, MAJOR), (True, MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(mean, np.roll(profile, tonic))[0, 1])
            found.append((r if np.isfinite(r) else -1.0, tonic, minor))
    found.sort(reverse=True)
    (best, tonic, minor), (second, _t, _m) = found[0], found[1]
    return f"{NAMES[tonic]} {'minor' if minor else 'major'}", max(0.0, min(1.0, best - second))


def structure(path: str) -> dict[str, object]:
    import librosa
    import numpy as np

    started = time.monotonic()
    beats, downbeats = _beats(path)
    y, _rate = _load(path, ANALYSIS_RATE)
    duration = float(len(y) / ANALYSIS_RATE)
    bar = _bar_length(beats, downbeats)
    intervals = np.diff(beats) if len(beats) > 1 else np.array([0.5])
    tempo = float(60.0 / np.median(intervals)) if len(intervals) else 120.0
    sections: list[dict[str, object]] = []
    if len(beats) >= 4 * bar:
        frames = librosa.time_to_frames(beats, sr=ANALYSIS_RATE, hop_length=HOP)
        features, starts = _features(y, beats, frames)
        similar = features.T @ features / features.shape[0]
        curve = _novelty(similar, KERNEL_BARS * bar)
        on_bars = {i for i, t in enumerate(starts) if any(abs(t - d) < 0.07 for d in downbeats)}
        cuts = _boundaries(curve, on_bars, bar)
        edges = [0, *cuts, similar.shape[0]]
        spans = list(pairwise(edges))
        letters = _labels(features, spans)
        times = [*starts, duration]  # column i starts at starts[i]; the last ends the song
        for (a, b), letter in zip(spans, letters, strict=True):
            sections.append(
                {"start": round(times[a], 3), "end": round(times[b], 3), "label": letter}
            )
        if sections:
            sections[0]["start"], sections[-1]["end"] = 0.0, round(duration, 3)
    key, clarity = _key(y)
    return {
        "tool": "structure",
        "analyser": f"beat_this {metadata.version('beat-this')} final0, Foote novelty",
        "duration": round(duration, 3),
        "tempo": round(tempo, 2),
        "beats_per_bar": bar,
        "beats": [round(b, 3) for b in beats],
        "downbeats": [round(d, 3) for d in downbeats],
        "sections": sections,
        "key": key,
        "key_clarity": round(clarity, 3),
        "seconds": time.monotonic() - started,
    }


def _voiced(path: str) -> object:
    """The file's voice at 16 kHz mono, its silences left out (a vocals stem is quiet between
    the lines; silence would pull every embedding toward the same place)."""
    import librosa
    import numpy as np

    signal, _rate = _load(path, SPEAKER_RATE)
    spans = librosa.effects.split(signal, top_db=QUIET_DB)
    if len(spans) == 0:
        raise ValueError(f"{Path(path).name} is silent")
    return np.ascontiguousarray(np.concatenate([signal[a:b] for a, b in spans]), dtype="float32")


def _embedder() -> object:
    """SpeechBrain's ECAPA-TDNN as its card's hyperparameters build it (`spkrec-ecapa-
    voxceleb`), its weights from the image; the features normalised per file, as there."""
    import torch
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    )
    model.load_state_dict(torch.load(ECAPA, map_location="cpu", weights_only=True))
    model.eval()
    fbank = Fbank(n_mels=80)

    def embed(signal: object) -> object:
        with torch.no_grad():
            feats = fbank(torch.from_numpy(signal).unsqueeze(0))  # type: ignore[arg-type]
            feats = feats - feats.mean(dim=1, keepdim=True)
            return model(feats).flatten()

    return embed


def similarity(reference: str, *paths: str) -> dict[str, object]:
    """Each file's voice against the reference's: the cosine of their speaker embeddings."""
    import torch

    started = time.monotonic()
    embed = _embedder()
    wanted = embed(_voiced(reference))
    found = {}
    for path in paths:
        other = embed(_voiced(path))
        found[Path(path).name] = float(torch.nn.functional.cosine_similarity(wanted, other, dim=0))
    return {
        "tool": "similarity",
        "model": f"speechbrain {metadata.version('speechbrain')} ecapa-voxceleb",
        "similarity": found,
        "seconds": time.monotonic() - started,
    }


def midi(*paths: str) -> dict[str, object]:
    """Each file's notes as `/out/<name>.mid`: Basic Pitch's ONNX model (its wheel carries it),
    every note with its pitch bends; the count per file."""
    import logging

    logging.disable(logging.WARNING)  # its import warns about the runtimes it does not find
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    started = time.monotonic()
    found = {}
    for path in paths:
        with contextlib.redirect_stdout(sys.stderr):  # it prints "Predicting MIDI for …"
            _output, notes, events = predict(path, ICASSP_2022_MODEL_PATH)
        name = f"{Path(path).stem}.mid"
        notes.write(str(OUT / name))
        found[Path(path).name] = {"midi": name, "notes": len(events)}
    return {
        "tool": "midi",
        "model": f"basic-pitch {metadata.version('basic-pitch')} (onnx)",
        "files": found,
        "seconds": time.monotonic() - started,
    }


def master(target: str, reference: str) -> dict[str, object]:
    """The mix matched to the reference (its loudness, frequency response, peak and width),
    then Matchering's own limiter, as `/out/master.wav` at the house format's rate."""
    import matchering as mg

    started = time.monotonic()
    said: list[str] = []
    mg.log(said.append)  # its messages stay off stdout, where the answer goes
    config = mg.Config(internal_sample_rate=MASTER_RATE)
    mg.process(target=target, reference=reference, results=[mg.pcm24(str(OUT / "master.wav"))],
               config=config)  # fmt: skip
    return {
        "tool": "master",
        "matchering": metadata.version("matchering"),
        "file": "master.wav",
        "notes": said[-20:],
        "seconds": time.monotonic() - started,
    }


def main(argv: list[str]) -> int:
    commands = {"score": score, "structure": structure}
    try:
        files = argv[1:2] if argv[:1] == ["tune"] else argv[1:]  # tune: then four words
        for name in files:
            if not Path(name).is_file():
                raise FileNotFoundError(f"no file {name}")
        if argv[:1] == ["versions"]:
            answer = versions()
        elif len(argv) == 2 and argv[0] in commands:
            answer = commands[argv[0]](argv[1])
        elif len(argv) >= 3 and argv[0] == "similarity":
            answer = similarity(argv[1], *argv[2:])
        elif len(argv) >= 2 and argv[0] == "midi":
            answer = midi(*argv[1:])
        elif len(argv) == 3 and argv[0] == "master":
            answer = master(argv[1], argv[2])
        elif len(argv) == 6 and argv[0] == "tune":
            from mnm_tune import tune  # beside this script in the image

            answer = tune(*argv[1:], out=OUT)
        else:
            raise ValueError(
                "usage: mnm-tools versions | score FILE | structure FILE | similarity REF FILE… "
                "| midi FILE… | master MIX REFERENCE | tune FILE TONIC major|minor STRENGTH SPEED"
            )
    except Exception as exc:  # one worded line for mnm, never a traceback it must parse
        print(json.dumps({"tool": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
