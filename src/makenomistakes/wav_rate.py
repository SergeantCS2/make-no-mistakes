"""A WAV brought to the rate and channels a model reads: the separators refuse any
other ("HTDemucs prepare() sample rate mismatch: expected 44100, got 48000", the `full-cpu`
image on 2026-10-31), and ACE-Step writes 48 kHz. 16-bit PCM in and out, stereo from mono
by copying the channel, the rate by linear interpolation: plain Python, no engine, enough to
split a take. The house format (ffmpeg in a container) replaces it for exports."""

from __future__ import annotations

import io
import math
import sys
import wave
from array import array


class WavError(ValueError):
    """A file that is not a 16-bit PCM WAV, worded for the user."""


def wav_facts(data: bytes) -> tuple[int, int, float]:
    """(rate, channels, seconds) of a WAV."""
    with _open(data) as reader:
        rate, channels, frames = reader.getframerate(), reader.getnchannels(), reader.getnframes()
    return rate, channels, frames / rate if rate else 0.0


def _open(data: bytes) -> wave.Wave_read:
    try:
        return wave.open(io.BytesIO(data), "rb")
    except (wave.Error, EOFError) as exc:
        raise WavError(f"not a WAV file mnm can read ({exc})") from None


def conformed(data: bytes, rate: int, channels: int = 2) -> bytes:
    """The WAV at `rate` with `channels`; the same bytes when it already is."""
    with _open(data) as reader:
        have_rate, have_channels = reader.getframerate(), reader.getnchannels()
        if reader.getsampwidth() != 2:
            raise WavError(f"a {8 * reader.getsampwidth()}-bit WAV: mnm reads 16-bit PCM here")
        if (have_rate, have_channels) == (rate, channels):
            return data
        samples = array("h", reader.readframes(reader.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    lanes = [samples[c::have_channels] for c in range(have_channels)]
    lanes = [_resampled(lane, have_rate, rate) for lane in lanes]
    lanes = (lanes * channels)[:channels] if len(lanes) < channels else lanes[:channels]
    out = array("h", bytes(2 * len(lanes[0]) * channels))
    for c, lane in enumerate(lanes):
        out[c::channels] = lane
    if sys.byteorder == "big":
        out.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(out.tobytes())
    return buffer.getvalue()


def _resampled(lane: array[int], have: int, want: int) -> array[int]:
    if have == want or not lane:
        return lane
    count = max(1, len(lane) * want // have)
    step = have / want
    last = len(lane) - 1
    out = array("h", bytes(2 * count))
    for i in range(count):
        at = i * step
        left = int(at)
        right = min(left + 1, last)
        frac = at - left
        out[i] = round(lane[min(left, last)] * (1 - frac) + lane[right] * frac)
    return out


def sample_mix(seconds: float = 5.0, rate: int = 44_100) -> bytes:
    """A made-up stereo mix for `mnm try` on a separator without `--audio`: a bass note, a
    beat of clicks and a wavering tone where a voice would be, so every stem has something."""
    count = int(seconds * rate)
    out = array("h", bytes(4 * count))
    for i in range(count):
        t = i / rate
        beat = t % 0.5
        kick = 0.5 * math.exp(-beat * 30) * math.sin(2 * math.pi * 60 * beat)
        tone = 220 * (1 + 0.02 * math.sin(2 * math.pi * 5 * t))
        value = (
            0.25 * math.sin(2 * math.pi * 55 * t) + kick + 0.2 * math.sin(2 * math.pi * tone * t)
        )
        out[2 * i] = out[2 * i + 1] = int(max(-1.0, min(1.0, value)) * 30_000)
    if sys.byteorder == "big":
        out.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(out.tobytes())
    return buffer.getvalue()
