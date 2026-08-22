#!/usr/bin/env python3
"""Generate the committed WAV fixtures under ``tests/fixtures/``.

Standard library only (``wave`` + ``math``), fully deterministic: the same
interpreter always produces byte-identical files, so the fixtures can be
committed and regenerated at will. Three small 16 kHz mono 16-bit PCM clips
are produced:

- ``tone.wav`` — a 220→880 Hz sine sweep followed by a silent tail;
- ``speechlike.wav`` — an amplitude-modulated harmonic stack with silent
  pauses, so voice-activity-detection paths see speech-like energy variation;
- ``silence.wav`` — pure digital silence.

Run ``python scripts/make_fixture.py`` from the repository root to rewrite
the fixtures in place, or pass ``--out-dir`` to generate them elsewhere
(``tests/test_integration.py`` does exactly that to prove the committed
fixtures match the generator).
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

SAMPLE_RATE: Final[int] = 16_000
"""Frames per second of every generated fixture."""

SAMPLE_WIDTH: Final[int] = 2
"""Bytes per sample: 16-bit signed little-endian PCM."""

CHANNELS: Final[int] = 1
"""Channel count: mono."""

PEAK: Final[int] = 24_575
"""Peak sample amplitude (~75% of int16 full scale, headroom against clipping)."""

DEFAULT_OUT_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
"""Where the committed fixtures live, relative to this script."""


def _clamp16(value: float) -> int:
    """Round *value* to the nearest sample and clamp it into int16 range."""
    return max(-32_768, min(32_767, round(value)))


def _silence(seconds: float) -> list[int]:
    """*seconds* of digital silence (all-zero samples)."""
    return [0] * round(seconds * SAMPLE_RATE)


def tone_samples() -> list[int]:
    """A 0.9 s linear sine sweep from 220 Hz to 880 Hz, then 0.3 s of silence."""
    sweep_seconds, low_hz, high_hz = 0.9, 220.0, 880.0
    samples: list[int] = []
    for frame in range(round(sweep_seconds * SAMPLE_RATE)):
        t = frame / SAMPLE_RATE
        # Instantaneous phase of a linear chirp: 2*pi * (f0*t + (f1-f0)*t^2 / (2*T)).
        phase = math.tau * (low_hz * t + (high_hz - low_hz) * t * t / (2.0 * sweep_seconds))
        samples.append(_clamp16(PEAK * math.sin(phase)))
    samples.extend(_silence(0.3))
    return samples


def speechlike_samples() -> list[int]:
    """1.6 s of syllable-like bursts: harmonics, 4 Hz modulation, silent pauses.

    Voiced spans carry a 140 Hz fundamental with two harmonics, amplitude-
    modulated at a 4 Hz syllable rate; leading, mid, and trailing silences
    give VAD-style consumers genuine speech/pause energy variation.
    """
    voiced_spans = ((0.1, 0.7), (0.9, 1.5))
    total_seconds = 1.6
    harmonics = ((140.0, 0.60), (280.0, 0.28), (420.0, 0.12))
    syllable_hz = 4.0
    samples = [0] * round(total_seconds * SAMPLE_RATE)
    for span_start, span_end in voiced_spans:
        for frame in range(round(span_start * SAMPLE_RATE), round(span_end * SAMPLE_RATE)):
            t = frame / SAMPLE_RATE
            envelope = math.sin(math.pi * syllable_hz * (t - span_start)) ** 2
            carrier = sum(weight * math.sin(math.tau * hz * t) for hz, weight in harmonics)
            samples[frame] = _clamp16(PEAK * envelope * carrier)
    return samples


def silence_samples() -> list[int]:
    """1.0 s of pure digital silence."""
    return _silence(1.0)


FIXTURES: Final[dict[str, Callable[[], list[int]]]] = {
    "tone.wav": tone_samples,
    "speechlike.wav": speechlike_samples,
    "silence.wav": silence_samples,
}
"""Fixture file names mapped to their sample generators."""


def write_wav(path: Path, samples: Sequence[int]) -> None:
    """Write *samples* to *path* as a 16 kHz mono 16-bit PCM WAV file."""
    frames = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(SAMPLE_WIDTH)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(frames)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate every fixture into the chosen directory; return the exit code."""
    parser = argparse.ArgumentParser(
        description="Generate the committed WAV fixtures under tests/fixtures/."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory to write the fixtures into (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(FIXTURES):
        samples = FIXTURES[name]()
        target = out_dir / name
        write_wav(target, samples)
        seconds = len(samples) / SAMPLE_RATE
        print(f"wrote {target} ({target.stat().st_size} bytes, {seconds:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
