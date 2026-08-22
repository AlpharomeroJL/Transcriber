"""Deterministic timestamp and duration formatting.

These helpers back the subtitle formatters (SRT/VTT) and human-facing output
(clock timestamps, humanised durations). All rounding is half-up and derived
from the float input in one deterministic step, and second/minute overflow is
carried (``59.9994`` seconds renders as ``00:00:59,999``, never ``…,60,000``
milliseconds). Negative inputs clamp to zero; non-finite inputs raise
``ValueError``.
"""

from __future__ import annotations

import math
from typing import Literal

__all__ = ["format_duration", "format_timestamp"]

TimestampStyle = Literal["srt", "vtt", "clock"]
"""The timestamp styles accepted by :func:`format_timestamp`."""


def format_timestamp(seconds: float, *, style: TimestampStyle) -> str:
    """Format a time offset in seconds as a timestamp string.

    Styles:
        * ``"srt"`` — ``HH:MM:SS,mmm`` (SubRip, comma millisecond separator).
        * ``"vtt"`` — ``HH:MM:SS.mmm`` (WebVTT, dot millisecond separator).
        * ``"clock"`` — ``H:MM:SS`` (no milliseconds, hours unpadded; for
          human display, rounded half-up to whole seconds).

    Hours are zero-padded to two digits for ``srt``/``vtt`` and grow beyond
    two digits as needed (``100:00:00,000`` at the 100-hour mark). Milliseconds
    are rounded half-up from the float input, with overflow carried into
    seconds. Negative values clamp to zero.

    Raises:
        ValueError: If ``seconds`` is not finite, or ``style`` is unknown.
    """
    seconds = _clamped_finite(seconds)
    if style == "srt":
        return _hms_with_millis(seconds, separator=",")
    if style == "vtt":
        return _hms_with_millis(seconds, separator=".")
    if style == "clock":
        hours, minutes, secs = _split_whole_seconds(math.floor(seconds + 0.5))
        return f"{hours}:{minutes:02d}:{secs:02d}"
    raise ValueError(f"unknown timestamp style: {style!r}")


def format_duration(seconds: float) -> str:
    """Humanise a duration in seconds.

    Durations under a minute render with tenths of a second (``"42.5s"``);
    longer ones as whole-second components (``"1m 30s"``, ``"1h 02m 03s"``),
    with minutes and seconds zero-padded below a larger unit. Rounding is
    half-up at the displayed precision, carrying overflow upward (``59.96``
    renders ``"1m 00s"``, not ``"60.0s"``). Negative values clamp to zero.

    Raises:
        ValueError: If ``seconds`` is not finite.
    """
    seconds = _clamped_finite(seconds)
    tenths = math.floor(seconds * 10.0 + 0.5) / 10.0
    if tenths < 60.0:
        return f"{tenths:.1f}s"
    hours, minutes, secs = _split_whole_seconds(math.floor(seconds + 0.5))
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _clamped_finite(seconds: float) -> float:
    """Reject non-finite input and clamp negatives to zero."""
    if not math.isfinite(seconds):
        raise ValueError(f"seconds must be finite, got {seconds!r}")
    return max(seconds, 0.0)


def _split_whole_seconds(total_seconds: int) -> tuple[int, int, int]:
    """Split non-negative whole seconds into (hours, minutes, seconds)."""
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return hours, minutes, secs


def _hms_with_millis(seconds: float, *, separator: str) -> str:
    """Render ``HH:MM:SS<separator>mmm`` with half-up millisecond rounding."""
    total_millis = math.floor(seconds * 1000.0 + 0.5)
    hours, minutes, secs = _split_whole_seconds(total_millis // 1000)
    millis = total_millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"
