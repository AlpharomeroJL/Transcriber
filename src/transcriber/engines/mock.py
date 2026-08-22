"""Deterministic offline engine for tests and dry runs.

:class:`MockEngine` is a first-class registered engine (``--engine mock``): it
produces stable, entirely synthetic transcripts without importing any
speech-recognition backend, so the CLI and pipeline can be exercised
end-to-end offline. Two escape hatches make it useful beyond smoke tests:

- a path whose stem contains ``FAIL`` raises
  :class:`~transcriber.errors.TranscriptionFailedError`, letting tests
  exercise per-file error isolation deterministically;
- the constructor accepts fixed segments to return verbatim, for tests that
  need exact known content.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Final

from transcriber.config import TranscriberConfig
from transcriber.engines.base import TranscriptionEngine
from transcriber.errors import TranscriptionFailedError
from transcriber.model import Segment, Transcript, Word

__all__ = ["FAIL_MARKER", "MockEngine"]

#: Substring of a path's stem that makes :class:`MockEngine` fail on purpose.
FAIL_MARKER: Final[str] = "FAIL"

#: File-size-to-duration scale: one synthetic second per this many bytes.
_BYTES_PER_SECOND: Final[int] = 1024


class MockEngine(TranscriptionEngine):
    """Engine that fabricates transcripts deterministically and offline.

    By default each file yields two segments of ``[mock transcription of
    <stem>]``-style text spanning a duration derived from the file's size
    (missing files count as empty), with language ``en`` unless the config
    requests another, and engine and model both reported as ``mock``. Equal
    inputs always produce equal transcripts.

    Args:
        segments: When given, returned verbatim in every transcript instead
            of the synthesised ones, with the duration taken from the last
            segment's end. The ``FAIL`` trigger still applies.
    """

    name: ClassVar[str] = "mock"

    def __init__(self, segments: Sequence[Segment] | None = None) -> None:
        self._fixed_segments: tuple[Segment, ...] | None = (
            None if segments is None else tuple(segments)
        )

    def transcribe(self, path: Path, config: TranscriberConfig) -> Transcript:
        """Fabricate a transcript for *path*; see the class docstring.

        Raises:
            TranscriptionFailedError: If ``path.stem`` contains
                :data:`FAIL_MARKER`.
        """
        if FAIL_MARKER in path.stem:
            raise TranscriptionFailedError(
                f"mock engine failed on purpose for {path} (its name contains {FAIL_MARKER!r})"
            )
        if self._fixed_segments is None:
            duration = _synthetic_duration(path)
            segments = _synthetic_segments(path.stem, duration, config)
        else:
            segments = self._fixed_segments
            duration = max((segment.end for segment in segments), default=0.0)
        return Transcript(
            segments=segments,
            language=config.language or "en",
            language_probability=1.0,
            duration=duration,
            source=str(path),
            engine=self.name,
            model="mock",
        )


def _synthetic_duration(path: Path) -> float:
    """Derive a stable positive duration in seconds from the file's size."""
    size = path.stat().st_size if path.is_file() else 0
    return max(size, 1) / _BYTES_PER_SECOND


def _synthetic_segments(
    stem: str, duration: float, config: TranscriberConfig
) -> tuple[Segment, ...]:
    """Build the two default segments covering ``[0, duration]``."""
    midpoint = duration / 2
    spans = ((1, 0.0, midpoint), (2, midpoint, duration))
    texts = (f"[mock transcription of {stem}]", f"[mock transcription of {stem}, part 2]")
    return tuple(
        Segment(
            id=segment_id,
            start=start,
            end=end,
            text=text,
            words=_synthetic_words(text, start, end) if config.word_timestamps else (),
        )
        for (segment_id, start, end), text in zip(spans, texts, strict=True)
    )


def _synthetic_words(text: str, start: float, end: float) -> tuple[Word, ...]:
    """Spread the whitespace-delimited words of *text* evenly over the span."""
    tokens = text.split()
    if not tokens:
        return ()
    step = (end - start) / len(tokens)
    return tuple(
        Word(
            start=start + index * step,
            end=start + (index + 1) * step,
            text=token,
            probability=1.0,
        )
        for index, token in enumerate(tokens)
    )
