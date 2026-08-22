"""Engine-agnostic transcript data model.

The classes in this module are the common currency of the whole library:
engines produce :class:`Transcript` values, formatters consume them, and the
JSON output format is exactly :meth:`Transcript.to_dict`. They are frozen,
slotted dataclasses — plain immutable values with no engine-specific baggage —
so transcripts can be compared, hashed into sets, serialised, and round-tripped
without surprises.

Serialisation is versioned: :meth:`Transcript.to_dict` emits a
``schema_version`` key (currently :data:`SCHEMA_VERSION`) and
:meth:`Transcript.from_dict` refuses dictionaries carrying any other version.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["SCHEMA_VERSION", "Segment", "Transcript", "Word"]

SCHEMA_VERSION: Final[int] = 1
"""Version of the dictionary/JSON schema produced by :meth:`Transcript.to_dict`."""


def _validate_span(start: float, end: float, *, label: str) -> None:
    """Validate a ``[start, end]`` time span in seconds, raising ``ValueError``."""
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"{label} times must be finite, got start={start!r}, end={end!r}")
    if start < 0:
        raise ValueError(f"{label} start must be >= 0, got {start!r}")
    if end < 0:
        raise ValueError(f"{label} end must be >= 0, got {end!r}")
    if end < start:
        raise ValueError(f"{label} end must be >= start, got start={start!r}, end={end!r}")


@dataclass(frozen=True, slots=True)
class Word:
    """A single recognised word with its time span.

    Attributes:
        start: Start time in seconds from the beginning of the media (>= 0).
        end: End time in seconds (>= ``start``).
        text: The word text, preserved exactly as produced by the engine.
        probability: Engine-reported confidence in ``[0, 1]``, or ``None``
            when the engine does not provide one.
    """

    start: float
    end: float
    text: str
    probability: float | None = None

    def __post_init__(self) -> None:
        _validate_span(self.start, self.end, label="word")


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous span of transcribed speech.

    Attributes:
        id: Segment identifier as assigned by the engine (typically 1-based).
        start: Start time in seconds (>= 0).
        end: End time in seconds (>= ``start``).
        text: Segment text; surrounding whitespace is stripped on construction.
        words: Per-word timings when word timestamps were requested, else ``()``.
        avg_logprob: Engine-reported average log-probability, if available.
        no_speech_prob: Engine-reported no-speech probability, if available.
    """

    id: int
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    def __post_init__(self) -> None:
        _validate_span(self.start, self.end, label="segment")
        # Frozen dataclass: normalise the stored text via object.__setattr__.
        object.__setattr__(self, "text", self.text.strip())


@dataclass(frozen=True, slots=True)
class Transcript:
    """The complete result of transcribing one media file.

    Attributes:
        segments: The transcribed segments, in playback order.
        language: Detected or requested language code (e.g. ``"en"``), or
            ``None`` when unknown.
        language_probability: Confidence of the language detection, or ``None``.
        duration: Duration of the source audio in seconds (>= 0).
        source: The input the transcript was produced from (path or name).
        engine: Name of the engine that produced it (e.g. ``"faster-whisper"``).
        model: Name of the model used (e.g. ``"small"``).
    """

    segments: tuple[Segment, ...]
    language: str | None
    language_probability: float | None
    duration: float
    source: str
    engine: str
    model: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration):
            raise ValueError(f"transcript duration must be finite, got {self.duration!r}")
        if self.duration < 0:
            raise ValueError(f"transcript duration must be >= 0, got {self.duration!r}")

    def __iter__(self) -> Iterator[Segment]:
        """Iterate over segments in playback order."""
        return iter(self.segments)

    @property
    def text(self) -> str:
        """All segment texts joined by newlines."""
        return "\n".join(segment.text for segment in self.segments)

    @property
    def is_empty(self) -> bool:
        """True when no segment carries any text (including zero segments)."""
        return not any(segment.text for segment in self.segments)

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words across all segment texts."""
        return sum(len(segment.text.split()) for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain, JSON-serialisable dictionary.

        Times and durations are raw float seconds (no ISO-8601 strings). The
        result carries ``schema_version`` :data:`SCHEMA_VERSION` and round-trips
        exactly through :meth:`from_dict`, including via ``json.dumps`` /
        ``json.loads``.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": [_segment_to_dict(segment) for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Transcript:
        """Reconstruct a :class:`Transcript` from :meth:`to_dict` output.

        Raises:
            ValueError: If ``schema_version`` is missing or unsupported, or a
                required key is absent.
        """
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported transcript schema_version {version!r}; expected {SCHEMA_VERSION}"
            )
        try:
            return cls(
                segments=tuple(_segment_from_dict(raw) for raw in data["segments"]),
                language=_opt_str(data["language"]),
                language_probability=_opt_float(data["language_probability"]),
                duration=float(data["duration"]),
                source=str(data["source"]),
                engine=str(data["engine"]),
                model=str(data["model"]),
            )
        except KeyError as exc:
            raise ValueError(f"transcript dict is missing required key {exc.args[0]!r}") from exc


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _word_to_dict(word: Word) -> dict[str, Any]:
    return {
        "start": word.start,
        "end": word.end,
        "text": word.text,
        "probability": word.probability,
    }


def _word_from_dict(data: Mapping[str, Any]) -> Word:
    return Word(
        start=float(data["start"]),
        end=float(data["end"]),
        text=str(data["text"]),
        probability=_opt_float(data["probability"]),
    )


def _segment_to_dict(segment: Segment) -> dict[str, Any]:
    return {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": [_word_to_dict(word) for word in segment.words],
        "avg_logprob": segment.avg_logprob,
        "no_speech_prob": segment.no_speech_prob,
    }


def _segment_from_dict(data: Mapping[str, Any]) -> Segment:
    return Segment(
        id=int(data["id"]),
        start=float(data["start"]),
        end=float(data["end"]),
        text=str(data["text"]),
        words=tuple(_word_from_dict(raw) for raw in data["words"]),
        avg_logprob=_opt_float(data["avg_logprob"]),
        no_speech_prob=_opt_float(data["no_speech_prob"]),
    )
