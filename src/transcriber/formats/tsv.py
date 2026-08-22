"""Tab-separated output matching openai-whisper's ``.tsv`` convention.

A ``start\\tend\\ttext`` header line followed by one row per segment, with
``start`` and ``end`` as integer milliseconds. Tabs, carriage returns, and
newlines inside segment text are collapsed to single spaces so every row is
always exactly three fields.
"""

from __future__ import annotations

import math
import re
from typing import ClassVar

from transcriber.formats.base import Formatter
from transcriber.model import Transcript

__all__ = ["TsvFormatter"]

_HEADER = "start\tend\ttext"

#: A run of characters that would corrupt the row structure of a TSV file.
_UNSAFE_RUN = re.compile(r"[\t\r\n]+")


class TsvFormatter(Formatter):
    """Renders one tab-separated row per segment under a header line.

    Times are integer milliseconds rounded half-up, matching the rounding in
    :mod:`transcriber.timestamps` (a row's times agree with the SRT/VTT
    timestamps rendered for the same segment). A transcript with no segments
    renders the header line alone.
    """

    name: ClassVar[str] = "tsv"
    extension: ClassVar[str] = ".tsv"

    def render(self, transcript: Transcript) -> str:
        """Render ``transcript`` as tab-separated values."""
        lines = [_HEADER]
        lines.extend(
            f"{_millis(segment.start)}\t{_millis(segment.end)}\t{_safe_text(segment.text)}"
            for segment in transcript
        )
        return "\n".join(lines) + "\n"


def _millis(seconds: float) -> int:
    """Convert seconds to whole milliseconds, rounding half-up."""
    return math.floor(seconds * 1000.0 + 0.5)


def _safe_text(text: str) -> str:
    """Collapse runs of tabs/newlines to single spaces, keeping rows intact."""
    return _UNSAFE_RUN.sub(" ", text)
