"""Plain-text output: one segment text per line."""

from __future__ import annotations

from typing import ClassVar

from transcriber.formats.base import Formatter
from transcriber.model import Transcript

__all__ = ["TxtFormatter"]


class TxtFormatter(Formatter):
    """Renders each segment's text on its own line, nothing else.

    A transcript with no segments renders as ``"\\n"``: an empty body plus
    the single trailing newline every formatter guarantees.
    """

    name: ClassVar[str] = "txt"
    extension: ClassVar[str] = ".txt"

    def render(self, transcript: Transcript) -> str:
        """Render ``transcript`` as plain text."""
        return transcript.text + "\n"
