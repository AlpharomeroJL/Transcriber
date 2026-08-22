"""Markdown output: a titled, timestamped reading view of the transcript."""

from __future__ import annotations

from typing import ClassVar

from transcriber.formats.base import Formatter
from transcriber.model import Transcript
from transcriber.timestamps import format_duration, format_timestamp

__all__ = ["MarkdownFormatter"]


class MarkdownFormatter(Formatter):
    """Renders a Markdown document: heading, metadata list, timestamped lines.

    Layout: ``# Transcript of <source>``, a blank line, one metadata bullet
    each for engine, model, language (``unknown`` when undetected), and the
    humanised duration; then — when there are segments — a blank line and one
    ``**[H:MM:SS]** text`` line per segment using clock-style start times.
    A transcript with no segments renders the heading and metadata alone.
    """

    name: ClassVar[str] = "md"
    extension: ClassVar[str] = ".md"

    def render(self, transcript: Transcript) -> str:
        """Render ``transcript`` as Markdown."""
        lines = [
            f"# Transcript of {transcript.source}",
            "",
            f"- Engine: {transcript.engine}",
            f"- Model: {transcript.model}",
            f"- Language: {transcript.language or 'unknown'}",
            f"- Duration: {format_duration(transcript.duration)}",
        ]
        if transcript.segments:
            lines.append("")
            lines.extend(_segment_line(segment.start, segment.text) for segment in transcript)
        return "\n".join(lines) + "\n"


def _segment_line(start: float, text: str) -> str:
    """One ``**[H:MM:SS]** text`` line (no trailing space when text is empty)."""
    return f"**[{format_timestamp(start, style='clock')}]** {text}".rstrip()
