"""WebVTT (``.vtt``) subtitle output.

WebVTT files open with a ``WEBVTT`` header line followed by a blank line;
cues carry ``HH:MM:SS.mmm --> HH:MM:SS.mmm`` time lines (dot millisecond
separator), no cue counters, and a blank line between consecutive cues.
"""

from __future__ import annotations

from typing import ClassVar

from transcriber.formats.base import Formatter
from transcriber.model import Transcript
from transcriber.timestamps import format_timestamp

__all__ = ["VttFormatter"]


class VttFormatter(Formatter):
    """Renders WebVTT subtitles, one cue per segment.

    A transcript with no segments renders the header alone: ``"WEBVTT\\n"``.
    """

    name: ClassVar[str] = "vtt"
    extension: ClassVar[str] = ".vtt"

    def render(self, transcript: Transcript) -> str:
        """Render ``transcript`` as WebVTT subtitles."""
        blocks = ["WEBVTT"]
        blocks.extend(
            f"{_time_line(segment.start, segment.end)}\n{segment.text}" for segment in transcript
        )
        return "\n\n".join(blocks) + "\n"


def _time_line(start: float, end: float) -> str:
    """The ``start --> end`` line of one cue."""
    return f"{format_timestamp(start, style='vtt')} --> {format_timestamp(end, style='vtt')}"
