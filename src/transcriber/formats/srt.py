"""SubRip (``.srt``) subtitle output.

The standard SubRip shape: cues numbered from 1, a
``HH:MM:SS,mmm --> HH:MM:SS,mmm`` time line (comma millisecond separator),
the cue text, and a blank line between consecutive cues.
"""

from __future__ import annotations

from typing import ClassVar

from transcriber.formats.base import Formatter
from transcriber.model import Transcript
from transcriber.timestamps import format_timestamp

__all__ = ["SrtFormatter"]


class SrtFormatter(Formatter):
    """Renders standard SubRip subtitles, one cue per segment.

    A transcript with no segments renders as ``"\\n"``: no cues, just the
    single trailing newline every formatter guarantees.
    """

    name: ClassVar[str] = "srt"
    extension: ClassVar[str] = ".srt"

    def render(self, transcript: Transcript) -> str:
        """Render ``transcript`` as SubRip subtitles."""
        cues = [
            f"{counter}\n{_time_line(segment.start, segment.end)}\n{segment.text}"
            for counter, segment in enumerate(transcript, start=1)
        ]
        return "\n\n".join(cues) + "\n"


def _time_line(start: float, end: float) -> str:
    """The ``start --> end`` line of one cue."""
    return f"{format_timestamp(start, style='srt')} --> {format_timestamp(end, style='srt')}"
