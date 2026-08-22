"""JSON output: the versioned transcript schema, pretty-printed.

The document is exactly :meth:`transcriber.model.Transcript.to_dict`
serialised with ``json.dumps(..., ensure_ascii=False, indent=2)`` — non-ASCII
text is preserved verbatim and the result round-trips through
:meth:`transcriber.model.Transcript.from_dict`.
"""

from __future__ import annotations

import json
from typing import ClassVar

from transcriber.formats.base import Formatter
from transcriber.model import Transcript

__all__ = ["JsonFormatter"]


class JsonFormatter(Formatter):
    """Renders the transcript's full versioned dictionary schema as JSON.

    Every transcript — including an empty one — renders a complete JSON
    object; ``segments`` is simply ``[]`` when there are none.
    """

    name: ClassVar[str] = "json"
    extension: ClassVar[str] = ".json"

    def render(self, transcript: Transcript) -> str:
        """Render ``transcript`` as pretty-printed JSON."""
        return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n"
