"""The formatter contract shared by every output format.

A formatter turns a :class:`~transcriber.model.Transcript` into the complete
text of an output document. Rendering is pure and deterministic: the same
transcript always produces byte-identical output, with ``\\n`` line endings
and exactly one trailing newline. Concrete formatters are stateless; the
registry in :mod:`transcriber.formats` holds one shared instance of each.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from transcriber.model import Transcript

__all__ = ["Formatter"]


class Formatter(ABC):
    """Renders transcripts in one output format.

    Attributes:
        name: The registry key and user-facing format name (e.g. ``"srt"``),
            unique across registered formatters.
        extension: The conventional file extension for the format, including
            the leading dot (e.g. ``".srt"``), ready to hand to
            :meth:`pathlib.PurePath.with_suffix`.
    """

    name: ClassVar[str]
    extension: ClassVar[str]

    @abstractmethod
    def render(self, transcript: Transcript) -> str:
        """Render ``transcript`` as the full text of an output document.

        Returns:
            The rendered document. Line endings are always ``"\\n"`` and the
            text ends with exactly one trailing newline, for every transcript
            including an empty one — a format whose body is empty renders as
            ``"\\n"`` alone. Each concrete formatter documents its minimal
            empty-transcript output.
        """
