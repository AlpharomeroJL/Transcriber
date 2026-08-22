"""Output formatters: transcripts rendered as txt, srt, vtt, json, tsv, or md.

Formatters are looked up by name in a fixed registry::

    from transcriber.formats import get_formatter

    formatter = get_formatter("srt")
    path.with_suffix(formatter.extension).write_text(formatter.render(transcript))

Rendering is pure and deterministic (see :class:`Formatter`): the same
transcript always yields byte-identical output, with ``\\n`` line endings and
exactly one trailing newline, and an empty transcript renders valid minimal
output in every format. Lookups are exact — format names are the lowercase
strings returned by :func:`available_formats`.
"""

from __future__ import annotations

from typing import Final

from transcriber.errors import OutputError
from transcriber.formats.base import Formatter
from transcriber.formats.json import JsonFormatter
from transcriber.formats.md import MarkdownFormatter
from transcriber.formats.srt import SrtFormatter
from transcriber.formats.tsv import TsvFormatter
from transcriber.formats.txt import TxtFormatter
from transcriber.formats.vtt import VttFormatter

__all__ = [
    "FORMATTERS",
    "Formatter",
    "JsonFormatter",
    "MarkdownFormatter",
    "SrtFormatter",
    "TsvFormatter",
    "TxtFormatter",
    "VttFormatter",
    "available_formats",
    "get_formatter",
]

#: Shared formatter instances keyed by format name. Treat as read-only; the
#: set of formats is fixed at import time.
FORMATTERS: Final[dict[str, Formatter]] = {
    formatter.name: formatter
    for formatter in (
        TxtFormatter(),
        SrtFormatter(),
        VttFormatter(),
        JsonFormatter(),
        TsvFormatter(),
        MarkdownFormatter(),
    )
}


def get_formatter(name: str) -> Formatter:
    """Return the registered formatter for ``name`` (exact match, e.g. ``"srt"``).

    Raises:
        OutputError: If ``name`` is not a registered format. The message
            lists the available format names.
    """
    try:
        return FORMATTERS[name]
    except KeyError:
        raise OutputError(
            f"unknown output format {name!r}; available formats: {', '.join(available_formats())}"
        ) from None


def available_formats() -> tuple[str, ...]:
    """All registered format names, sorted alphabetically."""
    return tuple(sorted(FORMATTERS))
