"""Media input discovery and validation.

This module decides *which* files a transcription run will process; it never
opens or decodes them — decoding is the engines' job. Discovery is therefore
purely name-based: a file is considered transcribable when its extension is in
:data:`SUPPORTED_EXTENSIONS` (audio formats plus the video containers the
engines can demux audio from).

:func:`collect_inputs` turns the raw paths a user supplies — files,
directories, or a mix — into a validated, deduplicated, deterministically
sorted list of media files:

- a path that does not exist raises
  :class:`~transcriber.errors.AudioNotFoundError` naming it;
- an explicitly named file with an unsupported extension raises
  :class:`~transcriber.errors.UnsupportedFormatError` listing the supported
  ones;
- a directory is scanned (recursively on request) keeping only supported
  files and silently skipping everything else, hidden entries included;
- an empty overall result raises
  :class:`~transcriber.errors.AudioNotFoundError`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

from transcriber.errors import AudioNotFoundError, UnsupportedFormatError

__all__ = ["SUPPORTED_EXTENSIONS", "collect_inputs", "is_supported"]

_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".aac",
        ".wma",
        ".aiff",
        ".aif",
        ".mka",
    }
)

#: Video containers the engines (via PyAV/FFmpeg) can demux audio from.
_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".ts",
    }
)

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS
"""File extensions transcriber accepts, lowercase with leading dot.

Covers common audio formats and the video containers the engines can demux
audio from. Matching is by suffix only — see :func:`is_supported`.
"""


def is_supported(path: Path) -> bool:
    """Return whether *path* has a supported media file extension.

    The check is by suffix alone, case-insensitively (``clip.WAV`` counts);
    the file need not exist and its contents are never inspected.
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_inputs(paths: Sequence[Path], *, recursive: bool = False) -> list[Path]:
    """Resolve user-supplied paths into the list of media files to transcribe.

    Each argument may be a file or a directory:

    - An existing file must be a supported format (:func:`is_supported`) or
      :class:`~transcriber.errors.UnsupportedFormatError` is raised listing
      the supported extensions. Explicitly named files are otherwise taken
      as-is — hidden (dot-prefixed) files included.
    - A directory is scanned — its immediate children by default, its whole
      tree with ``recursive=True`` — keeping only supported files. Anything
      else (unsupported files, subdirectories when not recursing) is silently
      skipped, as are hidden entries: files, and with ``recursive=True``
      whole directories, whose name starts with a dot below the scanned root.
    - A path that does not exist raises
      :class:`~transcriber.errors.AudioNotFoundError` naming it.

    Args:
        paths: File and/or directory paths as given by the user.
        recursive: Scan directories recursively instead of only their
            immediate children.

    Returns:
        Resolved absolute paths, deduplicated (so overlapping arguments such
        as a directory and a file inside it yield one entry) and sorted for
        deterministic, platform-independent processing order.

    Raises:
        AudioNotFoundError: If an argument does not exist, is neither a
            regular file nor a directory, or no supported media was found
            overall (including when *paths* is empty).
        UnsupportedFormatError: If an explicitly named file has an
            unsupported extension.
    """
    collected: set[Path] = set()
    for path in paths:
        if path.is_dir():
            collected.update(
                entry.resolve() for entry in _scan_directory(path, recursive=recursive)
            )
        elif path.is_file():
            if not is_supported(path):
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise UnsupportedFormatError(
                    f"cannot transcribe {path}: unsupported file extension "
                    f"{path.suffix!r} (supported: {supported})"
                )
            collected.add(path.resolve())
        elif not path.exists():
            raise AudioNotFoundError(f"input path does not exist: {path}")
        else:
            raise AudioNotFoundError(
                f"input path is neither a regular file nor a directory: {path}"
            )
    if not collected:
        searched = ", ".join(str(path) for path in paths)
        where = f" in: {searched}" if searched else " (no input paths were given)"
        raise AudioNotFoundError("no supported media found" + where)
    return sorted(collected)


def _scan_directory(directory: Path, *, recursive: bool) -> Iterator[Path]:
    """Yield the supported, non-hidden files under *directory*, unordered.

    Hidden means any dot-prefixed path component below *directory* itself, so
    a recursive scan also skips files inside hidden subdirectories. The
    directory's own name is never inspected — an explicitly named hidden
    directory is still scanned.
    """
    pattern = "**/*" if recursive else "*"
    for entry in directory.glob(pattern):
        parts = entry.relative_to(directory).parts
        if any(part.startswith(".") for part in parts):
            continue
        if entry.is_file() and is_supported(entry):
            yield entry
