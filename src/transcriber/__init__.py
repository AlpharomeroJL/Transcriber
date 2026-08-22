"""Local-first, privacy-preserving transcription of audio and video.

Transcriber turns media files into text — plain text, SRT/VTT subtitles,
JSON, TSV, or Markdown — using local speech-recognition engines with
language auto-detection, word timestamps, and voice-activity filtering.
No audio ever leaves the machine.

The top-level package deliberately re-exports very little for now; the
public API grows as the library modules land.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
