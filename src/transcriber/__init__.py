"""Local-first, privacy-preserving transcription of audio and video.

Transcriber turns media files into text — plain text, SRT/VTT subtitles,
JSON, TSV, or Markdown — using local speech-recognition engines with
language auto-detection, word timestamps, and voice-activity filtering.
No audio ever leaves the machine.

The essentials are re-exported here::

    from transcriber import Transcriber, TranscriberConfig

    config = TranscriberConfig(model="small", language="en")
    transcript = Transcriber(config).transcribe_file(Path("talk.mp3"))
    print(transcript.text)

- :class:`Transcriber` — the facade: single files or whole batches, with
  format rendering, atomic writes, and progress events
  (see :mod:`transcriber.pipeline` for batch results and events);
- :class:`TranscriberConfig` — immutable, validated settings
  (see :mod:`transcriber.config` for environment-variable support);
- :class:`Transcript` / :class:`Segment` / :class:`Word` — the engine-agnostic
  data model;
- :class:`TranscriptionEngine` — the engine interface, with :class:`MockEngine`
  as its deterministic offline implementation;
- the error taxonomy rooted at :class:`TranscriberError`.

Importing this package stays lightweight: heavyweight speech-recognition
backends (``faster_whisper``) are imported only when a real engine is
constructed, never at import time.
"""

from transcriber.config import TranscriberConfig
from transcriber.engines import MockEngine, TranscriptionEngine
from transcriber.errors import (
    AudioNotFoundError,
    ConfigError,
    EngineError,
    EngineNotAvailableError,
    InputError,
    ModelLoadError,
    OutputError,
    TranscriberError,
    TranscriptionFailedError,
    UnsupportedFormatError,
)
from transcriber.model import Segment, Transcript, Word
from transcriber.pipeline import Transcriber

__version__ = "1.0.0"

__all__ = [
    "AudioNotFoundError",
    "ConfigError",
    "EngineError",
    "EngineNotAvailableError",
    "InputError",
    "MockEngine",
    "ModelLoadError",
    "OutputError",
    "Segment",
    "Transcriber",
    "TranscriberConfig",
    "TranscriberError",
    "Transcript",
    "TranscriptionEngine",
    "TranscriptionFailedError",
    "UnsupportedFormatError",
    "Word",
    "__version__",
]
