"""Abstract interface every speech-recognition engine implements.

An engine turns one media file into a :class:`~transcriber.model.Transcript`
under the settings of a :class:`~transcriber.config.TranscriberConfig`. The
pipeline and CLI talk to engines exclusively through this interface, so new
backends plug in by subclassing :class:`TranscriptionEngine` and registering
in :mod:`transcriber.engines`.

Engines are constructed cheaply: anything expensive — importing a heavyweight
backend, downloading or loading a model — happens lazily on first use, and
whatever an engine caches is released by :meth:`TranscriptionEngine.close`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from transcriber.config import TranscriberConfig
from transcriber.model import Transcript

__all__ = ["TranscriptionEngine"]


class TranscriptionEngine(ABC):
    """Base class for speech-recognition engines.

    Subclasses set :attr:`name` (the identifier used in configuration and on
    the CLI's ``--engine`` option) and implement :meth:`transcribe`.
    """

    #: Engine identifier as used in :attr:`TranscriberConfig.engine`.
    name: ClassVar[str]

    @abstractmethod
    def transcribe(self, path: Path, config: TranscriberConfig) -> Transcript:
        """Transcribe the media file at *path* under *config*.

        Args:
            path: Media file to transcribe. Engines may assume the path was
                already validated by input discovery.
            config: Settings for the run (model, language, decoding options).

        Returns:
            The complete transcript of the file.

        Raises:
            transcriber.errors.EngineNotAvailableError: If the engine's
                backing package is not installed.
            transcriber.errors.ModelLoadError: If the model cannot be loaded.
            transcriber.errors.TranscriptionFailedError: If transcription of
                this file fails.
        """

    def close(self) -> None:  # noqa: B027 — deliberately a non-abstract no-op hook
        """Release any cached resources (loaded models, file handles).

        The default implementation does nothing. Engines stay usable after
        ``close()``; a later :meth:`transcribe` simply re-acquires whatever
        was released.
        """
