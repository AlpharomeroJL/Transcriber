"""Speech-recognition engines and their registry.

Engines implement :class:`~transcriber.engines.base.TranscriptionEngine` and
are looked up by the name used in :attr:`TranscriberConfig.engine` and on the
CLI's ``--engine`` option:

- ``faster-whisper`` — the production engine
  (:class:`~transcriber.engines.faster_whisper.FasterWhisperEngine`);
- ``mock`` — deterministic offline engine for tests and dry runs
  (:class:`~transcriber.engines.mock.MockEngine`).

Lookup is cheap and side-effect free: :func:`get_engine` returns the class
without constructing anything, so heavyweight backend imports happen only
when :func:`create_engine` actually instantiates an engine.
"""

from __future__ import annotations

from transcriber.engines.base import TranscriptionEngine
from transcriber.engines.faster_whisper import FasterWhisperEngine
from transcriber.engines.mock import MockEngine
from transcriber.errors import EngineError

__all__ = [
    "FasterWhisperEngine",
    "MockEngine",
    "TranscriptionEngine",
    "available_engines",
    "create_engine",
    "get_engine",
]

_REGISTRY: dict[str, type[TranscriptionEngine]] = {
    FasterWhisperEngine.name: FasterWhisperEngine,
    MockEngine.name: MockEngine,
}


def available_engines() -> tuple[str, ...]:
    """Return the names of all registered engines, sorted alphabetically."""
    return tuple(sorted(_REGISTRY))


def get_engine(name: str) -> type[TranscriptionEngine]:
    """Return the engine class registered under *name*, without constructing it.

    Raises:
        EngineError: If *name* is not a registered engine; the message names
            the available ones.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(available_engines())
        raise EngineError(f"unknown engine {name!r}; available engines: {available}") from None


def create_engine(name: str) -> TranscriptionEngine:
    """Construct and return the engine registered under *name*.

    Construction is where lazy backend imports happen, so this may raise
    :class:`~transcriber.errors.EngineNotAvailableError` in addition to the
    :class:`~transcriber.errors.EngineError` of an unknown name.
    """
    return get_engine(name)()
