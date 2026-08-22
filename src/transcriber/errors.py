"""Exception taxonomy for transcriber.

Every error the package raises on purpose derives from :class:`TranscriberError`,
so applications can catch a single type at their outermost boundary::

    try:
        ...
    except TranscriberError as exc:
        print(f"error: {exc}", file=sys.stderr)

The taxonomy mirrors the stages of a transcription run:

- :class:`ConfigError` — an invalid configuration value, option, or
  environment variable.
- :class:`InputError` — a problem with the media files to transcribe
  (:class:`AudioNotFoundError`, :class:`UnsupportedFormatError`).
- :class:`EngineError` — a failure in the speech-recognition engine
  (:class:`EngineNotAvailableError`, :class:`ModelLoadError`,
  :class:`TranscriptionFailedError`).
- :class:`OutputError` — a transcript could not be rendered or written.

Raise sites must supply human-actionable messages: say what went wrong, name
the offending value or path, and where possible say what to do about it.
"""

from __future__ import annotations

__all__ = [
    "AudioNotFoundError",
    "ConfigError",
    "EngineError",
    "EngineNotAvailableError",
    "InputError",
    "ModelLoadError",
    "OutputError",
    "TranscriberError",
    "TranscriptionFailedError",
    "UnsupportedFormatError",
]


class TranscriberError(Exception):
    """Base class for every error transcriber raises deliberately.

    The associated message is always written for humans; catching this type at
    an application boundary and printing ``str(exc)`` yields a sensible error
    report without a traceback.
    """


class ConfigError(TranscriberError):
    """An invalid configuration value, option, or environment variable.

    Args:
        message: Human-actionable description naming the offending value.
        field: Name of the :class:`~transcriber.config.TranscriberConfig`
            field the bad value was destined for, when known. Lets callers
            (for example :func:`transcriber.config.from_env`) map a validation
            failure back to the environment variable or CLI flag it came from.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field: str | None = field


class InputError(TranscriberError):
    """A problem with the media files given to transcribe."""


class AudioNotFoundError(InputError):
    """An input path does not exist or is not a readable file.

    Raise with a message naming the missing path.
    """


class UnsupportedFormatError(InputError):
    """An input file is not in a format transcriber can decode.

    Raise with a message naming the file and listing the supported formats.
    """


class EngineError(TranscriberError):
    """A failure inside a speech-recognition engine."""


class EngineNotAvailableError(EngineError):
    """A requested engine cannot be used because its backing package is missing.

    The default message includes the exact ``pip install`` command that fixes
    the problem, so this error can be raised bare from an ``ImportError``
    handler and still be actionable.

    Args:
        engine: Engine name as selected in configuration.
        package: Pip distribution that provides the engine. Defaults to the
            engine name, which is correct for ``faster-whisper``.
        message: Replaces the default message entirely, when given.
    """

    def __init__(
        self,
        engine: str = "faster-whisper",
        *,
        package: str | None = None,
        message: str | None = None,
    ) -> None:
        self.engine: str = engine
        self.package: str = engine if package is None else package
        if message is None:
            message = (
                f"transcription engine {self.engine!r} is not available because the "
                f"{self.package!r} package is not installed; "
                f"install it with: pip install {self.package}"
            )
        super().__init__(message)


class ModelLoadError(EngineError):
    """A speech-recognition model could not be loaded.

    Raise with a message naming the model and the underlying cause (missing
    download, corrupt cache, unsupported device, out of memory, ...).
    """


class TranscriptionFailedError(EngineError):
    """The engine failed while transcribing a file.

    Raise with a message naming the input file and the underlying cause.
    """


class OutputError(TranscriberError):
    """A transcript could not be rendered or written.

    Raise with a message naming the destination path and the cause.
    """
