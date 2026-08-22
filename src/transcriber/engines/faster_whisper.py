"""Production engine backed by faster-whisper (SYSTRAN, CTranslate2).

The ``faster_whisper`` package is heavyweight, so this module never imports it
at module level: the import happens lazily in :func:`_import_faster_whisper`
when a :class:`FasterWhisperEngine` is constructed, and a missing package
surfaces as :class:`~transcriber.errors.EngineNotAvailableError` with the
``pip install`` command that fixes it. Importing (and type-checking) this
module therefore works on machines without faster-whisper installed.

Option mapping (verified against faster-whisper 1.2.1): ``WhisperModel``
passes ``device`` and ``compute_type`` straight through to
``ctranslate2.models.Whisper``, which accepts the literals ``"auto"`` and
``"default"`` for both selection knobs — so config values need no aliasing.
Loaded models are cached per ``(model, device, compute_type)`` on the engine
instance and released by :meth:`FasterWhisperEngine.close`.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

from transcriber.config import TranscriberConfig
from transcriber.engines.base import TranscriptionEngine
from transcriber.errors import (
    EngineNotAvailableError,
    ModelLoadError,
    TranscriptionFailedError,
)
from transcriber.model import Segment, Transcript, Word

__all__ = ["FasterWhisperEngine"]


def _import_faster_whisper() -> ModuleType:
    """Import and return the ``faster_whisper`` module, lazily.

    Raises:
        EngineNotAvailableError: If the package is not installed; the message
            includes the exact ``pip install faster-whisper`` remedy.
    """
    try:
        import faster_whisper
    except ImportError as exc:
        raise EngineNotAvailableError("faster-whisper") from exc
    module: ModuleType = faster_whisper  # the package is untyped: pin the module type here
    return module


class FasterWhisperEngine(TranscriptionEngine):
    """Local Whisper transcription via faster-whisper / CTranslate2.

    Construction imports ``faster_whisper`` (raising
    :class:`~transcriber.errors.EngineNotAvailableError` when absent) but
    loads no model; models are loaded on first use per distinct
    ``(model, device, compute_type)`` and cached on the instance, so batches
    reuse one loaded model and a config change mid-run just loads another.
    """

    name: ClassVar[str] = "faster-whisper"

    def __init__(self) -> None:
        self._module: ModuleType = _import_faster_whisper()
        self._models: dict[tuple[str, str, str], Any] = {}

    def transcribe(self, path: Path, config: TranscriberConfig) -> Transcript:
        """Transcribe *path* with a (cached) ``WhisperModel`` under *config*.

        The segment generator faster-whisper returns is consumed fully here,
        so all transcription work happens inside this call and errors surface
        as :class:`~transcriber.errors.TranscriptionFailedError`.
        """
        model = self._load_model(config)
        try:
            segment_iter, info = model.transcribe(
                str(path),
                language=config.language,
                task=config.task,
                beam_size=config.beam_size,
                temperature=config.temperature,
                initial_prompt=config.initial_prompt,
                vad_filter=config.vad,
                word_timestamps=config.word_timestamps,
            )
            # Transcription happens while the generator is drained; drain it
            # here so failures are attributed to this file, not a later read.
            raw_segments = list(segment_iter)
        except Exception as exc:
            raise TranscriptionFailedError(f"transcription of {path} failed: {exc}") from exc
        return Transcript(
            segments=tuple(_convert_segment(raw) for raw in raw_segments),
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            source=str(path),
            engine=self.name,
            model=config.model,
        )

    def close(self) -> None:
        """Drop every cached model, releasing its memory to the allocator."""
        self._models.clear()

    def _load_model(self, config: TranscriberConfig) -> Any:
        """Return the cached ``WhisperModel`` for *config*, loading on first use.

        Raises:
            ModelLoadError: If ``WhisperModel`` construction fails (bad model
                name, missing download, unsupported device, ...).
        """
        key = (config.model, config.device, config.compute_type)
        model = self._models.get(key)
        if model is None:
            try:
                model = self._module.WhisperModel(
                    config.model,
                    device=config.device,
                    compute_type=config.compute_type,
                )
            except Exception as exc:
                raise ModelLoadError(
                    f"could not load model {config.model!r} "
                    f"(device={config.device!r}, compute_type={config.compute_type!r}): {exc}"
                ) from exc
            self._models[key] = model
        return model


def _convert_segment(raw: Any) -> Segment:
    """Convert a ``faster_whisper.transcribe.Segment`` to a model :class:`Segment`.

    Texts are stripped; a ``words`` of ``None`` (word timestamps off) becomes
    an empty tuple.
    """
    words = () if raw.words is None else tuple(_convert_word(word) for word in raw.words)
    return Segment(
        id=raw.id,
        start=raw.start,
        end=raw.end,
        text=raw.text.strip(),
        words=words,
        avg_logprob=raw.avg_logprob,
        no_speech_prob=raw.no_speech_prob,
    )


def _convert_word(raw: Any) -> Word:
    """Convert a ``faster_whisper.transcribe.Word`` (field ``word``) to a :class:`Word`."""
    return Word(
        start=raw.start,
        end=raw.end,
        text=raw.word.strip(),
        probability=raw.probability,
    )
