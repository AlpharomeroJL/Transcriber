"""The library facade: inputs -> engine -> formatters -> output files.

:class:`Transcriber` ties the other modules together. It discovers media via
:func:`transcriber.audio.collect_inputs`, transcribes each file with one
:class:`~transcriber.engines.base.TranscriptionEngine`, renders the requested
:mod:`transcriber.formats`, and writes the results next to the inputs or into
a chosen output directory.

Batch semantics are strict and predictable:

- Bad *requests* fail fast: unknown format names, invalid explicit input
  paths, or an uncreatable output directory raise before anything is
  transcribed.
- Bad *files* are isolated: an engine, input, or write failure for one file
  (:class:`~transcriber.errors.TranscriberError` or :class:`OSError`) is
  recorded in its :class:`FileResult` and the batch continues. Anything else
  is a bug and propagates.
- Writes are atomic: each output is rendered fully, written once to a hidden
  temporary file in the destination directory, and renamed into place, so a
  failure never leaves a partial or truncated output behind.
- Nothing is clobbered or skipped silently: an existing output file fails the
  input (unless ``overwrite=True``), and when two inputs would write the same
  output path the later one fails with an explanatory error.

Progress is observable through :class:`PipelineEvent` callbacks, and
:attr:`BatchResult.exit_code` maps directly onto the CLI contract
(0 = every file transcribed, 1 = at least one per-file failure).
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from transcriber.audio import collect_inputs
from transcriber.config import TranscriberConfig
from transcriber.engines import TranscriptionEngine, create_engine
from transcriber.errors import AudioNotFoundError, OutputError, TranscriberError
from transcriber.formats import Formatter, get_formatter
from transcriber.model import Transcript

__all__ = [
    "BatchResult",
    "FileResult",
    "PipelineEvent",
    "PipelineEventKind",
    "Transcriber",
]

PipelineEventKind = Literal["file_started", "file_finished", "file_failed"]
"""The kinds of :class:`PipelineEvent` a batch emits, in lifecycle order."""


@dataclass(frozen=True, slots=True)
class FileResult:
    """The outcome of processing one input file in a batch.

    Attributes:
        path: The input file, resolved and absolute (as discovered by
            :func:`transcriber.audio.collect_inputs`).
        ok: Whether the file was transcribed and every output was written.
        transcript: The transcript, when transcription succeeded — also set
            on a failed result whose failure happened *after* transcription
            (for example while writing an output); ``None`` otherwise.
        outputs: Output files actually written for this input, in the order
            of the requested formats. On failure this holds the outputs
            completed before the error.
        error: Human-readable failure description, or ``None`` when ``ok``.
        elapsed: Seconds spent transcribing and writing this file, measured
            with :func:`time.perf_counter`.
    """

    path: Path
    ok: bool
    transcript: Transcript | None
    outputs: list[Path]
    error: str | None
    elapsed: float


@dataclass(frozen=True, slots=True)
class BatchResult:
    """The outcome of a whole :meth:`Transcriber.transcribe_batch` run.

    Attributes:
        results: One :class:`FileResult` per discovered input, in the
            deterministic (sorted) processing order.
    """

    results: list[FileResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        """Number of inputs that were transcribed and written successfully."""
        return sum(1 for result in self.results if result.ok)

    @property
    def error_count(self) -> int:
        """Number of inputs that failed."""
        return sum(1 for result in self.results if not result.ok)

    @property
    def all_ok(self) -> bool:
        """Whether every input succeeded (vacuously true with no results)."""
        return self.error_count == 0

    @property
    def exit_code(self) -> int:
        """Process exit code for this batch: 0 all ok, 1 any per-file failure."""
        return 0 if self.all_ok else 1


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """A progress notification emitted while a batch runs.

    Every input emits exactly two events: ``file_started`` when processing
    begins, then either ``file_finished`` or ``file_failed``.

    Attributes:
        kind: What happened; see :data:`PipelineEventKind`.
        path: The input file the event is about.
        index: 1-based position of this file in the batch.
        total: Total number of files in the batch.
        message: The failure description for ``file_failed`` (identical to
            ``result.error``), ``None`` otherwise.
        result: The finished :class:`FileResult` for terminal events — the
            same object that ends up in :attr:`BatchResult.results` — and
            ``None`` for ``file_started``.
    """

    kind: PipelineEventKind
    path: Path
    index: int
    total: int
    message: str | None = None
    result: FileResult | None = None


class Transcriber:
    """High-level facade for transcribing media files.

    Args:
        config: Settings for every transcription this instance performs;
            defaults to ``TranscriberConfig()``.
        engine: Explicit engine instance to use. When given it wins outright;
            otherwise the engine named by ``config.engine`` is constructed
            via :func:`transcriber.engines.create_engine`.

    Raises:
        transcriber.errors.EngineError: If ``config.engine`` names no
            registered engine (only when *engine* is not injected).
        transcriber.errors.EngineNotAvailableError: If the configured
            engine's backing package is not installed.
    """

    def __init__(
        self,
        config: TranscriberConfig | None = None,
        *,
        engine: TranscriptionEngine | None = None,
    ) -> None:
        self._config = TranscriberConfig() if config is None else config
        self._engine = create_engine(self._config.engine) if engine is None else engine

    @property
    def config(self) -> TranscriberConfig:
        """The immutable configuration this instance transcribes under."""
        return self._config

    @property
    def engine(self) -> TranscriptionEngine:
        """The engine instance performing the transcriptions."""
        return self._engine

    def close(self) -> None:
        """Release the engine's cached resources (loaded models, handles).

        The instance stays usable; a later transcription re-acquires whatever
        was released.
        """
        self._engine.close()

    def transcribe_file(self, path: Path) -> Transcript:
        """Transcribe a single media file, raising on any failure.

        Args:
            path: An existing media file in a supported format.

        Returns:
            The complete transcript.

        Raises:
            transcriber.errors.AudioNotFoundError: If *path* does not exist
                or is a directory (use :meth:`transcribe_batch` for those).
            transcriber.errors.UnsupportedFormatError: If *path* is not a
                supported media format.
            transcriber.errors.EngineError: If the engine fails.
        """
        if path.is_dir():
            raise AudioNotFoundError(
                f"expected a single media file but {path} is a directory; "
                "use transcribe_batch to process directories"
            )
        resolved = collect_inputs([path])[0]
        return self._engine.transcribe(resolved, self._config)

    def transcribe_batch(
        self,
        inputs: Sequence[Path],
        *,
        formats: Sequence[str] = ("txt",),
        output_dir: Path | None = None,
        overwrite: bool = False,
        recursive: bool = False,
        on_event: Callable[[PipelineEvent], None] | None = None,
    ) -> BatchResult:
        """Transcribe many inputs and write their outputs, isolating failures.

        Inputs are resolved with :func:`transcriber.audio.collect_inputs`
        (files and directories, deduplicated, deterministically sorted) and
        the requested format names are validated up front, so a bad request
        fails before anything is transcribed. Each file is then transcribed
        and rendered to ``<output_dir or input's directory>/<stem><ext>`` for
        every requested format, atomically and in UTF-8.

        A failure confined to one file — any
        :class:`~transcriber.errors.TranscriberError` or :class:`OSError`
        from transcribing or writing it — is recorded in that file's
        :class:`FileResult` and the batch continues; unexpected exceptions
        propagate. An input whose output file already exists fails unless
        *overwrite* is true (nothing is skipped silently or clobbered), and
        when two inputs share an output stem targeting the same directory,
        every one after the first fails with an explanatory error.

        Args:
            inputs: Media files and/or directories to process.
            formats: Output format names (see
                :func:`transcriber.formats.available_formats`); duplicates
                are ignored, order is preserved.
            output_dir: Directory for all outputs, created (with parents) if
                missing. Defaults to writing next to each input.
            overwrite: Replace existing output files instead of failing the
                input that targets them.
            recursive: Scan input directories recursively.
            on_event: Callback invoked with a :class:`PipelineEvent` as each
                file starts and finishes; exceptions it raises propagate.

        Returns:
            A :class:`BatchResult` with one :class:`FileResult` per input in
            processing order.

        Raises:
            transcriber.errors.InputError: If an explicit input path is
                missing or unsupported, or no supported media is found.
            transcriber.errors.OutputError: If *formats* is empty or names
                an unknown format, or *output_dir* cannot be created.
        """
        files = collect_inputs(inputs, recursive=recursive)
        formatters = _resolve_formatters(formats)
        if output_dir is not None:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OutputError(f"cannot create output directory {output_dir}: {exc}") from exc
        collisions = _find_collisions(files, output_dir)

        total = len(files)
        results: list[FileResult] = []
        for position, path in enumerate(files, start=1):
            _emit(on_event, PipelineEvent("file_started", path, position, total))
            result = self._process_file(
                path,
                formatters,
                output_dir=output_dir,
                overwrite=overwrite,
                collision=collisions.get(path),
            )
            kind: PipelineEventKind = "file_finished" if result.ok else "file_failed"
            _emit(on_event, PipelineEvent(kind, path, position, total, result.error, result))
            results.append(result)
        return BatchResult(results)

    def _process_file(
        self,
        path: Path,
        formatters: Sequence[Formatter],
        *,
        output_dir: Path | None,
        overwrite: bool,
        collision: str | None,
    ) -> FileResult:
        """Transcribe one file and write its outputs, capturing per-file errors."""
        started = time.perf_counter()
        directory = path.parent if output_dir is None else output_dir
        targets = [directory / (path.stem + formatter.extension) for formatter in formatters]
        outputs: list[Path] = []
        transcript: Transcript | None = None
        error: str | None = None
        try:
            if collision is not None:
                raise OutputError(collision)
            if not overwrite:
                for target in targets:
                    if target.exists():
                        raise OutputError(
                            f"output file already exists: {target} "
                            "(pass overwrite=True to replace it)"
                        )
            transcript = self._engine.transcribe(path, self._config)
            for formatter, target in zip(formatters, targets, strict=True):
                _write_text_atomic(target, formatter.render(transcript))
                outputs.append(target)
        except (TranscriberError, OSError) as exc:
            error = str(exc)
        elapsed = time.perf_counter() - started
        return FileResult(
            path=path,
            ok=error is None,
            transcript=transcript,
            outputs=outputs,
            error=error,
            elapsed=elapsed,
        )


def _emit(on_event: Callable[[PipelineEvent], None] | None, event: PipelineEvent) -> None:
    """Invoke *on_event* with *event* when a callback was given."""
    if on_event is not None:
        on_event(event)


def _resolve_formatters(formats: Sequence[str]) -> list[Formatter]:
    """Resolve format names to formatter instances, failing fast on bad input.

    Duplicate names are dropped, first occurrence wins, order is preserved.

    Raises:
        OutputError: If *formats* is empty or contains an unknown name (the
            message lists the available formats).
    """
    names = list(dict.fromkeys(formats))
    if not names:
        raise OutputError("no output formats were requested; pass at least one format name")
    return [get_formatter(name) for name in names]


def _find_collisions(files: Sequence[Path], output_dir: Path | None) -> dict[Path, str]:
    """Map each input whose outputs another input would already claim to an error.

    Two inputs collide when they share a stem and target the same output
    directory (always the case for equal stems when *output_dir* is set).
    The first input in processing order keeps the output names; every later
    one is mapped here to an explanatory error message.
    """
    claimed: dict[tuple[Path, str], Path] = {}
    collisions: dict[Path, str] = {}
    for path in files:
        directory = path.parent if output_dir is None else output_dir
        first = claimed.setdefault((directory, path.stem), path)
        if first is not path:
            collisions[path] = (
                f"output name collision: {path} and {first} would both write "
                f"{directory / path.stem}.* — rename an input or use a different "
                "output directory per input"
            )
    return collisions


def _write_text_atomic(target: Path, text: str) -> None:
    """Write *text* to *target* in UTF-8, atomically.

    The text is written in a single call to a hidden temporary file in the
    target's directory (same filesystem), which is then renamed over the
    target. A failure therefore never leaves a partial or truncated *target*
    behind, and the temporary file is removed on a best-effort basis. Line
    endings are preserved exactly as rendered (no platform translation).

    Raises:
        OutputError: If the write fails; wraps the underlying :class:`OSError`
            and names the destination.
    """
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    except OSError as exc:
        raise OutputError(f"cannot write {target}: {exc}") from exc
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(tmp_path, target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise OutputError(f"cannot write {target}: {exc}") from exc
