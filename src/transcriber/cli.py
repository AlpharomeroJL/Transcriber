"""Command-line interface: the ``transcriber`` command.

The CLI is a thin, disciplined shell around :class:`transcriber.pipeline.Transcriber`:

- **Configuration precedence** — library defaults, overlaid with
  ``TRANSCRIBER_*`` environment variables (:func:`transcriber.config.from_env`),
  overlaid with command-line flags (:meth:`TranscriberConfig.replace`).
- **Exit codes** — 0: every file transcribed; 1: at least one per-file
  failure (batch isolation); 2: usage or configuration error. Request-level
  :class:`~transcriber.errors.ConfigError` / :class:`~transcriber.errors.InputError`
  / :class:`~transcriber.errors.OutputError` become UsageErrors (exit 2).
- **Stream discipline** — stdout carries transcript content only, and only
  under ``--stdout``; progress, hints, summaries, and errors go to stderr, so
  pipes stay clean. Informational subcommands (``formats``, ``models``,
  ``info``) print their listings to stdout, which *is* their content.
- **Progress** — a transient rich progress bar driven by pipeline events,
  automatically disabled when quiet, when stderr is not a terminal, or under
  ``--stdout``; errors are printed to stderr regardless.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import click
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from transcriber import __version__
from transcriber.config import (
    COMPUTE_TYPES,
    TASKS,
    TranscriberConfig,
    from_env,
    is_known_model,
)
from transcriber.engines import available_engines
from transcriber.errors import ConfigError, InputError, OutputError, TranscriberError
from transcriber.formats import available_formats, get_formatter
from transcriber.pipeline import BatchResult, PipelineEvent, Transcriber

__all__ = ["main"]

#: Device selectors offered on the command line (the library accepts any string).
_DEVICES: Final[tuple[str, ...]] = ("auto", "cpu", "cuda")

#: Library defaults, used to document effective defaults in ``--help`` output.
_DEFAULTS: Final[TranscriberConfig] = TranscriberConfig()

#: One-line description per registered output format, keyed by format name.
_FORMAT_DESCRIPTIONS: Final[dict[str, str]] = {
    "json": "Full transcript data (segments, words, metadata) as versioned, pretty-printed JSON.",
    "md": "Markdown reading view: title, metadata bullets, timestamped paragraphs.",
    "srt": "SubRip subtitles: numbered cues with HH:MM:SS,mmm timings.",
    "tsv": "Tab-separated start/end/text rows (openai-whisper convention, integer milliseconds).",
    "txt": "Plain text, one segment per line.",
    "vtt": "WebVTT subtitles: web-native cues with HH:MM:SS.mmm timings.",
}

#: ``(name, parameters, notes)`` per standard model alias, in KNOWN_MODELS order.
_MODEL_TABLE: Final[tuple[tuple[str, str, str], ...]] = (
    ("tiny", "39M", "Fastest, least accurate; 99 languages."),
    ("tiny.en", "39M", "English-only tiny."),
    ("base", "74M", "Very fast; 99 languages."),
    ("base.en", "74M", "English-only base."),
    ("small", "244M", "The default: best quality/speed balance on CPU."),
    ("small.en", "244M", "English-only small."),
    ("medium", "769M", "High accuracy, slower; 99 languages."),
    ("medium.en", "769M", "English-only medium."),
    ("large-v1", "1.5B", "Original large model."),
    ("large-v2", "1.5B", "Improved large model."),
    ("large-v3", "1.5B", "Most accurate; recommended on GPU."),
    ("large-v3-turbo", "809M", "large-v3 with a distilled decoder, about 4x faster."),
    ("turbo", "809M", "Alias of large-v3-turbo."),
    ("distil-small.en", "166M", "Distilled, English-only."),
    ("distil-medium.en", "394M", "Distilled, English-only."),
    ("distil-large-v2", "756M", "Distilled, English-only."),
    ("distil-large-v3", "756M", "Distilled, English-only."),
)


def _console(*, stderr: bool = False) -> Console:
    """A rich console with predictable plain-text behaviour.

    Markup and syntax highlighting are off so user-supplied strings (paths,
    error messages) are never interpreted; soft wrapping keeps log lines
    unwrapped so they stay greppable when redirected.
    """
    return Console(stderr=stderr, soft_wrap=True, highlight=False, markup=False)


def _print_error(console: Console, message: str) -> None:
    """Print one ``error:`` line to *console*; styling degrades to plain text."""
    console.print(Text(f"error: {message}", style="red"))


class _ProgressReporter:
    """Routes :class:`PipelineEvent` callbacks to the progress bar and stderr.

    The single progress task is created lazily on the first event (which
    carries the batch total). Failures are always printed to stderr; verbose
    mode additionally prints one completion line per file.
    """

    def __init__(self, progress: Progress, console: Console, *, verbose: bool) -> None:
        self._progress = progress
        self._console = console
        self._verbose = verbose
        self._task_id: TaskID | None = None

    def __call__(self, event: PipelineEvent) -> None:
        if self._task_id is None:
            self._task_id = self._progress.add_task("transcribing", total=event.total)
        if event.kind == "file_started":
            self._progress.update(self._task_id, description=f"transcribing {event.path.name}")
            return
        self._progress.advance(self._task_id)
        if event.kind == "file_failed":
            _print_error(self._console, f"{event.path}: {event.message}")
        elif self._verbose and event.result is not None:
            outputs = ", ".join(output.name for output in event.result.outputs) or "-"
            self._console.print(
                f"done: {event.path.name} ({event.result.elapsed:.2f}s) -> {outputs}"
            )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "--version", prog_name="transcriber")
def main() -> None:
    """Local-first, privacy-preserving transcription of audio and video.

    Media in, text out (txt / srt / vtt / json / tsv / md) — entirely on this
    machine. No audio ever leaves it.
    """


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option(
    "-f",
    "--format",
    "formats",
    multiple=True,
    default=("txt",),
    show_default=True,
    type=click.Choice(available_formats()),
    help="Output format; repeat the option to write several.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write all outputs into this directory (created if missing) instead of next to inputs.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="Whisper model alias, Hugging Face CTranslate2 repo id, or local model path "
    f"[default: {_DEFAULTS.model}].",
)
@click.option(
    "--engine",
    default=None,
    type=click.Choice(available_engines()),
    help=f"Transcription engine [default: {_DEFAULTS.engine}].",
)
@click.option(
    "-l",
    "--language",
    default=None,
    help="ISO 639 language code of the audio (e.g. en); auto-detected when omitted.",
)
@click.option(
    "--task",
    default=None,
    type=click.Choice(TASKS),
    help=f"Transcribe in the source language or translate to English [default: {_DEFAULTS.task}].",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(_DEVICES),
    help=f"Inference device [default: {_DEFAULTS.device}].",
)
@click.option(
    "--compute-type",
    default=None,
    type=click.Choice(COMPUTE_TYPES),
    help=f"Quantization used for inference [default: {_DEFAULTS.compute_type}].",
)
@click.option(
    "--beam-size",
    default=None,
    type=int,
    help=f"Beam width for decoding, >= 1 [default: {_DEFAULTS.beam_size}].",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help=f"Sampling temperature, 0.0-1.0 [default: {_DEFAULTS.temperature}].",
)
@click.option(
    "--initial-prompt",
    default=None,
    help="Text to prime the decoder with (names, jargon, spellings).",
)
@click.option(
    "--word-timestamps/--no-word-timestamps",
    "word_timestamps",
    default=None,
    help="Also produce per-word timestamps [default: off].",
)
@click.option(
    "--vad/--no-vad",
    "vad",
    default=None,
    help="Filter long silences with voice-activity detection [default: on].",
)
@click.option("-r", "--recursive", is_flag=True, help="Scan input directories recursively.")
@click.option("--overwrite", is_flag=True, help="Replace existing output files instead of failing.")
@click.option(
    "--stdout",
    "use_stdout",
    is_flag=True,
    help="Print the rendered transcript to stdout instead of writing files "
    "(requires exactly one input and one format).",
)
@click.option(
    "-q", "--quiet", is_flag=True, help="Suppress progress and summary; errors still print."
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print effective settings and per-file detail to stderr."
)
@click.pass_context
def transcribe(
    ctx: click.Context,
    inputs: tuple[Path, ...],
    formats: tuple[str, ...],
    output_dir: Path | None,
    model: str | None,
    engine: str | None,
    language: str | None,
    task: str | None,
    device: str | None,
    compute_type: str | None,
    beam_size: int | None,
    temperature: float | None,
    initial_prompt: str | None,
    word_timestamps: bool | None,
    vad: bool | None,
    recursive: bool,
    overwrite: bool,
    use_stdout: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Transcribe media files (or directories of them) to text.

    INPUTS are media files and/or directories to process. Outputs are written
    next to each input, or into ``-o/--output-dir``. Exit code 0 means every
    file was transcribed, 1 means at least one file failed (the rest were
    still processed), 2 means the request itself was invalid.
    """
    verbose = verbose and not quiet
    err = _console(stderr=True)
    if use_stdout and (len(inputs) != 1 or len(formats) != 1):
        raise click.UsageError(
            "--stdout requires exactly one input file and exactly one --format "
            f"(got {len(inputs)} input(s) and {len(formats)} format(s))"
        )
    if use_stdout and output_dir is not None:
        raise click.UsageError("--stdout writes no files; do not combine it with -o/--output-dir")
    try:
        config = from_env().replace(
            model=model,
            engine=engine,
            device=device,
            compute_type=compute_type,
            language=language,
            task=task,
            vad=vad,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            temperature=temperature,
            initial_prompt=initial_prompt,
        )
    except ConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    if verbose:
        err.print(_settings_line(config, formats))
    if not quiet and not is_known_model(config.model):
        err.print(
            f"note: {config.model!r} is not a standard Whisper model alias; assuming a "
            "Hugging Face CTranslate2 repo id or a local model path "
            "(run 'transcriber models' to list the standard aliases)"
        )
    try:
        app = Transcriber(config)
    except TranscriberError as exc:
        raise click.UsageError(str(exc)) from exc
    if use_stdout:
        ctx.exit(_run_stdout(app, inputs[0], formats[0], err))
    ctx.exit(
        _run_batch(
            app,
            inputs,
            formats,
            output_dir=output_dir,
            overwrite=overwrite,
            recursive=recursive,
            err=err,
            quiet=quiet,
            verbose=verbose,
        )
    )


def _settings_line(config: TranscriberConfig, formats: Sequence[str]) -> str:
    """The one-line effective-settings dump printed by ``--verbose``."""
    return (
        "settings: "
        f"engine={config.engine} model={config.model} device={config.device} "
        f"compute_type={config.compute_type} language={config.language or 'auto'} "
        f"task={config.task} vad={config.vad} word_timestamps={config.word_timestamps} "
        f"beam_size={config.beam_size} temperature={config.temperature} "
        f"formats={','.join(formats)}"
    )


def _run_stdout(app: Transcriber, source: Path, format_name: str, err: Console) -> int:
    """Transcribe one file and print its rendered transcript to stdout.

    Returns the process exit code: 0 on success, 1 when the file itself fails
    to transcribe. Request-level errors raise :class:`click.UsageError`
    (exit 2), mirroring batch mode's fail-fast semantics.
    """
    formatter = get_formatter(format_name)
    try:
        transcript = app.transcribe_file(source)
    except (ConfigError, InputError, OutputError) as exc:
        raise click.UsageError(str(exc)) from exc
    except TranscriberError as exc:
        _print_error(err, f"{source}: {exc}")
        return 1
    click.echo(formatter.render(transcript), nl=False)
    return 0


def _run_batch(
    app: Transcriber,
    inputs: Sequence[Path],
    formats: Sequence[str],
    *,
    output_dir: Path | None,
    overwrite: bool,
    recursive: bool,
    err: Console,
    quiet: bool,
    verbose: bool,
) -> int:
    """Run a batch with progress reporting and return its exit code."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
        transient=True,
        disable=quiet or not err.is_terminal,
    )
    reporter = _ProgressReporter(progress, err, verbose=verbose)
    try:
        with progress:
            batch = app.transcribe_batch(
                inputs,
                formats=formats,
                output_dir=output_dir,
                overwrite=overwrite,
                recursive=recursive,
                on_event=reporter,
            )
    except (ConfigError, InputError, OutputError) as exc:
        raise click.UsageError(str(exc)) from exc
    if not quiet:
        _print_summary(err, batch)
    return batch.exit_code


def _print_summary(console: Console, batch: BatchResult) -> None:
    """Print the after-batch summary table and counts line to *console*."""
    table = Table(box=box.SIMPLE, header_style="bold")
    table.add_column("File", overflow="fold")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Outputs", overflow="fold")
    for result in batch.results:
        status = Text("ok", style="green") if result.ok else Text("failed", style="red")
        duration = "-" if result.transcript is None else f"{result.transcript.duration:.1f}s"
        outputs = ", ".join(output.name for output in result.outputs) or "-"
        table.add_row(result.path.name, status, duration, outputs)
    console.print(table)
    console.print(f"{batch.ok_count} ok, {batch.error_count} failed")


@main.command()
def formats() -> None:
    """List the available output formats."""
    out = _console()
    table = Table(box=box.SIMPLE, header_style="bold")
    table.add_column("Format")
    table.add_column("Extension")
    table.add_column("Description")
    for name in available_formats():
        table.add_row(name, get_formatter(name).extension, _FORMAT_DESCRIPTIONS.get(name, ""))
    out.print(table)


@main.command()
def models() -> None:
    """List the standard Whisper model aliases."""
    out = _console()
    table = Table(box=box.SIMPLE, header_style="bold")
    table.add_column("Model")
    table.add_column("Parameters", justify="right")
    table.add_column("Notes")
    for name, parameters, notes in _MODEL_TABLE:
        table.add_row(name, parameters, notes)
    out.print(table)
    out.print(
        "Any Hugging Face CTranslate2 model repo id or local model directory path is also accepted."
    )


@main.command()
def info() -> None:
    """Show version, environment, and effective configuration."""
    out = _console()
    out.print(f"transcriber {__version__}")
    out.print(f"python {platform.python_version()} ({sys.executable})")
    out.print(f"platform {platform.platform()}")
    out.print(f"engines: {', '.join(available_engines())}")
    out.print(f"formats: {', '.join(available_formats())}")
    out.print(f"faster-whisper: {_faster_whisper_status()}")
    try:
        config = from_env()
    except ConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    out.print("config (defaults overlaid with TRANSCRIBER_* environment variables):")
    for field in dataclasses.fields(TranscriberConfig):
        out.print(f"  {field.name} = {getattr(config, field.name)!r}")


def _faster_whisper_status() -> str:
    """Availability of the production engine's backing package, without importing it."""
    if importlib.util.find_spec("faster_whisper") is None:
        return "not installed (pip install faster-whisper)"
    try:
        return f"installed (version {importlib.metadata.version('faster-whisper')})"
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover — importable but unpackaged
        return "installed (version unknown)"
