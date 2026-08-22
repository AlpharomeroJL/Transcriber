"""Tests for the ``transcriber`` command-line interface.

Everything here is offline and deterministic: transcription always runs the
mock engine (``--engine mock`` or ``TRANSCRIBER_ENGINE=mock``), inputs are
synthetic files under ``tmp_path``, and every test is isolated from ambient
``TRANSCRIBER_*`` variables and terminal quirks by an autouse fixture.

Click 8.4 is installed, where ``CliRunner`` always keeps stdout and stderr
separate (``mix_stderr`` was removed in click 8.2); tests read
``result.stdout`` and ``result.stderr`` directly.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import transcriber
from transcriber.cli import _FORMAT_DESCRIPTIONS, _MODEL_TABLE, main
from transcriber.config import KNOWN_MODELS, TranscriberConfig
from transcriber.engines import MockEngine
from transcriber.formats import available_formats, get_formatter

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stable_cli_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from ambient configuration and terminal detection.

    Ambient ``TRANSCRIBER_*`` variables would leak into ``from_env``;
    ``FORCE_COLOR`` could force rich into terminal mode; a wide ``COLUMNS``
    keeps rich tables from wrapping or truncating cell contents.
    """
    for var in list(os.environ):
        if var.startswith("TRANSCRIBER_"):
            monkeypatch.delenv(var)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def runner() -> CliRunner:
    """A fresh CliRunner (stdout and stderr captured separately)."""
    return CliRunner()


def make_media(directory: Path, name: str, size: int = 2048) -> Path:
    """Create a synthetic media file of *size* bytes and return its resolved path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\0" * size)
    return path.resolve()


def mock_render(path: Path, format_name: str = "txt", *, word_timestamps: bool = False) -> str:
    """Render *path* exactly as the CLI's mock-engine run should."""
    config = TranscriberConfig(engine="mock", word_timestamps=word_timestamps)
    return get_formatter(format_name).render(MockEngine().transcribe(path, config))


# ---------------------------------------------------------------------------
# Group basics: --version, --help
# ---------------------------------------------------------------------------


def test_version_option(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0, result.stderr
    assert transcriber.__version__ in result.stdout
    assert "transcriber" in result.stdout


def test_group_help_lists_commands(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("transcribe", "formats", "models", "info"):
        assert command in result.stdout


def test_transcribe_help_lists_key_options(runner: CliRunner) -> None:
    result = runner.invoke(main, ["transcribe", "--help"])
    assert result.exit_code == 0
    for option in ("--format", "--output-dir", "--model", "--engine", "--stdout", "--no-vad"):
        assert option in result.stdout


def test_transcribe_without_inputs_is_usage_error(runner: CliRunner) -> None:
    result = runner.invoke(main, ["transcribe"])
    assert result.exit_code == 2
    assert "INPUTS" in result.stderr


# ---------------------------------------------------------------------------
# Informational commands: formats, models, info
# ---------------------------------------------------------------------------


def test_formats_lists_every_format_with_extension(runner: CliRunner) -> None:
    result = runner.invoke(main, ["formats"])
    assert result.exit_code == 0, result.stderr
    for name in available_formats():
        assert name in result.stdout
        assert get_formatter(name).extension in result.stdout


def test_format_descriptions_stay_in_sync_with_registry() -> None:
    assert set(_FORMAT_DESCRIPTIONS) == set(available_formats())
    assert all(_FORMAT_DESCRIPTIONS[name] for name in available_formats())


def test_models_lists_every_known_model(runner: CliRunner) -> None:
    result = runner.invoke(main, ["models"])
    assert result.exit_code == 0, result.stderr
    for name in KNOWN_MODELS:
        assert name in result.stdout
    assert "39M" in result.stdout
    assert "1.5B" in result.stdout
    assert "809M" in result.stdout
    assert "also accepted" in result.stdout  # HF repo id / local path note


def test_model_table_stays_in_sync_with_known_models() -> None:
    assert tuple(name for name, _, _ in _MODEL_TABLE) == KNOWN_MODELS


def test_info_reports_version_environment_and_config(runner: CliRunner) -> None:
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0, result.stderr
    assert transcriber.__version__ in result.stdout
    assert platform.python_version() in result.stdout
    assert "faster-whisper:" in result.stdout
    assert "engines: faster-whisper, mock" in result.stdout
    assert "model = 'small'" in result.stdout
    assert "engine = 'faster-whisper'" in result.stdout


def test_info_reflects_environment_overrides(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIBER_MODEL", "tiny")
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "model = 'tiny'" in result.stdout


def test_info_bad_environment_is_usage_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIBER_TEMPERATURE", "woof")
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 2
    assert "TRANSCRIBER_TEMPERATURE" in result.stderr


# ---------------------------------------------------------------------------
# transcribe: successful batches
# ---------------------------------------------------------------------------


def test_transcribe_writes_outputs_for_every_format(runner: CliRunner, tmp_path: Path) -> None:
    alpha = make_media(tmp_path, "alpha.wav")
    beta = make_media(tmp_path, "beta.mp3", size=4096)
    result = runner.invoke(
        main,
        ["transcribe", str(alpha), str(beta), "--engine", "mock", "-f", "txt", "-f", "srt"],
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""  # stdout stays clean without --stdout
    for path in (alpha, beta):
        for format_name in ("txt", "srt"):
            target = tmp_path / (path.stem + "." + format_name)
            assert target.read_text(encoding="utf-8") == mock_render(path, format_name)
    assert "2 ok, 0 failed" in result.stderr


def test_transcribe_summary_table_lists_files_and_outputs(
    runner: CliRunner, tmp_path: Path
) -> None:
    make_media(tmp_path, "alpha.wav")
    result = runner.invoke(main, ["transcribe", str(tmp_path), "--engine", "mock"])
    assert result.exit_code == 0
    for header in ("File", "Status", "Duration", "Outputs"):
        assert header in result.stderr
    assert "alpha.wav" in result.stderr
    assert "alpha.txt" in result.stderr
    assert "2.0s" in result.stderr  # 2048 bytes -> 2.0 synthetic seconds
    assert "1 ok, 0 failed" in result.stderr


def test_transcribe_into_output_dir_creates_it(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path / "in", "clip.wav")
    out_dir = tmp_path / "out" / "nested"
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "-o", str(out_dir)])
    assert result.exit_code == 0, result.stderr
    assert (out_dir / "clip.txt").is_file()
    assert not (tmp_path / "in" / "clip.txt").exists()


def test_transcribe_directory_recursive(runner: CliRunner, tmp_path: Path) -> None:
    nested = make_media(tmp_path / "shows" / "deep", "episode.wav")
    result = runner.invoke(main, ["transcribe", str(tmp_path), "--engine", "mock", "-r"])
    assert result.exit_code == 0, result.stderr
    assert nested.with_suffix(".txt").is_file()


def test_word_timestamps_flag_reaches_engine(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(
        main,
        [
            "transcribe",
            str(media),
            "--engine",
            "mock",
            "-f",
            "json",
            "--stdout",
            "--word-timestamps",
        ],
    )
    assert result.exit_code == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["segments"][0]["words"], "word timestamps requested but none in output"
    plain = runner.invoke(
        main, ["transcribe", str(media), "--engine", "mock", "-f", "json", "--stdout"]
    )
    assert json.loads(plain.stdout)["segments"][0]["words"] == []


# ---------------------------------------------------------------------------
# transcribe: failures and exit codes
# ---------------------------------------------------------------------------


def test_per_file_failure_is_isolated_and_exits_1(runner: CliRunner, tmp_path: Path) -> None:
    good = make_media(tmp_path, "good.wav")
    make_media(tmp_path, "xFAILy.wav")
    result = runner.invoke(main, ["transcribe", str(tmp_path), "--engine", "mock"])
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert "xFAILy" in result.stderr
    assert good.with_suffix(".txt").is_file()  # the good file was still processed
    assert "1 ok, 1 failed" in result.stderr


def test_bad_format_name_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "-f", "nope"])
    assert result.exit_code == 2
    assert "nope" in result.stderr


def test_missing_input_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(main, ["transcribe", str(tmp_path / "ghost.wav"), "--engine", "mock"])
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_existing_output_fails_unless_overwrite(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    target = tmp_path / "clip.txt"
    target.write_text("sentinel\n", encoding="utf-8")
    refused = runner.invoke(main, ["transcribe", str(media), "--engine", "mock"])
    assert refused.exit_code == 1
    assert "already exists" in refused.stderr
    assert target.read_text(encoding="utf-8") == "sentinel\n"
    replaced = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "--overwrite"])
    assert replaced.exit_code == 0, replaced.stderr
    assert target.read_text(encoding="utf-8") == mock_render(media)


def test_bad_cli_temperature_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(
        main, ["transcribe", str(media), "--engine", "mock", "--temperature", "3.0"]
    )
    assert result.exit_code == 2
    assert "temperature" in result.stderr


def test_bad_environment_value_exits_2(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIBER_BEAM_SIZE", "zero")
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock"])
    assert result.exit_code == 2
    assert "TRANSCRIBER_BEAM_SIZE" in result.stderr


def test_unknown_environment_engine_exits_2(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIBER_ENGINE", "hypothetical")
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media)])
    assert result.exit_code == 2
    assert "unknown engine" in result.stderr


# ---------------------------------------------------------------------------
# transcribe: --stdout
# ---------------------------------------------------------------------------


def test_stdout_prints_rendered_txt_exactly_and_writes_nothing(
    runner: CliRunner, tmp_path: Path
) -> None:
    media = make_media(tmp_path, "solo.wav")
    before = set(tmp_path.iterdir())
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "--stdout"])
    assert result.exit_code == 0, result.stderr
    assert result.stdout == mock_render(media)
    assert result.stderr == ""  # no progress, no summary, no hints
    assert set(tmp_path.iterdir()) == before


def test_stdout_honours_selected_format(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "solo.wav")
    result = runner.invoke(
        main, ["transcribe", str(media), "--engine", "mock", "--stdout", "-f", "srt"]
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == mock_render(media, "srt")


def test_stdout_with_two_formats_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "solo.wav")
    result = runner.invoke(
        main,
        ["transcribe", str(media), "--engine", "mock", "--stdout", "-f", "txt", "-f", "srt"],
    )
    assert result.exit_code == 2
    assert "--stdout" in result.stderr


def test_stdout_with_two_inputs_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    first = make_media(tmp_path, "one.wav")
    second = make_media(tmp_path, "two.wav")
    result = runner.invoke(
        main, ["transcribe", str(first), str(second), "--engine", "mock", "--stdout"]
    )
    assert result.exit_code == 2
    assert "--stdout" in result.stderr


def test_stdout_with_output_dir_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "solo.wav")
    result = runner.invoke(
        main,
        ["transcribe", str(media), "--engine", "mock", "--stdout", "-o", str(tmp_path / "out")],
    )
    assert result.exit_code == 2
    assert "--stdout" in result.stderr


def test_stdout_failing_file_exits_1_with_clean_stdout(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "xFAILy.wav")
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "--stdout"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "xFAILy" in result.stderr


# ---------------------------------------------------------------------------
# Environment precedence, hints, verbosity, quiet
# ---------------------------------------------------------------------------


def test_environment_model_is_respected(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIBER_MODEL", "tiny")
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "-v"])
    assert result.exit_code == 0, result.stderr
    assert "model=tiny" in result.stderr


def test_cli_model_flag_beats_environment(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIBER_MODEL", "tiny")
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "-m", "base", "-v"])
    assert result.exit_code == 0, result.stderr
    assert "model=base" in result.stderr
    assert "model=tiny" not in result.stderr


def test_environment_engine_is_respected(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRANSCRIBER_ENGINE", "mock")
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media), "-v"])
    assert result.exit_code == 0, result.stderr
    assert "engine=mock" in result.stderr
    assert media.with_suffix(".txt").is_file()


def test_unknown_model_prints_hint(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(
        main, ["transcribe", str(media), "--engine", "mock", "-m", "acme/whisper-ct2"]
    )
    assert result.exit_code == 0, result.stderr
    assert "not a standard Whisper model alias" in result.stderr
    assert "acme/whisper-ct2" in result.stderr


def test_known_model_prints_no_hint(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "-m", "tiny"])
    assert result.exit_code == 0
    assert "not a standard" not in result.stderr


def test_quiet_success_prints_nothing(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(
        main, ["transcribe", str(media), "--engine", "mock", "-q", "-m", "acme/custom"]
    )
    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""  # summary, hint, and progress all suppressed


def test_quiet_still_prints_errors(runner: CliRunner, tmp_path: Path) -> None:
    make_media(tmp_path, "good.wav")
    make_media(tmp_path, "xFAILy.wav")
    result = runner.invoke(main, ["transcribe", str(tmp_path), "--engine", "mock", "-q"])
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert "xFAILy" in result.stderr
    assert "Status" not in result.stderr  # no summary table
    assert " ok, " not in result.stderr  # no counts line


def test_verbose_prints_per_file_detail(runner: CliRunner, tmp_path: Path) -> None:
    media = make_media(tmp_path, "clip.wav")
    result = runner.invoke(main, ["transcribe", str(media), "--engine", "mock", "-v"])
    assert result.exit_code == 0
    assert "settings:" in result.stderr
    assert "done: clip.wav" in result.stderr


# ---------------------------------------------------------------------------
# Packaging surface: python -m, lazy imports, public API
# ---------------------------------------------------------------------------


def test_python_dash_m_reports_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "transcriber", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert transcriber.__version__ in proc.stdout


def test_importing_package_is_lazy() -> None:
    """``import transcriber`` must not drag in faster_whisper or the CLI stack."""
    code = (
        "import sys\n"
        "import transcriber\n"
        "assert 'faster_whisper' not in sys.modules, 'faster_whisper imported eagerly'\n"
        "assert 'transcriber.cli' not in sys.modules, 'CLI imported eagerly'\n"
        "assert 'click' not in sys.modules, 'click imported eagerly'\n"
        "print(transcriber.__version__)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == transcriber.__version__


def test_public_api_exports() -> None:
    expected = {
        "Transcriber",
        "TranscriberConfig",
        "Transcript",
        "Segment",
        "Word",
        "TranscriptionEngine",
        "MockEngine",
        "TranscriberError",
        "ConfigError",
        "InputError",
        "AudioNotFoundError",
        "UnsupportedFormatError",
        "EngineError",
        "EngineNotAvailableError",
        "ModelLoadError",
        "TranscriptionFailedError",
        "OutputError",
        "__version__",
    }
    assert set(transcriber.__all__) == expected
    for name in expected:
        assert getattr(transcriber, name) is not None


def test_public_api_identity() -> None:
    from transcriber import config, engines, model, pipeline

    assert transcriber.Transcriber is pipeline.Transcriber
    assert transcriber.TranscriberConfig is config.TranscriberConfig
    assert transcriber.Transcript is model.Transcript
    assert transcriber.MockEngine is engines.MockEngine
