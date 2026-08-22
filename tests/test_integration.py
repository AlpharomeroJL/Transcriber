"""End-to-end integration tests over the committed synthetic media fixtures.

Unlike the unit suites, everything here exercises the real product surface:
the ``transcriber`` CLI driven through :class:`click.testing.CliRunner` over
the WAV fixtures committed under ``tests/fixtures/``, plus true
``python -m transcriber`` subprocesses (marked ``smoke``; deselect with
``-m "not smoke"``). Every test stays offline and deterministic by using the
mock engine (``--engine mock``); the fixtures themselves are generated — and
provably regenerable — by ``scripts/make_fixture.py``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import warnings
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest
from click.testing import CliRunner

import transcriber
from transcriber.cli import main
from transcriber.config import TranscriberConfig
from transcriber.engines import MockEngine
from transcriber.formats import available_formats, get_formatter
from transcriber.model import SCHEMA_VERSION, Transcript

# ``smoke`` marks the tests that spawn true ``python -m transcriber``
# subprocesses. The marker registry lives in pytest's ini options, which this
# suite does not own, so suppress only the unknown-mark warning for this one
# deliberate marker; ``-m smoke`` / ``-m "not smoke"`` selection still works.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    SMOKE: Final = pytest.mark.smoke

#: The committed fixtures directory this whole module transcribes from.
FIXTURES_DIR: Final[Path] = Path(__file__).resolve().parent / "fixtures"

#: Every committed fixture, in the sorted order batches process them.
FIXTURE_NAMES: Final[tuple[str, ...]] = ("silence.wav", "speechlike.wav", "tone.wav")

#: Samples per fixture second; must match ``scripts/make_fixture.py``.
SAMPLE_RATE: Final[int] = 16_000

_SRT_CUE = re.compile(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n.+")
_VTT_TIME_LINE = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}")

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stable_cli_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests (and their subprocesses) from ambient configuration.

    Ambient ``TRANSCRIBER_*`` variables would leak into ``from_env`` — both
    in-process and in the spawned ``python -m transcriber`` smoke tests,
    which inherit ``os.environ`` as patched here.
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


def all_format_args() -> list[str]:
    """``-f <name>`` CLI arguments requesting every registered format."""
    args: list[str] = []
    for name in available_formats():
        args.extend(["-f", name])
    return args


def mock_render(path: Path, format_name: str = "txt") -> str:
    """Render *path* exactly as a mock-engine CLI run should."""
    config = TranscriberConfig(engine="mock")
    return get_formatter(format_name).render(MockEngine().transcribe(path, config))


def read_wav(path: Path) -> tuple[int, int, int, list[int]]:
    """Return ``(channels, sample_width, frame_rate, samples)`` of a 16-bit mono WAV."""
    with wave.open(str(path), "rb") as reader:
        frames = reader.readframes(reader.getnframes())
        samples = list(struct.unpack(f"<{reader.getnframes()}h", frames))
        return reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), samples


def window_rms(samples: Sequence[int], window: int) -> list[float]:
    """Root-mean-square energy of consecutive *window*-sample slices."""
    return [
        math.sqrt(sum(value * value for value in samples[start : start + window]) / window)
        for start in range(0, len(samples) - window + 1, window)
    ]


# ---------------------------------------------------------------------------
# Fixture integrity: the committed WAVs and their generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_is_committed_with_canonical_parameters(name: str) -> None:
    path = FIXTURES_DIR / name
    assert path.is_file(), f"committed fixture missing: {path}"
    channels, sample_width, frame_rate, samples = read_wav(path)
    assert (channels, sample_width, frame_rate) == (1, 2, SAMPLE_RATE)
    assert 0.8 <= len(samples) / SAMPLE_RATE <= 2.0  # the spec'd ~1-2 s clips


def test_silence_fixture_is_pure_silence() -> None:
    _, _, _, samples = read_wav(FIXTURES_DIR / "silence.wav")
    assert samples, "silence.wav has no frames"
    assert all(value == 0 for value in samples)


def test_tone_fixture_has_energy_then_silent_tail() -> None:
    _, _, _, samples = read_wav(FIXTURES_DIR / "tone.wav")
    tail = round(0.2 * SAMPLE_RATE)
    assert max(abs(value) for value in samples[:-tail]) > 8_000  # the sweep is loud
    assert all(value == 0 for value in samples[-tail:])  # and ends in silence


def test_speechlike_fixture_has_speechlike_energy_variation() -> None:
    """VAD-style consumers must see bursts, pauses, and everything between."""
    _, _, _, samples = read_wav(FIXTURES_DIR / "speechlike.wav")
    profile = window_rms(samples, round(0.05 * SAMPLE_RATE))
    loud = [rms for rms in profile if rms > 5_000]
    silent = [rms for rms in profile if rms == 0.0]
    assert len(loud) >= 4, f"expected sustained speech-like bursts, got profile {profile}"
    assert len(silent) >= 2, f"expected silent pauses, got profile {profile}"
    assert profile[0] == 0.0, "expected a leading pause before the first burst"


def test_generator_reproduces_committed_fixtures(repo_root: Path, tmp_path: Path) -> None:
    """``scripts/make_fixture.py`` regenerates the committed bytes exactly."""
    out_dir = tmp_path / "regenerated"
    script = repo_root / "scripts" / "make_fixture.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(FIXTURE_NAMES)
    for name in FIXTURE_NAMES:
        regenerated = (out_dir / name).read_bytes()
        committed = (FIXTURES_DIR / name).read_bytes()
        assert regenerated == committed, f"{name} drifted from scripts/make_fixture.py"


# ---------------------------------------------------------------------------
# CLI end-to-end: every format at once over the fixtures directory
# ---------------------------------------------------------------------------


def test_batch_writes_every_format_for_every_fixture(runner: CliRunner, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        [
            "transcribe",
            str(FIXTURES_DIR),
            "--engine",
            "mock",
            "-o",
            str(out_dir),
            *all_format_args(),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""  # stdout stays clean without --stdout
    expected = {
        f"{Path(name).stem}{get_formatter(format_name).extension}"
        for name in FIXTURE_NAMES
        for format_name in available_formats()
    }
    assert {path.name for path in out_dir.iterdir()} == expected  # no extras, no temp litter
    assert f"{len(FIXTURE_NAMES)} ok, 0 failed" in result.stderr


def test_json_outputs_round_trip_through_transcript_from_dict(
    runner: CliRunner, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        ["transcribe", str(FIXTURES_DIR), "--engine", "mock", "-o", str(out_dir), "-f", "json"],
    )
    assert result.exit_code == 0, result.stderr
    for name in FIXTURE_NAMES:
        data = json.loads((out_dir / Path(name).stem).with_suffix(".json").read_text("utf-8"))
        assert data["schema_version"] == SCHEMA_VERSION
        transcript = Transcript.from_dict(data)
        assert transcript.to_dict() == data  # full round-trip, key for key
        assert Path(transcript.source) == (FIXTURES_DIR / name).resolve()
        assert (transcript.engine, transcript.model) == ("mock", "mock")
        assert transcript.duration > 0
        assert len(transcript.segments) == 2
        assert Path(name).stem in transcript.text


def test_srt_outputs_have_valid_cue_structure(runner: CliRunner, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        ["transcribe", str(FIXTURES_DIR), "--engine", "mock", "-o", str(out_dir), "-f", "srt"],
    )
    assert result.exit_code == 0, result.stderr
    for name in FIXTURE_NAMES:
        content = (out_dir / Path(name).stem).with_suffix(".srt").read_text("utf-8")
        assert content.endswith("\n") and not content.endswith("\n\n")
        cues = content.removesuffix("\n").split("\n\n")
        assert len(cues) == 2  # the mock engine always fabricates two segments
        for counter, cue in enumerate(cues, start=1):
            assert _SRT_CUE.fullmatch(cue), f"malformed SRT cue in {name}: {cue!r}"
            assert cue.startswith(f"{counter}\n"), "cues must be numbered from 1"


def test_vtt_outputs_have_header_and_dot_timestamps(runner: CliRunner, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        ["transcribe", str(FIXTURES_DIR), "--engine", "mock", "-o", str(out_dir), "-f", "vtt"],
    )
    assert result.exit_code == 0, result.stderr
    for name in FIXTURE_NAMES:
        content = (out_dir / Path(name).stem).with_suffix(".vtt").read_text("utf-8")
        assert content.startswith("WEBVTT\n\n"), f"{name}: missing WEBVTT header block"
        time_lines = [line for line in content.splitlines() if "-->" in line]
        assert len(time_lines) == 2
        for line in time_lines:
            assert _VTT_TIME_LINE.fullmatch(line), f"malformed VTT time line in {name}: {line!r}"


def test_tsv_outputs_have_header_and_integer_millisecond_rows(
    runner: CliRunner, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        ["transcribe", str(FIXTURES_DIR), "--engine", "mock", "-o", str(out_dir), "-f", "tsv"],
    )
    assert result.exit_code == 0, result.stderr
    for name in FIXTURE_NAMES:
        lines = (out_dir / Path(name).stem).with_suffix(".tsv").read_text("utf-8").splitlines()
        assert lines[0] == "start\tend\ttext"
        assert len(lines) == 3  # header + one row per mock segment
        for line in lines[1:]:
            start, end, text = line.split("\t")  # exactly three fields per row
            assert start.isdigit() and end.isdigit(), f"non-integer times in {name}: {line!r}"
            assert int(end) >= int(start)
            assert text


def test_txt_outputs_match_library_rendering_exactly(runner: CliRunner, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main,
        ["transcribe", str(FIXTURES_DIR), "--engine", "mock", "-o", str(out_dir), "-f", "txt"],
    )
    assert result.exit_code == 0, result.stderr
    for name in FIXTURE_NAMES:
        written = (out_dir / Path(name).stem).with_suffix(".txt").read_text("utf-8")
        assert written == mock_render((FIXTURES_DIR / name).resolve())


# ---------------------------------------------------------------------------
# Batch isolation: one failing file must not sink the batch
# ---------------------------------------------------------------------------


def test_failing_file_is_isolated_and_reported(runner: CliRunner, tmp_path: Path) -> None:
    work_dir = tmp_path / "media"
    work_dir.mkdir()
    for name in ("tone.wav", "speechlike.wav"):
        shutil.copyfile(FIXTURES_DIR / name, work_dir / name)
    shutil.copyfile(FIXTURES_DIR / "tone.wav", work_dir / "FAIL_case.wav")
    out_dir = tmp_path / "out"
    result = runner.invoke(
        main, ["transcribe", str(work_dir), "--engine", "mock", "-o", str(out_dir)]
    )
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert "FAIL_case.wav" in result.stderr  # the failure names the failing file
    assert (out_dir / "tone.txt").is_file()  # the good files were still produced
    assert (out_dir / "speechlike.txt").is_file()
    assert not (out_dir / "FAIL_case.txt").exists()
    assert "2 ok, 1 failed" in result.stderr


# ---------------------------------------------------------------------------
# Determinism: identical runs produce byte-identical outputs
# ---------------------------------------------------------------------------


def test_repeated_runs_produce_byte_identical_outputs(runner: CliRunner, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    args = [
        "transcribe",
        str(FIXTURES_DIR),
        "--engine",
        "mock",
        "-o",
        str(out_dir),
        "--overwrite",
        *all_format_args(),
    ]
    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.stderr
    snapshot = {path.name: path.read_bytes() for path in out_dir.iterdir()}
    assert len(snapshot) == len(FIXTURE_NAMES) * len(available_formats())
    second = runner.invoke(main, args)
    assert second.exit_code == 0, second.stderr
    assert {path.name: path.read_bytes() for path in out_dir.iterdir()} == snapshot


# ---------------------------------------------------------------------------
# Subprocess smoke: the real ``python -m transcriber`` entry path, still offline
# ---------------------------------------------------------------------------


@SMOKE
def test_module_entry_point_reports_version(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "transcriber", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "transcriber" in proc.stdout
    assert transcriber.__version__ in proc.stdout
    assert proc.stderr == ""


@SMOKE
def test_module_entry_point_transcribes_to_clean_stdout(tmp_path: Path) -> None:
    """``--stdout`` over a fixture: exact transcript on stdout, nothing else anywhere."""
    fixture = FIXTURES_DIR / "tone.wav"
    argv = [sys.executable, "-m", "transcriber", "transcribe", str(fixture)]
    argv += ["--engine", "mock", "--stdout", "-f", "txt"]
    proc = subprocess.run(
        argv,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == mock_render(fixture)
    assert proc.stderr == ""  # progress and summaries are suppressed off-terminal
    assert list(tmp_path.iterdir()) == []  # --stdout writes no files into the cwd
