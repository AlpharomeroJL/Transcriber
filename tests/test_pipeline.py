"""Tests for the transcription pipeline facade: batch runs, events, atomic writes.

Everything here is offline and deterministic: engines are :class:`MockEngine`
(or small subclasses of it injected for spying/fault injection), inputs are
synthetic files under ``tmp_path``, and no timing test asserts wall-clock
values — only that elapsed times exist and are non-negative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcriber.config import TranscriberConfig
from transcriber.engines import FasterWhisperEngine, MockEngine, TranscriptionEngine
from transcriber.errors import (
    AudioNotFoundError,
    EngineError,
    OutputError,
    TranscriptionFailedError,
    UnsupportedFormatError,
)
from transcriber.formats import SrtFormatter, get_formatter
from transcriber.model import Transcript
from transcriber.pipeline import (
    BatchResult,
    FileResult,
    PipelineEvent,
    Transcriber,
    _write_text_atomic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A fully resolved temporary directory, for exact path comparisons."""
    return tmp_path.resolve()


def make_media(directory: Path, name: str, size: int = 2048) -> Path:
    """Create a synthetic media file of *size* bytes and return its resolved path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\0" * size)
    return path.resolve()


def make_transcriber(engine: TranscriptionEngine | None = None) -> Transcriber:
    """A Transcriber wired to the given engine (a fresh MockEngine by default)."""
    return Transcriber(engine=engine if engine is not None else MockEngine())


class CountingEngine(MockEngine):
    """MockEngine that records every path it is asked to transcribe."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Path] = []

    def transcribe(self, path: Path, config: TranscriberConfig) -> Transcript:
        self.calls.append(path)
        return super().transcribe(path, config)


class BrokenReadEngine(MockEngine):
    """MockEngine that raises a plain OSError for stems containing ``oserr``."""

    def transcribe(self, path: Path, config: TranscriberConfig) -> Transcript:
        if "oserr" in path.stem:
            raise OSError("simulated unreadable media")
        return super().transcribe(path, config)


class ClosableEngine(MockEngine):
    """MockEngine that records whether close() was called."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_config_is_library_defaults(self) -> None:
        transcriber = make_transcriber()
        assert transcriber.config == TranscriberConfig()

    def test_explicit_config_is_kept(self) -> None:
        config = TranscriberConfig(engine="mock", model="tiny")
        transcriber = Transcriber(config)
        assert transcriber.config is config

    def test_engine_created_from_config_name(self) -> None:
        transcriber = Transcriber(TranscriberConfig(engine="mock"))
        assert isinstance(transcriber.engine, MockEngine)

    def test_default_engine_name_is_faster_whisper(self) -> None:
        transcriber = Transcriber()
        assert isinstance(transcriber.engine, FasterWhisperEngine)

    def test_injected_engine_wins_over_config(self) -> None:
        engine = MockEngine()
        # The config names an unregistered engine: if the constructor consulted
        # the registry despite the injection, this would raise EngineError.
        transcriber = Transcriber(
            TranscriberConfig(engine="not-a-registered-engine"), engine=engine
        )
        assert transcriber.engine is engine

    def test_unknown_engine_name_raises(self) -> None:
        with pytest.raises(EngineError, match="unknown engine"):
            Transcriber(TranscriberConfig(engine="not-a-registered-engine"))

    def test_close_delegates_to_engine(self) -> None:
        engine = ClosableEngine()
        make_transcriber(engine).close()
        assert engine.closed


# ---------------------------------------------------------------------------
# transcribe_file
# ---------------------------------------------------------------------------


class TestTranscribeFile:
    def test_returns_transcript_for_valid_file(self, root: Path) -> None:
        media = make_media(root, "clip.wav")
        transcript = make_transcriber().transcribe_file(media)
        assert transcript.source == str(media)
        assert transcript.engine == "mock"
        assert "[mock transcription of clip]" in transcript.text

    def test_missing_path_raises(self, root: Path) -> None:
        with pytest.raises(AudioNotFoundError):
            make_transcriber().transcribe_file(root / "nope.wav")

    def test_unsupported_extension_raises(self, root: Path) -> None:
        path = root / "notes.pdf"
        path.write_bytes(b"%PDF")
        with pytest.raises(UnsupportedFormatError):
            make_transcriber().transcribe_file(path)

    def test_directory_raises(self, root: Path) -> None:
        with pytest.raises(AudioNotFoundError, match="directory"):
            make_transcriber().transcribe_file(root)

    def test_engine_failure_propagates(self, root: Path) -> None:
        media = make_media(root, "clip_FAIL.wav")
        with pytest.raises(TranscriptionFailedError):
            make_transcriber().transcribe_file(media)


# ---------------------------------------------------------------------------
# transcribe_batch: happy path
# ---------------------------------------------------------------------------


class TestBatchHappyPath:
    def test_writes_all_formats_at_exact_paths(self, root: Path) -> None:
        alpha = make_media(root, "alpha.wav")
        bravo = make_media(root, "bravo.wav")
        batch = make_transcriber().transcribe_batch([root], formats=("txt", "srt", "json"))

        assert [result.path for result in batch.results] == [alpha, bravo]
        assert batch.ok_count == 2
        assert batch.error_count == 0
        assert batch.all_ok is True
        assert batch.exit_code == 0
        for result, stem in zip(batch.results, ("alpha", "bravo"), strict=True):
            expected = [root / f"{stem}.txt", root / f"{stem}.srt", root / f"{stem}.json"]
            assert result.outputs == expected
            assert result.ok is True
            assert result.error is None
            assert result.transcript is not None
            for target in expected:
                assert target.is_file()

    def test_default_format_is_txt_only(self, root: Path) -> None:
        media = make_media(root, "clip.wav")
        batch = make_transcriber().transcribe_batch([media])
        assert batch.results[0].outputs == [root / "clip.txt"]
        assert not (root / "clip.srt").exists()

    def test_written_content_matches_formatter_render(self, root: Path) -> None:
        make_media(root, "clip.wav")
        batch = make_transcriber().transcribe_batch([root], formats=("txt", "json"))
        transcript = batch.results[0].transcript
        assert transcript is not None
        txt = (root / "clip.txt").read_text(encoding="utf-8")
        assert txt == get_formatter("txt").render(transcript)
        assert txt == "[mock transcription of clip]\n[mock transcription of clip, part 2]\n"
        parsed = json.loads((root / "clip.json").read_text(encoding="utf-8"))
        assert parsed == transcript.to_dict()

    def test_elapsed_is_measured_not_asserted(self, root: Path) -> None:
        make_media(root, "clip.wav")
        result = make_transcriber().transcribe_batch([root]).results[0]
        assert isinstance(result.elapsed, float)
        assert result.elapsed >= 0.0

    def test_explicit_file_inputs_write_next_to_each_input(self, root: Path) -> None:
        one = make_media(root / "d1", "one.wav")
        two = make_media(root / "d2", "two.wav")
        batch = make_transcriber().transcribe_batch([one, two])
        assert batch.all_ok
        assert (root / "d1" / "one.txt").is_file()
        assert (root / "d2" / "two.txt").is_file()


# ---------------------------------------------------------------------------
# transcribe_batch: failure isolation
# ---------------------------------------------------------------------------


class TestBatchFailureIsolation:
    def test_failing_file_is_isolated_and_batch_continues(self, root: Path) -> None:
        alpha = make_media(root, "alpha.wav")
        failing = make_media(root, "bad_FAIL.wav")
        zulu = make_media(root, "zulu.wav")
        batch = make_transcriber().transcribe_batch([root])

        assert [result.path for result in batch.results] == [alpha, failing, zulu]
        assert batch.ok_count == 2
        assert batch.error_count == 1
        assert batch.all_ok is False
        assert batch.exit_code == 1

        failed = batch.results[1]
        assert failed.ok is False
        assert failed.transcript is None
        assert failed.outputs == []
        assert failed.error is not None
        assert "failed on purpose" in failed.error
        assert not (root / "bad_FAIL.txt").exists()
        assert (root / "alpha.txt").is_file()
        assert (root / "zulu.txt").is_file()

    def test_engine_oserror_is_recorded_not_raised(self, root: Path) -> None:
        make_media(root, "alpha.wav")
        make_media(root, "oserr_clip.wav")
        batch = make_transcriber(BrokenReadEngine()).transcribe_batch([root])
        assert batch.ok_count == 1
        assert batch.error_count == 1
        failed = next(result for result in batch.results if not result.ok)
        assert failed.error == "simulated unreadable media"
        assert batch.exit_code == 1

    def test_collection_errors_propagate(self, root: Path) -> None:
        with pytest.raises(AudioNotFoundError):
            make_transcriber().transcribe_batch([root / "missing.wav"])

    def test_unsupported_explicit_input_propagates(self, root: Path) -> None:
        path = root / "notes.pdf"
        path.write_bytes(b"%PDF")
        with pytest.raises(UnsupportedFormatError):
            make_transcriber().transcribe_batch([path])

    def test_empty_directory_propagates(self, root: Path) -> None:
        with pytest.raises(AudioNotFoundError):
            make_transcriber().transcribe_batch([root])


# ---------------------------------------------------------------------------
# transcribe_batch: overwrite policy
# ---------------------------------------------------------------------------


class TestOverwrite:
    def test_existing_target_refused_without_overwrite(self, root: Path) -> None:
        make_media(root, "alpha.wav")
        existing = root / "alpha.txt"
        existing.write_text("precious\n", encoding="utf-8")
        engine = CountingEngine()
        batch = make_transcriber(engine).transcribe_batch([root], formats=("txt", "srt"))

        result = batch.results[0]
        assert result.ok is False
        assert result.error is not None
        assert "already exists" in result.error
        assert str(existing) in result.error
        assert "overwrite=True" in result.error
        assert result.outputs == []
        assert batch.exit_code == 1
        # The clash was detected before any work: nothing transcribed or written.
        assert engine.calls == []
        assert existing.read_text(encoding="utf-8") == "precious\n"
        assert not (root / "alpha.srt").exists()

    def test_overwrite_true_replaces_existing_target(self, root: Path) -> None:
        make_media(root, "alpha.wav")
        existing = root / "alpha.txt"
        existing.write_text("precious\n", encoding="utf-8")
        batch = make_transcriber().transcribe_batch([root], overwrite=True)
        assert batch.all_ok
        content = existing.read_text(encoding="utf-8")
        assert content != "precious\n"
        assert "[mock transcription of alpha]" in content


# ---------------------------------------------------------------------------
# transcribe_batch: output directory
# ---------------------------------------------------------------------------


class TestOutputDir:
    def test_output_dir_created_with_parents(self, root: Path) -> None:
        media = make_media(root, "clip.wav")
        out = root / "nested" / "deep"
        batch = make_transcriber().transcribe_batch([media], output_dir=out)
        assert out.is_dir()
        assert batch.results[0].outputs == [out / "clip.txt"]
        assert (out / "clip.txt").is_file()
        assert not (root / "clip.txt").exists()

    def test_existing_output_dir_is_reused(self, root: Path) -> None:
        media = make_media(root, "clip.wav")
        out = root / "out"
        out.mkdir()
        batch = make_transcriber().transcribe_batch([media], output_dir=out)
        assert batch.all_ok
        assert (out / "clip.txt").is_file()

    def test_uncreatable_output_dir_fails_fast(self, root: Path) -> None:
        media = make_media(root, "clip.wav")
        blocker = root / "blocker"
        blocker.write_text("a file, not a directory\n", encoding="utf-8")
        engine = CountingEngine()
        with pytest.raises(OutputError, match="cannot create output directory"):
            make_transcriber(engine).transcribe_batch([media], output_dir=blocker)
        assert engine.calls == []


# ---------------------------------------------------------------------------
# transcribe_batch: duplicate output stems
# ---------------------------------------------------------------------------


class TestDuplicateStems:
    def test_later_duplicate_into_shared_output_dir_fails(self, root: Path) -> None:
        first = make_media(root / "d1", "x.wav", size=1024)
        second = make_media(root / "d2", "x.wav", size=4096)
        out = root / "out"
        engine = CountingEngine()
        batch = make_transcriber(engine).transcribe_batch([first, second], output_dir=out)

        assert [result.ok for result in batch.results] == [True, False]
        assert batch.exit_code == 1
        failed = batch.results[1]
        assert failed.error is not None
        assert "collision" in failed.error
        assert str(first) in failed.error
        assert str(second) in failed.error
        assert failed.outputs == []
        assert failed.transcript is None
        # Only the first claimant was transcribed; its output is in place.
        assert engine.calls == [first]
        assert (out / "x.txt").is_file()

    def test_same_directory_same_stem_different_extensions_collide(self, root: Path) -> None:
        mp3 = make_media(root, "x.mp3")
        wav = make_media(root, "x.wav")
        batch = make_transcriber().transcribe_batch([root])
        # Sorted processing order: x.mp3 first, so x.wav is the later duplicate.
        assert [result.path for result in batch.results] == [mp3, wav]
        assert [result.ok for result in batch.results] == [True, False]
        assert (root / "x.txt").is_file()

    def test_same_stems_in_different_parents_do_not_collide(self, root: Path) -> None:
        make_media(root / "d1", "x.wav")
        make_media(root / "d2", "x.wav")
        batch = make_transcriber().transcribe_batch([root / "d1", root / "d2"])
        assert batch.all_ok
        assert (root / "d1" / "x.txt").is_file()
        assert (root / "d2" / "x.txt").is_file()


# ---------------------------------------------------------------------------
# transcribe_batch: events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_event_sequence_ordering_and_totals(self, root: Path) -> None:
        alpha = make_media(root, "alpha.wav")
        failing = make_media(root, "bad_FAIL.wav")
        zulu = make_media(root, "zulu.wav")
        events: list[PipelineEvent] = []
        batch = make_transcriber().transcribe_batch([root], on_event=events.append)

        assert [event.kind for event in events] == [
            "file_started",
            "file_finished",
            "file_started",
            "file_failed",
            "file_started",
            "file_finished",
        ]
        assert [event.index for event in events] == [1, 1, 2, 2, 3, 3]
        assert all(event.total == 3 for event in events)
        assert [event.path for event in events] == [alpha, alpha, failing, failing, zulu, zulu]

        started = events[::2]
        terminal = events[1::2]
        assert all(event.result is None and event.message is None for event in started)
        for event, result in zip(terminal, batch.results, strict=True):
            assert event.result is result
        finished_ok = [terminal[0], terminal[2]]
        assert all(event.message is None for event in finished_ok)
        failed_event = terminal[1]
        assert failed_event.result is not None
        assert failed_event.message == failed_event.result.error
        assert failed_event.message is not None

    def test_no_callback_is_fine(self, root: Path) -> None:
        make_media(root, "clip.wav")
        assert make_transcriber().transcribe_batch([root]).all_ok

    def test_callback_exception_propagates(self, root: Path) -> None:
        make_media(root, "clip.wav")

        def explode(event: PipelineEvent) -> None:
            raise RuntimeError("observer bug")

        with pytest.raises(RuntimeError, match="observer bug"):
            make_transcriber().transcribe_batch([root], on_event=explode)


# ---------------------------------------------------------------------------
# transcribe_batch: format validation
# ---------------------------------------------------------------------------


class TestFormatValidation:
    def test_unknown_format_fails_before_any_work(self, root: Path) -> None:
        make_media(root, "alpha.wav")
        engine = CountingEngine()
        events: list[PipelineEvent] = []
        with pytest.raises(OutputError, match="unknown output format") as excinfo:
            make_transcriber(engine).transcribe_batch(
                [root], formats=("txt", "bogus"), on_event=events.append
            )
        assert "bogus" in str(excinfo.value)
        assert "available formats" in str(excinfo.value)
        assert engine.calls == []
        assert events == []
        assert not (root / "alpha.txt").exists()

    def test_empty_formats_rejected(self, root: Path) -> None:
        make_media(root, "alpha.wav")
        with pytest.raises(OutputError, match="no output formats"):
            make_transcriber().transcribe_batch([root], formats=())

    def test_duplicate_format_names_are_written_once(self, root: Path) -> None:
        make_media(root, "alpha.wav")
        batch = make_transcriber().transcribe_batch([root], formats=("txt", "txt"))
        assert batch.results[0].outputs == [root / "alpha.txt"]


# ---------------------------------------------------------------------------
# transcribe_batch: recursion passthrough
# ---------------------------------------------------------------------------


class TestRecursive:
    def test_non_recursive_misses_nested_media(self, root: Path) -> None:
        make_media(root / "sub", "inner.wav")
        with pytest.raises(AudioNotFoundError):
            make_transcriber().transcribe_batch([root])

    def test_recursive_finds_nested_media(self, root: Path) -> None:
        make_media(root / "sub", "inner.wav")
        batch = make_transcriber().transcribe_batch([root], recursive=True)
        assert batch.all_ok
        assert (root / "sub" / "inner.txt").is_file()


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_oserror_mid_batch_leaves_no_partial_file(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_media(root, "alpha.wav")

        def broken_render(self: SrtFormatter, transcript: Transcript) -> str:
            raise OSError("simulated render failure")

        monkeypatch.setattr(SrtFormatter, "render", broken_render)
        batch = make_transcriber().transcribe_batch([root], formats=("txt", "srt"))

        result = batch.results[0]
        assert result.ok is False
        assert result.error == "simulated render failure"
        # The txt output written before the failure survives; the srt target
        # was never created, not even partially, and the transcript is kept.
        assert result.outputs == [root / "alpha.txt"]
        assert (root / "alpha.txt").is_file()
        assert not (root / "alpha.srt").exists()
        assert result.transcript is not None
        # No stray temporary files remain anywhere in the output directory.
        assert {entry.name for entry in root.iterdir()} == {"alpha.wav", "alpha.txt"}

    def test_failed_overwrite_preserves_previous_output(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_media(root, "alpha.wav")
        old = root / "alpha.srt"
        old.write_text("previous subtitle content\n", encoding="utf-8")

        def broken_render(self: SrtFormatter, transcript: Transcript) -> str:
            raise OSError("simulated render failure")

        monkeypatch.setattr(SrtFormatter, "render", broken_render)
        batch = make_transcriber().transcribe_batch([root], formats=("srt",), overwrite=True)

        assert batch.exit_code == 1
        assert old.read_text(encoding="utf-8") == "previous subtitle content\n"

    def test_unexpected_formatter_exception_propagates(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_media(root, "alpha.wav")

        def buggy_render(self: SrtFormatter, transcript: Transcript) -> str:
            raise RuntimeError("formatter bug")

        monkeypatch.setattr(SrtFormatter, "render", buggy_render)
        with pytest.raises(RuntimeError, match="formatter bug"):
            make_transcriber().transcribe_batch([root], formats=("srt",))
        assert not (root / "alpha.srt").exists()

    def test_write_text_atomic_writes_exact_utf8_bytes(self, root: Path) -> None:
        target = root / "out.txt"
        _write_text_atomic(target, "héllo — 你好\nsecond line\n")
        assert target.read_bytes() == "héllo — 你好\nsecond line\n".encode()
        assert {entry.name for entry in root.iterdir()} == {"out.txt"}

    def test_write_text_atomic_replaces_existing_file(self, root: Path) -> None:
        target = root / "out.txt"
        target.write_text("old\n", encoding="utf-8")
        _write_text_atomic(target, "new\n")
        assert target.read_text(encoding="utf-8") == "new\n"

    def test_write_text_atomic_wraps_oserror(self, root: Path) -> None:
        target = root / "no-such-directory" / "out.txt"
        with pytest.raises(OutputError, match="cannot write") as excinfo:
            _write_text_atomic(target, "text\n")
        assert str(target) in str(excinfo.value)


# ---------------------------------------------------------------------------
# BatchResult arithmetic
# ---------------------------------------------------------------------------


def fabricated_result(ok: bool) -> FileResult:
    """A minimal FileResult for exercising BatchResult arithmetic."""
    return FileResult(
        path=Path("clip.wav"),
        ok=ok,
        transcript=None,
        outputs=[],
        error=None if ok else "boom",
        elapsed=0.0,
    )


class TestBatchResult:
    def test_all_ok(self) -> None:
        batch = BatchResult([fabricated_result(True), fabricated_result(True)])
        assert batch.ok_count == 2
        assert batch.error_count == 0
        assert batch.all_ok is True
        assert batch.exit_code == 0

    def test_mixed(self) -> None:
        batch = BatchResult([fabricated_result(True), fabricated_result(False)])
        assert batch.ok_count == 1
        assert batch.error_count == 1
        assert batch.all_ok is False
        assert batch.exit_code == 1

    def test_all_failed(self) -> None:
        batch = BatchResult([fabricated_result(False)])
        assert batch.ok_count == 0
        assert batch.error_count == 1
        assert batch.exit_code == 1

    def test_empty_is_vacuously_ok(self) -> None:
        batch = BatchResult()
        assert batch.ok_count == 0
        assert batch.error_count == 0
        assert batch.all_ok is True
        assert batch.exit_code == 0
