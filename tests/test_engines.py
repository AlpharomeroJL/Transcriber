"""Tests for the engine abstraction: registry, mock engine, faster-whisper engine.

Everything here is offline and deterministic. The faster-whisper engine is
tested against a fake ``faster_whisper`` module injected into ``sys.modules``
(the engine's lazy import then picks up the fake), and the missing-package
path is simulated by poisoning the ``sys.modules`` entry with ``None``, which
makes ``import faster_whisper`` raise ``ImportError``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from transcriber.config import TranscriberConfig
from transcriber.engines import (
    FasterWhisperEngine,
    MockEngine,
    TranscriptionEngine,
    available_engines,
    create_engine,
    get_engine,
)
from transcriber.errors import (
    EngineError,
    EngineNotAvailableError,
    ModelLoadError,
    TranscriptionFailedError,
)
from transcriber.model import Segment, Transcript

# ---------------------------------------------------------------------------
# Fake faster_whisper backend
# ---------------------------------------------------------------------------


def fake_word(start: float, end: float, word: str, probability: float = 0.9) -> SimpleNamespace:
    """A stand-in for ``faster_whisper.transcribe.Word`` (note the ``word`` field)."""
    return SimpleNamespace(start=start, end=end, word=word, probability=probability)


def fake_segment(
    segment_id: int,
    start: float,
    end: float,
    text: str,
    words: list[SimpleNamespace] | None = None,
    avg_logprob: float = -0.25,
    no_speech_prob: float = 0.05,
) -> SimpleNamespace:
    """A stand-in for ``faster_whisper.transcribe.Segment``."""
    return SimpleNamespace(
        id=segment_id,
        start=start,
        end=end,
        text=text,
        words=words,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


class FakeBackend:
    """A fake ``faster_whisper`` module that records every call made to it."""

    def __init__(self) -> None:
        self.constructed: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.transcribe_calls: list[tuple[Any, dict[str, Any]]] = []
        self.segments: Iterable[Any] = []
        self.info = SimpleNamespace(language="de", language_probability=0.97, duration=42.5)
        self.generator_finished: list[bool] = []
        self.construction_error: Exception | None = None
        self.transcribe_error: Exception | None = None
        backend = self

        class FakeWhisperModel:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                if backend.construction_error is not None:
                    raise backend.construction_error
                backend.constructed.append((args, kwargs))

            def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Iterator[Any], Any]:
                if backend.transcribe_error is not None:
                    raise backend.transcribe_error
                backend.transcribe_calls.append((audio, kwargs))
                return backend._generate_segments(), backend.info

        module = ModuleType("faster_whisper")
        module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
        self.module = module

    def _generate_segments(self) -> Iterator[Any]:
        """Yield the configured segments, then record that the generator finished."""
        yield from self.segments
        self.generator_finished.append(True)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Inject a fresh fake ``faster_whisper`` module for the duration of a test."""
    fake = FakeBackend()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake.module)
    return fake


@pytest.fixture
def media_file(tmp_path: Path) -> Path:
    """A 2048-byte media file, so the mock's synthetic duration is exactly 2.0 s."""
    path = tmp_path / "interview.wav"
    path.write_bytes(b"\0" * 2048)
    return path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_available_engines_lists_both_registered_engines() -> None:
    assert available_engines() == ("faster-whisper", "mock")


def test_engine_names_match_registry_keys() -> None:
    assert FasterWhisperEngine.name == "faster-whisper"
    assert MockEngine.name == "mock"


def test_get_engine_returns_class_without_constructing() -> None:
    assert get_engine("mock") is MockEngine
    assert get_engine("faster-whisper") is FasterWhisperEngine


def test_get_engine_does_not_import_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # With the import poisoned, class lookup must still succeed (it is lazy)...
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    engine_cls = get_engine("faster-whisper")
    assert engine_cls is FasterWhisperEngine
    # ...and only construction trips over the missing package.
    with pytest.raises(EngineNotAvailableError):
        create_engine("faster-whisper")


def test_create_engine_returns_instances() -> None:
    engine = create_engine("mock")
    assert isinstance(engine, MockEngine)
    assert isinstance(engine, TranscriptionEngine)


def test_unknown_engine_error_names_the_available_ones() -> None:
    with pytest.raises(EngineError, match=r"'whisperx'.*faster-whisper.*mock"):
        get_engine("whisperx")


def test_close_is_an_optional_noop(media_file: Path) -> None:
    engine = create_engine("mock")
    engine.close()  # default implementation: nothing to release, never raises
    assert isinstance(engine.transcribe(media_file, TranscriberConfig()), Transcript)


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------


def test_mock_engine_is_deterministic(media_file: Path) -> None:
    config = TranscriberConfig(engine="mock")
    first = MockEngine().transcribe(media_file, config)
    second = MockEngine().transcribe(media_file, config)
    assert first == second  # frozen dataclasses: full structural equality


def test_mock_engine_synthesises_two_segments_from_file_size(media_file: Path) -> None:
    transcript = MockEngine().transcribe(media_file, TranscriberConfig(engine="mock"))
    assert transcript.engine == "mock"
    assert transcript.model == "mock"
    assert transcript.language == "en"
    assert transcript.language_probability == 1.0
    assert transcript.source == str(media_file)
    assert transcript.duration == 2.0  # 2048 bytes at 1024 bytes/second
    assert [segment.id for segment in transcript.segments] == [1, 2]
    assert [(segment.start, segment.end) for segment in transcript.segments] == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]
    assert all("mock transcription of interview" in segment.text for segment in transcript)
    assert all(segment.words == () for segment in transcript)
    assert not transcript.is_empty


def test_mock_engine_tolerates_missing_file(tmp_path: Path) -> None:
    ghost = tmp_path / "not-there.wav"
    transcript = MockEngine().transcribe(ghost, TranscriberConfig(engine="mock"))
    assert transcript.duration == pytest.approx(1 / 1024)
    assert len(transcript.segments) == 2


def test_mock_engine_honours_configured_language(media_file: Path) -> None:
    transcript = MockEngine().transcribe(
        media_file, TranscriberConfig(engine="mock", language="de")
    )
    assert transcript.language == "de"


def test_mock_engine_word_timestamps(media_file: Path) -> None:
    config = TranscriberConfig(engine="mock", word_timestamps=True)
    transcript = MockEngine().transcribe(media_file, config)
    for segment in transcript:
        assert segment.words, "word_timestamps=True must produce words"
        assert " ".join(word.text for word in segment.words) == segment.text
        assert segment.words[0].start == segment.start
        assert segment.words[-1].end == pytest.approx(segment.end)
        assert all(word.probability == 1.0 for word in segment.words)


def test_mock_engine_fail_marker_raises(tmp_path: Path) -> None:
    doomed = tmp_path / "meeting-FAIL.wav"
    doomed.write_bytes(b"\0" * 100)
    with pytest.raises(TranscriptionFailedError, match="meeting-FAIL"):
        MockEngine().transcribe(doomed, TranscriberConfig(engine="mock"))


def test_mock_engine_fail_marker_is_case_sensitive(tmp_path: Path) -> None:
    fine = tmp_path / "failure-postmortem.wav"
    fine.write_bytes(b"\0" * 100)
    transcript = MockEngine().transcribe(fine, TranscriberConfig(engine="mock"))
    assert not transcript.is_empty


def test_mock_engine_fixed_segments(tmp_path: Path) -> None:
    path = tmp_path / "scripted.wav"
    path.write_bytes(b"\0" * 4096)
    fixed = (
        Segment(id=1, start=0.0, end=1.5, text="hello"),
        Segment(id=2, start=1.5, end=3.25, text="world"),
    )
    transcript = MockEngine(segments=fixed).transcribe(path, TranscriberConfig(engine="mock"))
    assert transcript.segments == fixed
    assert transcript.duration == 3.25  # from the last segment, not the file size


def test_mock_engine_fixed_segments_still_fail_on_marker(tmp_path: Path) -> None:
    engine = MockEngine(segments=(Segment(id=1, start=0.0, end=1.0, text="hi"),))
    with pytest.raises(TranscriptionFailedError):
        engine.transcribe(tmp_path / "FAIL.wav", TranscriberConfig(engine="mock"))


# ---------------------------------------------------------------------------
# Faster-whisper engine (against the fake backend)
# ---------------------------------------------------------------------------


def test_faster_whisper_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import now raises ImportError
    with pytest.raises(EngineNotAvailableError, match=r"pip install faster-whisper"):
        FasterWhisperEngine()


def test_faster_whisper_maps_config_to_transcribe_options(
    backend: FakeBackend, media_file: Path
) -> None:
    config = TranscriberConfig(
        model="large-v3",
        device="cuda",
        compute_type="int8",
        language="nl",
        task="translate",
        vad=False,
        word_timestamps=True,
        beam_size=3,
        temperature=0.4,
        initial_prompt="Names: Anna, Bram.",
    )
    FasterWhisperEngine().transcribe(media_file, config)

    (ctor_args, ctor_kwargs) = backend.constructed[0]
    assert ctor_args == ("large-v3",)
    assert ctor_kwargs == {"device": "cuda", "compute_type": "int8"}

    (audio, options) = backend.transcribe_calls[0]
    assert audio == str(media_file)
    assert options == {
        "language": "nl",
        "task": "translate",
        "beam_size": 3,
        "temperature": 0.4,
        "initial_prompt": "Names: Anna, Bram.",
        "vad_filter": False,
        "word_timestamps": True,
    }


def test_faster_whisper_default_config_options(backend: FakeBackend, media_file: Path) -> None:
    FasterWhisperEngine().transcribe(media_file, TranscriberConfig())

    (ctor_args, ctor_kwargs) = backend.constructed[0]
    assert ctor_args == ("small",)
    # "auto" is a literal both faster-whisper and ctranslate2 accept for both knobs.
    assert ctor_kwargs == {"device": "auto", "compute_type": "auto"}

    (_, options) = backend.transcribe_calls[0]
    assert options == {
        "language": None,
        "task": "transcribe",
        "beam_size": 5,
        "temperature": 0.0,
        "initial_prompt": None,
        "vad_filter": True,  # VAD on by default: Whisper hallucinates on silence
        "word_timestamps": False,
    }


def test_faster_whisper_converts_segments_and_info(backend: FakeBackend, media_file: Path) -> None:
    backend.segments = [
        fake_segment(
            1,
            0.0,
            2.0,
            "  Hello there.  ",
            words=[fake_word(0.0, 1.0, " Hello", 0.9), fake_word(1.0, 2.0, " there.", 0.8)],
            avg_logprob=-0.1,
            no_speech_prob=0.01,
        ),
        fake_segment(2, 2.0, 4.5, "\nSecond segment.\n", words=None),
    ]
    transcript = FasterWhisperEngine().transcribe(media_file, TranscriberConfig(model="small.en"))

    assert backend.generator_finished == [True], "segment generator must be drained fully"

    assert transcript.engine == "faster-whisper"
    assert transcript.model == "small.en"
    assert transcript.source == str(media_file)
    assert transcript.language == "de"
    assert transcript.language_probability == 0.97
    assert transcript.duration == 42.5

    first, second = transcript.segments
    assert (first.id, first.start, first.end) == (1, 0.0, 2.0)
    assert first.text == "Hello there."
    assert first.avg_logprob == -0.1
    assert first.no_speech_prob == 0.01
    assert [word.text for word in first.words] == ["Hello", "there."]
    assert [word.probability for word in first.words] == [0.9, 0.8]
    assert (first.words[0].start, first.words[0].end) == (0.0, 1.0)
    assert second.text == "Second segment."
    assert second.words == (), "words=None from the backend must become an empty tuple"


def test_faster_whisper_caches_models_per_configuration(
    backend: FakeBackend, media_file: Path
) -> None:
    engine = FasterWhisperEngine()
    config = TranscriberConfig()
    engine.transcribe(media_file, config)
    engine.transcribe(media_file, config)
    assert len(backend.constructed) == 1, "same (model, device, compute_type) must reuse the model"

    engine.transcribe(media_file, config.replace(compute_type="int8"))
    assert len(backend.constructed) == 2

    engine.close()
    engine.transcribe(media_file, config)
    assert len(backend.constructed) == 3, "close() must drop cached models"


def test_faster_whisper_wraps_model_construction_errors(
    backend: FakeBackend, media_file: Path
) -> None:
    backend.construction_error = RuntimeError("no such model on disk")
    with pytest.raises(ModelLoadError, match=r"'tiny'.*no such model on disk"):
        FasterWhisperEngine().transcribe(media_file, TranscriberConfig(model="tiny"))


def test_faster_whisper_wraps_transcribe_errors(backend: FakeBackend, media_file: Path) -> None:
    backend.transcribe_error = RuntimeError("decoder exploded")
    with pytest.raises(TranscriptionFailedError, match=r"interview\.wav.*decoder exploded"):
        FasterWhisperEngine().transcribe(media_file, TranscriberConfig())


def test_faster_whisper_wraps_errors_raised_while_draining(
    backend: FakeBackend, media_file: Path
) -> None:
    def exploding_segments() -> Iterator[Any]:
        yield fake_segment(1, 0.0, 1.0, "ok")
        raise RuntimeError("failed mid-stream")

    backend.segments = exploding_segments()
    with pytest.raises(TranscriptionFailedError, match="failed mid-stream"):
        FasterWhisperEngine().transcribe(media_file, TranscriberConfig())


def test_faster_whisper_error_chain_preserves_cause(backend: FakeBackend, media_file: Path) -> None:
    backend.transcribe_error = RuntimeError("decoder exploded")
    with pytest.raises(TranscriptionFailedError) as excinfo:
        FasterWhisperEngine().transcribe(media_file, TranscriberConfig())
    assert isinstance(excinfo.value.__cause__, RuntimeError)
