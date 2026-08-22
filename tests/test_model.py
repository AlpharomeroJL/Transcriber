"""Tests for the engine-agnostic transcript data model."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from transcriber.model import SCHEMA_VERSION, Segment, Transcript, Word


def make_transcript() -> Transcript:
    """A representative transcript: unicode text, word timings, metadata."""
    return Transcript(
        segments=(
            Segment(
                id=1,
                start=0.0,
                end=2.5,
                text="Héllo wörld — 你好 🎙️",
                words=(
                    Word(start=0.0, end=1.0, text=" Héllo", probability=0.98),
                    Word(start=1.0, end=2.5, text=" wörld", probability=None),
                ),
                avg_logprob=-0.25,
                no_speech_prob=0.01,
            ),
            Segment(id=2, start=2.5, end=4.75, text="second segment"),
        ),
        language="en",
        language_probability=0.93,
        duration=4.75,
        source="clip.wav",
        engine="mock",
        model="small",
    )


class TestWord:
    def test_defaults(self) -> None:
        word = Word(start=0.5, end=1.0, text="hi")
        assert word.probability is None

    def test_text_preserved_verbatim(self) -> None:
        # Engines may pass through leading spaces; Word does not normalise.
        assert Word(start=0.0, end=1.0, text=" hi ").text == " hi "

    def test_zero_width_span_allowed(self) -> None:
        Word(start=1.0, end=1.0, text="hi")

    @pytest.mark.parametrize(
        ("start", "end"),
        [(-0.1, 1.0), (0.0, -1.0), (2.0, 1.0), (math.nan, 1.0), (0.0, math.inf)],
    )
    def test_invalid_times_raise(self, start: float, end: float) -> None:
        with pytest.raises(ValueError, match="word"):
            Word(start=start, end=end, text="hi")

    def test_frozen_and_slotted(self) -> None:
        word = Word(start=0.0, end=1.0, text="hi")
        with pytest.raises(dataclasses.FrozenInstanceError):
            word.text = "bye"  # type: ignore[misc]
        assert not hasattr(word, "__dict__")


class TestSegment:
    def test_text_stored_stripped(self) -> None:
        segment = Segment(id=1, start=0.0, end=1.0, text="  hello there \n")
        assert segment.text == "hello there"

    def test_defaults(self) -> None:
        segment = Segment(id=1, start=0.0, end=1.0, text="hi")
        assert segment.words == ()
        assert segment.avg_logprob is None
        assert segment.no_speech_prob is None

    @pytest.mark.parametrize(
        ("start", "end"),
        [(-1.0, 0.0), (1.0, -1.0), (3.0, 2.9), (math.inf, math.inf)],
    )
    def test_invalid_times_raise(self, start: float, end: float) -> None:
        with pytest.raises(ValueError, match="segment"):
            Segment(id=1, start=start, end=end, text="hi")

    def test_frozen_and_slotted(self) -> None:
        segment = Segment(id=1, start=0.0, end=1.0, text="hi")
        with pytest.raises(dataclasses.FrozenInstanceError):
            segment.id = 2  # type: ignore[misc]
        assert not hasattr(segment, "__dict__")


class TestTranscript:
    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            Transcript(
                segments=(),
                language=None,
                language_probability=None,
                duration=-0.5,
                source="x.wav",
                engine="mock",
                model="small",
            )

    def test_non_finite_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Transcript(
                segments=(),
                language=None,
                language_probability=None,
                duration=math.nan,
                source="x.wav",
                engine="mock",
                model="small",
            )

    def test_text_joins_segments_with_newline(self) -> None:
        transcript = make_transcript()
        assert transcript.text == "Héllo wörld — 你好 🎙️\nsecond segment"

    def test_word_count_counts_whitespace_words(self) -> None:
        # "Héllo wörld — 你好 🎙️" -> 5 tokens, "second segment" -> 2.
        assert make_transcript().word_count == 7

    def test_is_empty(self) -> None:
        empty = Transcript(
            segments=(),
            language=None,
            language_probability=None,
            duration=0.0,
            source="x.wav",
            engine="mock",
            model="small",
        )
        assert empty.is_empty
        assert empty.text == ""
        assert empty.word_count == 0
        assert not make_transcript().is_empty
        # Segments whose text strips to nothing still count as empty.
        blank = dataclasses.replace(
            empty, segments=(Segment(id=1, start=0.0, end=1.0, text="   "),)
        )
        assert blank.is_empty

    def test_iter_yields_segments_in_order(self) -> None:
        transcript = make_transcript()
        assert tuple(transcript) == transcript.segments
        assert [segment.id for segment in transcript] == [1, 2]

    def test_frozen_and_slotted(self) -> None:
        transcript = make_transcript()
        with pytest.raises(dataclasses.FrozenInstanceError):
            transcript.language = "de"  # type: ignore[misc]
        assert not hasattr(transcript, "__dict__")

    def test_equality_is_structural(self) -> None:
        assert make_transcript() == make_transcript()


class TestSerialisation:
    def test_to_dict_shape(self) -> None:
        data = make_transcript().to_dict()
        assert data["schema_version"] == SCHEMA_VERSION == 1
        assert data["source"] == "clip.wav"
        assert data["engine"] == "mock"
        assert data["model"] == "small"
        assert data["language"] == "en"
        assert data["language_probability"] == 0.93
        assert data["duration"] == 4.75
        assert [segment["id"] for segment in data["segments"]] == [1, 2]
        first_word = data["segments"][0]["words"][0]
        assert first_word == {"start": 0.0, "end": 1.0, "text": " Héllo", "probability": 0.98}

    def test_to_dict_is_json_serialisable(self) -> None:
        json.dumps(make_transcript().to_dict(), ensure_ascii=False)

    def test_round_trip_exact(self) -> None:
        transcript = make_transcript()
        assert Transcript.from_dict(transcript.to_dict()) == transcript

    def test_round_trip_through_json_preserves_unicode(self) -> None:
        transcript = make_transcript()
        for ensure_ascii in (False, True):
            payload = json.dumps(transcript.to_dict(), ensure_ascii=ensure_ascii)
            restored = Transcript.from_dict(json.loads(payload))
            assert restored == transcript
            assert restored.segments[0].text == "Héllo wörld — 你好 🎙️"

    def test_round_trip_none_fields(self) -> None:
        transcript = Transcript(
            segments=(Segment(id=1, start=0.0, end=1.0, text="hi"),),
            language=None,
            language_probability=None,
            duration=1.0,
            source="x.wav",
            engine="mock",
            model="small",
        )
        assert Transcript.from_dict(transcript.to_dict()) == transcript

    def test_from_dict_coerces_integer_times(self) -> None:
        data = make_transcript().to_dict()
        data["duration"] = 5  # JSON writers may emit whole numbers as ints.
        data["segments"][1]["start"] = 2
        restored = Transcript.from_dict(data)
        assert restored.duration == 5.0
        assert isinstance(restored.duration, float)
        assert isinstance(restored.segments[1].start, float)

    @pytest.mark.parametrize("version", [None, 0, 2, "1"])
    def test_from_dict_rejects_bad_schema_version(self, version: object) -> None:
        data = make_transcript().to_dict()
        if version is None:
            del data["schema_version"]
        else:
            data["schema_version"] = version
        with pytest.raises(ValueError, match="schema_version"):
            Transcript.from_dict(data)

    @pytest.mark.parametrize("key", ["segments", "language", "duration", "source"])
    def test_from_dict_missing_key_raises_value_error(self, key: str) -> None:
        data = make_transcript().to_dict()
        del data[key]
        with pytest.raises(ValueError, match=key):
            Transcript.from_dict(data)

    def test_from_dict_missing_segment_key_raises_value_error(self) -> None:
        data = make_transcript().to_dict()
        del data["segments"][0]["end"]
        with pytest.raises(ValueError, match="end"):
            Transcript.from_dict(data)

    def test_from_dict_validates_reconstructed_values(self) -> None:
        data = make_transcript().to_dict()
        data["segments"][0]["start"] = -1.0
        with pytest.raises(ValueError, match="segment"):
            Transcript.from_dict(data)
