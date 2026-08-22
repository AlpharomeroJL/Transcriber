"""Golden tests for the output formatters and their registry."""

from __future__ import annotations

import json

import pytest

from transcriber.errors import OutputError
from transcriber.formats import FORMATTERS, Formatter, available_formats, get_formatter
from transcriber.model import Segment, Transcript, Word


def make_transcript() -> Transcript:
    """Two segments: unicode text, word timings, and a beyond-one-hour span."""
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
            Segment(id=2, start=3599.5, end=3723.456, text="Ünïcode across the hour"),
        ),
        language="en",
        language_probability=0.93,
        duration=3723.456,
        source="clip.wav",
        engine="mock",
        model="small",
    )


def make_empty_transcript() -> Transcript:
    """A transcript with no segments at all."""
    return Transcript(
        segments=(),
        language=None,
        language_probability=None,
        duration=0.0,
        source="silence.wav",
        engine="mock",
        model="small",
    )


FULL_TXT = "Héllo wörld — 你好 🎙️\nÜnïcode across the hour\n"

FULL_SRT = """\
1
00:00:00,000 --> 00:00:02,500
Héllo wörld — 你好 🎙️

2
00:59:59,500 --> 01:02:03,456
Ünïcode across the hour
"""

FULL_VTT = """\
WEBVTT

00:00:00.000 --> 00:00:02.500
Héllo wörld — 你好 🎙️

00:59:59.500 --> 01:02:03.456
Ünïcode across the hour
"""

FULL_TSV = (
    "start\tend\ttext\n0\t2500\tHéllo wörld — 你好 🎙️\n3599500\t3723456\tÜnïcode across the hour\n"
)

FULL_MD = """\
# Transcript of clip.wav

- Engine: mock
- Model: small
- Language: en
- Duration: 1h 02m 03s

**[0:00:00]** Héllo wörld — 你好 🎙️
**[1:00:00]** Ünïcode across the hour
"""

EMPTY_JSON = """\
{
  "schema_version": 1,
  "source": "silence.wav",
  "engine": "mock",
  "model": "small",
  "language": null,
  "language_probability": null,
  "duration": 0.0,
  "segments": []
}
"""

EMPTY_MD = """\
# Transcript of silence.wav

- Engine: mock
- Model: small
- Language: unknown
- Duration: 0.0s
"""


class TestRegistry:
    def test_available_formats_sorted(self) -> None:
        assert available_formats() == ("json", "md", "srt", "tsv", "txt", "vtt")

    def test_formatters_are_instances_keyed_by_name(self) -> None:
        assert set(FORMATTERS) == set(available_formats())
        for name, formatter in FORMATTERS.items():
            assert isinstance(formatter, Formatter)
            assert formatter.name == name

    @pytest.mark.parametrize(
        ("name", "extension"),
        [
            ("txt", ".txt"),
            ("srt", ".srt"),
            ("vtt", ".vtt"),
            ("json", ".json"),
            ("tsv", ".tsv"),
            ("md", ".md"),
        ],
    )
    def test_extensions_carry_a_leading_dot(self, name: str, extension: str) -> None:
        assert get_formatter(name).extension == extension

    def test_get_formatter_returns_the_registered_instance(self) -> None:
        assert get_formatter("srt") is FORMATTERS["srt"]

    def test_get_formatter_miss_raises_output_error_naming_formats(self) -> None:
        with pytest.raises(OutputError, match="unknown output format 'yaml'") as excinfo:
            get_formatter("yaml")
        assert "json, md, srt, tsv, txt, vtt" in str(excinfo.value)

    def test_formatter_abc_is_not_instantiable(self) -> None:
        with pytest.raises(TypeError):
            Formatter()  # type: ignore[abstract]


class TestFullTranscriptGoldens:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("txt", FULL_TXT),
            ("srt", FULL_SRT),
            ("vtt", FULL_VTT),
            ("tsv", FULL_TSV),
            ("md", FULL_MD),
        ],
    )
    def test_golden(self, name: str, expected: str) -> None:
        assert get_formatter(name).render(make_transcript()) == expected


class TestJsonFormat:
    def test_full_transcript_round_trips(self) -> None:
        transcript = make_transcript()
        output = get_formatter("json").render(transcript)
        assert Transcript.from_dict(json.loads(output)) == transcript

    def test_unicode_is_not_escaped(self) -> None:
        output = get_formatter("json").render(make_transcript())
        assert "你好 🎙️" in output
        assert "\\u" not in output

    def test_two_space_indent_and_schema_version(self) -> None:
        output = get_formatter("json").render(make_transcript())
        assert output.startswith('{\n  "schema_version": 1,\n')

    def test_empty_transcript_golden(self) -> None:
        assert get_formatter("json").render(make_empty_transcript()) == EMPTY_JSON


class TestEmptyTranscript:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("txt", "\n"),
            ("srt", "\n"),
            ("vtt", "WEBVTT\n"),
            ("json", EMPTY_JSON),
            ("tsv", "start\tend\ttext\n"),
            ("md", EMPTY_MD),
        ],
    )
    def test_minimal_valid_output(self, name: str, expected: str) -> None:
        assert get_formatter(name).render(make_empty_transcript()) == expected


class TestNewlineContract:
    @pytest.mark.parametrize("name", ["json", "md", "srt", "tsv", "txt", "vtt"])
    def test_single_trailing_newline_and_unix_line_endings(self, name: str) -> None:
        for transcript in (make_transcript(), make_empty_transcript()):
            output = FORMATTERS[name].render(transcript)
            assert output.endswith("\n")
            assert not output.endswith("\n\n")
            assert "\r" not in output


class TestTsvSafety:
    def test_tabs_and_newlines_in_text_collapse_to_spaces(self) -> None:
        transcript = Transcript(
            segments=(Segment(id=1, start=0.0, end=1.0, text="tab\there\nand\r\nreturn"),),
            language="en",
            language_probability=None,
            duration=1.0,
            source="tricky.wav",
            engine="mock",
            model="small",
        )
        output = get_formatter("tsv").render(transcript)
        lines = output.splitlines()
        assert lines == ["start\tend\ttext", "0\t1000\ttab here and return"]
        assert all(line.count("\t") == 2 for line in lines)
