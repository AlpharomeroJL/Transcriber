"""Golden tests for timestamp and duration formatting."""

from __future__ import annotations

import math
import re

import pytest

from transcriber.timestamps import format_duration, format_timestamp

SRT_PATTERN = re.compile(r"^\d{2,}:[0-5]\d:[0-5]\d,\d{3}$")
VTT_PATTERN = re.compile(r"^\d{2,}:[0-5]\d:[0-5]\d\.\d{3}$")


class TestSrtStyle:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (0.0005, "00:00:00,001"),  # half-up at the millisecond
            (1.5, "00:00:01,500"),
            (59.9994, "00:00:59,999"),
            (3599.9994, "00:59:59,999"),
            (3600.0, "01:00:00,000"),  # 1h boundary
            (3723.456, "01:02:03,456"),
            (359999.999, "99:59:59,999"),
            (360000.0, "100:00:00,000"),  # 100h: HH field grows
            (363723.456, "101:02:03,456"),
            (-5.0, "00:00:00,000"),  # negative clamps to zero
        ],
    )
    def test_goldens(self, seconds: float, expected: str) -> None:
        assert format_timestamp(seconds, style="srt") == expected

    def test_millisecond_overflow_carries_into_seconds(self) -> None:
        # 59.9994 must round down to 999 ms, never render as 60,000 ms...
        assert format_timestamp(59.9994, style="srt") == "00:00:59,999"
        # ...and 59.9996 must carry into the next minute.
        assert format_timestamp(59.9996, style="srt") == "00:01:00,000"


class TestVttStyle:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00.000"),
            (59.9994, "00:00:59.999"),
            (3600.0, "01:00:00.000"),
            (3723.456, "01:02:03.456"),
            (360000.0, "100:00:00.000"),
            (-1.0, "00:00:00.000"),
        ],
    )
    def test_goldens(self, seconds: float, expected: str) -> None:
        assert format_timestamp(seconds, style="vtt") == expected


class TestClockStyle:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0:00:00"),
            (0.4, "0:00:00"),
            (0.5, "0:00:01"),  # half-up at the second
            (59.4, "0:00:59"),
            (59.5, "0:01:00"),  # carries into the minute
            (3599.4, "0:59:59"),
            (3600.0, "1:00:00"),  # hours unpadded
            (3723.9, "1:02:04"),
            (360000.0, "100:00:00"),
            (-3.0, "0:00:00"),
        ],
    )
    def test_goldens(self, seconds: float, expected: str) -> None:
        assert format_timestamp(seconds, style="clock") == expected


class TestFormatTimestampInvariants:
    # A deterministic sweep of awkward floats near unit boundaries: fields
    # must stay in range (no 60-second minutes, no 1000-millisecond seconds).
    SWEEP = [k * 0.9994 for k in range(0, 130)] + [
        59.999,
        119.9995,
        3599.9996,
        86399.9994,
    ]

    @pytest.mark.parametrize("seconds", SWEEP)
    def test_fields_stay_in_range(self, seconds: float) -> None:
        assert SRT_PATTERN.match(format_timestamp(seconds, style="srt"))
        assert VTT_PATTERN.match(format_timestamp(seconds, style="vtt"))

    def test_srt_and_vtt_differ_only_in_separator(self) -> None:
        for seconds in (0.0, 61.25, 3661.5, 360000.75):
            srt = format_timestamp(seconds, style="srt")
            vtt = format_timestamp(seconds, style="vtt")
            assert srt.replace(",", ".") == vtt

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            format_timestamp(bad, style="srt")

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(ValueError, match="style"):
            format_timestamp(1.0, style="bogus")  # type: ignore[arg-type]


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0.0s"),
            (0.05, "0.1s"),
            (42.5, "42.5s"),
            (59.94, "59.9s"),
            (59.96, "1m 00s"),  # rounds past a minute: carries, never "60.0s"
            (60.0, "1m 00s"),
            (90.0, "1m 30s"),
            (119.6, "2m 00s"),
            (3723.0, "1h 02m 03s"),
            (7200.0, "2h 00m 00s"),
            (359999.5, "100h 00m 00s"),
            (360000.0, "100h 00m 00s"),
            (-2.0, "0.0s"),  # negative clamps to zero
        ],
    )
    def test_goldens(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            format_duration(bad)
