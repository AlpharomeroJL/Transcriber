"""Tests for media input discovery and validation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from transcriber.audio import SUPPORTED_EXTENSIONS, collect_inputs, is_supported
from transcriber.errors import AudioNotFoundError, InputError, UnsupportedFormatError

AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac"}
AUDIO |= {".wma", ".aiff", ".aif", ".mka"}
VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}


def make_file(path: Path) -> Path:
    """Create an empty file (discovery never reads contents, only names)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


class TestSupportedExtensions:
    def test_exact_contract(self) -> None:
        # The supported set is part of the public contract: pin it exactly.
        assert frozenset(AUDIO | VIDEO) == SUPPORTED_EXTENSIONS

    def test_entries_are_normalised(self) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")
            assert len(ext) > 1


class TestIsSupported:
    @pytest.mark.parametrize(
        "name",
        ["talk.wav", "TALK.WAV", "clip.Mp3", "video.MKV", "pod.opus", "show.webm"],
    )
    def test_supported_any_case(self, name: str) -> None:
        assert is_supported(Path(name))

    @pytest.mark.parametrize(
        "name",
        ["notes.txt", "archive.tar.gz", "noext", "song.wav.bak", "image.png"],
    )
    def test_unsupported(self, name: str) -> None:
        assert not is_supported(Path(name))

    def test_does_not_require_the_file_to_exist(self) -> None:
        assert is_supported(Path("/definitely/not/there/talk.flac"))


class TestExplicitFiles:
    def test_single_supported_file(self, tmp_path: Path) -> None:
        media = make_file(tmp_path / "talk.wav")
        assert collect_inputs([media]) == [media.resolve()]

    def test_uppercase_suffix_accepted(self, tmp_path: Path) -> None:
        media = make_file(tmp_path / "TALK.WAV")
        assert collect_inputs([media]) == [media.resolve()]

    def test_hidden_file_allowed_when_named_explicitly(self, tmp_path: Path) -> None:
        hidden = make_file(tmp_path / ".secret.mp3")
        assert collect_inputs([hidden]) == [hidden.resolve()]

    def test_unsupported_suffix_raises_listing_formats(self, tmp_path: Path) -> None:
        bogus = make_file(tmp_path / "notes.txt")
        with pytest.raises(UnsupportedFormatError) as excinfo:
            collect_inputs([bogus])
        message = str(excinfo.value)
        assert str(bogus) in message
        assert "'.txt'" in message
        for listed in (".wav", ".opus", ".mkv"):
            assert listed in message

    def test_extensionless_file_raises(self, tmp_path: Path) -> None:
        bogus = make_file(tmp_path / "README")
        with pytest.raises(UnsupportedFormatError, match="README"):
            collect_inputs([bogus])

    def test_missing_path_raises_naming_it(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.wav"
        with pytest.raises(AudioNotFoundError, match=re.escape(str(missing))):
            collect_inputs([missing])

    def test_errors_share_the_input_error_taxonomy(self) -> None:
        assert issubclass(AudioNotFoundError, InputError)
        assert issubclass(UnsupportedFormatError, InputError)


class TestDirectoryScans:
    def test_scan_filters_and_sorts(self, tmp_path: Path) -> None:
        make_file(tmp_path / "c.wav")
        make_file(tmp_path / "a.mp3")
        make_file(tmp_path / "b.flac")
        make_file(tmp_path / "notes.txt")
        make_file(tmp_path / "cover.png")
        result = collect_inputs([tmp_path])
        assert result == [
            (tmp_path / "a.mp3").resolve(),
            (tmp_path / "b.flac").resolve(),
            (tmp_path / "c.wav").resolve(),
        ]

    def test_non_recursive_ignores_subdirectories(self, tmp_path: Path) -> None:
        top = make_file(tmp_path / "top.wav")
        make_file(tmp_path / "sub" / "nested.wav")
        assert collect_inputs([tmp_path]) == [top.resolve()]

    def test_recursive_descends_and_sorts(self, tmp_path: Path) -> None:
        make_file(tmp_path / "sub" / "deep" / "z.ogg")
        make_file(tmp_path / "sub" / "b.mp4")
        make_file(tmp_path / "a.wav")
        make_file(tmp_path / "sub" / "skip.srt")
        result = collect_inputs([tmp_path], recursive=True)
        assert result == [
            (tmp_path / "a.wav").resolve(),
            (tmp_path / "sub" / "b.mp4").resolve(),
            (tmp_path / "sub" / "deep" / "z.ogg").resolve(),
        ]

    def test_scan_skips_hidden_files(self, tmp_path: Path) -> None:
        visible = make_file(tmp_path / "talk.wav")
        make_file(tmp_path / ".hidden.wav")
        assert collect_inputs([tmp_path]) == [visible.resolve()]

    def test_recursive_scan_skips_hidden_directories(self, tmp_path: Path) -> None:
        visible = make_file(tmp_path / "sub" / "talk.wav")
        make_file(tmp_path / ".cache" / "stale.wav")
        make_file(tmp_path / "sub" / ".old.mp3")
        assert collect_inputs([tmp_path], recursive=True) == [visible.resolve()]

    def test_explicitly_named_hidden_directory_is_scanned(self, tmp_path: Path) -> None:
        hidden_dir = tmp_path / ".stash"
        media = make_file(hidden_dir / "talk.wav")
        assert collect_inputs([hidden_dir]) == [media.resolve()]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AudioNotFoundError, match="no supported media found"):
            collect_inputs([tmp_path])

    def test_directory_with_only_unsupported_files_raises(self, tmp_path: Path) -> None:
        make_file(tmp_path / "notes.txt")
        make_file(tmp_path / "slides.pdf")
        with pytest.raises(AudioNotFoundError, match=re.escape(str(tmp_path))):
            collect_inputs([tmp_path])


class TestBatches:
    def test_directory_and_file_inside_it_deduplicate(self, tmp_path: Path) -> None:
        media = make_file(tmp_path / "talk.wav")
        make_file(tmp_path / "other.mp3")
        result = collect_inputs([tmp_path, media])
        assert result == [(tmp_path / "other.mp3").resolve(), media.resolve()]

    def test_same_file_twice_deduplicates(self, tmp_path: Path) -> None:
        media = make_file(tmp_path / "talk.wav")
        assert collect_inputs([media, media]) == [media.resolve()]

    def test_unnormalised_spelling_of_same_file_deduplicates(self, tmp_path: Path) -> None:
        media = make_file(tmp_path / "talk.wav")
        (tmp_path / "sub").mkdir()
        detour = tmp_path / "sub" / ".." / "talk.wav"
        assert collect_inputs([media, detour]) == [media.resolve()]

    def test_mixed_files_and_directories_sort_globally(self, tmp_path: Path) -> None:
        loose = make_file(tmp_path / "z-loose.wav")
        folder = tmp_path / "batch"
        make_file(folder / "a.mp3")
        result = collect_inputs([loose, folder])
        assert result == [(folder / "a.mp3").resolve(), loose.resolve()]

    def test_result_is_independent_of_argument_order(self, tmp_path: Path) -> None:
        first = make_file(tmp_path / "a.wav")
        second = make_file(tmp_path / "b.wav")
        assert collect_inputs([second, first]) == collect_inputs([first, second])

    def test_results_are_resolved_absolute_paths(self, tmp_path: Path) -> None:
        make_file(tmp_path / "talk.wav")
        result = collect_inputs([tmp_path])
        assert all(path.is_absolute() for path in result)
        assert all(path == path.resolve() for path in result)

    def test_one_missing_argument_fails_the_batch(self, tmp_path: Path) -> None:
        make_file(tmp_path / "talk.wav")
        with pytest.raises(AudioNotFoundError, match=re.escape("missing.wav")):
            collect_inputs([tmp_path, tmp_path / "missing.wav"])

    def test_one_unsupported_explicit_file_fails_the_batch(self, tmp_path: Path) -> None:
        make_file(tmp_path / "talk.wav")
        bogus = make_file(tmp_path / "notes.txt")
        with pytest.raises(UnsupportedFormatError):
            collect_inputs([tmp_path, bogus])

    def test_no_arguments_raises(self) -> None:
        with pytest.raises(AudioNotFoundError, match="no supported media found"):
            collect_inputs([])
