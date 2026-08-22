"""Shared pytest fixtures for the transcriber test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Absolute path of the repository root (the directory holding pyproject.toml)."""
    return Path(__file__).resolve().parent.parent
