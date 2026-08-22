#!/usr/bin/env python3
"""Mechanical documentation checks — keep the docs honest against the code.

Run from anywhere (paths are resolved relative to this file):

    .venv/bin/python scripts/check_docs.py

The script introspects the *working tree* package (``src/`` is put first on
``sys.path``) and asserts that:

- ``README.md`` exists and mentions every CLI subcommand registered on the
  ``transcriber`` click group, every available output format name, every
  ``TRANSCRIBER_*`` environment variable derived from the config fields, and
  the CI workflow path ``.github/workflows/ci.yml``;
- ``CHANGELOG.md`` exists and contains the current ``transcriber.__version__``;
- ``CONTRIBUTING.md`` exists and mentions ``make check``.

On failure it lists everything that is missing on stderr and exits nonzero,
so docs drift fails gates the same way broken code does. Only the standard
library is imported directly; the project package (and thus its declared
dependencies) is imported for introspection.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Path of the CI workflow the README must reference.
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"

# Check the working-tree sources, not whatever happens to be installed.
sys.path.insert(0, str(REPO_ROOT / "src"))


def _cli_subcommands() -> tuple[str, ...]:
    """Names of every subcommand registered on the ``transcriber`` click group."""
    from transcriber.cli import main as cli_group

    return tuple(sorted(cli_group.commands))


def _format_names() -> tuple[str, ...]:
    """Every registered output format name."""
    from transcriber.formats import available_formats

    return available_formats()


def _env_vars() -> tuple[str, ...]:
    """Every ``TRANSCRIBER_*`` environment variable defined by the config fields.

    Mirrors the naming rule documented on :func:`transcriber.config.from_env`:
    one variable per :class:`~transcriber.config.TranscriberConfig` field,
    ``TRANSCRIBER_`` plus the upper-cased field name.
    """
    from transcriber.config import TranscriberConfig

    return tuple(
        sorted(
            f"TRANSCRIBER_{field.name.upper()}" for field in dataclasses.fields(TranscriberConfig)
        )
    )


def _version() -> str:
    """The current package version."""
    import transcriber

    return transcriber.__version__


def _mentions(text: str, needle: str) -> bool:
    """Whether *needle* occurs in *text* as a standalone token.

    The needle must not be flanked by word characters or hyphens, so the
    subcommand ``transcribe`` is not satisfied by the word ``transcriber``,
    while ``transcribe`` inside ```` `transcriber transcribe talk.mp3` ````
    still counts.
    """
    return re.search(rf"(?<![\w-]){re.escape(needle)}(?![\w-])", text) is not None


def _read(path: Path, problems: list[str]) -> str | None:
    """Return *path*'s text, or record its absence in *problems*."""
    if not path.is_file():
        problems.append(f"{path.relative_to(REPO_ROOT)} is missing")
        return None
    return path.read_text(encoding="utf-8")


def _check_readme(problems: list[str]) -> None:
    """Check README.md against the introspected CLI, formats, and config."""
    text = _read(REPO_ROOT / "README.md", problems)
    if text is None:
        return
    for subcommand in _cli_subcommands():
        if not _mentions(text, subcommand):
            problems.append(f"README.md does not mention CLI subcommand {subcommand!r}")
    for format_name in _format_names():
        if not _mentions(text, format_name):
            problems.append(f"README.md does not mention output format {format_name!r}")
    for var in _env_vars():
        if not _mentions(text, var):
            problems.append(f"README.md does not mention environment variable {var}")
    if CI_WORKFLOW_PATH not in text:
        problems.append(f"README.md does not mention the CI workflow path {CI_WORKFLOW_PATH}")


def _check_changelog(problems: list[str]) -> None:
    """Check that CHANGELOG.md records the current package version."""
    text = _read(REPO_ROOT / "CHANGELOG.md", problems)
    if text is None:
        return
    version = _version()
    if version not in text:
        problems.append(f"CHANGELOG.md does not contain the current version {version}")


def _check_contributing(problems: list[str]) -> None:
    """Check that CONTRIBUTING.md exists and points at the local gate."""
    text = _read(REPO_ROOT / "CONTRIBUTING.md", problems)
    if text is None:
        return
    if "make check" not in text:
        problems.append("CONTRIBUTING.md does not mention 'make check'")


def main() -> int:
    """Run every check; report and return 1 when anything is missing."""
    problems: list[str] = []
    _check_readme(problems)
    _check_changelog(problems)
    _check_contributing(problems)
    if problems:
        print("documentation check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        "documentation check passed: "
        f"{len(_cli_subcommands())} subcommands, {len(_format_names())} formats, "
        f"{len(_env_vars())} environment variables, version {_version()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
