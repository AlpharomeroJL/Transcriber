# Contributing to Transcriber

Thanks for helping out. This project holds itself to a production bar — typed, linted,
tested, and documented, with every gate mechanical — and contributions are expected to clear
the same bar. The rules below are short; the tooling enforces most of them for you.

## Development setup

```sh
git clone https://github.com/AlpharomeroJL/Transcriber.git
cd Transcriber
python3 -m venv .venv
make install
```

`make install` runs `.venv/bin/pip install -e ".[dev]"` — an editable install of the package
plus the dev toolchain (ruff, mypy, pytest, pytest-cov, build, twine). Every make target
addresses tools through `.venv/bin`, so you never need to activate the virtualenv.

Python 3.10+ is required. No system ffmpeg is needed (PyAV bundles the FFmpeg libraries), and
no model download is needed for development — the test suite runs entirely offline.

## Everyday commands

| Target | What it runs |
| --- | --- |
| `make lint` | `ruff check .` and `ruff format --check .` |
| `make type` | `mypy` (strict mode) over `src` |
| `make test` | The pytest suite |
| `make cov` | Tests with coverage; the floor is 85% and the build fails below it |
| `make check` | lint + type + test — run this before every commit |
| `make build` | sdist + wheel via `python -m build` |
| `make gauntlet` | check + cov + build + a clean-venv install smoke — the full pre-release battery |

## Style

- **Formatting and linting**: PEP 8 via ruff (line length 100, rule set in `pyproject.toml`).
  Auto-format with `.venv/bin/ruff format .`; `make lint` must pass clean with no suppressions
  added casually — prefer fixing the code to silencing the rule.
- **Types**: mypy runs in strict mode. All code is fully annotated, including tests' helper
  functions; `Any` is a last resort reserved for genuinely untyped third-party boundaries.
- **Docstrings** on every public module, class, and function, in the imperative style used
  throughout `src/transcriber`.
- **Match the existing idioms**: frozen slotted dataclasses for values, lazy imports for
  heavyweight backends, errors raised from the `transcriber.errors` taxonomy (never bare
  `Exception`), stdout reserved for content and stderr for diagnostics in the CLI.

## Tests

- Code ships **with its tests in the same change**. A behavior change without a test that
  fails beforehand is incomplete.
- Tests are **offline and deterministic**: no network, no model downloads, no dependence on
  wall-clock time or unseeded randomness. CI must produce the same result every run.
- Fake external engines **by injection** — use `MockEngine`, or pass a fake module/engine
  through the seams the code already provides (`Transcriber(config, engine=...)`,
  the engine's injectable backend) — never by monkey-patching the internals of third-party
  packages at a distance.
- Keep the coverage floor honest: `make cov` fails under 85%. New modules should arrive well
  above it.

## Gates and CI

- The local gate is `make check` (lint + type + test). It must pass with a true zero exit
  before a change lands; `make gauntlet` adds coverage, the package build, and an install
  smoke in a clean venv.
- CI (`.github/workflows/ci.yml`) runs lint and type checks, the test matrix on Python
  3.10–3.13, and a package build on every push and pull request. Green CI is required —
  the badge on the README is the public face of it.
- Documentation is gated mechanically: `.venv/bin/python scripts/check_docs.py` verifies that
  the README covers every CLI subcommand, output format, and `TRANSCRIBER_*` environment
  variable, and that `CHANGELOG.md` records the current version. Touching the CLI surface,
  formats, config, or the docs themselves? Run it.

## Commits

- Small and focused: one logical change per commit, with `make check` passing at each one.
- Subject line in the imperative mood ("Add VTT formatter", not "Added"/"Adds"), at most
  about 72 characters; use the body to explain *why*, not to restate the diff.
- User-visible changes get a `CHANGELOG.md` entry (Keep a Changelog format) in the same
  commit.
- Do not commit generated artifacts, caches, or virtualenvs; `.gitignore` already covers the
  usual suspects.
