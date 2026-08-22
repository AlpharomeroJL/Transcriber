# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-22

### Added

- `transcriber` CLI with `transcribe`, `formats`, `models`, and `info` subcommands; exit-code
  contract (0 = all files transcribed, 1 = at least one per-file failure, 2 = usage or
  configuration error); transcript content on stdout only under `--stdout`, progress and
  diagnostics on stderr.
- Batch transcription of files and directories (optionally recursive) with per-file failure
  isolation, deterministic processing order, atomic UTF-8 output writes, output-collision
  detection, and explicit `--overwrite`.
- Six output formats — `txt`, `srt`, `vtt`, `json` (versioned schema), `tsv`, and `md` — with
  deterministic byte-identical rendering, writable in any combination per run.
- Production engine backed by faster-whisper / CTranslate2 with lazy backend import and model
  caching per (model, device, compute type); deterministic offline `mock` engine registered as
  a first-class engine for tests and dry runs.
- Silero voice-activity detection on by default (Whisper hallucinates on silence), language
  auto-detection across 99 languages, `translate` task, word-level timestamps, and beam size /
  temperature / initial prompt decoding controls.
- Immutable, validated `TranscriberConfig` resolved from library defaults, `TRANSCRIBER_*`
  environment variables, and CLI flags, in that precedence order, with errors naming the
  offending field or variable.
- Typed public Python API (`Transcriber` facade with batch results and progress events,
  `Transcript`/`Segment`/`Word` data model with versioned dict/JSON round-tripping, engine and
  formatter registries, `TranscriberError` taxonomy), shipping a `py.typed` marker.
- Packaging as the `transcriber-cli` distribution (hatchling, Python 3.10–3.13), ruff linting,
  strict mypy, an offline deterministic pytest suite with an 85% coverage floor, GitHub
  Actions CI (lint / type / test matrix / build), and a mechanical documentation checker
  (`scripts/check_docs.py`).
