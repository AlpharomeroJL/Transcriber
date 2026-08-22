# Transcriber 1.0 — product plan and research notes

Date: 2026-08-22. Campaign: `transcriber-1.0` (see `graph.json`).

## What this project is

**Transcriber** — a local-first, privacy-preserving transcription tool for audio and video:
a polished CLI (`transcriber`) plus a typed Python library (`import transcriber`). Media in,
text out (txt / srt / vtt / json / tsv / md), with batch processing, language auto-detection,
word timestamps, and voice-activity filtering. No audio ever leaves the machine.

The repository started empty (LICENSE only, MIT, © 2026 Josef D. Long); the scope above is the
smallest product that honestly earns "production state": typed and linted code, offline
deterministic tests with a coverage floor, end-to-end integration tests, CI, real packaging
proven by a clean-venv install smoke, and documentation checked mechanically against the code.

## Engine research (2026-08)

- **faster-whisper 1.2.1** (SYSTRAN, CTranslate2 backend) remains the production standard for
  Python: ~4× faster than reference Whisper, int8 quantization for CPU/GPU, built-in Silero
  VAD, word timestamps, `BatchedInferencePipeline`, supports `large-v3`, `large-v3-turbo`
  ("turbo", 809M distilled decoder, ~4× faster than large-v3) and `distil-*` checkpoints.
  Stable, not stalled (v1.2.1 released 2025-10-31). No torch dependency.
- **whisper.cpp** wins on Apple Silicon (Metal) but is a C++ binary, wrong fit for a Python
  library; **NVIDIA Parakeet-TDT** tops English-only leaderboards but is English-only and
  NeMo-shaped. Whisper via faster-whisper is the right default for a 99-language local tool.
- **VAD on by default**: Whisper hallucinates on long silences; Silero VAD filtering is the
  standard mitigation and is one flag in faster-whisper.
- Default model `small` (~244M, ~2 GB RAM, ~3.4% English WER): the accepted quality/speed
  sweet spot for CPU boxes; `turbo`/`large-v3` recommended on GPU; `.en`/`distil` variants for
  English-only speed.

## Architecture

```
cli.py ──► pipeline.py (Transcriber facade, batch, events, atomic writes)
              │                │
              ▼                ▼
        engines/ (ABC)    formats/ (ABC + registry)
        ├─ faster_whisper  ├─ txt srt vtt json tsv md
        └─ mock (offline)  │
              ▲            ▲
        config.py ── model.py (Word/Segment/Transcript) ── timestamps.py
        errors.py          audio.py (input discovery)
```

Key decisions:

- **Engine abstraction with lazy imports**: `faster_whisper` is imported only when the engine
  is constructed, so `import transcriber`, the mock engine, and the whole test suite work
  without model downloads or heavyweight imports. The mock engine is a first-class registered
  engine (`--engine mock`), which makes the CLI end-to-end testable offline and gives users a
  dry-run mode.
- **Tests are offline and deterministic**: the faster-whisper engine is tested against an
  injected fake module; a real-model smoke (`scripts/smoke_real.py`, downloads `tiny`) is
  env-gated, best-effort, and never part of CI or gates.
- **Stack**: Python ≥3.10, click 8 (CLI), rich (progress/tables), hatchling (build),
  ruff + mypy strict + pytest/pytest-cov (QA), coverage floor 85%, GitHub Actions CI
  (lint / type / test matrix 3.10–3.13 / build), distribution name `transcriber-cli`
  (import name `transcriber`).
- **CLI contract**: exit 0 = all files transcribed, 1 = at least one per-file failure
  (batch isolation), 2 = usage/config error. stdout carries transcript content only under
  `--stdout`; progress and logs go to stderr so pipes stay clean.

## How the work is scheduled

By the graph loop in this directory — see `README.md` for the protocol. 15 nodes; the wave
composition is computed by `graphloop.py frontier`, concurrency is legal only where owned
paths are disjoint, and every node closes with a true-exit gate recorded in `state.json`.
Expected shape: bootstrap → scaffold → {core, confer, audio} → {formats, engines} →
pipeline → cli → {docs, ci, integ} → pack → review (read-only, adversarial) → gauntlet.
