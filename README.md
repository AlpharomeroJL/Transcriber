# Transcriber

Local-first, privacy-preserving transcription of audio and video, powered by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Media in, text out — entirely on
your machine. No audio ever leaves it.

[![CI](https://github.com/AlpharomeroJL/Transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/AlpharomeroJL/Transcriber/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Features

- **Audio and video in** — wav, mp3, m4a, flac, ogg, opus, aac and more, plus video containers
  (mp4, mkv, mov, webm, ...) whose audio track is demuxed automatically.
- **Six output formats** — txt, srt, vtt, json, tsv, md — and any combination of them in one run.
- **Batch processing** — files, directories, or both (optionally recursive), with per-file
  failure isolation: one broken file never sinks the batch.
- **99-language transcription** with automatic language detection, plus a `translate` task that
  renders any source language as English text.
- **Word-level timestamps** on request (`--word-timestamps`).
- **Voice-activity detection on by default** — long silences are filtered out before decoding,
  because Whisper models are known to hallucinate text on silence.
- **Safe outputs** — atomic writes, no silent clobbering (`--overwrite` is explicit), and
  output-name collisions are detected and reported instead of overwritten.
- **A typed Python API** (`import transcriber`) mirroring everything the CLI can do, with
  progress events for embedding in your own tools.
- **Deterministic offline mock engine** (`--engine mock`) for dry runs and tests — no model
  download, no GPU, byte-identical results.
- **Scripting-friendly CLI** — configuration via flags or `TRANSCRIBER_*` environment
  variables, meaningful exit codes, and transcript content on stdout only under `--stdout`.

Privacy note: transcription runs entirely locally. The only network access ever performed is a
one-time download of the selected model's weights from Hugging Face on first use (cached under
`~/.cache/huggingface` and reused offline thereafter). Your audio is never uploaded anywhere.

## Installation

Transcriber is not on PyPI yet; install it from source. Python 3.10 or newer is required:

```sh
git clone https://github.com/AlpharomeroJL/Transcriber.git
cd Transcriber
pip install .
```

Prefer an isolated install of just the `transcriber` command? Use [pipx](https://pipx.pypa.io):

```sh
pipx install .
```

You do **not** need ffmpeg installed: audio and video are decoded through
[PyAV](https://github.com/PyAV-Org/PyAV), which bundles the FFmpeg libraries inside the Python
package that faster-whisper depends on.

The distribution is named `transcriber-cli`; it installs the `transcriber` command and the
`transcriber` Python package.

## Quickstart

```sh
# One file -> talk.txt written next to it
transcriber transcribe talk.mp3

# A directory tree -> SRT + VTT subtitles for every media file, into subs/
transcriber transcribe lectures/ -r -f srt -f vtt -o subs/

# Pipe a transcript through other tools: --stdout prints content only,
# progress and logs stay on stderr
transcriber transcribe interview.wav --stdout -f txt | wc -w

# Pick a model and pin the language instead of auto-detecting
transcriber transcribe talk.mp3 -m large-v3-turbo -l de

# Any source language -> English text
transcriber transcribe entrevista.mp3 --task translate

# Configure via the environment instead of flags
TRANSCRIBER_MODEL=medium TRANSCRIBER_VAD=false transcriber transcribe noisy.wav

# Dry-run the whole pipeline offline with the deterministic mock engine
transcriber transcribe talk.mp3 --engine mock --stdout
```

The CLI has four subcommands:

| Command | What it does |
| --- | --- |
| `transcriber transcribe` | Transcribe media files and/or directories to text outputs. |
| `transcriber formats` | List the available output formats. |
| `transcriber models` | List the standard Whisper model aliases. |
| `transcriber info` | Show version, environment, and the effective configuration. |

Exit codes are part of the contract: `0` — every file transcribed; `1` — at least one file
failed (the rest were still processed); `2` — the request itself was invalid (bad flag, bad
configuration, missing input).

## Choosing a model

The default model is `small` — the accepted quality/speed sweet spot on CPU. Figures below are
approximate; speed is relative to `large-v3` on the same hardware.

| Model | Parameters | RAM (approx.) | Speed | Quality |
| --- | --- | --- | --- | --- |
| `tiny` / `tiny.en` | 39M | ~1 GB | ~10x | Lowest; fine for rough drafts. |
| `base` / `base.en` | 74M | ~1 GB | ~7x | Basic. |
| `small` / `small.en` | 244M | ~2 GB | ~4x | **Default.** Best balance on CPU. |
| `medium` / `medium.en` | 769M | ~5 GB | ~2x | High. |
| `large-v3` | 1.5B | ~10 GB | 1x | Highest accuracy. |
| `large-v3-turbo` (`turbo`) | 809M | ~6 GB | ~8x | Near `large-v3`; distilled decoder. |
| `distil-small.en` ... `distil-large-v3` | 166M-756M | ~1-6 GB | fast | English-only distillations. |

Guidance:

- **On CPU**, stay with `small` (or `base` when speed matters more than accuracy).
- **On GPU**, use `large-v3-turbo` — about 4x faster than `large-v3` at nearly the same
  quality — or `large-v3` when accuracy is paramount.
- **English-only audio?** The `.en` and `distil-*` variants are faster at the same quality;
  try `distil-large-v3` for the best English speed/quality ratio.
- Any Hugging Face CTranslate2 model repo id or local model directory path is also accepted
  by `-m/--model`; run `transcriber models` for the live table.

## Output formats

| Format | Extension | Description |
| --- | --- | --- |
| `txt` | `.txt` | Plain text, one segment per line. |
| `srt` | `.srt` | SubRip subtitles: numbered cues with `HH:MM:SS,mmm` timings. |
| `vtt` | `.vtt` | WebVTT subtitles: web-native cues with `HH:MM:SS.mmm` timings. |
| `json` | `.json` | Full transcript data (segments, words, metadata) as versioned, pretty-printed JSON. |
| `tsv` | `.tsv` | Tab-separated start/end/text rows (openai-whisper convention, integer milliseconds). |
| `md` | `.md` | Markdown reading view: title, metadata bullets, timestamped paragraphs. |

Rendering is deterministic — the same transcript always yields byte-identical output — and
every write is atomic, so a failure never leaves a truncated file behind. Repeat `-f/--format`
to write several formats per input in one run.

## Python API

The whole CLI surface is available as a typed library (`py.typed` ships with the package):

```python
from pathlib import Path

from transcriber import Transcriber, TranscriberConfig
from transcriber.pipeline import PipelineEvent

config = TranscriberConfig(model="small", language="en", word_timestamps=True)
app = Transcriber(config)

# Single file -> in-memory Transcript (segments, words, language, duration)
transcript = app.transcribe_file(Path("talk.mp3"))
print(transcript.language, f"{transcript.duration:.1f}s", transcript.word_count, "words")
print(transcript.text)


# Batch -> rendered files on disk, with progress events as each file starts
# and finishes; failures are isolated per file, not raised
def on_event(event: PipelineEvent) -> None:
    print(f"{event.kind}: {event.path.name} ({event.index}/{event.total})")


batch = app.transcribe_batch(
    [Path("lectures/")],
    formats=("srt", "vtt"),
    output_dir=Path("subs"),
    recursive=True,
    on_event=on_event,
)
print(f"{batch.ok_count} ok, {batch.error_count} failed")
raise SystemExit(batch.exit_code)  # same 0/1 contract as the CLI
```

Inject an engine for tests — `Transcriber(config, engine=MockEngine())` — and every pipeline
guarantee (atomic writes, collision detection, batch isolation) is exercised without touching
a speech model. All failures derive from `transcriber.TranscriberError`; configuration,
input, and output problems have their own subclasses.

## Configuration

Settings are resolved in precedence order: library defaults, then `TRANSCRIBER_*` environment
variables, then command-line flags. Every config field maps to one variable:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRANSCRIBER_MODEL` | `small` | Model alias, Hugging Face CTranslate2 repo id, or local model path. |
| `TRANSCRIBER_ENGINE` | `faster-whisper` | Transcription engine: `faster-whisper` or `mock`. |
| `TRANSCRIBER_DEVICE` | `auto` | Inference device: `auto`, `cpu`, or `cuda`. |
| `TRANSCRIBER_COMPUTE_TYPE` | `auto` | Quantization: `auto`, `default`, `int8`, `int8_float16`, `int8_float32`, `int16`, `float16`, `float32`, `bfloat16`. |
| `TRANSCRIBER_LANGUAGE` | unset (auto-detect) | ISO 639 language code of the audio, e.g. `en`. |
| `TRANSCRIBER_TASK` | `transcribe` | `transcribe` (same-language text) or `translate` (to English). |
| `TRANSCRIBER_VAD` | `true` | Filter long silences with voice-activity detection. |
| `TRANSCRIBER_WORD_TIMESTAMPS` | `false` | Also produce per-word timestamps. |
| `TRANSCRIBER_BEAM_SIZE` | `5` | Beam width for decoding (>= 1). |
| `TRANSCRIBER_TEMPERATURE` | `0.0` | Sampling temperature, 0.0 (deterministic) to 1.0. |
| `TRANSCRIBER_INITIAL_PROMPT` | unset | Text to prime the decoder with (names, jargon, spellings). |

A variable that is unset or set to the empty string leaves the default in place. Booleans
accept `1/0/true/false/yes/no` in any case. An unparsable or invalid value fails the run with
exit code 2 and an error naming the variable. `transcriber info` prints the effective
configuration after the environment has been applied.

## How it works

- **Engine.** Transcription runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (SYSTRAN), a CTranslate2 reimplementation of OpenAI's Whisper — around 4x faster than the
  reference implementation at the same accuracy, with int8 quantization on CPU and GPU and no
  PyTorch dependency. Media decoding comes through PyAV's bundled FFmpeg libraries, so there is
  no system ffmpeg requirement.
- **VAD by default.** Whisper models hallucinate plausible-looking text when fed long stretches
  of silence. The standard mitigation — Silero voice-activity detection filtering the audio
  before decoding — is enabled by default and can be switched off with `--no-vad`.
- **Pipeline discipline.** Bad *requests* (unknown format, missing input path) fail fast before
  anything is transcribed; bad *files* are isolated and reported per file while the batch
  continues. Outputs are rendered fully and written atomically; existing files are never
  overwritten without `--overwrite`; two inputs that would claim the same output path are
  detected and the later one fails with an explanation.
- **Lazy and lightweight.** `import transcriber` imports no speech backend; `faster_whisper` is
  loaded only when the real engine is constructed, and loaded models are cached per
  (model, device, compute type) so batches reuse one model. The registered `mock` engine makes
  the entire pipeline runnable offline.

## Development

```sh
git clone https://github.com/AlpharomeroJL/Transcriber.git
cd Transcriber
python3 -m venv .venv
make install   # editable install with dev extras, into .venv
```

| Target | What it runs |
| --- | --- |
| `make lint` | `ruff check` and `ruff format --check`. |
| `make type` | `mypy` (strict) over `src`. |
| `make test` | The offline, deterministic pytest suite. |
| `make cov` | Tests with coverage, floor 85%. |
| `make check` | lint + type + test — the local gate. |
| `make build` | sdist + wheel. |
| `make gauntlet` | check + cov + build + clean-venv install smoke. |

CI (`.github/workflows/ci.yml`) runs the same lint and type gates, the test matrix on Python
3.10-3.13, and a package build. Documentation is checked mechanically:
`.venv/bin/python scripts/check_docs.py` fails whenever this README, `CHANGELOG.md`, or
`CONTRIBUTING.md` drifts from the code (CLI subcommands, output formats, environment
variables, version). See [CONTRIBUTING.md](CONTRIBUTING.md) for style and testing rules.

This repository was built to completion by an in-repo **graph loop**: the work was declared as
a dependency graph of gated nodes and executed wave by wave by a stdlib-only scheduler, with
each node's acceptance gate recorded as evidence. The machinery, plan, and per-node state live
in [`.graphloop/`](.graphloop/README.md).

## License

[MIT](LICENSE) — © 2026 Josef D. Long.
