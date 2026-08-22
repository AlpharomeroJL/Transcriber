"""Typed, validated configuration for transcription runs.

:class:`TranscriberConfig` is an immutable value object holding every knob a
transcription run understands. Instances validate themselves on construction
and raise :class:`~transcriber.errors.ConfigError` naming the offending value,
so a config that exists is a config that is usable.

Configs are built three ways, from lowest to highest precedence:

- ``TranscriberConfig()`` — the library defaults;
- :func:`from_env` — the defaults overlaid with ``TRANSCRIBER_*`` environment
  variables;
- :meth:`TranscriberConfig.replace` — a copy with explicit overrides, which is
  how the CLI applies command-line flags on top of the environment.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Mapping
from typing import Any

from transcriber.errors import ConfigError

__all__ = [
    "COMPUTE_TYPES",
    "KNOWN_MODELS",
    "TASKS",
    "TranscriberConfig",
    "from_env",
    "is_known_model",
]

#: Standard Whisper model aliases resolvable by the default engine. Unknown
#: names are still permitted in :attr:`TranscriberConfig.model` — they may be
#: Hugging Face repo ids or local model directories — but the CLI uses
#: :func:`is_known_model` to hint at likely typos.
KNOWN_MODELS: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
)

#: Compute (quantization) types accepted by faster-whisper / CTranslate2.
COMPUTE_TYPES: tuple[str, ...] = (
    "auto",
    "default",
    "int8",
    "int8_float16",
    "int8_float32",
    "int16",
    "float16",
    "float32",
    "bfloat16",
)

#: Tasks a Whisper model can perform.
TASKS: tuple[str, ...] = ("transcribe", "translate")

_LANGUAGE_RE = re.compile(r"[a-z]{2,3}")
_ENV_PREFIX = "TRANSCRIBER_"
_BOOL_FIELDS = frozenset({"vad", "word_timestamps"})
_INT_FIELDS = frozenset({"beam_size"})
_FLOAT_FIELDS = frozenset({"temperature"})
_TRUE_TOKENS = frozenset({"1", "true", "yes"})
_FALSE_TOKENS = frozenset({"0", "false", "no"})


def is_known_model(name: str) -> bool:
    """Return whether *name* is one of the standard aliases in :data:`KNOWN_MODELS`.

    Unknown names are valid configuration (Hugging Face repo ids, local model
    paths); this helper exists so the CLI can warn about likely typos without
    rejecting them.
    """
    return name in KNOWN_MODELS


@dataclasses.dataclass(frozen=True, slots=True)
class TranscriberConfig:
    """Immutable, validated settings for a transcription run.

    Every field is validated in :meth:`__post_init__`; invalid values raise
    :class:`~transcriber.errors.ConfigError` naming the offending field and
    value. Instances are frozen (and hashable), so they can be shared freely
    between the CLI, the pipeline, and engines.

    Attributes:
        model: Whisper model alias (see :data:`KNOWN_MODELS`), Hugging Face
            repo id, or path to a local model directory.
        engine: Name of the transcription engine to use.
        device: Device selector passed to the engine (for example ``auto``,
            ``cpu``, or ``cuda``).
        compute_type: Quantization used for inference; one of
            :data:`COMPUTE_TYPES`.
        language: ISO 639 language code (2-3 lowercase letters) of the audio,
            or ``None`` to auto-detect.
        task: ``transcribe`` (same-language text) or ``translate``
            (to English).
        vad: Filter out long silences with voice-activity detection before
            transcribing. On by default because Whisper hallucinates on
            silence.
        word_timestamps: Also produce per-word timestamps.
        beam_size: Beam width for decoding; at least 1.
        temperature: Sampling temperature between 0.0 (greedy, deterministic)
            and 1.0.
        initial_prompt: Optional text to prime the decoder with (names,
            jargon, spelling), or ``None`` for no priming.
    """

    model: str = "small"
    engine: str = "faster-whisper"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    task: str = "transcribe"
    vad: bool = True
    word_timestamps: bool = False
    beam_size: int = 5
    temperature: float = 0.0
    initial_prompt: str | None = None

    def __post_init__(self) -> None:
        """Validate every field, raising :class:`ConfigError` naming the bad value."""
        for name in ("model", "engine", "device"):
            value: str = getattr(self, name)
            if not value.strip():
                raise ConfigError(f"{name} must be a non-empty string, got {value!r}", field=name)
        if self.compute_type not in COMPUTE_TYPES:
            raise ConfigError(
                f"compute_type must be one of {'/'.join(COMPUTE_TYPES)}, got {self.compute_type!r}",
                field="compute_type",
            )
        if self.language is not None and _LANGUAGE_RE.fullmatch(self.language) is None:
            raise ConfigError(
                f"language must be a 2-3 letter lowercase code such as 'en', got {self.language!r}",
                field="language",
            )
        if self.task not in TASKS:
            raise ConfigError(
                f"task must be one of {'/'.join(TASKS)}, got {self.task!r}",
                field="task",
            )
        if self.beam_size < 1:
            raise ConfigError(f"beam_size must be >= 1, got {self.beam_size!r}", field="beam_size")
        if not 0.0 <= self.temperature <= 1.0:
            raise ConfigError(
                f"temperature must be between 0.0 and 1.0, got {self.temperature!r}",
                field="temperature",
            )

    def replace(self, **overrides: Any) -> TranscriberConfig:
        """Return a new validated config with the given fields replaced.

        Overrides whose value is ``None`` are ignored, so callers — the CLI in
        particular — can pass every option straight through and only the ones
        a user actually set take effect. Consequently ``replace`` cannot reset
        :attr:`language` or :attr:`initial_prompt` back to ``None``; construct
        a fresh config for that.

        Raises:
            ConfigError: If an override name is not a config field, or the
                resulting config fails validation.
        """
        known = {field.name for field in dataclasses.fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ConfigError("unknown config option(s): " + ", ".join(unknown))
        changed = {name: value for name, value in overrides.items() if value is not None}
        return dataclasses.replace(self, **changed)


def _env_var(field_name: str) -> str:
    """Return the environment variable that configures *field_name*."""
    return _ENV_PREFIX + field_name.upper()


def _parse_bool(value: str, var: str, field: str) -> bool:
    """Parse an environment boolean, accepting 1/0/true/false/yes/no (any case)."""
    token = value.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ConfigError(
        f"{var}: cannot parse {value!r} as a boolean (expected 1/0/true/false/yes/no)",
        field=field,
    )


def _parse_int(value: str, var: str, field: str) -> int:
    """Parse an environment integer, raising :class:`ConfigError` naming *var*."""
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"{var}: cannot parse {value!r} as an integer", field=field) from None


def _parse_float(value: str, var: str, field: str) -> float:
    """Parse an environment float, raising :class:`ConfigError` naming *var*."""
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"{var}: cannot parse {value!r} as a number", field=field) from None


def from_env(environ: Mapping[str, str] | None = None) -> TranscriberConfig:
    """Build a config from ``TRANSCRIBER_*`` environment variables.

    Each :class:`TranscriberConfig` field maps to one variable —
    ``TRANSCRIBER_MODEL``, ``TRANSCRIBER_ENGINE``, ``TRANSCRIBER_DEVICE``,
    ``TRANSCRIBER_COMPUTE_TYPE``, ``TRANSCRIBER_LANGUAGE``,
    ``TRANSCRIBER_TASK``, ``TRANSCRIBER_VAD``,
    ``TRANSCRIBER_WORD_TIMESTAMPS``, ``TRANSCRIBER_BEAM_SIZE``,
    ``TRANSCRIBER_TEMPERATURE``, and ``TRANSCRIBER_INITIAL_PROMPT``.

    A variable that is unset or set to the empty string leaves the library
    default in place. Booleans accept ``1/0/true/false/yes/no`` in any case.
    A value that cannot be parsed, or that fails config validation, raises
    :class:`~transcriber.errors.ConfigError` naming the variable.

    Args:
        environ: Mapping to read instead of ``os.environ``; useful for tests
            and embedding applications.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}
    for field in dataclasses.fields(TranscriberConfig):
        var = _env_var(field.name)
        value = env.get(var)
        if value is None or value == "":
            continue
        if field.name in _BOOL_FIELDS:
            overrides[field.name] = _parse_bool(value, var, field.name)
        elif field.name in _INT_FIELDS:
            overrides[field.name] = _parse_int(value, var, field.name)
        elif field.name in _FLOAT_FIELDS:
            overrides[field.name] = _parse_float(value, var, field.name)
        else:
            overrides[field.name] = value
    try:
        return TranscriberConfig(**overrides)
    except ConfigError as exc:
        if exc.field is not None and exc.field in overrides:
            raise ConfigError(f"{_env_var(exc.field)}: {exc}", field=exc.field) from exc
        raise
