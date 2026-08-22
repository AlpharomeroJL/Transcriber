"""Tests for the error taxonomy (transcriber.errors) and configuration (transcriber.config)."""

from __future__ import annotations

import dataclasses
import math
import os

import pytest

from transcriber.config import (
    COMPUTE_TYPES,
    KNOWN_MODELS,
    TASKS,
    TranscriberConfig,
    from_env,
    is_known_model,
)
from transcriber.errors import (
    AudioNotFoundError,
    ConfigError,
    EngineError,
    EngineNotAvailableError,
    InputError,
    ModelLoadError,
    OutputError,
    TranscriberError,
    TranscriptionFailedError,
    UnsupportedFormatError,
)

ALL_ENV_VARS: tuple[str, ...] = tuple(
    f"TRANSCRIBER_{field.name.upper()}" for field in dataclasses.fields(TranscriberConfig)
)


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (TranscriberError, Exception),
        (ConfigError, TranscriberError),
        (InputError, TranscriberError),
        (AudioNotFoundError, InputError),
        (UnsupportedFormatError, InputError),
        (EngineError, TranscriberError),
        (EngineNotAvailableError, EngineError),
        (ModelLoadError, EngineError),
        (TranscriptionFailedError, EngineError),
        (OutputError, TranscriberError),
    ],
)
def test_error_hierarchy(child: type[Exception], parent: type[Exception]) -> None:
    assert issubclass(child, parent)


@pytest.mark.parametrize(
    "exc_type",
    [
        ConfigError,
        InputError,
        AudioNotFoundError,
        UnsupportedFormatError,
        EngineError,
        ModelLoadError,
        TranscriptionFailedError,
        OutputError,
    ],
)
def test_errors_carry_message_and_catch_as_transcriber_error(exc_type: type[Exception]) -> None:
    with pytest.raises(TranscriberError, match="something actionable"):
        raise exc_type("something actionable")


def test_engine_not_available_default_message_has_pip_hint() -> None:
    err = EngineNotAvailableError()
    assert err.engine == "faster-whisper"
    assert err.package == "faster-whisper"
    assert "'faster-whisper'" in str(err)
    assert "pip install faster-whisper" in str(err)


def test_engine_not_available_custom_engine_and_package() -> None:
    err = EngineNotAvailableError("whisperx", package="whisperx[gpu]")
    assert err.engine == "whisperx"
    assert err.package == "whisperx[gpu]"
    assert "pip install whisperx[gpu]" in str(err)


def test_engine_not_available_explicit_message_wins() -> None:
    err = EngineNotAvailableError("mock", message="the mock engine is disabled")
    assert str(err) == "the mock engine is disabled"
    assert err.engine == "mock"


def test_config_error_field_attribute() -> None:
    assert ConfigError("boom").field is None
    assert ConfigError("boom", field="model").field == "model"


# ---------------------------------------------------------------------------
# TranscriberConfig defaults and invariants
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    cfg = TranscriberConfig()
    assert cfg.model == "small"
    assert cfg.engine == "faster-whisper"
    assert cfg.device == "auto"
    assert cfg.compute_type == "auto"
    assert cfg.language is None
    assert cfg.task == "transcribe"
    assert cfg.vad is True
    assert cfg.word_timestamps is False
    assert cfg.beam_size == 5
    assert cfg.temperature == 0.0
    assert cfg.initial_prompt is None


def test_config_is_frozen() -> None:
    cfg = TranscriberConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model = "base"  # type: ignore[misc]


def test_config_equality_and_hash() -> None:
    assert TranscriberConfig() == TranscriberConfig()
    assert TranscriberConfig(model="base") != TranscriberConfig()
    assert hash(TranscriberConfig(language="de")) == hash(TranscriberConfig(language="de"))


def test_constants_match_engine_contract() -> None:
    assert KNOWN_MODELS == (
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
    assert set(COMPUTE_TYPES) == {
        "auto",
        "default",
        "int8",
        "int8_float16",
        "int8_float32",
        "int16",
        "float16",
        "float32",
        "bfloat16",
    }
    assert TASKS == ("transcribe", "translate")


@pytest.mark.parametrize(
    ("name", "known"),
    [
        ("small", True),
        ("large-v3-turbo", True),
        ("distil-large-v3", True),
        ("Small", False),
        ("openai/whisper-large-v3", False),
        ("/models/my-finetune", False),
        ("", False),
    ],
)
def test_is_known_model(name: str, known: bool) -> None:
    assert is_known_model(name) is known


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compute_type", COMPUTE_TYPES)
def test_every_documented_compute_type_is_accepted(compute_type: str) -> None:
    assert TranscriberConfig(compute_type=compute_type).compute_type == compute_type


@pytest.mark.parametrize("language", ["en", "de", "zh", "yue", "haw"])
def test_valid_language_codes_accepted(language: str) -> None:
    assert TranscriberConfig(language=language).language == language


@pytest.mark.parametrize("task", TASKS)
def test_valid_tasks_accepted(task: str) -> None:
    assert TranscriberConfig(task=task).task == task


@pytest.mark.parametrize("temperature", [0.0, 0.4, 1.0])
def test_temperature_bounds_inclusive(temperature: float) -> None:
    assert TranscriberConfig(temperature=temperature).temperature == temperature


@pytest.mark.parametrize(
    ("kwargs", "field", "needle"),
    [
        ({"model": ""}, "model", "model"),
        ({"model": "   "}, "model", "model"),
        ({"engine": ""}, "engine", "engine"),
        ({"device": ""}, "device", "device"),
        ({"compute_type": "fp16"}, "compute_type", "fp16"),
        ({"language": "English"}, "language", "English"),
        ({"language": "EN"}, "language", "language"),
        ({"language": "e"}, "language", "language"),
        ({"language": ""}, "language", "language"),
        ({"task": "detect"}, "task", "detect"),
        ({"beam_size": 0}, "beam_size", "beam_size"),
        ({"beam_size": -3}, "beam_size", "beam_size"),
        ({"temperature": -0.01}, "temperature", "temperature"),
        ({"temperature": 1.01}, "temperature", "temperature"),
    ],
)
def test_validation_failures_name_the_offending_value(
    kwargs: dict[str, object], field: str, needle: str
) -> None:
    with pytest.raises(ConfigError, match=needle) as excinfo:
        TranscriberConfig(**kwargs)  # type: ignore[arg-type]
    assert excinfo.value.field == field


def test_nan_temperature_rejected() -> None:
    with pytest.raises(ConfigError, match="temperature"):
        TranscriberConfig(temperature=math.nan)


# ---------------------------------------------------------------------------
# replace()
# ---------------------------------------------------------------------------


def test_replace_returns_new_validated_config() -> None:
    base = TranscriberConfig()
    out = base.replace(model="turbo", beam_size=2)
    assert isinstance(out, TranscriberConfig)
    assert out is not base
    assert out.model == "turbo"
    assert out.beam_size == 2
    assert out.engine == base.engine
    assert base.model == "small"  # the original is untouched


def test_replace_filters_none_valued_overrides() -> None:
    base = TranscriberConfig()
    out = base.replace(model=None, beam_size=None, language="de", vad=None)
    assert out.model == "small"
    assert out.beam_size == 5
    assert out.vad is True
    assert out.language == "de"


def test_replace_cannot_reset_optional_fields_to_none() -> None:
    base = TranscriberConfig(language="en", initial_prompt="Hello.")
    out = base.replace(language=None, initial_prompt=None)
    assert out == base


def test_replace_with_no_overrides_is_an_equal_copy() -> None:
    base = TranscriberConfig(model="base.en")
    assert base.replace() == base


def test_replace_validates_the_result() -> None:
    with pytest.raises(ConfigError, match="beam_size"):
        TranscriberConfig().replace(beam_size=0)


def test_replace_rejects_unknown_options() -> None:
    with pytest.raises(ConfigError, match="modle"):
        TranscriberConfig().replace(modle="small")


# ---------------------------------------------------------------------------
# from_env()
# ---------------------------------------------------------------------------


def test_from_env_empty_mapping_gives_defaults() -> None:
    assert from_env({}) == TranscriberConfig()


def test_from_env_reads_every_variable() -> None:
    env = {
        "TRANSCRIBER_MODEL": "large-v3",
        "TRANSCRIBER_ENGINE": "mock",
        "TRANSCRIBER_DEVICE": "cpu",
        "TRANSCRIBER_COMPUTE_TYPE": "int8",
        "TRANSCRIBER_LANGUAGE": "de",
        "TRANSCRIBER_TASK": "translate",
        "TRANSCRIBER_VAD": "no",
        "TRANSCRIBER_WORD_TIMESTAMPS": "1",
        "TRANSCRIBER_BEAM_SIZE": "3",
        "TRANSCRIBER_TEMPERATURE": "0.25",
        "TRANSCRIBER_INITIAL_PROMPT": "Names: Ada, Linus.",
    }
    assert set(env) == set(ALL_ENV_VARS)
    assert from_env(env) == TranscriberConfig(
        model="large-v3",
        engine="mock",
        device="cpu",
        compute_type="int8",
        language="de",
        task="translate",
        vad=False,
        word_timestamps=True,
        beam_size=3,
        temperature=0.25,
        initial_prompt="Names: Ada, Linus.",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
        ("TRUE", True),
        ("False", False),
        ("YES", True),
        ("No", False),
        (" true ", True),
    ],
)
def test_from_env_bool_coercion(raw: str, expected: bool) -> None:
    cfg = from_env({"TRANSCRIBER_VAD": raw, "TRANSCRIBER_WORD_TIMESTAMPS": raw})
    assert cfg.vad is expected
    assert cfg.word_timestamps is expected


def test_from_env_numeric_parsing_tolerates_surrounding_whitespace() -> None:
    cfg = from_env({"TRANSCRIBER_BEAM_SIZE": " 8 ", "TRANSCRIBER_TEMPERATURE": " 0.5 "})
    assert cfg.beam_size == 8
    assert cfg.temperature == 0.5


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("TRANSCRIBER_MODEL", "   "),
        ("TRANSCRIBER_COMPUTE_TYPE", "fp16"),
        ("TRANSCRIBER_LANGUAGE", "English"),
        ("TRANSCRIBER_TASK", "detect"),
        ("TRANSCRIBER_VAD", "maybe"),
        ("TRANSCRIBER_WORD_TIMESTAMPS", "2"),
        ("TRANSCRIBER_BEAM_SIZE", "three"),
        ("TRANSCRIBER_BEAM_SIZE", "3.5"),
        ("TRANSCRIBER_BEAM_SIZE", "0"),
        ("TRANSCRIBER_TEMPERATURE", "warm"),
        ("TRANSCRIBER_TEMPERATURE", "1.5"),
    ],
)
def test_from_env_bad_values_raise_config_error_naming_the_variable(var: str, value: str) -> None:
    with pytest.raises(ConfigError, match=var):
        from_env({var: value})


def test_from_env_empty_string_means_unset() -> None:
    # Every variable set-but-empty behaves exactly like unset: defaults apply
    # and nothing is parsed or validated (contrast: language="" passed directly
    # to TranscriberConfig is rejected, and TRANSCRIBER_VAD=maybe is an error).
    env = dict.fromkeys(ALL_ENV_VARS, "")
    cfg = from_env(env)
    assert cfg == TranscriberConfig()
    assert cfg.language is None
    assert cfg.vad is True


def test_from_env_ignores_unrelated_variables() -> None:
    env = {"PATH": "/usr/bin", "TRANSCRIBERMODEL": "base", "MODEL": "base"}
    assert from_env(env) == TranscriberConfig()


def test_from_env_reads_process_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in list(os.environ):
        if var.startswith("TRANSCRIBER_"):
            monkeypatch.delenv(var)
    monkeypatch.setenv("TRANSCRIBER_MODEL", "base.en")
    monkeypatch.setenv("TRANSCRIBER_WORD_TIMESTAMPS", "yes")
    cfg = from_env()
    assert cfg.model == "base.en"
    assert cfg.word_timestamps is True
    assert cfg.engine == "faster-whisper"


def test_from_env_explicit_mapping_shields_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBER_MODEL", "large-v2")
    assert from_env({}).model == "small"
