#!/usr/bin/env bash
# Packaging smoke test: prove the distribution is real and installs cleanly.
#
#   1. Build the sdist and wheel into a fresh dist/ with `python -m build`.
#   2. `twine check` both archives' metadata.
#   3. Verify the wheel ships transcriber/py.typed and none of .graphloop/,
#      tests/, or scripts/; verify the sdist ships LICENSE and README.md.
#   4. Install the wheel into a throwaway venv (dependencies resolve from
#      PyPI) and exercise the installed CLI end to end: --version, --help,
#      formats, and a mock-engine transcription of a freshly generated WAV.
#
# Run from the repository root:
#
#   bash scripts/smoke_install.sh
#
# Prints PASS and exits 0 on success; any failure aborts with a FAIL line.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
TWINE="$REPO_ROOT/.venv/bin/twine"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

step() {
  printf '\n== %s\n' "$*" >&2
}

[ -x "$PYTHON" ] || fail "missing $PYTHON (the project virtualenv is required)"
[ -x "$TWINE" ] || fail "missing $TWINE (install the dev extras: pip install -e '.[dev]')"

cd "$REPO_ROOT"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

step "building sdist and wheel"
rm -rf dist
"$PYTHON" -m build

shopt -s nullglob
wheels=(dist/*.whl)
sdists=(dist/*.tar.gz)
shopt -u nullglob
[ "${#wheels[@]}" -eq 1 ] || fail "expected exactly one wheel in dist/, found ${#wheels[@]}"
[ "${#sdists[@]}" -eq 1 ] || fail "expected exactly one sdist in dist/, found ${#sdists[@]}"
WHEEL="${wheels[0]}"
SDIST="${sdists[0]}"

# dist/transcriber_cli-<version>-py3-none-any.whl -> <version>
VERSION="$(basename "$WHEEL" | cut -d- -f2)"
[ -n "$VERSION" ] || fail "could not parse a version out of $WHEEL"

step "twine check"
"$TWINE" check dist/*

step "verifying wheel contents ($WHEEL)"
wheel_listing="$("$PYTHON" -m zipfile -l "$WHEEL")"
printf '%s\n' "$wheel_listing" | grep -Eq '^transcriber/py\.typed([[:space:]]|$)' \
  || fail "wheel does not ship transcriber/py.typed"
if printf '%s\n' "$wheel_listing" | grep -Eq '^(\.graphloop|tests|scripts)/'; then
  fail "wheel leaks .graphloop/, tests/, or scripts/ entries"
fi

step "verifying sdist contents ($SDIST)"
sdist_listing="$("$PYTHON" -m tarfile -l "$SDIST")"
printf '%s\n' "$sdist_listing" | grep -Eq '^[^/]+/LICENSE([[:space:]]|$)' \
  || fail "sdist does not ship LICENSE"
printf '%s\n' "$sdist_listing" | grep -Eq '^[^/]+/README\.md([[:space:]]|$)' \
  || fail "sdist does not ship README.md"
if printf '%s\n' "$sdist_listing" | grep -Eq '^[^/]+/\.graphloop/'; then
  fail "sdist leaks .graphloop/ entries"
fi

step "installing the wheel into a throwaway venv"
VENV_DIR="$TMP_DIR/venv"
"$PYTHON" -m venv "$VENV_DIR"
if ! "$VENV_DIR/bin/pip" install --quiet "$WHEEL"; then
  fail "pip could not install $WHEEL into a clean venv"
fi
CLI="$VENV_DIR/bin/transcriber"
[ -x "$CLI" ] || fail "installing the wheel produced no $CLI entry point"

step "transcriber --version"
version_output="$("$CLI" --version)" || fail "transcriber --version exited nonzero"
printf '%s\n' "$version_output"
case "$version_output" in
  *"$VERSION"*) ;;
  *) fail "--version output does not mention the built version $VERSION" ;;
esac

step "transcriber --help"
if ! "$CLI" --help > /dev/null; then
  fail "transcriber --help exited nonzero"
fi

step "transcriber formats"
formats_output="$("$CLI" formats)" || fail "transcriber formats exited nonzero"
printf '%s\n' "$formats_output"
for format_name in json md srt tsv txt vtt; do
  printf '%s\n' "$formats_output" | grep -Eq "(^|[[:space:]])${format_name}([[:space:]]|$)" \
    || fail "formats listing does not mention ${format_name}"
done

step "mock-engine transcription of a generated WAV"
WAV="$TMP_DIR/sample.wav"
"$VENV_DIR/bin/python" - "$WAV" <<'PY'
"""Write a tiny deterministic 16 kHz mono 16-bit PCM WAV (0.5 s, 440 Hz)."""
import math
import struct
import sys
import wave

path = sys.argv[1]
rate = 16_000
samples = [round(20_000 * math.sin(math.tau * 440.0 * frame / rate)) for frame in range(rate // 2)]
with wave.open(path, "wb") as writer:
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(rate)
    writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
PY
[ -s "$WAV" ] || fail "failed to generate $WAV"

OUT_DIR="$TMP_DIR/out"
if ! "$CLI" transcribe "$WAV" --engine mock --format txt --output-dir "$OUT_DIR" --quiet; then
  fail "installed CLI failed the mock-engine transcription (nonzero exit)"
fi
TRANSCRIPT="$OUT_DIR/sample.txt"
[ -f "$TRANSCRIPT" ] || fail "expected transcript $TRANSCRIPT was not written"
grep -q "mock transcription of sample" "$TRANSCRIPT" \
  || fail "$TRANSCRIPT does not contain the mock engine's transcript text"

step "cleaning up the throwaway venv"
rm -rf "$TMP_DIR"
trap - EXIT

printf '\nPASS: %s and %s build, check, install, and run cleanly\n' \
  "$(basename "$WHEEL")" "$(basename "$SDIST")"
