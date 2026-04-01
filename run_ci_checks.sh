#!/usr/bin/env bash
set -euo pipefail

SKIP_SYNC=false

for arg in "$@"; do
  case "$arg" in
    --skip-sync)
      SKIP_SYNC=true
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash run_ci_checks.sh [--skip-sync]" >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

if [ "$SKIP_SYNC" = false ]; then
  echo "==> Syncing dependencies"
  uv sync --group dev
fi

echo "==> Ruff lint"
uv run ruff check .

echo "==> BasedPyright type check"
uv run basedpyright --level error .

if [ -d tests ]; then
  echo "==> Pytest"
  uv run pytest -q
else
  echo "==> Pytest"
  echo "No tests directory found. Skipping pytest."
fi

compile_targets=()
for path in train scripts inference smoke_test_running; do
  if [ -d "$path" ]; then
    compile_targets+=("$path")
  fi
done

if [ "${#compile_targets[@]}" -gt 0 ]; then
  echo "==> Compile smoke check"
  uv run python -m compileall "${compile_targets[@]}"
else
  echo "No Python source directories found for compileall."
fi
