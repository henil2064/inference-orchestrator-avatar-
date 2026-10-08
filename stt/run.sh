#!/usr/bin/env bash
set -euo pipefail

service_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
env_file="${STT_ENV_FILE:-$service_dir/.env}"
venv_dir="${STT_VENV:-$service_dir/.venv-sravaani}"

load_env_file() {
  local line key value
  [ -f "$1" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -n "$line" ] && [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$1"
}

load_env_file "$env_file"

runtime="${STT_RUNTIME:-transformers}"
case "$runtime" in
  transformers|torchscript|pytorch) requirements_file="$service_dir/requirements-sravaani.txt" ;;
  onnx) requirements_file="$service_dir/requirements-sravaani-onnx.txt" ;;
  *) echo "STT_RUNTIME must be transformers or onnx (got: $runtime)" >&2; exit 2 ;;
esac

if [ ! -x "$venv_dir/bin/python" ]; then
  "$python_bin" -m venv "$venv_dir"
fi

venv_python="$venv_dir/bin/python"
requirements_hash="$("$python_bin" - <<'PY' "$requirements_file"
from pathlib import Path
import hashlib
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
stamp_file="$venv_dir/.stt_requirements.sha256"
installed_hash="$( [ -f "$stamp_file" ] && tr -d '\n' < "$stamp_file" || true )"

if [ "${STT_SKIP_INSTALL:-0}" != "1" ] && [ "$installed_hash" != "$requirements_hash" ]; then
  "$venv_python" -m pip install --upgrade pip
  # onnxruntime and onnxruntime-gpu expose the same Python package.  Remove
  # the CPU wheel before installing the GPU one so an old environment cannot
  # retain a CPU-only shared library while STT_ONNX_PROVIDER=cuda is selected.
  if [ "$runtime" = "onnx" ]; then
    "$venv_python" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
  fi
  "$venv_python" -m pip install -r "$requirements_file"
  printf '%s\n' "$requirements_hash" > "$stamp_file"
fi

if ! "$venv_python" -c 'import torch' >/dev/null 2>&1; then
  echo "PyTorch is required but not installed in $venv_dir." >&2
  echo "Install the CUDA build first, e.g.: $venv_python -m pip install torch --index-url https://download.pytorch.org/whl/cu124" >&2
  exit 1
fi

cd "$service_dir"
exec "$venv_python" server.py "$@"
