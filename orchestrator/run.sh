#!/usr/bin/env bash
set -euo pipefail

service_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
env_file="${ORCHESTRATOR_ENV_FILE:-$service_dir/.env}"
venv_dir="${ORCHESTRATOR_VENV:-$service_dir/.venv}"

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

if [ ! -x "$venv_dir/bin/python" ]; then
  "$python_bin" -m venv "$venv_dir"
fi

venv_python="$venv_dir/bin/python"
requirements_hash="$("$python_bin" - <<'PY' "$service_dir/requirements.txt"
from pathlib import Path
import hashlib
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
stamp_file="$venv_dir/.orchestrator_requirements.sha256"
installed_hash="$( [ -f "$stamp_file" ] && tr -d '\n' < "$stamp_file" || true )"

if [ "${ORCHESTRATOR_SKIP_INSTALL:-0}" != "1" ] && [ "$installed_hash" != "$requirements_hash" ]; then
  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install -r "$service_dir/requirements.txt"
  printf '%s\n' "$requirements_hash" > "$stamp_file"
fi

cd "$service_dir"
exec "$venv_python" server.py "$@"
