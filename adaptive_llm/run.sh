#!/usr/bin/env bash
set -euo pipefail

service_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$service_dir/.." && pwd)"
python_bin="${PYTHON:-python3}"
env_file="${ADAPTIVE_LLM_ENV_FILE:-$service_dir/.env}"

if [ -f "$env_file" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [ -z "$line" ] || [[ "$line" == \#* ]] || [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    if [[ "$value" == \"*\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' ]]; then
      value="${value:1:${#value}-2}"
    else
      # Keep literal hashes in values (HF tokens may contain them), while
      # retaining conventional whitespace-separated inline comments.
      value="${value%%[[:space:]]\#*}"
      value="${value%"${value##*[![:space:]]}"}"
    fi
    if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$env_file"
fi

venv_dir="${ADAPTIVE_LLM_VENV:-$service_dir/.venv}"

if [ ! -x "$venv_dir/bin/python" ]; then
  "$python_bin" -m venv "$venv_dir"
fi

venv_python="$venv_dir/bin/python"
overrides_file="$service_dir/requirements-overrides.txt"
requirements_hash="$("$python_bin" - <<'PY' "$service_dir/requirements.txt" "$overrides_file"
from pathlib import Path
import hashlib
import sys

digest = hashlib.sha256()
for argument in sys.argv[1:]:
    path = Path(argument)
    digest.update(path.read_bytes() if path.is_file() else b"")
print(digest.hexdigest())
PY
)"
stamp_file="$venv_dir/.adaptive_llm_requirements.sha256"
installed_hash=""
if [ -f "$stamp_file" ]; then
  installed_hash="$(tr -d '\n' < "$stamp_file")"
fi

if [ "${ADAPTIVE_LLM_SKIP_INSTALL:-0}" != "1" ] && [ "$installed_hash" != "$requirements_hash" ]; then
  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install -r "$service_dir/requirements.txt"
  # Deliberate version overrides go in second, with --no-deps. pip cannot
  # resolve one of these against the pin it contradicts: asking for both in
  # one transaction ends in ResolutionImpossible and installs nothing at all.
  # --no-deps applies only to the overridden packages, so everything
  # requirements.txt pulled in stays exactly as resolved above.
  if [ -f "$overrides_file" ]; then
    "$venv_python" -m pip install --no-deps --upgrade -r "$overrides_file"
  fi
  printf '%s\n' "$requirements_hash" > "$stamp_file"
fi

cd "$repo_root"
exec "$venv_python" -m adaptive_llm.main "$@"
