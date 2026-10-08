#!/usr/bin/env bash
set -euo pipefail

service_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${AVATAR_ENV_FILE:-$service_dir/.env}"

load_env_file() {
  local line key value
  [ -f "$1" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"; line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"; [ -n "$line" ] && [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; value="${line#*=}"; key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"; value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!key+x}" ]; then export "$key=$value"; fi
  done < "$1"
}
load_env_file "$env_file"

# AVATAR_PYTHON runs the service outside the upstream environment. Only the
# synthetic backend works that way; it is how the edge's video path gets
# exercised on a machine with no GPU (see SETUP.md).
if [ -n "${AVATAR_PYTHON:-}" ]; then
  cd "$service_dir"
  exec "$AVATAR_PYTHON" server.py "$@"
fi

upstream_dir="${AVATAR_UPSTREAM_DIR:-$service_dir/third_party/avtr-1}"
export PATH="$HOME/.pixi/bin:$PATH"
if [ ! -d "$upstream_dir/.pixi" ] || ! command -v pixi >/dev/null; then
  echo "avatar: upstream environment not found at $upstream_dir -- run avatar/setup.sh first" >&2
  exit 1
fi
# Run from the upstream checkout, like its own scripts: pixi resolves the
# environment from the working directory. server.py finds its own packages
# and .env by file path, so the working directory does not matter to it.
cd "$upstream_dir"
exec pixi run python "$service_dir/server.py" "$@"
