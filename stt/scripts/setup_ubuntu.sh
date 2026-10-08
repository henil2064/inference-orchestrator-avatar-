#!/usr/bin/env bash
# One-shot setup for SraVaani STT on an Ubuntu GPU box: system deps (apt) + venv
# (pip). Idempotent — safe to re-run after a `git pull`. Its environment must not
# be shared with the orchestrator.
set -euo pipefail
cd "$(dirname "$0")"

# --- system deps ---
need_apt=()
command -v python3 >/dev/null || need_apt+=(python3)
python3 -c "import venv" 2>/dev/null || need_apt+=(python3-venv)
if [ "${#need_apt[@]}" -gt 0 ]; then
  echo "installing system deps: ${need_apt[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${need_apt[@]}"
fi

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi not found — no GPU driver visible, SraVaani GPU STT will not run here" >&2
  exit 1
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- venv + python deps ---
if [ ! -d .venv-sravaani ]; then
  echo "creating venv"
  python3 -m venv .venv-sravaani
fi
source .venv-sravaani/bin/activate
pip install -q -U pip
pip install -q -r requirements-sravaani.txt
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print('torch', torch.__version__, '- cuda available:', torch.cuda.is_available())"

echo
echo "done. activate with: source stt/.venv-sravaani/bin/activate"
