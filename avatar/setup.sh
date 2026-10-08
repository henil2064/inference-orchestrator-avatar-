#!/usr/bin/env bash
# One-time setup for the avatar service on the GPU box that will run it.
#
#   HF_TOKEN=hf_... bash avatar/setup.sh
#
# Clones upstream AVTR-1 at a pinned commit, installs its pixi environment,
# adds this service's dependencies, downloads the model artifacts, and builds
# the TensorRT engines ON THIS GPU. Safe to re-run: every step is skipped when
# its output already exists. Afterwards start the service with avatar/run.sh.
#
# TensorRT engines only run on the GPU model and TensorRT version that built
# them, which is why they are built here rather than shipped. Engines built
# elsewhere (a Kaggle T4, say) must not be copied onto a different card.
set -euo pipefail

service_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream_dir="${AVATAR_UPSTREAM_DIR:-$service_dir/third_party/avtr-1}"
upstream_repo="${AVATAR_UPSTREAM_REPO:-https://github.com/avaturn-live/avtr-1.git}"
# Pinned: this service calls upstream internals (Pipeline, Chunk, RenderOptions)
# that a later commit is free to change.
upstream_ref="${AVATAR_UPSTREAM_REF:-eb1e5a89ed7bec3debe712b090efd676b3a420bd}"

step() { printf '\n==> %s\n' "$*"; }
warn() { printf 'setup: WARNING: %s\n' "$*" >&2; }
die() { printf 'setup: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "AVTR-1 runs on Linux only (found $(uname -s))"
command -v git >/dev/null || die "git is required"
command -v curl >/dev/null || die "curl is required"

step "Checking the GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; install the NVIDIA driver first"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[ -n "$gpu_name" ] || die "nvidia-smi reports no GPU"
# compute_cap needs a recent driver; treat a failed query as unknown, not fatal.
compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "GPU: $gpu_name (compute capability ${compute_cap:-unknown})"
case "${compute_cap%%.*}" in
  ''|*[!0-9]*) ;;
  *) [ "${compute_cap%%.*}" -ge 8 ] || warn "upstream requires Ampere or newer (compute capability 8.0+). \
A T4 (7.5) measured ~430 ms per 200 ms chunk -- less than half real time." ;;
esac

step "Installing pixi"
export PATH="$HOME/.pixi/bin:$PATH"
if ! command -v pixi >/dev/null; then
  curl -fsSL https://pixi.sh/install.sh | sh
fi
pixi --version

step "Fetching upstream AVTR-1 at ${upstream_ref:0:12}"
if [ ! -d "$upstream_dir/.git" ]; then
  mkdir -p "$(dirname "$upstream_dir")"
  git clone "$upstream_repo" "$upstream_dir"
fi
if ! git -C "$upstream_dir" cat-file -e "$upstream_ref^{commit}" 2>/dev/null; then
  git -C "$upstream_dir" fetch --quiet origin
fi
git -C "$upstream_dir" checkout --quiet --detach "$upstream_ref"

# Everything below runs inside upstream's default pixi environment, from its
# checkout -- the same way upstream's own scripts expect to be run.
py() { (cd "$upstream_dir" && pixi run python "$@"); }

step "Installing the upstream environment (downloads several GB the first time)"
(cd "$upstream_dir" && pixi install)

step "Adding avatar service dependencies"
py -m ensurepip --upgrade >/dev/null
py -m pip install --no-cache-dir --quiet -r "$service_dir/requirements-service.txt"

step "Downloading model artifacts from Hugging Face"
# The model repository is gated: accept its terms at
# https://huggingface.co/avaturn-live/avtr-1 with the account the token belongs to.
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
elif ! py -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)"; then
  die "no Hugging Face credentials: re-run as HF_TOKEN=hf_... bash avatar/setup.sh"
fi
py scripts/download_artifacts.py

storage="$(py -c "from avtr1_renderer.avtr1_artifact_manager import get_storage_root; print(get_storage_root())" | tail -n1)"
[ -d "$storage" ] || die "artifact storage root not found at $storage"

if compgen -G "$service_dir/portraits/*.png" >/dev/null; then
  step "Installing custom portraits"
  mkdir -p "$storage/avatars_artifacts/reference_frames"
  cp -v "$service_dir"/portraits/*.png "$storage/avatars_artifacts/reference_frames/"
fi

s2m="$storage/speech2motion_runtime_artifacts_cc"
rnd="$storage/renderer_runtime_artifacts_cc"
stamp="$storage/.engines_built_on"

# Optional: bring engines built elsewhere instead of building here, in the
# layout  <dir>/speech2motion/*.engine  and  <dir>/renderer/*.engine.
if [ -n "${AVATAR_ENGINES_DIR:-}" ]; then
  step "Importing engines from $AVATAR_ENGINES_DIR"
  [ -d "$AVATAR_ENGINES_DIR/speech2motion" ] && [ -d "$AVATAR_ENGINES_DIR/renderer" ] \
    || die "AVATAR_ENGINES_DIR must contain speech2motion/ and renderer/"
  warn "imported engines only work if they were built on a '$gpu_name' with this TensorRT version"
  mkdir -p "$s2m" "$rnd"
  cp -v "$AVATAR_ENGINES_DIR"/speech2motion/*.engine "$s2m/"
  cp -v "$AVATAR_ENGINES_DIR"/renderer/*.engine "$rnd/"
  printf 'imported (built elsewhere; expected %s)\n' "$gpu_name" > "$stamp"
fi

# Engines left over from a different card (a copied artifacts directory, a
# swapped GPU) would fail at load or, worse, run with wrong timings.
if [ -f "$stamp" ] && ! grep -q '^imported' "$stamp" && [ "$(cat "$stamp")" != "$gpu_name" ]; then
  warn "existing engines were built on '$(cat "$stamp")', not '$gpu_name'; rebuilding them"
  rm -f "$s2m"/*.engine "$rnd"/*.engine "$stamp"
fi

have_all() { local f; for f in "$@"; do [ -f "$f" ] || return 1; done; }

# The normalizer is written by the AVTR-1 build, so a missing one triggers it
# even when the two engines were imported.
if ! have_all "$s2m/avtr1_encode_fp16.engine" "$s2m/avtr1_decode_fp16.engine" "$storage/avtr1_normalizer.safetensors"; then
  step "Building AVTR-1 motion engines (a few minutes)"
  py scripts/build_avtr1_engines.py
fi
if ! have_all "$rnd/decoder_b5_fp16.engine" "$rnd/warp_network_b5_fp16.engine" \
              "$rnd/modnet_b5_fp16.engine" "$rnd/stitch_network_b5_fp16.engine"; then
  step "Building renderer engines (a few minutes)"
  py scripts/build_renderer_engines.py
fi
if ! have_all "$s2m/hubert_lbs_fp16.engine"; then
  step "Building the HuBERT engine"
  py scripts/build_hubert_engine.py
fi
[ -f "$stamp" ] || printf '%s\n' "$gpu_name" > "$stamp"

step "Verifying"
have_all "$s2m/avtr1_encode_fp16.engine" "$s2m/avtr1_decode_fp16.engine" "$s2m/hubert_lbs_fp16.engine" \
         "$rnd/decoder_b5_fp16.engine" "$rnd/warp_network_b5_fp16.engine" \
         "$rnd/modnet_b5_fp16.engine" "$rnd/stitch_network_b5_fp16.engine" \
         "$storage/avtr1_normalizer.safetensors" \
  || die "some engines are still missing under $storage"
ls -lh "$s2m"/*.engine "$rnd"/*.engine

[ -f "$service_dir/.env" ] || cp "$service_dir/.env.example" "$service_dir/.env"

cat <<EOF

Avatar setup complete on $gpu_name.

  Start the service:   bash avatar/run.sh
  Point the edge at it: EDGE_AVATAR_TARGET=localhost:50056 (media-edge/.env)
  Or launch everything: AVATAR_ENABLED=1 python3 scripts/run_pipeline.py

The service prints its measured ms/chunk at startup; real time needs < 200.
EOF
