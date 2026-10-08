#!/usr/bin/env bash
# One-shot setup + Linux build for media-edge on an Ubuntu box: Go toolchain
# (matching go.mod exactly), build-essential (cgo needs a C compiler for
# onnxruntime_go — see README "Voice activity detection"), libopus-dev +
# libopusfile-dev + pkg-config (cgo needs both libraries for the outbound
# agent Opus track),
# onnxruntime's shared lib (for Silero VAD at runtime), then the build itself.
# Idempotent — safe to re-run after a `git pull`.
set -euo pipefail
cd "$(dirname "$0")"

REQUIRED_GO=$(grep -oP '^go \K[0-9.]+' go.mod)

# --- Go toolchain, pinned to what go.mod declares ---
installed_go=""
if command -v go >/dev/null; then
  installed_go=$(go version | grep -oP 'go\K[0-9.]+')
fi
if [ -z "$installed_go" ] || [ "$(printf '%s\n%s\n' "$REQUIRED_GO" "$installed_go" | sort -V | head -1)" != "$REQUIRED_GO" ]; then
  echo "installing Go ${REQUIRED_GO} (have: ${installed_go:-none})"
  TMP=$(mktemp -d)
  curl -fsSL "https://go.dev/dl/go${REQUIRED_GO}.linux-amd64.tar.gz" -o "$TMP/go.tar.gz"
  sudo rm -rf /usr/local/go
  sudo tar -C /usr/local -xzf "$TMP/go.tar.gz"
  rm -rf "$TMP"
  echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee /etc/profile.d/go.sh >/dev/null
  export PATH="$PATH:/usr/local/go/bin"
fi
go version

# --- cgo needs a C compiler (onnxruntime_go, opus) plus libopus and
#     libopusfile for the outbound agent Opus track (via pkg-config) ---
need_pkgs=()
if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
  need_pkgs+=(build-essential)
fi
if ! command -v pkg-config >/dev/null; then
  need_pkgs+=(pkg-config)
fi
if ! pkg-config --exists opus 2>/dev/null; then
  need_pkgs+=(libopus-dev)
fi
if ! pkg-config --exists opusfile 2>/dev/null; then
  need_pkgs+=(libopusfile-dev)
fi
if [ ${#need_pkgs[@]} -gt 0 ]; then
  echo "installing: ${need_pkgs[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${need_pkgs[@]}"
fi

# --- onnxruntime shared lib, for Silero VAD at runtime (build itself doesn't need it) ---
ORT_VERSION=1.29.0
ORT_DIR=third_party/onnxruntime
if ! ls "$ORT_DIR"/libonnxruntime.so* >/dev/null 2>&1; then
  echo "fetching onnxruntime v${ORT_VERSION} (linux-x64)"
  mkdir -p "$ORT_DIR"
  TMP=$(mktemp -d)
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" -o "$TMP/ort.tgz"
  tar -xzf "$TMP/ort.tgz" -C "$TMP"
  cp "$TMP"/onnxruntime-linux-x64-"${ORT_VERSION}"/lib/libonnxruntime.so* "$ORT_DIR"/
  rm -rf "$TMP"
fi

# --- build (Linux, overriding build-media-edge.sh's windows default) ---
GOOS=linux GOARCH=amd64 ../scripts/build-media-edge.sh

echo
echo "built. run with, e.g.:"
echo "  ../build/media-edge/media-edge-latest --no-tls --orchestrator localhost:50051 \\"
echo "    --vad-model models/silero_vad.onnx --onnx-lib ${ORT_DIR}/libonnxruntime.so"
