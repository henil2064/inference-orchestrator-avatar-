#!/usr/bin/env bash
# Publishes the already-built media-edge exe (see build-media-edge.sh) as a
# GitHub release tagged v<VERSION>. Does not rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(tr -d ' \t\r\n' < media-edge/VERSION)
GOOS=${GOOS:-windows}
GOARCH=${GOARCH:-amd64}
EXT=""
[ "$GOOS" = "windows" ] && EXT=".exe"

BIN="build/media-edge/media-edge-v${VERSION}-${GOOS}-${GOARCH}${EXT}"
[ -f "$BIN" ] || { echo "missing $BIN — run scripts/build-media-edge.sh first" >&2; exit 1; }

gh release create "v${VERSION}" "$BIN" \
  --title "media-edge v${VERSION}" \
  --notes "media-edge v${VERSION}" \
  --repo arbatmondee/inference-orchestrator
