#!/usr/bin/env bash
# Builds the media-edge binary and drops it in build/media-edge/, named with
# its VERSION and platform, plus a floating "latest" copy for quick testing.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(tr -d ' \t\r\n' < media-edge/VERSION)
GOOS=${GOOS:-windows}
GOARCH=${GOARCH:-amd64}
EXT=""
[ "$GOOS" = "windows" ] && EXT=".exe"

OUT_DIR="build/media-edge"
NAME="media-edge-v${VERSION}-${GOOS}-${GOARCH}${EXT}"
mkdir -p "$OUT_DIR"

(cd media-edge && GOOS="$GOOS" GOARCH="$GOARCH" go build \
  -ldflags "-X main.version=v${VERSION}" \
  -o "../$OUT_DIR/$NAME" \
  ./cmd/media-edge)

cp -f "$OUT_DIR/$NAME" "$OUT_DIR/media-edge-latest${EXT}"
echo "built $OUT_DIR/$NAME"
