#!/usr/bin/env bash
# Runs the phase-1 ingress pipeline for manual testing: starts the Python
# orchestrator, waits for its gRPC port, then runs media-edge in the
# foreground pointed at it. Ctrl+C stops both. Extra args pass through to
# media-edge (e.g. `./dev-run.sh --no-tls`).
set -euo pipefail
cd "$(dirname "$0")/.."

ORCH_PORT=50051
orch_pid=""
cleanup() { [ -n "$orch_pid" ] && kill "$orch_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

(cd orchestrator && python server.py) &
orch_pid=$!

for _ in $(seq 1 50); do
  (exec 3<>"/dev/tcp/localhost/$ORCH_PORT") 2>/dev/null && exec 3<&- 3>&- && break
  sleep 0.2
done

(cd media-edge && go run ./cmd/media-edge -orchestrator "localhost:$ORCH_PORT" "$@")
