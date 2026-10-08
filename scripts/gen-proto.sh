#!/usr/bin/env bash
# Regenerate gRPC stubs for bridge, STT, LLM, TTS and avatar contracts.
# Requires: protoc, protoc-gen-go, protoc-gen-go-grpc (Go); python -m grpc_tools.protoc (Python).
set -euo pipefail
cd "$(dirname "$0")/.."

# Go stubs -> media-edge/internal/edgepb (go_package option sets the subpath).
protoc -I proto \
  --go_out=media-edge --go_opt=module=mondee/media-edge \
  --go-grpc_out=media-edge --go-grpc_opt=module=mondee/media-edge \
  proto/edge.proto

# Python stubs -> orchestrator/proto/
python -m grpc_tools.protoc -I proto \
  --python_out=orchestrator/proto --grpc_python_out=orchestrator/proto \
  proto/edge.proto

# avatar.proto is Go<->Python (media-edge <-> avatar service).
protoc -I proto \
  --go_out=media-edge --go_opt=module=mondee/media-edge \
  --go-grpc_out=media-edge --go-grpc_opt=module=mondee/media-edge \
  proto/avatar.proto
python -m grpc_tools.protoc -I proto \
  --python_out=avatar/proto --grpc_python_out=avatar/proto \
  proto/avatar.proto
sed -i 's/^import avatar_pb2 as avatar__pb2$/from . import avatar_pb2 as avatar__pb2/' \
  avatar/proto/avatar_pb2_grpc.py

# asr.proto is Python<->Python only (orchestrator <-> STT). Keep the STT copy
# in its generated-bindings package instead of mixing it with service code.
python -m grpc_tools.protoc -I proto \
  --python_out=orchestrator/proto --grpc_python_out=orchestrator/proto \
  proto/asr.proto
python -m grpc_tools.protoc -I proto \
  --python_out=stt/proto --grpc_python_out=stt/proto \
  proto/asr.proto
# grpc_tools emits an absolute sibling import. STT's bindings are a package,
# so make that import relative to keep `from proto import asr_pb2_grpc` valid.
sed -i 's/^import asr_pb2 as asr__pb2$/from . import asr_pb2 as asr__pb2/' \
  stt/proto/asr_pb2_grpc.py

# tts.proto is Python<->Python only (orchestrator <-> TTS).
for out in orchestrator/proto tts/proto; do
  python -m grpc_tools.protoc -I proto \
    --python_out="$out" --grpc_python_out="$out" \
    proto/tts.proto
done

# llm.proto is Python<->Python only. Stubs go to the orchestrator, legacy LLM
# service, and adaptive runtime facade/workers.
for out in orchestrator/proto llm adaptive_llm/proto; do
  python -m grpc_tools.protoc -I proto \
    --python_out="$out" --grpc_python_out="$out" \
    proto/llm.proto
done
# adaptive_llm/proto is a package; generated gRPC imports must therefore be relative.
sed -i 's/^import llm_pb2 as llm__pb2$/from . import llm_pb2 as llm__pb2/' \
  adaptive_llm/proto/llm_pb2_grpc.py
# All orchestrator bindings share one package; generated gRPC modules otherwise
# import their sibling message modules as top-level modules.
for binding in asr edge llm tts; do
  sed -i "s/^import ${binding}_pb2 as ${binding}__pb2$/from . import ${binding}_pb2 as ${binding}__pb2/" \
    "orchestrator/proto/${binding}_pb2_grpc.py"
done
# TTS bindings are also a package; keep their generated sibling import relative.
sed -i 's/^import tts_pb2 as tts__pb2$/from . import tts_pb2 as tts__pb2/' \
  tts/proto/tts_pb2_grpc.py

echo "generated: media-edge/internal/{edgepb,avatarpb}/*.go, avatar/proto/avatar_pb2*.py, orchestrator/proto/{edge,asr,llm,tts}_pb2*.py, stt/proto/asr_pb2*.py, tts/proto/tts_pb2*.py, llm/llm_pb2*.py, adaptive_llm/proto/llm_pb2*.py"
