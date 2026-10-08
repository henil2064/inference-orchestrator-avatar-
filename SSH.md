# Datacenter SSH Runbook

This guide runs the inference pipeline inside a restricted Ubuntu instance where Docker and systemd are unavailable. Services run as foreground processes in separate SSH terminals. The persistent filesystem is /mnt/sfs.

~~~bash
REPO=/mnt/sfs/home/atharv/inference-orchestrator
BASE=/mnt/sfs/home/atharv/infra
~~~

## Persistence

Keep binaries, models, virtual environments, caches, configuration, logs, and
service data under /mnt/sfs:

~~~text
$BASE/bin
$BASE/go
$BASE/go-work
$BASE/redis
$BASE/clickhouse
$BASE/prometheus/data
$BASE/grafana
$REPO/*/.venv
$REPO/*/models
$REPO/run/logs
~~~

APT files under /usr and /etc may disappear when an instance is recreated.
The data and binaries under /mnt/sfs are the durable installation.

## Every SSH session

~~~bash
export REPO=/mnt/sfs/home/atharv/inference-orchestrator
export BASE=/mnt/sfs/home/atharv/infra

export GOROOT="$BASE/go"
export GOPATH="$BASE/go-work"
export GOMODCACHE="$GOPATH/pkg/mod"
export GOCACHE="$GOPATH/cache"
export PATH="$GOROOT/bin:$GOPATH/bin:$PATH"

NVIDIA_LIB_DIRS=$(
  find "$REPO/tts/.venv/lib/python3.12/site-packages/nvidia" \
    -type f -name '*.so*' -printf '%h\n' 2>/dev/null |
    sort -u | paste -sd:
)
if [ -n "$NVIDIA_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS"
fi
~~~

Save these commands in $BASE/env.sh if desired, then run:

~~~bash
source /mnt/sfs/home/atharv/infra/env.sh
~~~

Do not put an incompatible system CUDA/NCCL directory before the TTS virtual
environment libraries.

## One-time prerequisites

~~~bash
apt-get update
apt-get install -y \
  build-essential cmake pkg-config git curl wget ca-certificates \
  libopus-dev libopusfile-dev libgomp1 \
  python3.12 python3.12-venv python3.12-dev

nvidia-smi
nvcc --version
~~~

Use Python 3.12 and a separate virtual environment for STT, TTS, adaptive-LLM,
and orchestrator.

## Persistent Go and media-edge

Install the Go version from media-edge/go.mod under $BASE/go:

~~~bash
cd "$REPO/media-edge"
GO_VERSION=$(awk '$1 == "go" {print $2; exit}' go.mod)
mkdir -p "$BASE/src"
cd "$BASE/src"
wget "https://go.dev/dl/go$GO_VERSION.linux-amd64.tar.gz"
tar -xzf "go$GO_VERSION.linux-amd64.tar.gz" -C "$BASE"

export PATH="$BASE/go/bin:$PATH"
cd "$REPO"
./media-edge/setup_ubuntu.sh
~~~

The setup script keeps ONNX Runtime under:

~~~text
$REPO/media-edge/third_party/onnxruntime
~~~

It also uses:

~~~text
$REPO/media-edge/models/silero_vad.onnx
$REPO/media-edge/profile.json
~~~

Optionally preserve the compiled binary:

~~~bash
mkdir -p "$BASE/bin"
cp "$REPO/build/media-edge/media-edge-latest" "$BASE/bin/media-edge"
chmod +x "$BASE/bin/media-edge"
~~~

## Native Redis

Start the persistent Redis installation in its own terminal:

~~~bash
"$BASE/redis/bin/redis-server" "$BASE/redis/etc/redis.conf"
~~~

Verify:

~~~bash
"$BASE/redis/bin/redis-cli" ping
~~~

Expected: PONG. Redis data must be under $BASE/redis/data. The orchestrator
uses redis://127.0.0.1:6379/0. Optional Redis modules are not required.

## Native ClickHouse

If the instance was recreated and the service user is missing:

~~~bash
id clickhouse >/dev/null 2>&1 || \
  useradd --system --no-create-home --shell /usr/sbin/nologin clickhouse
chown -R clickhouse:clickhouse "$BASE/clickhouse"
~~~

Start it:

~~~bash
sudo -u clickhouse "$BASE/bin/clickhouse" server \
  --config-file="$BASE/clickhouse/etc/config.xml"
~~~

Verify:

~~~bash
"$BASE/bin/clickhouse" client --query 'SELECT 1'
~~~

Initialize the project schema once per data directory:

~~~bash
cd "$REPO"
"$BASE/bin/clickhouse" client --multiquery < monitoring/clickhouse/init.sql
"$BASE/bin/clickhouse" client --query 'SHOW TABLES FROM pipeline'
~~~

ClickHouse data must be under $BASE/clickhouse/data.

## Prometheus

Edit monitoring/prometheus.yml to replace Docker names with 127.0.0.1 targets.
Remove the cAdvisor job when no containers are running.

Start Prometheus:

~~~bash
mkdir -p "$BASE/prometheus/data"
"$BASE/bin/prometheus" \
  --config.file="$REPO/monitoring/prometheus.yml" \
  --storage.tsdb.path="$BASE/prometheus/data" \
  --web.listen-address=127.0.0.1:9090
~~~

Optional host metrics:

~~~bash
node_exporter --web.listen-address=127.0.0.1:9100
~~~

Verify:

~~~bash
curl http://127.0.0.1:9090/-/ready
~~~

cAdvisor is omitted because it monitors containers.

## Grafana

Grafana is installed under $BASE/grafana/app. Keep data, logs, and plugins
under $BASE/grafana. Keep dashboards and provisioning files in the repository.

The dashboard provider must point directly to:

~~~yaml
options:
  path: /mnt/sfs/home/atharv/inference-orchestrator/monitoring/grafana/dashboards
~~~

The datasource file must use Prometheus at http://127.0.0.1:9090 and ClickHouse
at host 127.0.0.1, port 8123.

Install the ClickHouse plugin once:

~~~bash
"$BASE/grafana/app/bin/grafana" cli \
  --pluginsDir="$BASE/grafana/plugins" \
  plugins install grafana-clickhouse-datasource
~~~

Start Grafana:

~~~bash
mkdir -p "$BASE/grafana"/{data,logs,plugins}
GF_PATHS_DATA="$BASE/grafana/data" \
GF_PATHS_LOGS="$BASE/grafana/logs" \
GF_PATHS_PLUGINS="$BASE/grafana/plugins" \
GF_PATHS_PROVISIONING="$REPO/monitoring/grafana/provisioning" \
"$BASE/grafana/app/bin/grafana" server \
  --homepath="$BASE/grafana/app"
~~~

Grafana checks the repository dashboard directory every 10 seconds. Changes to
dashboard JSON files are reflected without copying them.

## Environment files

Create them once:

~~~bash
cd "$REPO"
cp -n media-edge/.env.example media-edge/.env
cp -n orchestrator/.env.example orchestrator/.env
cp -n stt/.env.example stt/.env
cp -n tts/.env.example tts/.env
cp -n adaptive_llm/.env.example adaptive_llm/.env
~~~

### orchestrator/.env

~~~ini
REDIS_URL=redis://127.0.0.1:6379/0
CLICKHOUSE_ENABLED=1
CLICKHOUSE_URL=http://127.0.0.1:8123
CLICKHOUSE_DATABASE=pipeline
STT_TARGET=127.0.0.1:50052
LLM_TARGET=127.0.0.1:50053
TTS_TARGET=127.0.0.1:50054
ORCH_METRICS_HOST=127.0.0.1
ORCH_METRICS_PORT=9102
~~~

### media-edge/.env

~~~ini
EDGE_ADDR=:8443
EDGE_ORCHESTRATOR=127.0.0.1:50051
EDGE_METRICS_ADDR=127.0.0.1:9101
EDGE_NO_TLS=0
EDGE_CERT_DIR=/mnt/sfs/home/atharv/inference-orchestrator/media-edge/certs
EDGE_PROFILE=/mnt/sfs/home/atharv/inference-orchestrator/media-edge/profile.json
EDGE_ONNX_LIB=/mnt/sfs/home/atharv/inference-orchestrator/media-edge/third_party/onnxruntime/libonnxruntime.so
EDGE_VAD_MODEL=/mnt/sfs/home/atharv/inference-orchestrator/media-edge/models/silero_vad.onnx
~~~

### stt/.env

Use FP16 ONNX when the INT8 export fails with ConvInteger not implemented:

~~~ini
STT_RUNTIME=onnx
STT_ONNX_VARIANT=fp16
STT_ONNX_MODEL_DIR=/mnt/sfs/home/atharv/inference-orchestrator/stt/models/sravaani-16-onnx
STT_ONNX_PROVIDER=cpu
STT_ADDR=127.0.0.1:50052
STT_METRICS_HOST=127.0.0.1
STT_METRICS_PORT=9103
STT_CPU_THREADS=4
~~~

The FP16 model directory must contain config.json, preproc.pt, tokenizer.model,
encoder.fp16.onnx, and decoder_joint.fp16.onnx.

### tts/.env

For mixed CPU/GPU Indic-Mio GGUF execution:

~~~ini
TTS_MODEL_DIR=/mnt/sfs/home/atharv/inference-orchestrator/tts/models/indic-mio-q8/indic-mio-q8.gguf
TTS_FORMAT=gguf
TTS_PROMPT_WAV=/mnt/sfs/home/atharv/inference-orchestrator/tts/sample/harvard.wav
TTS_N_GPU_LAYERS=8
TTS_NUM_THREADS=4
TTS_DEVICE=auto
TTS_ADDR=127.0.0.1:50054
TTS_METRICS_HOST=127.0.0.1
TTS_METRICS_PORT=9104
TTS_WARMUP=1
~~~

TTS_N_GPU_LAYERS offloads part of the GGUF backbone to the GPU; remaining
layers use the CPU. Confirm the CUDA-enabled llama-cpp-python build:

~~~bash
cd "$REPO/tts"
./.venv/bin/python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
./.venv/bin/python -c 'import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'
~~~

Do not reinstall the normal CPU llama-cpp-python wheel afterward.

### adaptive_llm/.env

Use persistent model paths and loopback service addresses:

~~~ini
LLM_MODEL_DIR=/mnt/sfs/home/atharv/inference-orchestrator/adaptive_llm/models
LLM_ADDR=127.0.0.1:50053
ADAPTIVE_LLM_WORKER_ADDR=127.0.0.1:50055
LLM_METRICS_HOST=127.0.0.1
LLM_METRICS_PORT=9106
LLM_WORKER_METRICS_HOST=127.0.0.1
LLM_WORKER_METRICS_PORT=9114
~~~

Set LLM_MODEL_REF, LLM_ENGINE, HF_TOKEN, and GPU memory settings for the
selected model. Never commit HF_TOKEN.

## Python environments

Each launcher creates a persistent environment:

~~~bash
cd "$REPO"
./stt/run.sh
./tts/run.sh
./adaptive_llm/run.sh
./orchestrator/run.sh
~~~

Stop each process with Ctrl-C after dependencies and models are ready. If a
virtual environment is broken, recreate only that service's environment:

~~~bash
rm -rf "$REPO/stt/.venv-sravaani"
python3.12 -m venv --copies "$REPO/stt/.venv-sravaani"
~~~

Do not run pip install -r tts/requirements.txt after installing the custom CUDA
llama-cpp-python wheel unless you intentionally reinstall the CUDA wheel.

## Startup order

Use one SSH terminal per foreground process:

1. Redis
2. ClickHouse
3. Prometheus and Node Exporter
4. Grafana
5. Adaptive LLM
6. STT
7. TTS
8. Orchestrator
9. Media Edge

In every terminal run:

~~~bash
source /mnt/sfs/home/atharv/infra/env.sh
~~~

Then use separate terminals for:

~~~bash
# Adaptive LLM
cd "$REPO"
./adaptive_llm/run.sh

# STT
cd "$REPO"
./stt/run.sh

# TTS
cd "$REPO/tts"
./run.sh

# Orchestrator
cd "$REPO"
unset STT_TARGETS TTS_TARGETS
./orchestrator/run.sh

# Media Edge
cd "$REPO"
./media-edge/run.sh
~~~

Do not run scripts/run_pipeline.py at the same time as these individual
launchers; it starts the same services and causes port conflicts.

## Verification

~~~bash
ss -lntp | grep -E '3000|6379|8123|8443|9090|9100|9101|9102|9103|9104|9106|9114|50051|50052|50053|50054'

curl http://127.0.0.1:9101/metrics
curl http://127.0.0.1:9102/metrics
curl http://127.0.0.1:9103/metrics
curl http://127.0.0.1:9104/metrics
curl http://127.0.0.1:9106/metrics
curl http://127.0.0.1:9090/-/ready
~~~

The orchestrator may report NOT_SERVING until STT, TTS, and adaptive-LLM are
healthy. It should become SERVING after all required backends are ready.

Open:

~~~text
https://<server-ip>:8443
http://127.0.0.1:3000
http://127.0.0.1:9090
~~~

From a developer PC:

~~~bash
ssh -N \
  -L 3000:127.0.0.1:3000 \
  -L 9090:127.0.0.1:9090 \
  -L 8443:127.0.0.1:8443 \
  <server-user>@<server-address>
~~~

## Troubleshooting

- systemctl fails: expected; PID 1 is not systemd.
- STT ConvInteger error: use the FP16 ONNX model and verify all required files.
- TTS generic transformers error: test imports in backend order:

~~~bash
cd "$REPO/tts"
./.venv/bin/python -c '
from llama_cpp import Llama, llama_supports_gpu_offload
from transformers import AutoTokenizer
from miocodec import MioCodecModel, load_audio
import torch
print("TTS imports OK")
'
~~~

- TTS CUDA/NCCL error: unset LD_LIBRARY_PATH and rebuild it from
  tts/.venv/lib/python3.12/site-packages/nvidia.
- Port conflict:

~~~bash
ss -lntp | grep ':8443\|:50051\|:50052\|:50053\|:50054'
~~~

Do not start individual launchers and scripts/run_pipeline.py together.

Use tmux when services must remain alive after SSH disconnect:

~~~bash
tmux new -s inference
~~~

Detach with Ctrl-b d and reconnect with:

~~~bash
tmux attach -t inference
~~~

Manual processes must be started again if the provider recreates the instance,
but binaries, models, environments, configuration, and data under /mnt/sfs do
not need to be downloaded again.

