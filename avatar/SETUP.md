# Avatar Setup

## Prerequisites

- Linux with an NVIDIA GPU, Ampere or newer (compute capability 8.0+), and a
  current NVIDIA driver. An L40S is the intended target.
- `git` and `curl`. `setup.sh` installs `pixi` itself.
- A Hugging Face token for an account that has accepted the terms of the gated
  [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1) repository.
- About 20 GB of free disk: the pixi environment is ~12 GB, artifacts and
  engines ~4 GB.

## Install

On the GPU box, from the repository root:

```bash
HF_TOKEN=hf_... bash avatar/setup.sh
```

This clones upstream AVTR-1 at a pinned commit into `avatar/third_party/`,
installs its environment, adds this service's dependencies, downloads the
models, and builds the seven TensorRT engines on the GPU it is running on.
Re-running skips every step whose output already exists.

TensorRT engines only run on the GPU model and TensorRT version that built
them. Do not copy engines built on another card (a Kaggle T4, for example)
onto this one. `setup.sh` records the GPU it built on and rebuilds if it later finds
itself on a different one.

To use engines you built yourself on an identical GPU, point `setup.sh` at a
directory laid out as `speech2motion/*.engine` and `renderer/*.engine`:

```bash
AVATAR_ENGINES_DIR=/path/to/engines HF_TOKEN=hf_... bash avatar/setup.sh
```

### Custom portraits

Put `{avatar_id}.png` files in `avatar/portraits/` before running `setup.sh`,
then select one with `AVATAR_DEFAULT_ID`. A portrait letterboxed onto a
1920x1080 transparent canvas is known to work.

## Configure

`setup.sh` creates `avatar/.env` from `.env.example`. Important settings:

| Variable | Purpose |
| --- | --- |
| `AVATAR_ADDR` | gRPC listener, default `[::]:50056`. |
| `AVATAR_METRICS_PORT` | Prometheus metrics port, default `9105`. |
| `AVATAR_BACKEND` | `avtr1` or `synthetic`. |
| `AVATAR_DEFAULT_ID` | Portrait used when the edge does not name one, default `maria`. |
| `AVATAR_DEFAULT_BG` | Background id, default `plain_white`. |
| `AVATAR_MAX_SESSIONS` | Concurrent sessions, default `2`. |
| `AVATAR_VIDEO_CODECS` | Encoders to try in order, default `h264_nvenc,libx264`. |
| `AVATAR_VIDEO_BITRATE` | Target bitrate in bits per second, default `1500000`. |

Then tell media-edge where the service is, in `media-edge/.env`:

```bash
EDGE_AVATAR_TARGET=localhost:50056
```

## Run

```bash
bash avatar/run.sh
```

Or with the rest of the pipeline; this also points media-edge at it:

```bash
AVATAR_ENABLED=1 python3 scripts/run_pipeline.py
```

The service prints one line when it is ready:

```text
AVATAR SERVICE | READY | gRPC [::]:50056 | engine avtr-1 | 1280x720 @25 fps | encoder h264_nvenc | avatar maria | warmup 84 ms/chunk | max sessions 2
```

`warmup ... ms/chunk` is measured on this GPU. Real time needs it under 200;
the service prints a warning when it is not.

## Run without a GPU

The synthetic backend lets you test the edge's video track and the lip-sync
delay on any machine, including before the GPU box exists:

```bash
python3 -m venv avatar/.venv
avatar/.venv/bin/pip install grpcio grpcio-health-checking "protobuf>=7.35.1" prometheus-client numpy av
AVATAR_BACKEND=synthetic AVATAR_PYTHON=avatar/.venv/bin/python bash avatar/run.sh
```

Start media-edge with `EDGE_AVATAR_TARGET=localhost:50056` and open the page:
a bar should rise and fall with the agent's voice, in step with the audio.

## Verify

```bash
bash -n avatar/setup.sh avatar/run.sh
./test avatar
(cd media-edge && go test ./...)
```

For a full check, start the pipeline with the avatar enabled, open the edge's
page, and confirm the video appears, the mouth follows the voice, and both stop
together when you interrupt.

## Tuning

- `edge_avatar_late_frames_total` climbing means video is arriving after its
  audio has played. Raise `EDGE_AVATAR_RENDER_MARGIN_MS` on the edge.
- If it stays at zero, lower the margin to take latency off every reply.
- `avatar_chunks_dropped_total` climbing means the GPU cannot keep up. Lower
  `AVATAR_MAX_SESSIONS` or stop sharing the GPU.
- `avatar_chunk_render_seconds` is the number capacity planning should read.

## Operational Notes

- Rebuild media-edge after pulling these changes; the avatar client is compiled
  into it.
- The LLM and the avatar compete for the same GPU. Measure render time while
  the LLM is generating, not only at idle.
- Startup loads seven engines and the configured portraits. `run_pipeline.py`
  allows a service 180 s to become healthy before restarting it.
- After a GPU swap, driver upgrade that changes TensorRT, or upstream ref
  change, re-run `setup.sh`.
