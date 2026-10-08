# vLLM tuning guide for the 4 GB GTX 1650

This guide controls the vLLM worker used by Adaptive LLM. It is written for
operators: changing a value requires no Python knowledge, but **always restart
the service and benchmark after changing one value**.

The bundled GTX 1650 profiles target small models such as Gemma 3 1B and
Qwen3 0.6B. A 4 GB card cannot use the 50-session L40S configuration from the
utilization research. Start with two to four active voice requests, not dozens.

## Where to set a value

1. Edit `adaptive_llm/.env` when running `./adaptive_llm/run.sh` locally.
2. Put a value in the shell environment for a one-off override, for example
   `LLM_MAX_NUM_SEQS=2 ./adaptive_llm/run.sh`.
3. Leave a value absent to use the selected model profile or vLLM default.

Shell values win over `.env`; `.env` values win over the runtime profile; a
runtime profile only fills values that are otherwise unset. `run.sh` must be
used, or the same variables must be exported by your service manager. Settings
are read when the worker starts, so changing `.env` never affects an already
running worker.

## Recommended starting point

```dotenv
LLM_GPU_MEMORY_UTILIZATION=0.70
LLM_MAX_NUM_BATCHED_TOKENS=512
LLM_MAX_NUM_SEQS=4
LLM_ENABLE_PREFIX_CACHING=1
LLM_ENABLE_CHUNKED_PREFILL=1
LLM_ENFORCE_EAGER=0
LLM_KV_CACHE_DTYPE=auto
LLM_DISABLE_LOG_STATS=0
VLLM_USE_V1=0
LLM_SWAP_SPACE_GIB=0
LLM_CPU_OFFLOAD_GB=0
ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS=4
ADAPTIVE_LLM_MAX_QUEUED_REQUESTS=16
LLM_MAX_MODEL_LEN=2048
LLM_MAX_TOKENS=128
```

`LLM_MAX_TOKENS=128` is recommended for a voice response. The shipped `.env`
leaves context and output limits under profile/config control; set them
explicitly if you need a reproducible test.

## Capacity and memory knobs

| Variable | Safe GTX 1650 range | What it changes internally | Behaviour when raised |
| --- | --- | --- | --- |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.60`–`0.80`; start `0.70` | vLLM sizes its model executor and KV-cache pool as this fraction of currently free VRAM. | More room for cached tokens and concurrent requests. Too high can fail startup or leave no desktop/driver headroom. Increase by `0.05` only after a stable run. |
| `LLM_MAX_MODEL_LEN` | `1024` or `2048` | Maximum prompt plus generated-token length vLLM accepts and reserves capacity for. | Longer conversations consume KV cache per active request. Halving it is the simplest way to make room for more simultaneous requests. |
| `LLM_CONTEXT_SAFETY_TOKENS` | `128` | Tokens deliberately reserved between the rendered prompt and output limit by the worker's context trimmer. | Raising it prevents last-minute context overflows but shortens usable conversation history. |
| `LLM_MAX_TOKENS` | `64`–`128` for voice | Maximum generated tokens passed to vLLM for every reply. | A smaller answer frees KV sooner and improves tail latency. It does not make individual model tokens faster. |
| `LLM_MAX_NUM_SEQS` | `2`–`4`; test `6` only after measurement | Scheduler cap for sequences processed in one iteration. | More sequences permits continuous batching and higher aggregate throughput, but divides decode capacity and increases latency/memory pressure. It is not a promise that every request starts immediately. |
| `LLM_MAX_NUM_BATCHED_TOKENS` | `256`, `512`, then `768` if stable | Per-engine-iteration token budget. It bounds prefill work and helps vLLM interleave long prompts with decoding requests. | Larger improves prompt throughput and GPU occupancy but increases activation/scratch demand and can worsen first-token latency. |
| `ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS` | Match `LLM_MAX_NUM_SEQS`, normally `4` | Facade admission slots before a request reaches vLLM. | Higher than `MAX_NUM_SEQS` creates a second queue and higher latency; it does not create GPU capacity. |
| `ADAPTIVE_LLM_MAX_QUEUED_REQUESTS` | `4`–`16` | Additional bounded requests allowed to wait at the facade. | Limits overload. Excess requests receive `RESOURCE_EXHAUSTED` rather than consuming unlimited RAM. Unary and streaming requests now use this same policy. |

## Performance knobs

| Variable | Recommended value | What it changes internally | When to change it |
| --- | --- | --- | --- |
| `LLM_ENFORCE_EAGER` | `0` | `0` lets vLLM capture and replay CUDA graphs for common execution shapes; `1` forces ordinary eager CUDA launches. | Keep `0` when startup and load tests are stable; use `1` to diagnose graph-capture OOMs or version-specific CUDA failures. CUDA graphs use some extra VRAM but often reduce CPU launch overhead. |
| `LLM_ENABLE_CHUNKED_PREFILL` | `1` | Splits long prompt processing across scheduler iterations so decode work can continue. | Keep on for mixed or interactive traffic. Set `0` only to compare latency on very short, single-user prompts. The worker forces `0` for head-size-256 models on pre-Ampere GPUs -- see below. |
| `LLM_ENABLE_PREFIX_CACHING` | `1` | Reuses KV blocks for identical token prefixes, such as the system prompt and unchanged conversation history. | It helps repeated session prefixes; it cannot help unrelated prompts. It shares the same limited KV pool, so measure TTFT before relying on it. The worker forces `0` for head-size-256 models on pre-Ampere GPUs -- see below. |
| `VLLM_USE_V1` | unset (automatic) | Selects the vLLM execution engine before vLLM is imported. | vLLM 0.8.x requires compute capability 8.0+ for V1. The worker automatically uses V0 on older GPUs such as the GTX 1650; explicitly set `0` only to force V0 on newer hardware. |
| `LLM_DISABLE_LOG_STATS` | `0` | Keeps vLLM periodic statistics and Prometheus metrics available. | Use `1` only if log/metric overhead matters more than observability. The worker metrics endpoint is `http://<machine-ip>:9114/metrics` by default. |

## Gemma 3 and MedGemma on a pre-Ampere GPU

These models use an attention head size of 256. Prefix caching and chunked
prefill both route a prompt through vLLM's Triton prefix-prefill kernel, which
asks for 72 KiB of shared memory per block at that head size; a Turing SM such
as the GTX 1650's offers only 64 KiB, so the request dies mid-generation with:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 73728, Hardware limit: 65536.
```

The worker detects this combination at startup, disables both features, and
raises `max_num_batched_tokens` to `LLM_MAX_MODEL_LEN` (vLLM rejects a smaller
batch budget once chunking is off). It logs a warning when it does. Head size
128 models such as Qwen3 need only 36 KiB and keep both features. Nothing needs
to change on an Ampere-or-newer GPU.

## Response-style knobs

These change the model's token selection, not GPU capacity. Keep them stable
while measuring hardware performance so that each benchmark has comparable
response lengths.

| Variable | Typical range / default | What it does internally | Behaviour when changed |
| --- | --- | --- | --- |
| `LLM_TEMPERATURE` | `0`–`1`; default `0.7` | Scales token probabilities before vLLM samples the next token. | `0` is more repeatable; higher values are more varied. It does not improve GPU speed. |
| `LLM_TOP_P` | `0.8`–`1.0`; default `1.0` | Keeps the smallest set of next-token candidates whose combined probability reaches this value. | Lower values usually make answers more focused. Use `1.0` to disable this filter. |
| `LLM_TOP_K` | `-1` or `20`–`100`; default `-1` | Limits sampling to the K most likely next tokens. | `-1` disables the limit. Combining a small K with low top-p can make responses repetitive. |
| `LLM_MIN_P` | `0.0`–`0.1`; default `0.0` | Drops candidates whose probability is too small relative to the best candidate. | `0` disables it. Raise carefully; it changes answer quality and style. |
| `LLM_REPEAT_PENALTY` | `1.0`–`1.15`; default `1.0` | Penalizes tokens already present in the generated sequence before sampling. | Values above `1` reduce repeated phrases; too high makes wording unnatural. |

## Precision and offload knobs

| Variable | GTX 1650 value | What it changes internally | Warning |
| --- | --- | --- | --- |
| `LLM_DTYPE` | `auto` | Weight/compute dtype selection when the model permits it. | Do not force a dtype unless the model and installed vLLM version document it. |
| `LLM_QUANTIZATION` | empty | Selects a vLLM quantization backend, for example a compatible AWQ/GPTQ checkpoint. | Do **not** set `fp8` on Turing/RTX 1650. FP8 tuning in the L40S research is for newer hardware. A non-empty value must match the model checkpoint and installed vLLM support. |
| `LLM_KV_CACHE_DTYPE` | `auto` | Data type of stored key/value attention cache. | Keep `auto` on this GPU. Alternate values can save memory but may not be supported or may affect quality. Benchmark and validate outputs before changing it. |
| `LLM_SWAP_SPACE_GIB` | `0` | Host-memory swap reservation used by vLLM for request swapping. | It can avoid rejection under pressure but is much slower than VRAM. Do not use it for interactive voice unless intentionally testing overload behaviour. |
| `LLM_CPU_OFFLOAD_GB` | `0` | Keeps part of model weights in system RAM and transfers them over PCIe for inference. | It can make a larger model start, but normally lowers token speed substantially. It is capacity recovery, not a performance optimization. |

## Runtime and observability knobs

| Variable | Recommended value | What it does internally | Warning |
| --- | --- | --- | --- |
| `LLM_LANGUAGE_MODEL_ONLY` | `1` | Prevents a vision-language model from reserving image/video encoder resources because this gRPC API carries text only. | Do not set `0` until the public request contract actually supports multimodal inputs. |
| `LLM_WORKER_METRICS_HOST` | `0.0.0.0` | Interface where the worker metrics endpoint binds. | Set to `127.0.0.1` for local-only metrics. |
| `LLM_WORKER_METRICS_PORT` | `9114` | HTTP port where the worker exposes Prometheus counters and vLLM statistics. | Keep firewall/access controls in mind when exposing it on a LAN. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Asks PyTorch to use expandable allocation segments, reducing fragmentation during changing batch shapes. | This is a PyTorch setting, not a vLLM scheduler setting. Remove it only when diagnosing allocator-specific issues. |

## Tuning procedure

1. Close other GPU-heavy applications. Start the service and check worker logs
   and the metrics endpoint.
2. Run a warm-up request, then a five-minute mixed streaming test at concurrency
   1, 2, and 4 using normal prompt lengths.
3. Record p50/p95 time-to-first-token, end-to-end latency, output tokens/sec,
   GPU utilization, VRAM usage, and errors.
4. Change one variable only. Try memory utilization first (`0.70 → 0.75 →
   0.80`), then batched tokens (`256 → 512 → 768`), then sequences (`2 → 4`).
5. Keep a value only if it improves the metric you care about without OOM,
   worker restart, rising p95 latency, or sustained facade queueing.
6. Revert the last change immediately after an OOM. A setting that barely
   starts is not a stable operating point.

## Troubleshooting

- **Fails while starting:** lower `LLM_GPU_MEMORY_UTILIZATION` to `0.65`, set
  `LLM_ENFORCE_EAGER=1`, then reduce context to `1024`.
- **Fails only under concurrent traffic:** reduce `LLM_MAX_NUM_SEQS` first,
  then `LLM_MAX_NUM_BATCHED_TOKENS`.
- **GPU is underused and no queue exists:** raise `MAX_NUM_BATCHED_TOKENS` one
  step, then test one more sequence. Do not jump to L40S-scale values.
- **GPU is full but responses are slow:** reduce context/output budget and
  sequence count; check whether prefix caching is actually hitting.
- **Queue grows:** do not simply raise facade inflight requests. First confirm
  the selected `MAX_NUM_SEQS` and GPU cache can sustain more active sequences.
