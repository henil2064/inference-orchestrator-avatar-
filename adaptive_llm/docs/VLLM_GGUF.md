# Serving a GGUF checkpoint with vLLM

Selecting a GGUF model no longer forces the `llama.cpp` worker. vLLM has its
own single-file GGUF loader, and on a CUDA GPU the planner normally prefers it:
vLLM's continuous batching and paged KV cache are worth more than the kernel
gap, since its GGUF kernels are ported from llama.cpp rather than tuned the way
its own quantization paths are.

Pick `llama.cpp` instead when you need CPU-only inference, partial GPU offload,
or a GGUF split across several files — vLLM's loader takes exactly one file.

```ini
LLM_MODEL_REF=unsloth/medgemma-27b-it-GGUF
LLM_MODEL_FILE=medgemma-27b-it-Q4_K_M.gguf
LLM_ENGINE=vllm
```

The one extra requirement is a tokenizer. A GGUF carries no chat template in a
form Transformers can apply, and the same repository also supplies the config
fields the conversion drops (below), so the worker resolves one in this order:

1. `LLM_TOKENIZER`, if set.
2. A matching `model-profiles.json` entry's `tokenizer`.
3. The repository the GGUF itself records as `general.base_model.0.repo_url`.

Step 3 means swapping to another GGUF of a known architecture usually needs no
new configuration at all. It is read strictly — only an
`https://huggingface.co/<owner>/<name>` URL is accepted, and the uploader's own
`general.repo_url` is ignored, because that is set to an organization page
(`https://huggingface.co/unsloth`) and would silently resolve to the wrong
thing. If nothing resolves, the worker refuses to start rather than render
prompts against a tokenizer the model was not tuned on.

Note that these repositories are usually gated: accept the model's terms on
Hugging Face and set `HF_TOKEN`, or startup fails with a 403 naming the repo.

## What vLLM 0.8.3 cannot do unassisted

Five problems sit between a stock vLLM 0.8.3 and a working Gemma 3 GGUF. Each
is handled in `engines/common.py` and applied by `engines/vllm_worker.py` only
when the selected model is a GGUF; unit coverage is in
`tests/test_gguf_vllm_compat.py`. Four of the five fail loudly. The fifth does
not, which is why it is worth reading even if you never touch this path.

| Problem | Symptom without the fix |
| --- | --- |
| V1 cannot serve GGUF | Hard startup failure; `V1_UNSUPPORTED_QUANT` contains `gguf`, and it *raises* rather than downgrading when `VLLM_USE_V1=1` was set explicitly — which the worker does on any Ampere-or-newer GPU. `ensure_v0_for_gguf` selects V0 first. |
| Architecture name mismatch | `RuntimeError: Unknown gguf model_type: gemma3_text`. A Gemma 3 GGUF holds only the language model, so Transformers reports `gemma3_text`, while the `gguf` package spells the architecture `gemma3`. `ensure_gguf_gemma3_arch_alias` aligns the two. |
| `sliding_window` is moved aside | `AttributeError: 'Gemma3TextConfig' object has no attribute 'sliding_window'`. vLLM renames it to `interleaved_sliding_window`, but its GGUF loader builds a *Transformers* `Gemma3ForCausalLM` to enumerate parameter names, and that constructor reads the original. `ensure_gguf_gemma3_sliding_window` restores it for that one call. |
| Embedding gets no quant config | `KeyError: 'embed_tokens.qweight_type'`. vLLM's Gemma 3 builds its `VocabParallelEmbedding` without `quant_config` (its Llama implementation passes it), so the quantized `token_embd` has nowhere to land. `ensure_gguf_gemma3_quantized_embedding` binds it for the model's construction. |
| **Gemma norm weights carry a baked-in +1** | **Nothing errors.** See below. |

### The silent one

llama.cpp's HF→GGUF conversion stores every Gemma RMSNorm weight as `w + 1`,
because its own kernel computes `x * w` while Gemma's definition is
`x * (1 + w)`. Transformers undoes this when it reads a GGUF (its
`Gemma2TensorProcessor`); vLLM's GGUF loader streams tensors verbatim and has no
per-architecture equivalent, while its `GemmaRMSNorm` applies the `(1 + w)`
form. Every norm therefore ends up scaled by `(2 + w_original)`.

The model loads. The engine reports itself healthy. Generation returns fluent
English that is complete nonsense and runs to `LLM_MAX_TOKENS` every time,
because it never produces an end-of-turn token:

```
Q: What is 17 minus 9? Reply with just the number.
A: - -
-2345, 234, 2, 2, and 204, 205, 0,000,0,0000,0000, 000000000000000...   [1024 tokens]
```

`ensure_gguf_gemma_norm_offset` subtracts the 1 back, for any `gemma`/`gemma2`/
`gemma3` GGUF, decided from the file's own recorded architecture. Verified
directly against the bf16 checkpoint: each norm weight in this Q4_K_M export is
exactly its safetensors counterpart plus one.

### Config fields the conversion loses

A GGUF header cannot express everything a `config.json` does, and Transformers'
GGUF reader does not map back everything llama.cpp does write. Left at the
Transformers defaults, two of them are silently *wrong* rather than missing:

| Field | GGUF-derived | Checkpoint |
| --- | --- | --- |
| `query_pre_attn_scalar` | 256 (default) | 168 |
| `rope_scaling` | `None` | `{"factor": 8.0, "rope_type": "linear"}` |

`gguf_config_overrides` reads these from the same repository the tokenizer comes
from and passes them to vLLM as `hf_overrides`. It raises rather than falling
back: this runs at startup, the tokenizer is fetched from that repository
moments later anyway, and the alternative to a clear error is a model that
serves slightly wrong attention forever.

## Dependency note: the `gguf` override

vLLM 0.8.3 requires `gguf==0.10.0`, and that version cannot be used here at
all. Both halves of the gap are load-bearing: 0.10.0's `GGUFReader` calls
`ndarray.newbyteorder()`, removed in NumPy 2, so it fails to open any GGUF
file; and it predates Gemma 3, whose architecture the loader looks up by name.
The API vLLM actually uses (`GGUFReader`, `GGML_QUANT_SIZES`,
`get_tensor_name_map`, `GGMLQuantizationType`) is unchanged between 0.10.0 and
0.17.x, and `gguf` is pure Python, so overriding it is safe.

The override cannot live in `requirements.txt`. pip resolves the whole file at
once, sees `gguf>=0.17.1` against vLLM's `gguf==0.10.0`, and fails with
`ResolutionImpossible` — installing *nothing*, not just skipping the conflict.
It therefore sits in `requirements-overrides.txt`, which `run.sh` installs
immediately after `requirements.txt` with `--no-deps`:

```bash
python -m pip install -r adaptive_llm/requirements.txt
python -m pip install --no-deps --upgrade -r adaptive_llm/requirements-overrides.txt
```

`--no-deps` applies only to the overridden package, so everything
`requirements.txt` resolved stays as it was. `run.sh` hashes both files
together, so editing either one re-triggers the install.

If you build the venv by hand and run only the first command, GGUF support is
broken. The worker checks the installed version at startup and refuses to load
a GGUF with the second command in the error, rather than letting it fail later
inside vLLM as an unrelated NumPy byte-order `AttributeError`.

## Measured: MedGemma 27B IT, bf16 vs Q4_K_M

One A100 80 GB, vLLM 0.8.3 V0, `LLM_MAX_MODEL_LEN=9162`, `LLM_MAX_TOKENS=1024`,
`LLM_MAX_NUM_SEQS=16`, `LLM_TEMPERATURE=0`. Four arms, because CUDA graphs turn
out to matter more than the quantization does, and comparing one eager arm
against one captured arm would have credited the wrong change. Reproduce with
`tools/model_ab_bench.py` (see below); raw results in `run/bench/`.

| | bf16 eager | bf16 + CUDA graphs | Q4_K_M eager | Q4_K_M + CUDA graphs |
| --- | --- | --- | --- | --- |
| `LLM_GPU_MEMORY_UTILIZATION` | 0.80 | 0.80 | 0.80 | **0.45** |
| Weights on disk | ~54 GiB | ~54 GiB | 16.5 GiB | 16.5 GiB |
| VRAM held | 62.3 GiB | 62.4 GiB | 63.2 GiB | **35.6 GiB** |
| Startup to first request | 45 s | 69 s | 185 s | 196 s |
| TTFT mean / p95 | 74 / 81 ms | 115 / 122 ms | 91 / 194 ms | 118 / 223 ms |
| Decode mean | 22.3 tok/s | 28.2 tok/s | 21.8 tok/s | **42.5 tok/s** |
| End-to-end mean / p95 | 1.24 / 5.85 s | 1.05 / 4.83 s | 1.20 / 5.87 s | **0.69 / 3.08 s** |
| Aggregate tok/s @ 1 | 20.2 | 24.6 | 19.3 | **37.0** |
| Aggregate tok/s @ 4 | 69.5 | **93.2** | 63.5 | 73.4 |
| Aggregate tok/s @ 8 | 131.6 | **175.1** | 87.3 | 114.2 |
| Smoke checks | 18/18 | 18/18 | 18/18 | 18/18 |

### Quality

At temperature 0, the Q4_K_M arm matches bf16 on 6 of 10 single-turn answers
byte-for-byte and 4 of 6 multi-turn ones; every remaining difference is a
paraphrase of the same content. Multi-turn recall is correct on all six turns in
both arms, and the chest-pain triage and insulin-dosing refusal behave the same
way. The quantized arm's answers are byte-identical with CUDA graphs on and off
(10/10), confirming the tuning changes speed only.

```
bf16 : This sounds like a potential heart attack, which is a medical emergency. Call emergency services immediately.
Q4KM : This sounds like a potential heart attack. Call emergency services immediately.
```

### Performance

- **Turn on CUDA graphs regardless of the model.** Eager mode pays per-kernel
  launch overhead 62 times per token on this architecture. Capturing them is
  worth +26% decode on bf16 and +95% on Q4_K_M — the single largest change
  measured here, and it costs only a longer startup.
- **At one conversation at a time, Q4_K_M is the faster model**: 42.5 vs 28.2
  tok/s decode and 0.69 s vs 1.05 s mean end-to-end. Decoding a single sequence
  is memory-bandwidth bound, so a third of the weight bytes is a direct win, and
  vLLM's MMVQ kernel handles batch size 1 natively.
- **Under batching, bf16 pulls ahead**: 175 vs 114 tok/s at 8 concurrent
  requests. Past batch size 1, vLLM switches to its MMQ kernels, which are
  compute-bound and no match for bf16 tensor cores on an A100.
- **TTFT is bf16's** on the tail (122 ms vs 223 ms p95). Prefill is compute-bound
  and hits the same MMQ path.
- **Startup is ~3x slower** for the GGUF, dominated by reading and unpacking one
  16.5 GiB file; the safetensors path memory-maps twelve shards instead.
- **VRAM is the decisive difference.** The eager rows all sit near 62-63 GiB only
  because `LLM_GPU_MEMORY_UTILIZATION=0.80` lets the KV cache expand into
  whatever the weights leave behind. What actually changes is the weights, ~54
  GiB to ~17 GiB, which is what makes 0.45 a viable setting at all — and bf16
  cannot run there, because its weights alone exceed that budget. The tuned
  Q4_K_M arm serves the same model in 35.6 GiB and leaves ~44 GiB of the card
  for TTS, headroom, or a second model.

The concurrency rows are therefore not a controlled comparison: each arm runs at
the largest KV cache it can practically hold, which is the operationally honest
framing but means the Q4_K_M arm is also working with a smaller cache.

### Recommendation for this pipeline

Q4_K_M with CUDA graphs. The voice path is one turn at a time per session, which
is exactly where the quantized model wins — a third off mean response latency —
and it hands back 27 GiB of the GPU. Stay on bf16 if this box ever needs to
sustain 8+ concurrent generations, where the MMQ kernels give up more than the
bandwidth saves.

## Running the benchmark

One process per arm; each starts a clean service, waits for the model to load,
and writes a JSON result.

```bash
python -m adaptive_llm.tools.model_ab_bench --arm bf16 \
  --set LLM_MODEL_REF=google/medgemma-27b-it --set LLM_MODEL_FILE= \
  --set LLM_TEMPERATURE=0 --out adaptive_llm/run/bench/bf16.json

python -m adaptive_llm.tools.model_ab_bench --arm q4km \
  --set LLM_TEMPERATURE=0 --out adaptive_llm/run/bench/q4km.json

python -m adaptive_llm.tools.model_ab_bench --compare \
  adaptive_llm/run/bench/bf16.json adaptive_llm/run/bench/q4km.json
```

Each result keeps every generated answer verbatim, so a quantization change can
be judged on what it said and not only on tokens per second. The `expect_pass`
flags are a smoke check for an answer that collapsed into nonsense; they are not
a clinical quality metric.
