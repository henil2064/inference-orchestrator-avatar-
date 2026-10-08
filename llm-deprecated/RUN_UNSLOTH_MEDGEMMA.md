# Unsloth MedGemma 1.5 4B GGUF on llama.cpp

`server_unsloth.py` hosts `unsloth/medgemma-1.5-4b-it-GGUF` behind the same
`llm.v1.Llm/Generate` gRPC service used by the orchestrator.

This backend is intended for the 4 GB GTX 1650 dev box. It uses llama.cpp GGUF
loading with partial GPU offload instead of vLLM.

## Setup

```bash
cd llm
python3 -m venv .venv-llama
source .venv-llama/bin/activate
pip install -U pip
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
pip install -r requirements-llama.txt
cp .env.unsloth.example .env.unsloth
```

The default model file is `medgemma-1.5-4b-it-Q4_K_S.gguf`, downloaded from
Hugging Face into `llm/models/unsloth/medgemma-1.5-4b-it-GGUF`.

## Run

```bash
cd llm
source .venv-llama/bin/activate
python server_unsloth.py
```

The service listens on `:50053`, so the orchestrator can keep
`LLM_TARGET=localhost:50053`.

## Tuning on 4 GB VRAM

If startup OOMs, lower `LLM_N_GPU_LAYERS` in `.env.unsloth` from `20` to `16`,
then `12`.

If prompt processing OOMs, lower `LLM_N_BATCH=64` and `LLM_N_UBATCH=32`.

If there is spare VRAM in `nvidia-smi`, raise `LLM_N_GPU_LAYERS` gradually.
Full offload may be possible only when the display and other processes leave
enough free VRAM.

Keep `LLM_N_CTX=2048` and `LLM_MAX_TOKENS=256` for first bring-up. Raise them
only after the model is stable.
