FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib \
    LLM_MODEL_DIR=/app/adaptive_llm/models \
    LLM_MODEL_REF=Qwen/Qwen3-0.6B \
    LLM_AUTO_DOWNLOAD=1
WORKDIR /app
COPY adaptive_llm/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}" \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir nvidia-cuda-nvrtc
COPY adaptive_llm/ ./adaptive_llm/
CMD ["python", "-m", "adaptive_llm.main"]
