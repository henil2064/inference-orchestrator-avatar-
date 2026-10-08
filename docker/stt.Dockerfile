FROM python:3.12-slim

ARG STT_RUNTIME=transformers
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/stt \
    STT_MODEL_DIR=/app/stt/models/sravaani \
    SRAVAANI_MODEL=/app/stt/models/sravaani
WORKDIR /app/stt
COPY stt/requirements-sravaani.txt stt/requirements-sravaani-onnx.txt /tmp/
RUN pip install --no-cache-dir torch --index-url "${TORCH_INDEX_URL}" \
    && if [ "${STT_RUNTIME}" = "onnx" ]; then \
         pip install --no-cache-dir -r /tmp/requirements-sravaani-onnx.txt; \
       else \
         pip install --no-cache-dir -r /tmp/requirements-sravaani.txt; \
       fi
COPY stt/ ./
EXPOSE 50052 9103
CMD ["python", "server.py"]
