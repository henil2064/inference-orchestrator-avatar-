FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/tts \
    TTS_MODEL_DIR=/app/tts/models/DhVaani-0.5-fp16 \
    TTS_PROMPT_WAV=/app/tts/sample/harvard.wav
WORKDIR /app/tts
COPY tts/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch torchaudio --index-url "${TORCH_INDEX_URL}" \
    && pip install --no-cache-dir -r /tmp/requirements.txt
COPY tts/ ./
EXPOSE 50054 9104
CMD ["python", "server.py"]
