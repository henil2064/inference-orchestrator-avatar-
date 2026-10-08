FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/orchestrator
WORKDIR /app/orchestrator
COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY orchestrator/ ./
EXPOSE 50051 9102
CMD ["python", "server.py"]
