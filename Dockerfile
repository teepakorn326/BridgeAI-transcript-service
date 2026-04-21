# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

# Cache directory for faster-whisper models; persists across runs if mounted.
ENV HF_HOME=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache

RUN useradd --create-home --uid 1001 app \
 && mkdir -p /app/.cache \
 && chown -R app:app /app
USER app

EXPOSE 8081

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
