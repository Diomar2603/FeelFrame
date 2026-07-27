# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Bibliotecas de sistema exigidas em runtime por opencv-python/opencv-contrib-python
# e mediapipe (não têm relação com GPU/driver, são libs gráficas "headless").
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala as dependências Python antes de copiar o código para aproveitar
# o cache de camadas do Docker (só reinstala se requirements.txt mudar).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Código da aplicação. A pasta app/credentials/ NÃO é copiada para a imagem
# (ver .dockerignore) — a serviceAccountKey.json é injetada em runtime via
# Secret Manager (ver DEPLOY.md).
COPY app ./app

# Baixa os pesos do modelo de emoção do DeepFace durante o build da imagem,
# em vez de na primeira requisição em produção (evita cold start de ~1min
# e evita falha caso a instância do Cloud Run não tenha saída de rede liberada).
RUN python -c "from deepface import DeepFace; DeepFace.build_model(model_name='Emotion', task='facial_attribute')"

# Usuário não-root
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser

EXPOSE 8080

# O Cloud Run injeta a variável de ambiente PORT (padrão 8080) — o container
# precisa escutar nela. workers=1: cada worker carregaria os modelos de ML
# (TensorFlow/MediaPipe/DeepFace) de novo, multiplicando o uso de memória;
# a concorrência de I/O já é resolvida internamente pelo asyncio/ThreadPoolExecutor
# da aplicação (ver VIDEO_POOL_WORKERS). --timeout 0 desativa o timeout do
# worker gunicorn; o limite de tempo de requisição real é configurado no
# `gcloud run deploy` (--timeout), que cobre o processamento de vídeo.
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:${PORT:-8080} --timeout 0 --graceful-timeout 30"]
