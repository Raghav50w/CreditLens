# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder
# Installs dependencies and trains the models from the CSVs in this repo, so
# the image is built from source rather than from artifacts someone uploaded.
FROM python:3.13-slim AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

COPY creditlens/ ./creditlens/
COPY datasets/ ./datasets/

ENV PYTHONPATH=/install/lib/python3.13/site-packages:/build \
    PATH=/install/bin:$PATH

RUN python -m creditlens.ml.preprocess && python -m creditlens.ml.train

# ----------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

# Non-root: the app only ever reads its artifacts.
RUN useradd --create-home --shell /usr/sbin/nologin creditlens

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/models /app/models
COPY creditlens/ /app/creditlens/
COPY static/ /app/static/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=5000

USER creditlens
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/healthz').read()"

# Render (and most hosts) inject $PORT, so bind to it rather than a fixed value.
CMD ["sh", "-c", "uvicorn creditlens.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
