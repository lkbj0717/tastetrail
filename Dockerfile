FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=2 \
    HOME=/home/tastetrail \
    V11_PRIVATE=/data/culture_v11 \
    V13_PRIVATE=/data/culture_v13 \
    TASTETRAIL_PRIVATE_DIR=/data \
    TASTETRAIL_CREDENTIALS=/run/secrets/tastetrail.json \
    TASTETRAIL_HOST=0.0.0.0 \
    PORT=8773 \
    PYTHONPATH=/app:/app/work/culture_v13:/app/work/culture_v11:/app/work/culture_v11/src

WORKDIR /app
COPY work/culture_v13/requirements-runtime-lock.txt /tmp/requirements-runtime-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime-lock.txt \
    && useradd --create-home --uid 10001 tastetrail \
    && mkdir -p /data /run/secrets \
    && chown tastetrail:tastetrail /data
COPY work/culture_v11 /app/work/culture_v11
COPY work/culture_v13 /app/work/culture_v13
COPY deploy /app/deploy
USER 10001:10001
EXPOSE 8773
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8773/health',timeout=3)"
ENTRYPOINT ["sh", "/app/deploy/run.sh"]
