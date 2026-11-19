# Default LeakLens demo image: the API on :8000 and the Streamlit UI on :8501.
# It runs as an unprivileged user (uid 10001) and declares a healthcheck on the
# API's /health endpoint; no application path needs to be writable at runtime.
FROM python:3.11.18-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY leaklens ./leaklens

RUN pip install --no-cache-dir ".[arena,app]" \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin leaklens

USER leaklens

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "leaklens.api:app", "--host", "0.0.0.0", "--port", "8000"]
