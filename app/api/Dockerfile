FROM python:3.12-slim

# Installer uv depuis l'image officielle (beaucoup plus rapide que pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependances systeme minimales pour LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installer les dependances Python avec uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY main.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
