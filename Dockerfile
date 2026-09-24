FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf_cache \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

COPY app.py api.py pipeline.py fusion_model.py test_pipeline.py README.md ./
COPY artifacts ./artifacts
COPY assets ./assets
COPY data/images ./data/images
COPY data/splits.json ./data/splits.json
COPY data/hf_cache ./data/hf_cache

EXPOSE 7860 8080
CMD ["python", "app.py"]
