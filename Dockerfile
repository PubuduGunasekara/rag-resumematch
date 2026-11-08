# App image (Streamlit + pipeline). Ollama runs as its own container
# (official ollama/ollama image) -- see docker-compose.*.yml.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/opt/hf

# CPU-only torch, installed BEFORE requirements.txt so sentence-transformers
# doesn't pull the default CUDA build on x86 (~2.5 GB more). The app only
# embeds a handful of short resume sections per request; the LLM, which
# does need a GPU on GCP, runs in the Ollama container.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image (~130 MB) so a container
# never needs Hugging Face at request time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Code last, so editing code doesn't invalidate the dependency layers.
COPY . .
RUN pip install --no-deps -e . && chmod +x /app/entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/app/entrypoint.sh"]
