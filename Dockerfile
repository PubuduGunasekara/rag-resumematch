# Single image for the app side (Streamlit + pipeline code). Ollama runs
# as its own container (see docker-compose.*.yml) using the official
# ollama/ollama image — no reason to reinvent that image ourselves.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
