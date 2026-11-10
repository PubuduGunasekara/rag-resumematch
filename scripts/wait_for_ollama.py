"""Container start-up helper: wait until Ollama answers, then make sure
OLLAMA_MODEL is pulled. The ollama/ollama container starts with NO
models, and docker-compose `depends_on` only orders start-up -- without
this, the first resume upload fails with "model not found".

Usage: python scripts/wait_for_ollama.py   (reads OLLAMA_HOST, OLLAMA_MODEL)
"""

import os
import sys
import time

import requests

host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

for _attempt in range(60):
    try:
        version = requests.get(f"{host}/api/version", timeout=5).json()["version"]
        print(f"Ollama {version} is up at {host}.")
        break
    except (requests.RequestException, KeyError, ValueError):
        time.sleep(2)
else:
    sys.exit(f"Ollama never became reachable at {host}.")

installed = {m["name"] for m in requests.get(f"{host}/api/tags", timeout=10).json().get("models", [])}
if model in installed or f"{model}:latest" in installed:
    print(f"Model {model} already present.")
else:
    print(f"Pulling {model} (first run only; ~2 GB for llama3.2:3b, ~5 GB for llama3.1:8b)...")
    with requests.post(f"{host}/api/pull", json={"model": model}, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        last = None
        for line in resp.iter_lines():
            status = line.decode(errors="replace")
            if '"error"' in status:
                sys.exit(f"Pull failed: {status}")
            if '"status"' in status and status != last and "pulling" not in status:
                print(status)
                last = status
    print(f"Model {model} ready.")
