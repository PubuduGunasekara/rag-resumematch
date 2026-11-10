"""Startup checks: is everything the pipeline needs in place?

Each problem comes with a one-line fix, so the person running the app
knows what to do instead of meeting a stack trace on their first upload.
Used by the Streamlit app (shown above the upload box) and by the API's
/api/health endpoint (shown by the Next.js site).
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from resumematch.generation.ollama_client import DEFAULT_MODEL

VOCAB_PATH = Path("data/processed/domain_vocab.json")
DETAILS_PATH = Path("data/processed/posting_details.json")


@dataclass
class Problem:
    code: str
    message: str      # one line, ends with the command to run
    blocking: bool    # True: no resume can be analyzed until it's fixed


def _index_chunks(persist_dir: str) -> int:
    if not Path(persist_dir).exists():
        return 0  # don't let the check itself create an empty index folder
    from resumematch.retrieval.vector_store import VectorStore
    try:
        return VectorStore(persist_dir=persist_dir).count()
    except Exception:  # noqa: BLE001 - a corrupt or foreign folder counts as "no index"
        return 0


def check(persist_dir: str, host: str | None = None, model: str | None = None, timeout: float = 2.0) -> list[Problem]:
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    problems = []

    if _index_chunks(persist_dir) == 0:
        problems.append(Problem("no_index", "No job postings index found. Build a small one with: "
                                            "python scripts/build_index.py --limit 2000", True))
    try:
        tags = requests.get(f"{host}/api/tags", timeout=timeout).json().get("models", [])
        names = {t.get("name") for t in tags}
        if model not in names and f"{model}:latest" not in names:
            problems.append(Problem("no_model", f"The model {model} isn't downloaded. Run: ollama pull {model}", True))
    except (requests.RequestException, ValueError):
        problems.append(Problem("model_down", f"Ollama isn't running at {host}. Start it with: ollama serve", True))

    if not VOCAB_PATH.exists():
        problems.append(Problem("no_vocab", "Typo correction is off because data/processed/domain_vocab.json is "
                                            "missing. It is written by: python scripts/build_index.py", False))
    if not DETAILS_PATH.exists():
        problems.append(Problem("no_details", "Job cards won't show location or pay. Add them with: "
                                              "python scripts/build_posting_details.py", False))
    return problems


def as_dicts(problems: list[Problem]) -> list[dict]:
    return [asdict(p) for p in problems]
