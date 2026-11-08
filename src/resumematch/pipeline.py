"""Query-time pipeline: resume in, cited gap analysis out.

Assumes scripts/build_index.py has already run — this file does NOT
rebuild the index on every call, since that would mean re-processing
124k+ postings per resume upload. If the index is empty, fail loudly
(see run_pipeline's check) rather than silently returning no matches.
"""

import os

import yaml

from resumematch.ingest.parse_resume import parse_resume
from resumematch.preprocess.query_correction import build_domain_vocabulary, correct_query
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.vector_store import VectorStore
from resumematch.generation.ollama_client import OllamaClient
from resumematch.generation.gap_analysis import run_gap_analysis, GapAnalysisResult


class IndexNotBuiltError(RuntimeError):
    """Raised when the vector store is empty — run scripts/build_index.py first."""


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(resume_path: str, config_path: str = "config.yaml", domain_vocabulary: dict[str, float] | None = None) -> GapAnalysisResult:
    """
    Input:
        resume_path: path to the uploaded resume (.pdf or .docx).
        domain_vocabulary: precomputed via build_domain_vocabulary on a
            sample of job postings — passed in rather than rebuilt here,
            since rebuilding it on every query would mean re-reading a
            posting sample on every resume upload. The Streamlit app
            builds this once at startup and reuses it.
    """
    config = load_config(config_path)

    resume_text = parse_resume(resume_path)

    store = VectorStore(persist_dir=config.get("chroma_persist_dir", "./chroma_db"))
    if store.count() == 0:
        raise IndexNotBuiltError(
            "The job-postings index is empty. Run `python scripts/build_index.py` first."
        )

    if domain_vocabulary is not None:
        corrected_text, _corrections = correct_query(
            resume_text, domain_vocabulary, min_confidence_margin=config["query_correction"]["min_confidence_margin"]
        )
    else:
        corrected_text = resume_text  # vocabulary not available (e.g. a quick dev run) — skip correction rather than fail

    # Must match whatever EMBEDDING_MODEL scripts/build_index.py was run
    # with -- a different model here would embed queries into a space
    # that doesn't match the index, silently returning garbage matches
    # rather than erroring.
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedder = Embedder(model_name=embedding_model)
    query_embedding = embedder.embed([corrected_text])[0]
    retrieved = store.query(query_embedding, top_k=config["retrieval"]["top_k"])

    client = OllamaClient()
    return run_gap_analysis(
        resume_text=resume_text,
        retrieved_chunks=retrieved,
        client=client,
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_tokens"],
    )
