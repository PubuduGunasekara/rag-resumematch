"""Query-time pipeline: resume in, cited gap analysis out.

Assumes scripts/build_index.py has already run — this file does NOT
rebuild the index on every call, since that would mean re-processing
124k+ postings per resume upload. If the index is empty, fail loudly
(see run_pipeline's check) rather than silently returning no matches.
"""

import os
from dataclasses import dataclass

import yaml

from resumematch.ingest.parse_resume import parse_resume
from resumematch.preprocess.query_correction import CorrectionResult, correct_query
from resumematch.preprocess.resume_sections import ResumeSection, split_resume
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.fusion import retrieve_for_resume
from resumematch.retrieval.vector_store import VectorStore
from resumematch.generation.ollama_client import OllamaClient
from resumematch.generation.gap_analysis import run_gap_analysis, GapAnalysisResult


class IndexNotBuiltError(RuntimeError):
    """Raised when the vector store is empty — run scripts/build_index.py first."""


@dataclass
class PipelineResult:
    gap_analysis: GapAnalysisResult
    # Every typo fix applied to the resume before embedding, so the UI
    # can show "pyhton -> python" instead of correcting invisibly.
    # Empty when correction was off or nothing needed fixing.
    corrections: list[CorrectionResult]
    # The resume as it was queried: one entry per section, each within
    # the embedder's token window.
    sections: list[ResumeSection]
    # The postings handed to the LLM, one chunk per posting, with
    # "matched_sections" saying which resume sections retrieved each.
    retrieved: list[dict]


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_persist_dir(config: dict) -> str:
    """Where the Chroma index lives. CHROMA_PERSIST_DIR (what
    build_index.py writes to, and what Docker sets) wins over
    config.yaml, so the indexer and the app can never silently point at
    different folders -- that would look like an empty index."""
    return os.environ.get("CHROMA_PERSIST_DIR") or config.get("chroma_persist_dir", "./chroma_db")


def run_pipeline(
    resume_path: str,
    config_path: str = "config.yaml",
    domain_vocabulary: dict[str, float] | None = None,
    apply_correction: bool = True,
) -> PipelineResult:
    """
    Input:
        resume_path: path to the uploaded resume (.pdf or .docx).
        domain_vocabulary: precomputed via build_domain_vocabulary on a
            sample of job postings — passed in rather than rebuilt here,
            since rebuilding it on every query would mean re-reading a
            posting sample on every resume upload. The Streamlit app
            builds this once at startup and reuses it.
        apply_correction: False skips typo correction even when a
            vocabulary is available (UI toggle / A-B comparison).
    """
    config = load_config(config_path)

    resume_text = parse_resume(resume_path)

    store = VectorStore(persist_dir=resolve_persist_dir(config))
    if store.count() == 0:
        raise IndexNotBuiltError(
            "The job-postings index is empty. Run `python scripts/build_index.py` first."
        )

    corrections: list[CorrectionResult] = []
    if apply_correction and domain_vocabulary is not None:
        corrected_text, considered = correct_query(resume_text, domain_vocabulary, **config["query_correction"])
        corrections = [c for c in considered if c.was_corrected]
    else:
        # Correction off, or no vocabulary yet (index not built with the
        # current build_index.py) -- skip rather than fail.
        corrected_text = resume_text

    # Must match whatever EMBEDDING_MODEL scripts/build_index.py was run
    # with -- a different model here would embed queries into a space
    # that doesn't match the index, silently returning garbage matches
    # rather than erroring.
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedder = Embedder(model_name=embedding_model)

    # One query per resume section, never the whole resume at once: the
    # embedder reads at most max_tokens (512) and silently drops the
    # rest, which cut the test resume's EXPERIENCE and SKILLS entirely.
    # Results are fused per posting with reciprocal rank fusion.
    sections = split_resume(
        corrected_text, count_tokens=embedder.count_tokens, max_tokens=embedder.max_tokens - 32
    )
    retrieved = retrieve_for_resume(
        sections,
        embedder,
        store,
        top_k=config["retrieval"]["top_k"],
        per_section_k=config["retrieval"]["per_section_k"],
        max_per_company=config["retrieval"]["max_per_company"],
    )

    client = OllamaClient()
    # The LLM gets the ORIGINAL resume text: correction exists to make
    # retrieval robust, and the model reads typos fine on its own.
    gap_analysis = run_gap_analysis(
        resume_text=resume_text,
        retrieved_chunks=retrieved,
        client=client,
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_tokens"],
        num_ctx=config["generation"]["num_ctx"],
    )
    return PipelineResult(
        gap_analysis=gap_analysis, corrections=corrections, sections=sections, retrieved=retrieved
    )
