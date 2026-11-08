"""Splits a job posting's description (+ skills_desc, if present) into
retrievable chunks with citation metadata attached.

This is the indexed side of the RAG system: every chunk must carry
enough metadata to render a clickable citation back to the exact
posting and section it came from (job_id -> job_posting_url, plus
which field the text came from) — "Verifiability" in the rubric isn't
optional, so this isn't optional either.
"""

from dataclasses import dataclass

from resumematch.ingest.job_postings_loader import JobPosting


@dataclass
class Chunk:
    text: str
    chunk_id: str          # f"{job_id}_{section}_{index}" — unique and traceable
    job_id: str
    section: str           # "description" or "skills_desc"
    job_posting_url: str | None
    title: str


def chunk_job_posting(posting: JobPosting, chunk_size: int = 300, chunk_overlap: int = 50) -> list[Chunk]:
    """
    Input:
        posting: a JobPosting (from job_postings_loader).
        chunk_size: target chunk size in tokens (whitespace-split is
            fine here, doesn't need a real tokenizer).
        chunk_overlap: overlap between consecutive chunks within the
            same field, so a requirement split across a chunk boundary
            isn't lost.
    Output:
        One or more Chunk, each carrying job_id + job_posting_url so the
        UI can render a working citation link, not just an opaque ID.

    Verify before trusting this at scale: pull 5 real postings, print
    their chunks, and read them. A chunk cut mid-bullet-point is a bug,
    not an acceptable artifact of "it technically ran."
    """
    chunks: list[Chunk] = []
    chunks.extend(_chunk_field(posting, posting.description, "description", chunk_size, chunk_overlap))
    if posting.skills_desc:
        chunks.extend(_chunk_field(posting, posting.skills_desc, "skills_desc", chunk_size, chunk_overlap))
    return chunks


def _chunk_field(posting: JobPosting, text: str, section: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    tokens = text.split()
    if not tokens:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
    stride = chunk_size - chunk_overlap

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = " ".join(tokens[start:end])
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=f"{posting.job_id}_{section}_{index}",
            job_id=posting.job_id,
            section=section,
            job_posting_url=posting.job_posting_url,
            title=posting.title,
        ))
        index += 1
        if end == len(tokens):
            break
        start += stride

    return chunks
