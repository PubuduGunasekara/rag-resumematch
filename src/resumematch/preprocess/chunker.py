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
from resumematch.preprocess.boilerplate import is_boilerplate, split_sentences


@dataclass
class Chunk:
    text: str
    chunk_id: str          # f"{job_id}_{section}_{index}" — unique and traceable
    job_id: str
    section: str           # "description" or "skills_desc"
    job_posting_url: str | None
    title: str
    company_name: str | None = None


# A trailing chunk shorter than this is folded into the previous one: a
# 1-30 word fragment (13 of the first 4,180 chunks) is a noisy query
# target that can still win a retrieval slot.
MIN_CHUNK_WORDS = 40


def chunk_job_posting(
    posting: JobPosting,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    boilerplate: set[str] | None = None,
) -> list[Chunk]:
    """
    Input:
        posting: a JobPosting (from job_postings_loader).
        chunk_size: max words per chunk. 300 words is ~400 wordpieces,
            inside the embedder's 512-token window.
        chunk_overlap: approximate words of WHOLE sentences repeated at
            the start of the next chunk, so a requirement near a boundary
            is fully present in at least one chunk.
        boilerplate: frequent-sentence set from find_boilerplate; matching
            sentences (and legal EEO wording) are removed before chunking.
            None = remove legal wording only.
    Output:
        Chunks carrying job_id + job_posting_url for citations. A field
        that is entirely boilerplate produces no chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
    frequent = boilerplate or set()
    description, skills = posting.description, posting.skills_desc
    if skills and len(skills.split()) < MIN_CHUNK_WORDS:
        # A short skills list ("Azure"; "Salesforce, Apex, Agile") as its own
        # chunk is a 1-5 word retrieval target -- about half of all chunks
        # under 40 words. Fold it into the description instead.
        description = f"{description}\nSkills: {skills.strip()}"
        skills = None
    chunks: list[Chunk] = []
    for section, text in (("description", description), ("skills_desc", skills)):
        if not text:
            continue
        sentences = [s for s in split_sentences(text) if not is_boilerplate(s, frequent)]
        for index, chunk_text in enumerate(_pack_sentences(sentences, chunk_size, chunk_overlap)):
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{posting.job_id}_{section}_{index}",
                job_id=posting.job_id,
                section=section,
                job_posting_url=posting.job_posting_url,
                title=posting.title,
                company_name=posting.company_name,
            ))
    return chunks


def _words(text: str) -> int:
    return len(text.split())


def _pack_sentences(sentences: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily packs whole sentences into chunks of <= chunk_size words,
    so chunks end at sentence/bullet boundaries, never mid-sentence. The
    last sentences of each chunk (up to ~chunk_overlap words) start the
    next. A single sentence longer than chunk_size (a run-on with no
    punctuation) is the only thing ever split mid-way, on words.

    Works on sentence POSITIONS, not text, so a sentence that repeats
    within a posting is never mistaken for overlap."""
    units: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= chunk_size:
            units.append(sentence)
        else:
            stride = chunk_size - chunk_overlap
            units.extend(" ".join(words[i:i + chunk_size]) for i in range(0, len(words), stride))
    if not units:
        return []
    sizes = [len(u.split()) for u in units]

    spans: list[tuple[int, int]] = []  # [start, end) into units
    start = 0
    while True:
        end, total = start, 0
        while end < len(units) and (end == start or total + sizes[end] <= chunk_size):
            total += sizes[end]
            end += 1
        spans.append((start, end))
        if end == len(units):
            break
        # next chunk starts with whole trailing sentences as overlap...
        nxt, overlap = end, 0
        while nxt - 1 > start and overlap + sizes[nxt - 1] <= chunk_overlap:
            nxt -= 1
            overlap += sizes[nxt]
        # ...unless the next sentence wouldn't fit beside them (no progress)
        if overlap + sizes[end] > chunk_size:
            nxt = end
        start = nxt

    # Fold a short tail into the previous chunk (allowing a modest overshoot).
    if len(spans) >= 2:
        (prev_start, prev_end), (_, last_end) = spans[-2], spans[-1]
        tail_words = sum(sizes[prev_end:last_end])
        if tail_words < MIN_CHUNK_WORDS and sum(sizes[prev_start:last_end]) <= chunk_size + MIN_CHUNK_WORDS:
            spans[-2:] = [(prev_start, last_end)]

    return [" ".join(units[a:b]) for a, b in spans]
