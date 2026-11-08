"""Multi-query retrieval: one query per resume section, fused per posting.

Each resume section (retrieval queries can't exceed the embedder's
512-token window -- see preprocess/resume_sections.py) retrieves its own
ranked list of chunks. Those lists are combined with Reciprocal Rank
Fusion (RRF; Cormack et al., SIGIR 2009):

    score(posting) = sum over sections s of  1 / (RRF_K + rank_s(posting))

where rank_s is the posting's rank in section s's list (1 = best) and a
section that didn't retrieve the posting contributes nothing.

Why RRF over averaging similarity scores:
  - It uses ranks only, so it doesn't care that distances from a short
    SKILLS list and a long PROJECTS paragraph sit on different scales.
  - A posting that several sections agree on (skills AND experience AND
    projects) outranks one that a single section matches very closely --
    which is what "this job fits the whole resume" means.
  - No parameters to tune beyond RRF_K = 60, the standard default.

Why rank POSTINGS, not chunks: chunk-level top-k let one long posting
(or one company's repeated boilerplate) fill every slot. Grouping by
job_id first means top_k results are top_k distinct postings, each
represented by its single best-matching chunk.

Why also cap postings per COMPANY: distinct job_ids can still be the
same job -- one staffing agency (CodeMonk) posted five ~55%-identical
"Senior Software Engineer" variants that filled half the top 10. Exact
de-duplication at index time can't catch partial copies; a per-company
cap (default 2) keeps the results about different employers.
"""

from collections import defaultdict

from resumematch.preprocess.resume_sections import ResumeSection
from resumematch.retrieval.embedder import Embedder
from resumematch.retrieval.vector_store import VectorStore

RRF_K = 60


def fuse_rankings(
    per_section_hits: list[tuple[str, list[dict]]], top_k: int, max_per_company: int | None = 2
) -> list[dict]:
    """
    Input:
        per_section_hits: [(section heading, chunk hits from
            VectorStore.query, closest first), ...]
        top_k: number of distinct postings to return.
        max_per_company: at most this many postings from one company
            (postings without a company name are never capped). None = no cap.
    Output:
        Up to top_k chunk dicts (same shape as VectorStore.query), one
        per posting -- that posting's closest chunk across all sections --
        plus "fusion_score" and "matched_sections" (headings whose query
        retrieved the posting, best-ranked first).
    """
    scores: dict[str, float] = defaultdict(float)
    best_chunk: dict[str, dict] = {}
    matched: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for heading, hits in per_section_hits:
        seen: set[str] = set()
        for hit in hits:
            job_id = hit["job_id"]
            if job_id not in best_chunk or hit["distance"] < best_chunk[job_id]["distance"]:
                best_chunk[job_id] = hit
            if job_id in seen:
                continue  # rank postings, not chunks: a posting scores once per section
            seen.add(job_id)
            rank = len(seen)
            scores[job_id] += 1.0 / (RRF_K + rank)
            matched[job_id].append((rank, heading))

    ranked = sorted(scores, key=lambda j: (-scores[j], best_chunk[j]["distance"]))
    results = []
    per_company: dict[str, int] = defaultdict(int)
    for job_id in ranked:
        if len(results) == top_k:
            break
        company = best_chunk[job_id].get("company_name")
        if company and max_per_company is not None:
            if per_company[company] >= max_per_company:
                continue
            per_company[company] += 1
        results.append({
            **best_chunk[job_id],
            "fusion_score": scores[job_id],
            "matched_sections": [heading for _, heading in sorted(matched[job_id])],
        })
    return results


def retrieve_for_resume(
    sections: list[ResumeSection],
    embedder: Embedder,
    store: VectorStore,
    top_k: int = 5,
    per_section_k: int = 30,
    max_per_company: int | None = 2,
) -> list[dict]:
    """
    Embeds every section in one batch, retrieves per_section_k chunks
    for each, and fuses them into top_k distinct postings.

    per_section_k is in CHUNKS (postings have ~2 chunks on average), and
    must be well above top_k: a posting only earns fusion credit from a
    section if it appears in that section's list at all.
    """
    if not sections:
        return []
    vectors = embedder.embed([s.text for s in sections])
    per_section_hits = [
        (section.heading, store.query(vector, top_k=per_section_k))
        for section, vector in zip(sections, vectors)
    ]
    return fuse_rankings(per_section_hits, top_k=top_k, max_per_company=max_per_company)
