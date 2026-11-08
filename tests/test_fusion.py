"""Tests for reciprocal rank fusion across resume sections."""

from resumematch.retrieval.fusion import RRF_K, fuse_rankings


def hit(job_id: str, distance: float, chunk: int = 0) -> dict:
    return {"job_id": job_id, "distance": distance, "text": f"{job_id} chunk {chunk}",
            "section": "description", "job_posting_url": None, "title": job_id}


def test_posting_found_by_several_sections_beats_one_strong_match():
    hits = [
        ("SKILLS", [hit("strong_once", 0.10), hit("broad", 0.40)]),
        ("EXPERIENCE", [hit("broad", 0.45), hit("other", 0.50)]),
        ("PROJECTS", [hit("broad", 0.42), hit("other2", 0.50)]),
    ]
    ranked = [r["job_id"] for r in fuse_rankings(hits, top_k=3)]
    assert ranked[0] == "broad"


def test_results_are_distinct_postings_with_their_best_chunk():
    hits = [
        ("SKILLS", [hit("a", 0.30, 0), hit("a", 0.20, 1), hit("b", 0.35)]),
        ("PROJECTS", [hit("a", 0.25, 2), hit("c", 0.40)]),
    ]
    results = fuse_rankings(hits, top_k=5)
    assert [r["job_id"] for r in results].count("a") == 1
    a = next(r for r in results if r["job_id"] == "a")
    assert a["distance"] == 0.20  # closest chunk of "a" across every section


def test_repeated_chunks_of_one_posting_do_not_push_others_down():
    # "a" has 3 chunks at the top of SKILLS; "b" must still rank 2nd in
    # that list (rank 2, not rank 4), since postings are ranked, not chunks.
    hits = [("SKILLS", [hit("a", 0.1, 0), hit("a", 0.11, 1), hit("a", 0.12, 2), hit("b", 0.2)])]
    results = fuse_rankings(hits, top_k=2)
    b = next(r for r in results if r["job_id"] == "b")
    assert b["fusion_score"] == 1 / (RRF_K + 2)


def test_matched_sections_are_reported_best_rank_first():
    hits = [
        ("SKILLS", [hit("x", 0.3), hit("a", 0.31)]),   # a at rank 2
        ("PROJECTS", [hit("a", 0.2)]),                 # a at rank 1
    ]
    a = next(r for r in fuse_rankings(hits, top_k=2) if r["job_id"] == "a")
    assert a["matched_sections"] == ["PROJECTS", "SKILLS"]


def test_top_k_limits_postings():
    hits = [("S", [hit(str(i), i / 10) for i in range(10)])]
    assert len(fuse_rankings(hits, top_k=3)) == 3
