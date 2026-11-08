"""Tests for query correction. Mirrors the verification discipline used
on Assignment 3: known input, hand-checked expected output.

Direction: the RESUME is the query now (OCR/typo artifacts from PDF
extraction), corrected against a vocabulary built from a SAMPLE of
job-posting text (so tech jargon survives as "known").
"""

from resumematch.preprocess.query_correction import correct_query, build_domain_vocabulary


def test_known_typo_is_corrected():
    vocab = {"python": 0.4, "kubernetes": 0.3, "docker": 0.3}
    corrected, results = correct_query("Pyhton and Dcoker experience", vocab)
    assert "python" in corrected.lower()
    assert "docker" in corrected.lower()


def test_unknown_domain_term_is_not_mangled():
    # "faiss" needs to survive once it's IN the sample vocabulary, even
    # though it's not a generic English word — this is the guard against
    # correction "fixing" real jargon into garbage.
    vocab = {"python": 0.5, "faiss": 0.5}
    corrected, results = correct_query("faiss vector search", vocab)
    assert "faiss" in corrected.lower()


def test_build_domain_vocabulary_includes_posting_terms():
    sample_texts = [
        "Experience with Kubernetes and distributed systems design.",
        "Kubernetes orchestration and container deployment required.",
    ]
    vocab = build_domain_vocabulary(sample_texts, min_count_threshold=2)
    assert "kubernetes" in vocab


def test_build_domain_vocabulary_excludes_below_threshold():
    # "kubernetes" appears once -> below threshold=2 -> should be excluded,
    # same OOV-robustness tradeoff documented in Assignment 4's StyleGram.
    sample_texts = ["Experience with Kubernetes required."]
    vocab = build_domain_vocabulary(sample_texts, min_count_threshold=2)
    assert "kubernetes" not in vocab
