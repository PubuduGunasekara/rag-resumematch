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


# --- Guard rails. Each case below is a false positive actually observed
# --- on a clean test resume before the rule existed.

def test_url_and_email_spans_are_untouched():
    vocab = {"whitaker": 0.5, "example": 0.5}
    text = "github.com/dwhitaker-demo and dwhitaker@example.com"
    corrected, results = correct_query(text, vocab)
    assert corrected == text  # was: dwhitaker -> whitaker
    assert results == []


def test_acronyms_are_not_corrected():
    vocab = {"sbm": 0.5, "eoe": 0.5}
    corrected, _ = correct_query("NSBM EXE competition", vocab)
    assert "NSBM" in corrected and "EXE" in corrected  # was: NSBM -> SBM


def test_mixed_case_brand_names_are_not_corrected():
    vocab = {"openam": 0.5, "python": 0.5}
    corrected, _ = correct_query("Used OpenAI models", vocab)
    assert "OpenAI" in corrected  # was: OpenAI -> Openam


def test_short_words_are_not_corrected():
    vocab = {"la": 0.5, "erp": 0.5}
    corrected, _ = correct_query("Lua and ESP", vocab)
    assert corrected == "Lua and ESP"  # was: Lua -> La, ESP -> ERP


def test_edit_distance_two_needs_a_long_word():
    vocab = {"hours": 0.5, "kubernetes": 0.5}
    corrected, results = correct_query("Khoury College, kubrnetis", vocab)
    assert "Khoury" in corrected  # 6 letters, 2 edits: was Khoury -> Hours
    assert "kubernetes" in corrected  # 9 letters: missing "e" + "e"->"i"
    fixed = [r for r in results if r.was_corrected]
    assert [(r.original_token, r.edit_distance) for r in fixed] == [("kubrnetis", 2)]


def test_capitalised_typo_at_bullet_start_is_still_corrected():
    vocab = {"developed": 1.0}
    corrected, _ = correct_query("- Devloped a service", vocab)
    assert corrected == "- Developed a service"


def test_build_domain_vocabulary_accepts_a_generator():
    vocab = build_domain_vocabulary((t for t in ["redis cache", "redis queue"]))
    assert "redis" in vocab
