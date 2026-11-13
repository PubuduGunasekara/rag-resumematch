"""Tests for query correction. Mirrors the verification discipline used
on hand-picked cases: known input, hand-checked expected output.

Direction: the RESUME is the query now (OCR/typo artifacts from PDF
extraction), corrected against a vocabulary built from a SAMPLE of
job-posting text (so tech jargon survives as "known").
"""

import pytest

from resumematch.preprocess.query_correction import build_domain_vocabulary, correct_query


def test_known_typo_is_corrected():
    vocab = {"python": 0.4, "kubernetes": 0.3, "docker": 0.3}
    corrected, results = correct_query("Pyhton and Dcoker experience", vocab)
    assert "python" in corrected.lower()
    assert "docker" in corrected.lower()


def test_unknown_domain_term_is_not_mangled():
    # "faiss" needs to survive once it's IN the sample vocabulary, even
    # though it's not a generic English word; this is the guard against
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
    # a known tradeoff of frequency-based vocabularies.
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
    vocab = build_domain_vocabulary(t for t in ["redis cache", "redis queue"])
    assert "redis" in vocab


def test_mid_line_capitalized_word_gets_no_edit2_fix():
    # Real: "Midstate Metal Forming" -> "Mistake Metal Forming" (2 edits).
    vocab = {"mistake": 0.5, "metal": 0.3, "forming": 0.2}
    corrected, _ = correct_query("Process Engineer at Midstate Metal Forming", vocab)
    assert "Midstate" in corrected


def test_mid_line_capitalized_word_still_gets_edit1_fix():
    vocab = {"tools": 0.3, "docker": 0.3, "kubernetes": 0.4}
    corrected, _ = correct_query("Tools: Docker, Kubernets", vocab)
    assert corrected.endswith("Kubernetes")


def test_line_initial_capitalized_word_keeps_edit2():
    vocab = {"kubernetes": 1.0}
    corrected, _ = correct_query("- Kuberntes clusters", vocab)  # bullet start: not a name
    assert corrected.startswith("- Kubernetes")


def test_dash_separator_is_not_a_sentence_start():
    vocab = {"mistake": 0.5, "process": 0.3, "engineer": 0.2}
    corrected, _ = correct_query("Process Engineer — Midstate Metal Forming", vocab)
    assert "Midstate" in corrected


# --- Rules from false corrections on the synthetic resumes (eval/resumes/) ---

VOCAB = {"daniel": 0.1, "marquee": 0.1, "brightline": 0.1, "payments": 0.1, "retro": 0.1, "retries": 0.1,
         "host": 0.1, "repost": 0.1, "debug": 0.1, "debugger": 0.1, "skewed": 0.1, "sew": 0.1,
         "kubernetes": 0.1, "docker": 0.1, "tools": 0.1, "built": 0.1, "clusters": 0.1,
         "engineer": 0.1, "software": 0.1, "mailboxes": 0.1, "migrated": 0.1}


def test_name_on_the_first_line_is_left_alone():
    # Real: "Daniela Marquez" -> "Daniel Marquee".
    corrected, _ = correct_query("Daniela Marquez\nSoftware Engineer", VOCAB)
    assert corrected.startswith("Daniela Marquez")


def test_first_line_that_is_not_a_name_is_still_corrected():
    corrected, _ = correct_query("Tools: Docker, Kubernets", VOCAB)
    assert corrected.endswith("Kubernetes")


def test_multi_word_proper_name_is_left_alone():
    # Real: "Brightlane Payments" -> "Brightline Payments".
    corrected, _ = correct_query("Name\nSoftware Engineer II, Brightlane Payments", VOCAB)
    assert "Brightlane Payments" in corrected


def test_capitalized_line_start_does_not_shield_the_next_word():
    corrected, _ = correct_query("Name\nBuilt Kubernets clusters", VOCAB)
    assert "Kubernetes clusters" in corrected


@pytest.mark.parametrize("word", ["retry", "rehost", "Debugged", "skew"])
def test_real_words_with_a_known_form_are_left_alone(word):
    # Real: retry -> retro, rehost -> repost, Debugged -> Debugger, skew -> sew.
    corrected, _ = correct_query(f"Name\nWe {word} it", VOCAB)
    assert word in corrected


def test_genuine_typos_are_still_corrected():
    corrected, _ = correct_query("Name\nMigrated 300 mailbxoes", VOCAB)
    assert "mailboxes" in corrected


# Corpus counts from data/processed/domain_vocab.json, as probabilities.
COUNTS = {"active": 13034, "directory": 3000, "brightline": 19, "payments": 900, "environments": 8423,
          "enviroment": 17, "retro": 47, "retries": 2, "cloud": 9000}
FREQ_VOCAB = {w: c / sum(COUNTS.values()) for w, c in COUNTS.items()}


def test_typo_in_a_product_name_is_fixed_when_the_fix_is_common():
    # "Acitve Directory" -> active (13,034 postings); "Brightlane" -> brightline (19) stays.
    corrected, _ = correct_query("Name\nAcitve Directory, Brightlane Payments", FREQ_VOCAB)
    assert "Active Directory" in corrected and "Brightlane Payments" in corrected


def test_rare_corpus_variant_does_not_shield_a_typo():
    # "enviroment" is a corpus typo (17); environments (8,423) is ~495x more frequent.
    corrected, _ = correct_query("Name\ncloud enviroments and retry logic", FREQ_VOCAB)
    assert "cloud environments" in corrected and "retry" in corrected
