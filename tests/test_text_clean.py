"""LLM text must not carry em dashes or emoji into the UI or the PDF."""

import pytest

from resumematch.generation.text_clean import clean_generated


@pytest.mark.parametrize("raw, clean", [
    ("Strong fit — strong ICU background", "Strong fit, strong ICU background"),
    ("Strong fit—strong ICU background", "Strong fit, strong ICU background"),
    ("Experience 2018–2021", "Experience 2018 to 2021"),
    ("Fast learner -- adaptable", "Fast learner, adaptable"),
    ("Great match 🎯 for this role ✅", "Great match for this role"),
    ("— Kubernetes", "Kubernetes"),
    ("Python, — and Go", "Python, and Go"),
])
def test_dashes_and_emoji_are_removed(raw, clean):
    assert clean_generated(raw) == clean


def test_hyphenated_words_are_kept():
    assert clean_generated("Full-time, medical-surgical, end-to-end") == "Full-time, medical-surgical, end-to-end"


def test_no_dash_characters_survive():
    out = clean_generated("A — B – C — 1–2 🙂")
    assert "—" not in out and "–" not in out
