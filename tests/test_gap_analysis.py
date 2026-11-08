"""Tests for gap_analysis.parse_response — the LLM-output parser, which
docs/design.md flags as the riskiest piece (local models drift from a
requested format more than a hosted frontier model would). Written
against a hand-constructed "model response" string so these don't
depend on Ollama actually being installed.
"""

import pytest

from resumematch.generation.gap_analysis import parse_response


def make_chunks():
    return [
        {"job_id": "1", "section": "description", "job_posting_url": "https://x/1", "text": "Requires 3+ years Python."},
        {"job_id": "2", "section": "skills_desc", "job_posting_url": "https://x/2", "text": "Must know Kubernetes."},
    ]


def test_parses_matched_and_missing_with_citations():
    raw = """MATCHED:
- 3+ years Python experience -- evidenced by resume, required by [1]

MISSING:
- Kubernetes experience -- not evidenced anywhere in the resume excerpts [2]

SUMMARY:
Strong Python match, missing Kubernetes experience.
"""
    result = parse_response(raw, make_chunks())

    assert len(result.matched) == 1
    assert result.matched[0].job_id == "1"
    assert result.matched[0].job_posting_url == "https://x/1"
    assert "Python" in result.matched[0].requirement

    assert len(result.missing) == 1
    assert result.missing[0].job_id == "2"

    assert "Kubernetes" in result.overall_summary


def test_missing_headers_raises_with_raw_response_included():
    with pytest.raises(ValueError) as exc_info:
        parse_response("not the right format at all", make_chunks())
    assert "not the right format at all" in str(exc_info.value)


def test_out_of_range_citation_raises():
    raw = """MATCHED:
- something -- required by [99]

SUMMARY:
n/a
"""
    with pytest.raises(ValueError) as exc_info:
        parse_response(raw, make_chunks())
    assert "out of range" in str(exc_info.value)


def test_missing_citation_marker_raises():
    raw = """MATCHED:
- something with no bracket reference at all

SUMMARY:
n/a
"""
    with pytest.raises(ValueError) as exc_info:
        parse_response(raw, make_chunks())
    assert "No [N] citation" in str(exc_info.value)
