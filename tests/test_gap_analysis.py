"""Tests for gap_analysis: the JSON parser, resume-evidence verification,
the response schema, and the retry. Written against hand-constructed
model responses so they don't need Ollama running.
"""

import json

import pytest

from resumematch.generation.gap_analysis import (
    MAX_ITEMS,
    evidence_in_resume,
    parse_response,
    response_schema,
    run_gap_analysis,
)
from resumematch.generation.ollama_client import GenerationTruncatedError


def make_chunks():
    return [
        {"job_id": "1", "section": "description", "job_posting_url": "https://x/1", "text": "Requires 3+ years Python."},
        {"job_id": "2", "section": "skills_desc", "job_posting_url": "https://x/2", "text": "Must know Kubernetes."},
    ]


RESUME = """TECHNICAL SKILLS
Languages: Java, Python, SQL
Backend: Spring Boot,
Node.js"""


def answer(matched=(), missing=(), summary="ok") -> str:
    return json.dumps({
        "matched": [{"requirement": r, "resume_quote": q, "excerpt": n} for r, q, n in matched],
        "missing": [{"requirement": r, "excerpt": n} for r, n in missing],
        "summary": summary,
    })


GOOD = answer(
    matched=[("Python programming", "Languages: Java, Python", 1),
             ("Kubernetes", "Deployed services on Kubernetes", 2)],  # quote not in RESUME
    missing=[("Kubernetes", 2)],
    summary="Good Python fit, no Kubernetes.",
)


# --- Citations resolve to the real retrieved record ---

def test_excerpt_numbers_resolve_to_retrieved_postings():
    result = parse_response(GOOD, make_chunks())
    assert result.matched[0].job_id == "1"
    assert result.matched[0].job_posting_url == "https://x/1"
    assert result.matched[0].supporting_excerpt == "Requires 3+ years Python."
    assert result.missing[0].job_id == "2"
    assert "Kubernetes" in result.overall_summary


def test_out_of_range_excerpt_raises():
    with pytest.raises(ValueError, match="out of range"):
        parse_response(answer(missing=[("Go", 99)]), make_chunks())


def test_invalid_json_raises_with_raw_response_included():
    with pytest.raises(ValueError) as exc_info:
        parse_response("not json at all", make_chunks())
    assert "not json at all" in str(exc_info.value)


def test_missing_field_raises():
    with pytest.raises(ValueError):
        parse_response(json.dumps({"matched": [], "summary": "x"}), make_chunks())  # no "missing"
    with pytest.raises(ValueError, match="Malformed item"):
        parse_response(json.dumps({"matched": [{"requirement": "x"}], "missing": []}), make_chunks())


def test_empty_lists_are_valid():
    result = parse_response(answer(summary="No overlap."), make_chunks(), resume_text=RESUME)
    assert result.matched == [] and result.missing == [] and result.unverified == []


def test_duplicate_requirements_are_collapsed():
    raw = answer(missing=[("Kubernetes", 2), ("kubernetes", 2), ("Go", 1)])
    assert [c.requirement for c in parse_response(raw, make_chunks()).missing] == ["Kubernetes", "Go"]


# --- Matches must quote the resume, verified in code ---

def test_quote_found_in_resume_is_a_verified_match():
    result = parse_response(GOOD, make_chunks(), resume_text=RESUME)
    assert [c.requirement for c in result.matched] == ["Python programming"]
    assert result.matched[0].resume_evidence == "Languages: Java, Python"


def test_quote_not_in_resume_goes_to_unverified_not_matched():
    # The real failure: a 3B model "quoted" job-posting text as resume evidence.
    result = parse_response(GOOD, make_chunks(), resume_text=RESUME)
    assert [c.requirement for c in result.unverified] == ["Kubernetes"]


def test_empty_quote_is_unverified():
    raw = answer(matched=[("Python", "  ", 1)])
    result = parse_response(raw, make_chunks(), resume_text=RESUME)
    assert result.matched == [] and len(result.unverified) == 1


def test_evidence_matching_ignores_case_punctuation_and_line_wraps():
    assert evidence_in_resume("backend: spring boot, node.js", RESUME)  # wraps across a line in RESUME
    assert not evidence_in_resume("Pyth", RESUME)  # whole words only
    assert not evidence_in_resume("   ", RESUME)


# --- The schema used for constrained decoding ---

def test_schema_caps_lists_and_bounds_excerpts_to_what_was_retrieved():
    schema = response_schema(num_excerpts=5)
    matched = schema["properties"]["matched"]
    assert matched["maxItems"] == MAX_ITEMS
    assert matched["items"]["properties"]["excerpt"] == {"type": "integer", "minimum": 1, "maximum": 5}
    assert set(matched["items"]["required"]) == {"requirement", "resume_quote", "excerpt"}
    assert schema["properties"]["missing"]["items"]["properties"]["excerpt"]["maximum"] == 5


# --- Retry ---

class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_schema_is_sent_to_the_model():
    client = FakeClient([GOOD])
    run_gap_analysis(RESUME, make_chunks(), client)
    sent = client.calls[0][1]["schema"]
    assert sent["properties"]["matched"]["items"]["properties"]["excerpt"]["maximum"] == 2


def test_unusable_answer_is_retried_with_a_reminder():
    client = FakeClient(["{not json", GOOD])
    result = run_gap_analysis(RESUME, make_chunks(), client)
    assert result.attempts == 2
    assert "could not be used" in client.calls[1][0]


def test_truncated_output_is_retried():
    client = FakeClient([GenerationTruncatedError('{"matched": [{"requ', 1024), GOOD])
    assert run_gap_analysis(RESUME, make_chunks(), client).attempts == 2


def test_gives_up_after_max_attempts_with_the_last_error():
    client = FakeClient(["garbage", "still garbage"])
    with pytest.raises(ValueError, match="still garbage"):
        run_gap_analysis(RESUME, make_chunks(), client, max_attempts=2)
