"""Tests for gap_analysis.parse_response — the LLM-output parser, which
docs/design.md flags as the riskiest piece (local models drift from a
requested format more than a hosted frontier model would). Written
against a hand-constructed "model response" string so these don't
depend on Ollama actually being installed.
"""

import pytest

from resumematch.generation.gap_analysis import evidence_in_resume, parse_response, run_gap_analysis
from resumematch.generation.ollama_client import GenerationTruncatedError


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


# --- Verified resume evidence (MATCHED claims must quote the resume) ---

RESUME = """TECHNICAL SKILLS
Languages: Java, Python, SQL
Backend: Spring Boot,
Node.js"""

NEW_FORMAT = """MATCHED:
- Python programming | resume: "Languages: Java, Python" | [1]
- Kubernetes | resume: "Deployed services on Kubernetes" | [2]
- Backend frameworks | [1]

MISSING:
- Kubernetes | [2]

SUMMARY:
Good Python fit.
"""


def test_quote_found_in_resume_is_a_verified_match():
    result = parse_response(NEW_FORMAT, make_chunks(), resume_text=RESUME)
    assert [c.requirement for c in result.matched] == ["Python programming"]
    assert result.matched[0].resume_evidence == "Languages: Java, Python"


def test_invented_or_missing_quote_goes_to_unverified_not_matched():
    # The real failure: a 3B model "quoted" job-posting text as resume evidence.
    result = parse_response(NEW_FORMAT, make_chunks(), resume_text=RESUME)
    assert [c.requirement for c in result.unverified] == ["Kubernetes", "Backend frameworks"]
    assert result.unverified[1].resume_evidence is None


def test_evidence_matching_ignores_case_punctuation_and_line_wraps():
    assert evidence_in_resume("backend: spring boot, node.js", RESUME)  # wraps across a line in RESUME
    assert not evidence_in_resume("Pyth", RESUME)  # whole words only
    assert not evidence_in_resume("   ", RESUME)


def test_none_bullet_means_empty_list():
    raw = "MATCHED:\n- none\n\nMISSING:\n- None.\n\nSUMMARY:\nNo overlap.\n"
    result = parse_response(raw, make_chunks(), resume_text=RESUME)
    assert result.matched == [] and result.missing == [] and result.unverified == []


def test_markdown_bold_headers_are_accepted():
    raw = "**MATCHED:**\n- Python | resume: \"Java, Python\" | [1]\n\n**SUMMARY:**\nok\n"
    assert len(parse_response(raw, make_chunks(), resume_text=RESUME).matched) == 1


# --- Retry with a format reminder ---

class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_unusable_first_answer_is_retried_with_a_reminder():
    client = FakeClient(["MATCHED:\n- Python with no citation\n\nSUMMARY:\nx", NEW_FORMAT])
    result = run_gap_analysis(RESUME, make_chunks(), client)
    assert result.attempts == 2
    assert len(result.matched) == 1
    assert "could not be used" in client.prompts[1]
    assert "No [N] citation" in client.prompts[1]  # the reminder names the actual problem


def test_truncated_output_keeps_complete_lines_without_retrying():
    partial = 'MATCHED:\n- Python | resume: "Java, Python" | [1]\n- Kubern'
    client = FakeClient([GenerationTruncatedError(partial, 1024)])
    result = run_gap_analysis(RESUME, make_chunks(), client)
    assert result.truncated and result.attempts == 1
    assert [c.requirement for c in result.matched] == ["Python"]


def test_truncated_output_with_nothing_usable_is_retried():
    client = FakeClient([GenerationTruncatedError("MATCH", 1024), NEW_FORMAT])
    result = run_gap_analysis(RESUME, make_chunks(), client)
    assert result.attempts == 2 and not result.truncated


def test_gives_up_after_max_attempts_with_the_last_error():
    client = FakeClient(["garbage", "still garbage"])
    with pytest.raises(ValueError, match="still garbage"):
        run_gap_analysis(RESUME, make_chunks(), client, max_attempts=2)
